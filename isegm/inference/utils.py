from datetime import timedelta
from pathlib import Path

import torch
import numpy as np

from isegm.data.datasets import GrabCutDataset, BerkeleyDataset, DavisDataset, SBDEvaluationDataset, PascalVocDataset, \
    Davis585Dataset, COCOMValDataset, CocoDataset, Davis2017Dataset, SBDDataset
from isegm.data.datasets.coco import CocoValDataset
from isegm.data.datasets.cocomval import COCOMValMultiDataset

from isegm.utils.serialization import load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, **kwargs)
        models = [load_single_is_model(x, device, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, **kwargs)


def load_single_is_model(state_dict, device, **kwargs):
    #print(state_dict['config'], **kwargs )
    model = load_model(state_dict['config'], **kwargs)
    model.load_state_dict(state_dict['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_dataset(dataset_name, cfg):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH)
    elif dataset_name == 'DAVIS17':
        dataset = Davis2017Dataset(cfg.DAVIS17_PATH, images_dir_name="JPEGImages/480p",
                                   masks_dir_name="Annotations/480p")
    elif dataset_name == 'DAVIS17Val':
        dataset = Davis2017Dataset(cfg.DAVIS17_PATH, images_dir_name="JPEGImages/480p",
                                   masks_dir_name="Annotations/480p", imset_file="ImageSets/2017/val.txt")
    elif dataset_name == 'SBD':
        dataset = SBDDataset(cfg.SBD_PATH, split='val')
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train')
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(cfg.PASCALVOC_PATH, split='val')
    elif dataset_name == 'COCO_MVal':
        dataset = COCOMValDataset(cfg.COCO_MVAL_PATH)
    elif dataset_name == "COCO_MVal_Multi":
        dataset = COCOMValMultiDataset(dataset_path=cfg.COCO_MVAL_PATH, coco_path=cfg.COCO_PATH)
    elif dataset_name == 'COCO':
        dataset = CocoDataset(cfg.COCO_PATH, split="val2017")
    elif dataset_name == 'COCOVal':
        dataset = CocoValDataset(cfg.COCO_PATH, split="val2017")
    elif dataset_name == 'D585_SP':
        dataset = Davis585Dataset(cfg.DAVIS585_PATH, init_mask_mode='sp')
    elif dataset_name == 'D585_STM':
        dataset = Davis585Dataset(cfg.DAVIS585_PATH, init_mask_mode='stm')
    elif dataset_name == 'D585_ZERO':
        dataset = Davis585Dataset(cfg.DAVIS585_PATH, init_mask_mode='zero')
    else:
        dataset = None
    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = np.array(iou_arr) >= iou_thr
        noc = np.argmax(vals) + 1 if np.any(vals) else max_clicks
        return min(noc, max_clicks)

    noc_list = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int)

        score = scores_arr.mean()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        over_max_list.append(over_max)

    return noc_list, over_max_list


def compute_nci_metric(result_per_image, iou_thrs):
    def _get_noc(iou_arr, iou_thr, max_clicks_per_image):
        vals = np.array(iou_arr) >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks_per_image

    nci_list = np.ones((len(result_per_image), len(iou_thrs))) * -1
    nof_objects = []
    nof_images = []
    for _i, _result_dict in enumerate(result_per_image):
        nof_objects_per_thresh = []
        nof_images_per_thresh = []
        for _j, iou_thr in enumerate(iou_thrs):
            num_instances = _result_dict['num_instances']
            max_clicks = _result_dict['max_clicks']
            ious = _result_dict['ious']
            scores_arr = np.array([_get_noc(iou_arr, iou_thr, max_clicks)
                                   for iou_arr in ious], dtype=np.int)

            cum_scores = np.cumsum(scores_arr)
            num_failed_objects = (cum_scores >= max_clicks).sum()
            num_success = (cum_scores < max_clicks).sum()
            score = scores_arr[:num_success].mean() if num_success > 0 else max_clicks

            nci_list[_i, _j] = score
            # failed_objects = over_max + (num_instances - len(ious))
            nof_objects_per_thresh.append(num_failed_objects)
            nof_images_per_thresh.append(int(num_failed_objects > 0))
        nof_objects.append(np.array(nof_objects_per_thresh))
        nof_images.append(np.array(nof_images_per_thresh))
    return nci_list.mean(axis=0), np.stack(nof_objects).sum(axis=0), np.stack(nof_images).sum(axis=0)


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time,
                      n_clicks=20, model_name=None, instance_count=None):
    table_header = (f'|{"Pipeline":^13}|{"Dataset":^11}|{"#Instances":^9}|'
                    f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|'
                    f'{">="+str(n_clicks)+"@85%":^9}|{">="+str(n_clicks)+"@90%":^9}|'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|{instance_count:^9}'
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{over_max_list[1]:^9}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row


def get_nic_results_table(nci_list, nof_images, nof_objects, brs_type, dataset_name, mean_spc, elapsed_time,
                          model_name=None, instance_count=None):
    table_header = (f'|{"Pipeline":^13}|{"Dataset":^11}|{"#Instances":^9}|'
                    f'{"NCI@80%":^9}|{"NCI@85%":^9}|{"NCI@90%":^9}| {"NOF Images@85%":^9} | {"NOF Objects@85%":^9}'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|{instance_count:^9}'
    table_row += f'{nci_list[0]:^9.2f}|'
    table_row += f'{nci_list[1]:^9.2f}|' if len(nci_list) > 1 else f'{"?":^9}|'
    table_row += f'{nci_list[2]:^9.2f}|' if len(nci_list) > 2 else f'{"?":^9}|'
    table_row += f'{nof_images[1]:^9}|' if len(nof_images) > 1 else f'{"?":^9}|'
    table_row += f'{nof_objects[1]:^9}|' if len(nof_objects) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row
