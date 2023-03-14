from copy import deepcopy
from time import time

import numpy as np
import torch
from pycocotools import mask
from tqdm import tqdm

from isegm.inference import utils
from isegm.inference.clicker import Clicker
from isegm.inference.evaluation import Progressive_Merge


def get_obj_id(running_ious, thresh, strategy="best"):
    """
    :running_ious - the updated iou buffer
    :pool - the available iou pool
    :thresh - iou threshold
    :strategy - one of [best, worst, random]
    """
    if strategy == "random":
        obj_ids_ordered = np.random.permutation(np.arange(len(running_ious)))
    else:
        obj_ids_ordered = np.argsort(running_ious) \
            if strategy =="worst" else np.argsort(running_ious)[::-1]

    for _id in obj_ids_ordered:
        if running_ious[_id] < thresh:
                return _id

    return None


def get_fused_iou(pred_masks, gt_masks):
    #     fuse masks
    fused_preds = np.zeros_like(gt_masks[0])
    fused_gt = np.zeros_like(gt_masks[0])
    pixel_areas = [_m.sum() for _m in pred_masks]
    # smallest object comes in the front
    ids = np.argsort(pixel_areas)[::-1]
    for _iter, _id in enumerate(ids):
        obj_mask = pred_masks[_id]
        obj_gt_mask = gt_masks[_id]

        fused_preds[obj_mask != 0] = _id + 1
        fused_gt[obj_gt_mask != 0] = _id + 1

    return utils.get_fused_iou(fused_preds, fused_gt)


def get_fused_iou_argmax(pred_probs, gt_masks):
    #     fuse masks
    pred_probs = np.stack(pred_probs)
    bg_prob = 1-(np.max(pred_probs,axis=0))
    pred_probs = np.concatenate((bg_prob[None], pred_probs), axis=0)
    fused_preds = np.argmax(pred_probs, axis=0)

    gt_fused = np.stack(gt_masks)
    gt_bg = (np.sum(gt_fused, axis=0) == 0).astype(gt_fused.dtype)
    gt_fused = np.concatenate((gt_bg[None], gt_fused), axis=0)
    gt_fused = np.argmax(gt_fused, axis=0)

    return utils.get_fused_iou(fused_preds, gt_fused)


def dynamite_evaluation(dataset, predictor, max_iou_thr, min_clicks, max_clicks, strategy, vis=True,
                        fuse_every_step=True):
    avg_click_times_per_image = []
    instance_count = 0

    start_time = time()
    print(f"Number of images in dataset is {len(dataset)}")
    max_clicks_per_objects = max_clicks
    clicks_per_image = []
    failed_images = []
    per_image_times = []
    ious_objects_per_interaction = {}
    fused_ious_dict = {}
    pred_masks_dict = {}
    gt_masks_dict = {}
    result_dict = {}
    # print_data_stats(dataset)
    for index in tqdm(range(len(dataset)), leave=False):
    # for index in tqdm(range(10), leave=False):
        per_image_start_time = time()
        sample = dataset.get_sample(index)
        num_instance = len(sample.objects_ids)
        max_clicks_per_image = num_instance * max_clicks_per_objects
        clicks_per_image.append(max_clicks_per_image)
        ids = sample.objects_ids
        all_ious_per_image = []
        clickers = []
        pred_masks = {}
        pred_probs = {}
        gt_masks = {}
        with torch.no_grad():
            predictor.set_input_image(sample.image)
            for t_step in range((max_clicks_per_image - num_instance) + 1):
                if t_step == 0:
                    ious = []
                    for obj_id in ids:
                        new_iou, probs, obj_mask, clicker = evaluate_sample(
                            predictor=predictor, gt_mask=sample.get_object_mask(obj_id),
                            pred_mask=np.zeros_like(sample.get_object_mask(obj_id)),
                            clicker=Clicker(gt_mask=sample.get_object_mask(obj_id)))
                        ious.append(new_iou[0])
                        pred_masks[obj_id] = obj_mask
                        pred_probs[obj_id] = probs
                        gt_masks[obj_id] = sample.get_object_mask(obj_id)
                        clickers.append(clicker)
                else:
                    selected_obj_id = get_obj_id(all_ious_per_image[t_step-1], thresh=max_iou_thr,
                                                 strategy=strategy)
                    ious = deepcopy(all_ious_per_image[-1])
                    new_iou, probs, obj_mask, clicker = evaluate_sample(
                        predictor=predictor, gt_mask=sample.get_object_mask(selected_obj_id),
                        pred_mask=pred_masks[selected_obj_id], clicker=clickers[selected_obj_id])
                    # copy over the iou from previous iteration
                    # all_ious_per_image[t_step] = all_ious_per_image[t_step-1]
                    ious[selected_obj_id] = new_iou[0]
                    # update the iou for selected object
                    # all_ious_per_image.append(ious)
                    pred_masks[selected_obj_id] = obj_mask
                    pred_probs[selected_obj_id] = probs
                    clickers[selected_obj_id] = clicker

                if fuse_every_step:
                    # fused_ious = get_fused_iou(list(pred_masks.values()), list(gt_masks.values()))
                    fused_ious = get_fused_iou_argmax(list(pred_probs.values()), list(gt_masks.values()))
                    all_ious_per_image.append(fused_ious)
                else:
                    all_ious_per_image.append(ious)

                if np.all([_iou >= max_iou_thr for _iou in all_ious_per_image[t_step]]):
                    break

        # all_ious_per_image = all_ious_per_image[:t_step]
        # fused_ious = get_fused_iou(list(pred_masks.values()), list(gt_masks.values()))
        fused_ious = get_fused_iou_argmax(pred_probs.values(), gt_masks.values())
        pred_masks_rle = []
        for _p in pred_masks.values():
            _rle = mask.encode(np.asfortranarray(_p))
            _rle['counts'] = str(_rle['counts'], encoding='utf-8')
            pred_masks_rle.append(_rle)

        gt_masks_rle = []
        for _gt in gt_masks.values():
            _rle = mask.encode(np.asfortranarray(_gt.astype(np.uint8)))
            _rle['counts'] = str(_rle['counts'], encoding='utf-8')
            gt_masks_rle.append(_rle)
        print(f"Processed file {sample.filename} with {len(sample.objects_ids)} instances.")

        # fused_ious_per_image = utils.get_fused_iou(pred_fused, gt_fused)
        # fused_ious.append(fused_ious_per_image)
        # all_ious_per_image[-1]['fused_ious_per_image'] = fused_ious_per_image
        # assert len(fused_ious[-1]) == len(ids), print(fused_ious)
        elapsed_per_image_time = time() - per_image_start_time
        per_image_times.append(elapsed_per_image_time)
        instance_count += len(sample.objects_ids)

        ious_objects_per_interaction[sample.filename] = all_ious_per_image
        fused_ious_dict[sample.filename] = fused_ious
        pred_masks_dict[sample.filename] = pred_masks_rle
        gt_masks_dict[sample.filename] = gt_masks_rle

    result_dict={'ious_objects_per_interaction': ious_objects_per_interaction,
                 'final_fused_ious': fused_ious_dict,
                 'final_pred_masks': pred_masks_dict,
                 'gt_masks': gt_masks_dict}

    end_time = time()
    elapsed_time = end_time - start_time

    print(f"Avg time per image is {np.mean(np.array(per_image_times))}")
    print(f"Average time per click {np.mean(avg_click_times_per_image)}")
    return result_dict


def evaluate_sample(gt_mask, predictor, pred_mask, clicker,
                    pred_thr=0.49, progressive_mode=True,
                    ):
    # clicker = Clicker(gt_mask=gt_mask)
    prev_mask = pred_mask
    ious_list = []
    click_times = []

    num_pm = 999

    clicker.make_next_click(pred_mask)
    time_per_click = time()
    pred_probs = predictor.get_prediction(clicker)
    pred_mask = pred_probs > pred_thr

    if progressive_mode:
        clicks = clicker.get_clicks()
        if len(clicks) >= num_pm:
            last_click = clicks[-1]
            last_y, last_x = last_click.coords[0], last_click.coords[1]
            pred_mask = Progressive_Merge(pred_mask, prev_mask, last_y, last_x)
            predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask, 0), 0)

    elapsed_time_per_click = time() - time_per_click
    click_times.append(elapsed_time_per_click)
    iou = utils.get_iou(gt_mask, pred_mask)
    ious_list.append(iou)
    prev_mask = pred_mask

    return np.array(ious_list, dtype=np.float32), pred_probs, pred_mask, clicker
