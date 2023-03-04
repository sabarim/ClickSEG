from time import time

import numpy as np
import torch
import os

from pycocotools import mask

from isegm.data.sample import DSample
from isegm.inference import utils
from isegm.inference.clicker import Clicker
import shutil
import cv2
from isegm.utils.vis import add_tag



try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def print_data_stats(dataset):
    num_instances = 0
    for index in tqdm(range(len(dataset)), leave=False):
        num_instances += len(dataset.get_sample(index).objects_ids)
    print(f"Number of instances in the input data is {num_instances}")


def evaluate_dataset_multi_instance(dataset, predictor, vis = True, vis_path = './experiments/vis_val/',**kwargs):
    all_ious = []
    all_ious_per_image = []
    fused_ious = []
    avg_click_times_per_image = []
    instance_count = 0
    if vis:
        save_dir =  vis_path + dataset.name + '/'
        #save_dir = '/home/admin/workspace/project/data/logs/'+ dataset.name + '/'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        save_dir = None

    start_time = time()
    print(f"Number of images in dataset is {len(dataset)}")
    max_clicks_per_objects = kwargs['max_clicks']
    clicks_per_image = []
    failed_images = []
    per_image_times = []
    # print_data_stats(dataset)
    for index in tqdm(range(len(dataset)), leave=False):
    # for index in tqdm(range(10), leave=False):
        per_image_start_time = time()
        sample = dataset.get_sample(index)
        max_clicks_per_image = len(sample.objects_ids) * max_clicks_per_objects
        all_ious_per_image.append(
            {'ious': [], 'num_instances': len(sample.objects_ids),
             'max_clicks': max_clicks_per_image, 'filename': sample.filename,
             'pred_masks_per_object': [], 'gt_mask_per_object': []}
        )
        # if len(sample._objects) > 0:
        clicks_per_image.append(max_clicks_per_image)
        # ids = np.random.permutation(sample.objects_ids)
        ids = sample.objects_ids
        pred_fused = np.zeros_like(sample.get_object_mask(ids[0]))
        gt_fused = np.zeros_like(sample.get_object_mask(ids[0]))
        preds_per_object = []
        gt_per_object = []

        for _i, obj_id in enumerate(ids):
            if max_clicks_per_image == 0:
                break
            gt_mask = sample.get_object_mask(obj_id)
            kwargs['max_clicks'] = max_clicks_per_image
            _, sample_ious, pred_probs, click_times, masks_per_thresh = evaluate_sample(
                sample.image, gt_mask, sample.init_mask, predictor,
                sample_id=index, vis=vis, save_dir=save_dir,
                index=index, **kwargs)
            # print(f"Average time per click {np.mean(click_times)}")

            # max_clicks_per_image-=len(sample_ious)
            pred_fused[pred_probs > kwargs['pred_thr']] = _i + 1

            gt_fused[gt_mask == 1] = _i + 1
            sample_ious = sample_ious.tolist()
            all_ious.append(sample_ious)

            # dict that stores per image click counts
            all_ious_per_image[-1]['ious'].append(sample_ious)
            masks_per_thresh = {k: mask.encode(np.asfortranarray(v)) for k, v in masks_per_thresh.items()}
            for k, v in masks_per_thresh.items():
                v['counts'] = str(v['counts'], encoding = 'utf-8')
                masks_per_thresh[k] = v

            all_ious_per_image[-1]['pred_masks_per_object'].append(masks_per_thresh)

            # add gt rles
            gt_rle = mask.encode(np.asfortranarray(gt_mask.astype(np.uint8)))
            gt_rle['counts'] = str(gt_rle['counts'], encoding='utf-8')
            all_ious_per_image[-1]['gt_mask_per_object'].append(gt_rle)

            final_pred_rle = mask.encode(np.asfortranarray(pred_probs > kwargs['pred_thr']))
            final_pred_rle['counts'] = str(final_pred_rle['counts'], encoding='utf-8')
            avg_click_times_per_image.append(np.mean(click_times))
            # image_level_ious = np.concatenate((image_level_ious, sample_ious))
            # if max_clicks_per_image <= 0:
            #     break

        print(f"Processed file {sample.filename} with {len(sample.objects_ids)} instances.")

        fused_ious_per_image = utils.get_fused_iou(pred_fused, gt_fused)
        fused_ious.append(fused_ious_per_image)
        all_ious_per_image[-1]['fused_ious_per_image'] = fused_ious_per_image
        # assert len(fused_ious[-1]) == len(ids), print(fused_ious)
        elapsed_per_image_time = time()-per_image_start_time
        per_image_times.append(elapsed_per_image_time)
        instance_count += len(sample.objects_ids)
    end_time = time()
    elapsed_time = end_time - start_time
    
    print(f"Avg time per image is {np.mean(np.array(per_image_times))}")
    print(f"Average time per click {np.mean(avg_click_times_per_image)}")
    return {'all_ious': all_ious,
            'ious_per_image': all_ious_per_image,
            'fused_ious': fused_ious,
            'time': elapsed_time,
            'instance_count': instance_count}


def evaluate_dataset(dataset, predictor, vis = True, vis_path = './experiments/vis_val/',**kwargs):
    all_ious = []
    all_ious_per_image = []
    avg_click_times_per_image = []
    instance_count = 0
    if vis:
        save_dir =  vis_path + dataset.name + '/'
        #save_dir = '/home/admin/workspace/project/data/logs/'+ dataset.name + '/'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        save_dir = None

    start_time = time()
    print(f"Number of images in dataset is {len(dataset)}")
    for index in tqdm(range(len(dataset)), leave=False):
    # for index in tqdm(range(3), leave=False):
        image_level_ious = np.array([])
        sample = dataset.get_sample(index)
        # if len(sample._objects) > 0:
        for _i, obj_id in (sample.objects_ids):
            gt_mask = sample.get_object_mask(obj_id)
            _, sample_ious, pred_probs, click_times, pred_masks_per_thresh = \
                evaluate_sample(sample.image, gt_mask, sample.init_mask, predictor,
                                sample_id=index, vis=vis, save_dir=save_dir,
                                index=index, **kwargs)
        # else:
        #     _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask, sample.init_mask, predictor,
        #                                         sample_id=index, vis= vis, save_dir = save_dir,
        #                                         index = index, **kwargs)
            all_ious.append(sample_ious.tolist())
            image_level_ious = np.concatenate((image_level_ious, sample_ious))
            avg_click_times_per_image.append(np.mean(click_times))

        all_ious_per_image.append(image_level_ious)
        print(f"Processed file {sample.filename} with {len(sample.objects_ids)} instances.")
        instance_count += len(sample.objects_ids)
    end_time = time()
    elapsed_time = end_time - start_time

    print(f"Average time per click {np.mean(avg_click_times_per_image)}")

    return {'all_ious': all_ious,
            # 'ious_per_image': all_ious_per_image,
            'time': elapsed_time,
            'instance_count': instance_count}


def Progressive_Merge(pred_mask, previous_mask, y, x):
    diff_regions = np.logical_xor(previous_mask, pred_mask)
    num, labels = cv2.connectedComponents(diff_regions.astype(np.uint8))
    label = labels[y,x]
    corr_mask = labels == label
    if previous_mask[y,x] == 1:
        progressive_mask = np.logical_and( previous_mask, np.logical_not(corr_mask))
    else:
        progressive_mask = np.logical_or( previous_mask, corr_mask)
    return progressive_mask


def evaluate_sample(image, gt_mask, init_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, vis = True, save_dir = None, index = 0,  callback=None,
                    progressive_mode = True,
                    ):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    prev_mask = pred_mask
    ious_list = []
    click_times = []

    with torch.no_grad():
        predictor.set_input_image(image)
        if init_mask is not None:
            predictor.set_prev_mask(init_mask)
            pred_mask = init_mask
            prev_mask = init_mask
            num_pm = 0
        else:
            num_pm = 999

        pred_masks_per_thresh = {}
        thrs = [0.8, 0.85, 0.9]

        for click_indx in range(max_clicks):
            vis_pred = prev_mask
            clicker.make_next_click(pred_mask)
            time_per_click = time()
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if progressive_mode:
                clicks = clicker.get_clicks()
                if len(clicks) >= num_pm: 
                    last_click = clicks[-1]
                    last_y, last_x = last_click.coords[0], last_click.coords[1]
                    pred_mask = Progressive_Merge(pred_mask, prev_mask,last_y, last_x)
                    predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask,0),0)
            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            elapsed_time_per_click = time() - time_per_click
            click_times.append(elapsed_time_per_click)
            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)
            prev_mask = pred_mask

            for th in thrs:
                if iou > th and th not in pred_masks_per_thresh:
                    pred_masks_per_thresh[th] = pred_mask

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        if vis:
            if predictor.focus_roi is not None:
                focus_roi = predictor.focus_roi
                global_roi = predictor.global_roi
                clicks_list = clicker.get_clicks()
                last_y, last_x = predictor.last_y, predictor.last_x
                focus_refined = predictor.focus_refined
                focus_coarse = predictor.focus_coarse

                out_image, focus_image = vis_result_refine(image, pred_mask, gt_mask, init_mask, iou,click_indx+1,clicks_list,focus_roi, global_roi, vis_pred, last_y, last_x, focus_refined, focus_coarse)
                cv2.imwrite(save_dir+str(index)+'.png', out_image)
                cv2.imwrite(save_dir+str(index)+'_focus.png', focus_image)
                
            else:
                clicks_list = clicker.get_clicks()
                last_y, last_x = predictor.last_y, predictor.last_x
                out_image = vis_result_base(image, pred_mask, gt_mask, init_mask, iou,click_indx+1,clicks_list, vis_pred, last_y, last_x)
                cv2.imwrite(save_dir+str(index)+'.png', out_image)
        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), \
            pred_probs, np.array(click_times, dtype=np.float32), pred_masks_per_thresh


def vis_result_base(image, pred_mask, instances_mask, init_mask,  iou, num_clicks,  clicks_list, prev_prediction, last_y, last_x):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    pred_mask = pred_mask.astype(np.float32)
    prev_mask = prev_prediction.astype(np.float32)
    instances_mask = instances_mask.astype(np.float32)
    image = image.astype(np.float32)

    pred_mask_3 = np.repeat(pred_mask[...,np.newaxis],3,2)
    prev_mask_3 = np.repeat(prev_mask[...,np.newaxis],3,2)
    gt_mask_3 = np.repeat( instances_mask[...,np.newaxis],3,2  )

    color_mask_gt = np.zeros_like(pred_mask_3)
    color_mask_gt[:,:,0] = instances_mask * 255

    color_mask_pred = np.zeros_like(pred_mask_3) #+ 255
    color_mask_pred[:,:,0] = pred_mask * 255

    color_mask_prev = np.zeros_like(prev_mask_3) #+ 255
    color_mask_prev[:,:,0] = prev_mask * 255


    fusion_pred = image * 0.4 + color_mask_pred * 0.6
    fusion_pred = image * (1-pred_mask_3) + fusion_pred * pred_mask_3

    fusion_prev = image * 0.4 + color_mask_prev * 0.6
    fusion_prev = image * (1-prev_mask_3) + fusion_prev * prev_mask_3


    fusion_gt = image * 0.4 + color_mask_gt * 0.6

    color_mask_init = np.zeros_like(pred_mask_3)
    if init_mask is not None:
        color_mask_init[:,:,0] = init_mask * 255

    fusion_init = image * 0.4 + color_mask_init * 0.6
    fusion_init = image * (1-color_mask_init) + fusion_init * color_mask_init


    #cv2.putText( image, 'click num: '+str(num_clicks)+ '  iou: '+ str(round(iou,3)), (50,50),
    #            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255 ), 1   )

    for i in range(len(clicks_list)):
        click_tuple =  clicks_list[i]

        if click_tuple.is_positive:
            color = (0,0,255)
        else:
            color = (0,255,0)

        coord = click_tuple.coords
        x,y = coord[1], coord[0]
        if x < 0 or y< 0:
            continue
        cv2.circle(fusion_pred,(x,y),4,color,-1)
        #cv2.putText(fusion_pred, str(i+1), (x-10, y-10),  cv2.FONT_HERSHEY_COMPLEX, 0.6 , color,1 )

    cv2.circle(fusion_pred,(last_x,last_y),2,(255,255,255),-1)
    image = add_tag(image, 'nclicks:'+str(num_clicks)+ '  iou:'+ str(round(iou,3)))
    fusion_init = add_tag(fusion_init,'init mask')
    fusion_pred = add_tag(fusion_pred,'pred')
    fusion_gt = add_tag(fusion_gt,'gt')
    fusion_prev = add_tag(fusion_prev,'prev pred')

    h,w = image.shape[0],image.shape[1]
    if h < w:
        out_image = cv2.hconcat([image.astype(np.float32),fusion_init.astype(np.float32),fusion_pred.astype(np.float32), fusion_gt.astype(np.float32),fusion_prev.astype(np.float32)])
    else:
        out_image = cv2.hconcat([image.astype(np.float32),fusion_init.astype(np.float32), fusion_pred.astype(np.float32), fusion_gt.astype(np.float32),fusion_prev.astype(np.float32)])

    return out_image


def vis_result_refine(image, pred_mask, instances_mask, init_mask,  iou, num_clicks,  clicks_list, focus_roi, global_roi, prev_prediction, last_y, last_x, focus_refined, focus_coarse):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    pred_mask = pred_mask.astype(np.float32)
    prev_mask = prev_prediction.astype(np.float32)
    instances_mask = instances_mask.astype(np.float32)
    focus_refined = focus_refined.astype(np.float32)
    focus_coarse = focus_coarse.astype(np.float32)
    image = image.astype(np.float32)

    pred_mask_3 = np.repeat(pred_mask[...,np.newaxis],3,2)
    prev_mask_3 = np.repeat(prev_mask[...,np.newaxis],3,2)
    gt_mask_3 = np.repeat( instances_mask[...,np.newaxis],3,2  )
    focus_refined_3 = np.repeat( focus_refined[...,np.newaxis],3,2  )
    focus_coarse_3 = np.repeat( focus_coarse[...,np.newaxis],3,2  )

    color_mask_gt = np.zeros_like(pred_mask_3)
    color_mask_gt[:,:,0] = instances_mask * 255

    color_mask_pred = np.zeros_like(pred_mask_3) #+ 255
    color_mask_pred[:,:,0] = pred_mask * 255

    color_mask_prev = np.zeros_like(prev_mask_3) #+ 255
    color_mask_prev[:,:,0] = prev_mask * 255


    fusion_pred = image * 0.4 + color_mask_pred * 0.6
    fusion_pred = image * (1-pred_mask_3) + fusion_pred * pred_mask_3

    fusion_prev = image * 0.4 + color_mask_prev * 0.6
    fusion_prev = image * (1-prev_mask_3) + fusion_prev * prev_mask_3


    fusion_gt = image * 0.4 + color_mask_gt * 0.6

    color_mask_init = np.zeros_like(pred_mask_3)
    if init_mask is not None:
        color_mask_init[:,:,0] = init_mask * 255

    fusion_init = image * 0.4 + color_mask_init * 0.6
    fusion_init = image * (1-color_mask_init) + fusion_init * color_mask_init


    #cv2.putText( image, 'click num: '+str(num_clicks)+ '  iou: '+ str(round(iou,3)), (50,50),
    #            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255 ), 1   )


    for i in range(len(clicks_list)):
        click_tuple =  clicks_list[i]

        if click_tuple.is_positive:
            color = (0,0,255)
        else:
            color = (0,255,0)

        coord = click_tuple.coords
        x,y = coord[1], coord[0]
        if x < 0 or y< 0:
            continue
        cv2.circle(fusion_pred,(x,y),4,color,-1)
        #cv2.putText(fusion_pred, str(i+1), (x-10, y-10),  cv2.FONT_HERSHEY_COMPLEX, 0.6 , color,1 )

    cv2.circle(fusion_pred,(last_x,last_y),2,(255,255,255),-1)

    y1,y2,x1,x2 = focus_roi
    cv2.rectangle(image, (x1+1,y1+1), (x2-1,y2-1), (0,255,0), 1)

    y1,y2,x1,x2 = global_roi
    cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 1)

    h,w = image.shape[0],image.shape[1]
    image = add_tag(image, 'nclicks:'+str(num_clicks)+ '  iou:'+ str(round(iou,3)))
    fusion_init = add_tag(fusion_init,'init mask')
    fusion_pred = add_tag(fusion_pred,'pred')
    fusion_gt = add_tag(fusion_gt,'gt')
    fusion_prev = add_tag(fusion_prev,'prev pred')
    focus_coarse_3 = add_tag(focus_coarse_3, 'focus coarse')
    focus_refined_3 = add_tag(focus_refined_3, 'focus refined')
    if h < w:
        out_image = cv2.hconcat([image.astype(np.float32),fusion_init.astype(np.float32),fusion_pred.astype(np.float32), fusion_gt.astype(np.float32),fusion_prev.astype(np.float32)])
    else:
        out_image = cv2.hconcat([image.astype(np.float32),fusion_init.astype(np.float32), fusion_pred.astype(np.float32), fusion_gt.astype(np.float32),fusion_prev.astype(np.float32)])
    
    focus_image = cv2.hconcat( [focus_coarse_3, focus_refined_3] )
    return out_image, focus_image
