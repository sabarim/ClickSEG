import argparse
import json
import random
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_json', type=str, required=True,
                        help='The path to the json file containing results of the interactive segmentation model.')
    parser.add_argument('--n_clicks', type=int, default=20,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--n_iter', type=int, default=5,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--iou_thresh', type=float, default=0.85,
                        help='IOU threshold for evaluating clicks.')

    args = parser.parse_args()

    return args


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
            obj_ids = np.random.permutation(np.arange(0, num_instances))

            scores_arr = np.array([_get_noc(ious[_id], iou_thr, max_clicks)
                                   for _id in obj_ids], dtype=np.int)

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


def main():
    args = parse_args()
    start = time.time()

    results = json.load(open(args.result_json, "r"))
    per_instance_ious = results['all_ious']
    image_level_res = results['ious_per_image'] # list[{'ious':[obj1_ious,...]*n_obj,
                                                # 'num_instances': int, 'max_clicks': int, filename:str}]
    np.random.seed(1987)

    scores = np.stack([compute_nci_metric(image_level_res, [0.85]) for _iter in range(args.n_iter)]).squeeze(-1)
    mean_nci, mean_nof, mean_nfi = scores.mean(0)
    best_nci, best_nof, best_nfi = scores.min(0)
    worst_nci, worst_nof, worst_nfi = scores.max(0)

    print(f"Mean NCI: {mean_nci}\nMean NOF: {mean_nof}\nMean NFI: {mean_nfi}")
    print(f"\nWorst NCI: {worst_nci}\nWorst NOF: {worst_nof}\nWorst NFI: {worst_nfi}")
    print(f"\nBest NCI: {best_nci}\nBest NOF: {best_nof}\nBest NFI: {best_nfi}")


if __name__ == '__main__':
    main()