import argparse
import json
import random
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_json', type=str, required=True,
                        help='The path to the json file containing results of the interactive segmentation model.')
    parser.add_argument('--n_clicks', type=int, default=10,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--n_iter', type=int, default=5,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--iou_thresh', type=float, default=0.85,
                        help='IOU threshold for evaluating clicks.')

    args = parser.parse_args()

    return args


def get_results_table(nci, nfo, nfi, iou, eval_type, model_name=None, instance_count=None):
    table_header = (f'|{"Eval Type":^13}|{"#Instances":^9}|'
                    f'{"NCI@85%":^9}'
                    f'{"NFO@85%":^9}|'
                    f'{"NFI@85%":^9}|'
                    f'{"IOU@85%":^9}')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    table_row = f'|{eval_type:^13}|{instance_count:^9}|'
    table_row += f'{nci:^9.2f}|'
    table_row += f'{nfo:^9.2f}|'
    table_row += f'{nfi:^9.2f}|'
    table_row += f'{iou:^9.2f}|'

    return header, table_row


def compute_nci_metric(result_per_image, iou_thrs, object_ordering='random', n_clicks_per_object=10):
    def _get_noc(iou_arr, iou_thr, max_clicks_per_image):
        vals = np.array(iou_arr) >= iou_thr
        noc = np.argmax(vals) + 1 if np.any(vals) else max_clicks_per_image
        iou_at_noc = np.array(iou_arr)[vals][0] if np.any(vals) else 0.
        return noc, iou_at_noc

    nci_list = np.ones((len(result_per_image), len(iou_thrs))) * -1
    nof_objects = []
    nof_images = []
    avg_iou = []
    total_instance_count = 0
    for _i, _result_dict in enumerate(result_per_image):
        nof_objects_per_thresh = []
        nof_images_per_thresh = []
        avg_iou_per_thresh = []
        for _j, iou_thr in enumerate(iou_thrs):
            num_instances = _result_dict['num_instances']
            total_instance_count+=num_instances
            max_clicks = _result_dict['max_clicks']
            assert max_clicks == num_instances*n_clicks_per_object
            ious = _result_dict['ious']
            obj_ids = np.random.permutation(np.arange(0, num_instances))

            scores_arr = np.array([_get_noc(ious[_id], iou_thr, max_clicks)
                                   for _id in obj_ids], dtype=np.float)
            noc = scores_arr[:, 0].astype(int)
            iou_at_noc = scores_arr[:, 1]
            if object_ordering == 'best':
                noc = np.sort(noc)

            cum_scores = np.cumsum(noc)
            num_failed_objects = (cum_scores >= max_clicks).sum()
            num_success = (cum_scores < max_clicks)
            nci = noc[num_success].mean() if num_success.sum() > 0 else max_clicks
            avg_iou_per_image = np.concatenate(
                [iou_at_noc[num_success],
                 np.zeros(num_failed_objects)]
            ).mean()

            nci_list[_i, _j] = nci
            # failed_objects = over_max + (num_instances - len(ious))
            nof_objects_per_thresh.append(num_failed_objects)
            nof_images_per_thresh.append(int(num_failed_objects > 0))
            avg_iou_per_thresh.append(avg_iou_per_image)

        nof_objects.append(np.array(nof_objects_per_thresh))
        nof_images.append(np.array(nof_images_per_thresh))
        avg_iou.append(avg_iou_per_thresh)

    return nci_list.mean(axis=0), np.stack(nof_objects).sum(axis=0), \
        np.stack(nof_images).sum(axis=0), [np.mean(avg_iou)], [total_instance_count]


def nci_per_object_limit(result_per_image, iou_thrs, n_clicks_per_object):
    def _get_noc(iou_arr, iou_thr, max_clicks_per_image):
        vals = np.array(iou_arr) >= iou_thr
        noc = np.argmax(vals) + 1 if np.any(vals) else max_clicks_per_image
        iou_at_noc = np.array(iou_arr)[vals][0] if np.any(vals) else 0.
        return noc, iou_at_noc

    nci_list = np.ones((len(result_per_image), len(iou_thrs))) * -1
    nof_objects = []
    nof_images = []
    avg_iou = []
    total_instance_count = 0
    for _i, _result_dict in enumerate(result_per_image):
        nof_objects_per_thresh = []
        nof_images_per_thresh = []
        avg_iou_per_thresh = []
        for _j, iou_thr in enumerate(iou_thrs):
            num_instances = _result_dict['num_instances']
            total_instance_count +=num_instances
            max_clicks = _result_dict['max_clicks']
            assert max_clicks == n_clicks_per_object*num_instances

            ious = _result_dict['ious']
            obj_ids = np.random.permutation(np.arange(0, num_instances))

            scores_arr = np.array([_get_noc(ious[_id], iou_thr, n_clicks_per_object)
                                   for _id in obj_ids], dtype=np.float)
            noc = scores_arr[:, 0].astype(int)
            iou_at_noc = scores_arr[:, 1]

            cum_scores = np.cumsum(noc)
            num_failed_objects = (noc >= n_clicks_per_object).sum()
            success = (noc < n_clicks_per_object)
            nci = noc.mean()
            avg_iou_per_image = iou_at_noc.mean()

            nci_list[_i, _j] = nci
            # failed_objects = over_max + (num_instances - len(ious))
            nof_objects_per_thresh.append(num_failed_objects)
            nof_images_per_thresh.append(int(num_failed_objects > 0))
            avg_iou_per_thresh.append(avg_iou_per_image)

        nof_objects.append(np.array(nof_objects_per_thresh))
        nof_images.append(np.array(nof_images_per_thresh))
        avg_iou.append(avg_iou_per_thresh)

    return nci_list.mean(axis=0), np.stack(nof_objects).sum(axis=0), \
        np.stack(nof_images).sum(axis=0), [np.mean(avg_iou)], [total_instance_count]


def main():
    args = parse_args()
    start = time.time()

    results = json.load(open(args.result_json, "r"))
    per_instance_ious = results['all_ious']
    image_level_res = results['ious_per_image'] # list[{'ious':[obj1_ious,...]*n_obj,
                                                # 'num_instances': int, 'max_clicks': int, filename:str}]
    np.random.seed(1987)

    # scores = np.stack([
    #     nci_per_object_limit(image_level_res, [0.85], n_clicks_per_object=args.n_clicks)
    #     for _iter in range(args.n_iter)
    # ]).squeeze(-1)
    scores = np.stack([
        compute_nci_metric(image_level_res, [0.85], n_clicks_per_object=args.n_clicks,
                           object_ordering='best' if _iter == args.n_iter-1 else 'random')
        for _iter in range(args.n_iter)
    ]).squeeze(-1)

    nci_easy_object_first, nof_easy_object_first, nfi_easy_object_first, iou_easy_object_first, _ = scores[-1]
    scores = scores[:-1]
    mean_nci, mean_nof, mean_nfi, mean_iou, instance_count = scores.mean(0)
    best_nci, best_nof, best_nfi, best_iou, instance_count = scores.min(0)
    worst_nci, worst_nof, worst_nfi, worst_iou, instance_count = scores.max(0)

    header, row1 = get_results_table(mean_nci, mean_nof, mean_nfi, mean_iou,
                                    eval_type='Mean', instance_count=instance_count)
    _, row2 = get_results_table(worst_nci, worst_nof, worst_nfi, worst_iou,
                                    eval_type='Worst', instance_count=instance_count)
    _, row3 = get_results_table(best_nci, best_nof, best_nfi, best_iou,
                                    eval_type='Best', instance_count=instance_count)
    _, row4 = get_results_table(nci_easy_object_first, nof_easy_object_first, nfi_easy_object_first,
                                iou_easy_object_first, eval_type='Easy first', instance_count=instance_count)
    print(f'{header}\n{row1}\n{row2}\n{row3}\n{row4}')

    # print(f"Mean NCI: {mean_nci}\nMean NFO: {mean_nof}\nMean NFI: {mean_nfi}\nMean IOU: {mean_iou}")
    # print(f"\nWorst NCI: {worst_nci}\nWorst NFO: {worst_nof}\nWorst NFI: {worst_nfi}\nWorst IOU: {worst_iou}")
    # print(f"\nBest NCI: {best_nci}\nBest NFO: {best_nof}\nBest NFI: {best_nfi}\nBest IOU: {best_iou}")
    # print(f"\nNCI easy object first: {nci_easy_object_first}\nNFO easy object first: {nof_easy_object_first}\n"
    #       f"NFI easy object first: {nfi_easy_object_first}\nIOU easy object first: {iou_easy_object_first}")


if __name__ == '__main__':
    main()