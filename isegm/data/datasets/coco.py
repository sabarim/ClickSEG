import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from pycocotools import _mask as coco_mask
from tqdm import tqdm

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        # area = coco_mask.area(rles)
        # print(area)
        # if area < min_area:
        #     continue
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class CocoDataset(ISDataset):
    def __init__(self, dataset_path, split='train', stuff_prob=0.0, **kwargs):
        super(CocoDataset, self).__init__(**kwargs)
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.stuff_prob = stuff_prob

        self.load_samples()

    def load_samples(self):
        annotation_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}.json'
        self.labels_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}'
        self.images_path = self.dataset_path / self.split

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        self.dataset_samples = annotation['annotations']

        self._categories = annotation['categories']
        self._stuff_labels = [x['id'] for x in self._categories if x['isthing'] == 0]
        self._things_labels = [x['id'] for x in self._categories if x['isthing'] == 1]
        self._things_labels_set = set(self._things_labels)
        self._stuff_labels_set = set(self._stuff_labels)

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]

        image_path = self.images_path / self.get_image_name(dataset_sample['file_name'])
        label_path = self.labels_path / dataset_sample['file_name']

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
        label = 256 * 256 * label[:, :, 0] + 256 * label[:, :, 1] + label[:, :, 2]

        instance_map = np.full_like(label, 0)
        things_ids = []
        stuff_ids = []

        for segment in dataset_sample['segments_info']:
            class_id = segment['category_id']
            obj_id = segment['id']
            if class_id in self._things_labels_set:
                if segment['iscrowd'] == 1:
                    continue
                things_ids.append(obj_id)
            else:
                stuff_ids.append(obj_id)

            instance_map[label == obj_id] = obj_id

        if self.stuff_prob > 0 and random.random() < self.stuff_prob:
            instances_ids = things_ids + stuff_ids
        else:
            instances_ids = things_ids

            for stuff_id in stuff_ids:
                instance_map[instance_map == stuff_id] = 0

        mapped_ids = []
        for _i, obj_id in enumerate(instances_ids):
            instance_map[instance_map == obj_id] = _i + 1
            mapped_ids.append(_i+1)

        return DSample(image, instance_map, objects_ids=mapped_ids)

    @classmethod
    def get_image_name(cls, panoptic_name):
        return panoptic_name.replace('.png', '.jpg')


class CocoValDataset(ISDataset):
    def __init__(self, dataset_path, split='val', stuff_prob=0.0, **kwargs):
        super(CocoValDataset, self).__init__(**kwargs)
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.stuff_prob = stuff_prob
        self.load_samples()

    def load_samples(self):
        # self.annotation_file = self.dataset_path / 'annotations' / f'panoptic_{self.split}.json' \
        #     if 'train' in self.split else f'instances_{self.split}.json'
        self.annotation_file = self.dataset_path / 'annotations' / f'instances_{self.split}.json'
        self.labels_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}'
        self.images_path = self.dataset_path / self.split
        self.min_area = -1
        self.init_coco()

        with open(self.annotation_file, 'r') as f:
            annotation = json.load(f)

        self.dataset_samples = self.create_sample_list()

        self._categories = annotation['categories']
        # self._stuff_labels = [x['id'] for x in self._categories if x['isthing'] == 0]
        self._stuff_labels = []
        # self._things_labels = [x['id'] for x in self._categories if x['isthing'] == 1]
        self._things_labels = [x['id']-1 for x in self._categories]
        self._things_labels_set = set(self._things_labels)
        self._stuff_labels_set = set(self._stuff_labels)

    def init_coco(self):
        # only import this dependency on demand
        import pycocotools.coco as coco
        self.coco = coco.COCO(self.annotation_file)
        ann_ids = self.coco.getAnnIds([])
        self.anns = self.coco.loadAnns(ann_ids)
        self.label_map = {k - 1: v for k, v in self.coco.cats.items()}
        self.filename_to_anns = dict()
        self.build_filename_to_anns_dict()

    def build_filename_to_anns_dict(self):
        for ann in self.anns:
            img_id = ann['image_id']
            img = self.coco.loadImgs(img_id)
            file_name = img[0]['file_name']
            if file_name in self.filename_to_anns:
                self.filename_to_anns[file_name].append(ann)
            else:
                self.filename_to_anns[file_name] = [ann]
                # self.filename_to_anns[file_name] = ann
        self.filter_anns()

    def filter_anns(self):
        # filter annotations with too small boxes
        if self.min_area != -1.0:
            self.filename_to_anns = {f: [ann for ann in anns if ann["area"] >= self.min_area] for f, anns in
                                     self.filename_to_anns.items()}

        # remove annotations with crowd regions
        self.filename_to_anns = {f: [ann for ann in anns if not ann["iscrowd"]]
                                for f, anns in self.filename_to_anns.items()}

        # filter out images without annotations
        self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items() if len(anns) > 0}
        n_before = len(self.anns)
        self.anns = []
        for anns in self.filename_to_anns.values():
            self.anns += anns
        n_after = len(self.anns)
        print("filtered annotations:", n_before, "->", n_after)

    def create_sample_list(self):
        # Filtering the image file names since some of them do not have annotations.
        # Since we use the minival split, validation files could be part of the training set
        imgs = [os.path.join('%s/%s/' % (self.dataset_path, self.split), fn)
                for fn in self.filename_to_anns.keys()]
        samples = []
        for line in imgs:
            sample = {'INFO': {}, 'IMAGES': [], 'TARGETS': []}
            sample['IMAGES'] = line
            sample['INFO']['num_frames'] = 1
            samples += [sample]

        return samples

    def read_image(self, sample):
        path = sample['IMAGES']
        img = np.array(Image.open(path).convert('RGB'))
        return img

    def read_target(self, sample):
        img_filename = sample['IMAGES']
        anns = self.filename_to_anns[img_filename.split("/")[-1]]
        img = self.coco.loadImgs(anns[0]['image_id'])[0]
        gt_classes = np.zeros(len(anns))
        boxes = np.zeros((len(anns), 4))

        height = img['height']
        width = img['width']
        sample['INFO']['shape'] = (height, width)

        label = np.zeros((height, width))
        # if len(anns) > self.max_objects:
        #   anns = np.random.choice(anns, self.max_objects)

        sample['INFO']['num_objects'] = len(anns)
        # sort annotations based on area
        anns = sorted(anns, key=lambda d: d['area'])
        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann)[:, :]
            # mask out pixels overlapping with existing instances
            mask[label != 0] = 0
            label[mask != 0] = i + 1
            gt_classes[i] = ann['category_id'] - 1  # map category ids to be from 0 to n_classes-1

        if len(np.unique(label)) == 1:
            print("GT contains only background. Filename {} #Annotations {}".format(img_filename, len(anns)))

        return {'TARGET_MASK': label.astype(np.uint8), 'TARGET_CLASSES': gt_classes}

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]
        image = self.read_image(dataset_sample)
        target = self.read_target(dataset_sample)

        # image_path = self.images_path / self.get_image_name(dataset_sample['file_name'])
        # label_path = self.labels_path / dataset_sample['file_name']
        #
        # image = cv2.imread(str(image_path))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
        # label = 256 * 256 * label[:, :, 0] + 256 * label[:, :, 1] + label[:, :, 2]

        instance_map = np.full_like(target['TARGET_MASK'], 0)
        things_ids = []
        stuff_ids = []
        mask = target['TARGET_MASK']
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]
        assert len(obj_ids) == len(target['TARGET_CLASSES'])

        for obj_id, class_id in zip(obj_ids, target['TARGET_CLASSES']):
            # class_id = segment['category_id']
            # obj_id = segment['id']
            assert class_id in self._things_labels_set
            things_ids.append(obj_id)

            instance_map[mask == obj_id] = obj_id
            assert (mask == obj_id).sum() > 0

        if self.stuff_prob > 0 and random.random() < self.stuff_prob:
            instances_ids = things_ids + stuff_ids
        else:
            instances_ids = things_ids

            for stuff_id in stuff_ids:
                instance_map[instance_map == stuff_id] = 0

        return DSample(image, instance_map, objects_ids=instances_ids, filename=dataset_sample['IMAGES'])

    @classmethod
    def get_image_name(cls, panoptic_name):
        return panoptic_name.replace('.png', '.jpg')


class CocoDetectronValDataset(ISDataset):
    def __init__(self, dataset_path, split='val', stuff_prob=0.0, **kwargs):
        super(CocoDetectronValDataset, self).__init__(**kwargs)
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.stuff_prob = stuff_prob
        self.dataset_samples = self.load_samples()

    def load_samples(self):
        dataset_dict = DatasetCatalog.get("coco_2017_val")
        samples = []
        num_instances=0
        for _i, _sample in enumerate(dataset_dict):
            image_shape = (_sample['height'], _sample['width'])
            if "annotations" in _sample:
                # USER: Modify this if you want to keep them for some reason.
                for anno in _sample["annotations"]:
                    # Let's always keep mask
                    # if not self.mask_on:
                    #     anno.pop("segmentation", None)
                    anno.pop("keypoints", None)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    obj for obj in _sample.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                # NOTE: does not support BitMask due to augmentation
                # Current BitMask cannot handle empty objects
                instances = utils.annotations_to_instances(annos, image_shape)
                # After transforms such as cropping are applied, the bounding box may no longer
                # tightly bound the object. As an example, imagine a triangle object
                # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                # the intersection of original bounding box and the cropping box.
                if not hasattr(instances, 'gt_masks'):
                    continue
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                # boxes_area = instances.gt_boxes.area()
                # Need to filter empty instances first (due to augmentation)
                instances = utils.filter_empty_instances(instances)

                if len(instances) == 0:
                    continue
                # Generate masks from polygon
                h, w = instances.image_size

                if hasattr(instances, 'gt_masks'):
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)

                    # Make smaller object in front in case of overlapping masks
                    mask_areas = torch.sum(gt_masks, (1, 2))
                    gt_masks = gt_masks.to(dtype=torch.uint8)
                    gt_masks = gt_masks[sorted(range(len(mask_areas)), key=mask_areas.__getitem__, reverse=True)]

                    instance_map = torch.zeros((gt_masks.shape[-2:]), dtype=torch.int16)
                    num_objects = gt_masks.shape[0]
                    instances_ids = np.arange(1, num_objects + 1)

                    for _id, _m in enumerate(gt_masks):
                        instance_map[_m == 1] = _id + 1
                        assert (_m != 0).sum() > 0

                    new_gt_masks = []
                    for _id in instances_ids:
                        _m = (instance_map == _id).to(dtype=torch.uint8)
                        if _m.sum() > 0:
                            new_gt_masks.append(_m)
                    if not len(new_gt_masks):
                        continue
                    new_gt_masks = torch.stack(new_gt_masks, dim=0)
                    assert num_objects == new_gt_masks.shape[0]

                    _sample['semantic_map'] = instance_map

                    # sample = {'INFO': {}, 'IMAGES': [], 'TARGETS': [instance_map]}
                    # sample['IMAGES'] = _sample['file_name']
                    # sample['INFO']['num_frames'] = 1
                    samples.append(_sample)
                    num_instances+=num_objects

        print(f"Total instances {num_instances}")
        return samples

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]
        image = np.array(Image.open(dataset_sample['file_name']).convert('RGB'))
        target = dataset_sample['semantic_map']
        instance_ids = np.unique(target)
        instance_ids = instance_ids[instance_ids!=0]

        return DSample(image, target.numpy(), objects_ids=instance_ids, filename=dataset_sample['file_name'])

    @classmethod
    def get_image_name(cls, panoptic_name):
        return panoptic_name.replace('.png', '.jpg')


if __name__ == '__main__':
    ds = CocoDetectronValDataset('')
    for data in tqdm(ds):
        print(data.keys())
