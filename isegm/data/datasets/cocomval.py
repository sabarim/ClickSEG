import json
import random
from pathlib import Path

import cv2
import numpy as np
import os
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class COCOMValDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='img', masks_dir_name='gt',
                 init_mask_mode = None, **kwargs):
        super(COCOMValDataset, self).__init__(**kwargs)
        self.name = 'COCOMVal'
        image_mask_dict = {}

        mval_root = dataset_path + '/'
        image_dir = mval_root + 'img/'
        gt_dir = mval_root + 'gt/'
        file_lst = os.listdir(image_dir)
        image_lst = [image_dir+i for i in file_lst if '.jpg' in i]
        for i in image_lst:
            mask_path = i.replace('.jpg','.png').replace('/img/','/gt/')
            image_mask_dict[i] = mask_path
        
        self.image_mask_dict = image_mask_dict
        self.dataset_samples = list(self.image_mask_dict.keys() )

    def get_sample(self, index) -> DSample:
        image_path = self.dataset_samples[index]
        mask_path = self.image_mask_dict[image_path]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path).astype(np.int32)
        if len(instances_mask.shape) == 3:
            instances_mask = instances_mask[:,:,0]
        instances_mask = instances_mask > 128
        instances_mask = instances_mask.astype(np.int32)


        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)


class COCOMValMultiDataset(ISDataset):
    def __init__(self, dataset_path, coco_path,
                 images_dir_name='img', masks_dir_name='gt',
                 init_mask_mode=None, **kwargs):
        super(COCOMValMultiDataset, self).__init__(**kwargs)
        self.name = 'COCOMVal'
        image_mask_dict = {}
        self.coco_path = coco_path

        mval_root = dataset_path + '/'
        image_dir = mval_root + 'img/'
        gt_dir = mval_root + 'gt/'
        file_lst = os.listdir(image_dir)
        image_lst = [image_dir + i for i in file_lst if '.jpg' in i]
        for i in image_lst:
            mask_path = i.replace('.jpg', '.png').replace('/img/', '/gt/')
            image_mask_dict[i] = mask_path

        self.image_mask_dict = image_mask_dict
        self.dataset_samples = list(self.image_mask_dict.keys())
        self.load_samples()
        self.dataset_samples = [_f for _f in self.dataset_samples if _f.split("_")[-1] in self.filename_to_anns]

    def load_samples(self):
        # self.annotation_file = self.dataset_path / 'annotations' / f'panoptic_{self.split}.json' \
        #     if 'train' in self.split else f'instances_{self.split}.json'
        self.annotation_file = os.path.join(self.coco_path, 'annotations', 'instances_val2017.json')
        # self.labels_path = self.coco_path / 'annotations' / f'panoptic_{self.split}'
        # self.images_path = self.coco_path / self.split
        self.min_area = 1000
        self.init_coco()

        # with open(self.annotation_file, 'r') as f:
        #     annotation = json.load(f)
        #
        # self.dataset_samples = self.create_sample_list()
        #
        # self._categories = annotation['categories']
        # # self._stuff_labels = [x['id'] for x in self._categories if x['isthing'] == 0]
        # self._stuff_labels = []
        # # self._things_labels = [x['id'] for x in self._categories if x['isthing'] == 1]
        # self._things_labels = [x['id']-1 for x in self._categories]
        # self._things_labels_set = set(self._things_labels)
        # self._stuff_labels_set = set(self._stuff_labels)

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
        # exclude all images which contain a crowd
        # if self.filter_crowd_images:
        #     self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items()
        #                              if not any([an["iscrowd"] for an in anns])}
        # filter annotations with too small boxes
        if self.min_area != -1.0:
            self.filename_to_anns = {f: [ann for ann in anns if ann["area"] >= self.min_area] for f, anns in
                                     self.filename_to_anns.items()}

        # remove annotations with crowd regions
        self.filename_to_anns = {f: [ann for ann in anns if not ann["iscrowd"] and not ann['category_id'] > 80]
                                 for f, anns in self.filename_to_anns.items()}
        # restrict images to contain considered categories
        # if self.restricted_image_category_list is not None and len(self.restricted_image_category_list) > 0:
        #     print("filtering images to contain categories", self.restricted_image_category_list)
        #     self.filename_to_anns = {f: [ann for ann in anns if self.label_map[ann["category_id"] - 1][
        #         "name"] in self.restricted_image_category_list]
        #                              for f, anns in self.filename_to_anns.items()
        #                              if any([self.label_map[ann["category_id"] - 1]["name"]
        #                                      in self.restricted_image_category_list for ann in anns])}
        #     for cat in self.restricted_image_category_list:
        #         n_imgs_for_cat = sum([1 for anns in self.filename_to_anns.values() if
        #                               any([self.label_map[ann["category_id"] - 1]["name"] == cat for ann in anns])])
        #         print("number of images containing", cat, ":", n_imgs_for_cat)

        # filter out images without annotations
        self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items() if len(anns) > 0}
        n_before = len(self.anns)
        self.anns = []
        for anns in self.filename_to_anns.values():
            self.anns += anns
        n_after = len(self.anns)
        print("filtered annotations:", n_before, "->", n_after)

    def read_target(self, img_filename):
        # img_filename = sample['IMAGES']
        anns = self.filename_to_anns[img_filename]
        img = self.coco.loadImgs(anns[0]['image_id'])[0]
        gt_classes = np.zeros(len(anns))
        boxes = np.zeros((len(anns), 4))

        height = img['height']
        width = img['width']

        label = np.zeros((height, width))
        # if len(anns) > self.max_objects:
        #   anns = np.random.choice(anns, self.max_objects)

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
        image_path = self.dataset_samples[index]
        coco_image_filename = image_path.split("_")[-1]
        target = self.read_target(coco_image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instance_map = np.full_like(target['TARGET_MASK'], 0)
        instance_ids = []
        stuff_ids = []
        mask = target['TARGET_MASK']
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]
        assert len(obj_ids) == len(target['TARGET_CLASSES'])

        for obj_id, class_id in zip(obj_ids, target['TARGET_CLASSES']):
            # class_id = segment['category_id']
            # obj_id = segment['id']
            # if class_id in self._things_labels_set:
            #     things_ids.append(obj_id + 1)
            # else:
            #     stuff_ids.append(obj_id + 1)
            instance_ids.append(obj_id)
            instance_map[mask == obj_id] = obj_id

        # if self.stuff_prob > 0 and random.random() < self.stuff_prob:
        #     instances_ids = things_ids + stuff_ids
        # else:
        #     instances_ids = things_ids
        #
        #     for stuff_id in stuff_ids:
        #         instance_map[instance_map == stuff_id] = 0

        return DSample(image, instance_map, objects_ids=instance_ids)
