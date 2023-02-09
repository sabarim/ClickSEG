import os

import cv2
import json
import random
import numpy as np
from pathlib import Path

from PIL import Image

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


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
