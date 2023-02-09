from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import glob
import os
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class DavisDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='img', masks_dir_name='gt',
                 init_mask_mode = None, **kwargs):
        super(DavisDataset, self).__init__(**kwargs)
        self.name = 'Davis'
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}
        self.init_mask_mode = init_mask_mode

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = np.max(cv2.imread(mask_path).astype(np.int32), axis=2)
        instances_mask[instances_mask > 0] = 1

        init_mask_dir = None
        if self.init_mask_mode == 'low':
            init_mask_dir = '/home/admin/workspace/project/data/datasets/DAVIS_Edit/DAVIS_low/'
        elif self.init_mask_mode == 'mid':
            init_mask_dir = '/home/admin/workspace/project/data/datasets/DAVIS_Edit/DAVIS_mid/'
        elif self.init_mask_mode == 'high':
            init_mask_dir = '/home/admin/workspace/project/data/datasets/DAVIS_Edit/DAVIS_high/'

        if init_mask_dir is not None:
            init_mask_path = init_mask_dir + image_name.replace('.jpg','.png')
            init_mask = cv2.imread(init_mask_path)[:,:,0] > 0
        else:
            init_mask = None

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index, init_mask=init_mask)


class Davis2017Dataset(DavisDataset):
    def __init__(self, dataset_path,
                 images_dir_name='img', masks_dir_name='gt', imset_file=None,
                 init_mask_mode = None, **kwargs):
        super(Davis2017Dataset, self).__init__(dataset_path, images_dir_name, masks_dir_name, init_mask_mode, **kwargs)
        self.name = "davis17"
        if imset_file is None:
            video_paths = glob.glob(str(self._images_path / "*"))
        else:
            with open(os.path.join(dataset_path, imset_file), "r") as lines:
                # for _video_id, line in enumerate(['judo']):
                video_paths = [str(self._images_path / line.rstrip('\n')) for line in lines]

        # for v in glob.glob(str(self._images_path / "*")):
        for v in video_paths:
            video = v.split("/")[-1]
            self.dataset_samples += list(
                glob.glob(os.path.join(self._images_path, video, "*.*"))
            )
            self._masks_paths.update(
                dict([(video + "_" + _m.split("/")[-1].split(".png")[0], _m)
                      for _m in glob.glob(os.path.join(self._insts_path, video, "*.*"))])
            )

    def get_sample(self, index) -> DSample:
        image_path = self.dataset_samples[index]
        video = image_path.split("/")[-2]
        _f_name = image_path.split("/")[-1].split(".jpg")[0]
        # image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[video + "_" + _f_name])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = np.array(Image.open(mask_path).convert("P"))
        # instances_mask[instances_mask > 0] = 1
        instance_ids = np.unique(instances_mask)
        # fg instances
        instance_ids = instance_ids[instance_ids != 0]

        return DSample(image, instances_mask, objects_ids=instance_ids, sample_id=index, init_mask=None,
                       filename=os.path.join(video, _f_name))