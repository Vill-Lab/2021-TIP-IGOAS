from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import glob
import re
import warnings

from torchreid.data.datasets import ImageDataset


class PartialREID(ImageDataset):
    dataset_dir = 'partial-reid'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join('/home/ubuntu/reid/reid-data/market1501/Market-1501-v15.09.15', 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'PartialREID/occluded_body_images')
        # self.query_dir = osp.join(self.dataset_dir, 'PartialREID/partial_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'PartialREID/whole_body_images')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(PartialREID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:

            pid, _ = map(int, pattern.search(img_path).groups())

            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)

        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())


#            if pid == -1:
#                continue # junk images are just ignored
#            assert 0 <= pid <= 60  # pid == 0 means background
#            assert 0 <= camid <= 5
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data



