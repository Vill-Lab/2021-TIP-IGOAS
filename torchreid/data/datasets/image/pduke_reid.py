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
from torchreid.utils import read_image
import cv2
import numpy as np


class P_Dukereid(ImageDataset):
    def __init__(self, root='', **kwargs):
        dataset_dir = 'P-DukeMTMC-reid'
        self.root = osp.abspath(osp.expanduser(root))
        # self.dataset_dir = self.root
        data_dir = osp.join(self.root, dataset_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated.')
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'test', 'occluded_body_images')
        self.gallery_dir = osp.join(self.data_dir, 'test', 'whole_body_images')

        train = self.process_train_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False, is_query=False)
        super(P_Dukereid, self).__init__(train, query, gallery, **kwargs)

    def process_train_dir(self, dir_path, relabel=True):
        img_paths = glob.glob(osp.join(dir_path, 'whole_body_images', '*', '*.jpg'))
        camid = 1
        pattern = re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')
        pid_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern.search(img_path).groups())
            # print(pid)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        data = []
        for img_path in img_paths:
            pid, _, _ = map(int, pattern.search(img_path).groups())
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        img_paths = glob.glob(osp.join(dir_path, 'occluded_body_images', '*', '*.jpg'))
        camid = 0
        for img_path in img_paths:
            pid, _, _ = map(int, pattern.search(img_path).groups())
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        return data

    def process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path, '*', '*.jpg'))
        pattern = re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')
        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, _, _ = map(int, pattern.search(img_path).groups())
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        return data
