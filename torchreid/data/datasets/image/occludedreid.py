from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import glob
import re
import random

from torchreid.data.datasets import ImageDataset


class OccludedREID(ImageDataset):
    dataset_dir = 'occluded-reid'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        
        self.train_dir = osp.join(self.dataset_dir, 'Occluded_REID')
        self.query_dir = osp.join(self.dataset_dir, 'Occluded_REID/occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'Occluded_REID/whole_body_images')

        required_files = [
            self.dataset_dir,
            self.query_dir,
            self.gallery_dir
        ]
        self.check_before_run(required_files)

        all_pid = []
        for i in range(1, 201):
            all_pid.append(i)

        # train_pid = random.sample(all_pid, 100)
        train_pid = []
        for i in range(88, 188):
            train_pid.append(i)
        test_pid = []
        for m in all_pid:
            if m not in train_pid:
                test_pid.append(m)

        train = self.process_train_dir(self.train_dir, train_pid, relabel=True)
        query = self.process_dir(self.query_dir, test_pid, relabel=False)
        gallery = self.process_dir(self.gallery_dir, test_pid, relabel=False)

        super(OccludedREID, self).__init__(train, query, gallery, **kwargs)

    def process_train_dir(self, dir_path, pid_list, relabel=True):
        img_paths = glob.glob(osp.join(dir_path, 'whole_body_images', '*.tif'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            # print(pid, _)
            if pid in pid_list:
                pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # print(pid)
            if pid in pid_list:
                if relabel:
                    pid = pid2label[pid]
                data.append((img_path, pid, camid))

        img_paths = glob.glob(osp.join(dir_path, 'occluded_body_images', '*.tif'))
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid in pid_list:
                if relabel:
                    pid = pid2label[pid]
                data.append((img_path, pid, camid))
        return data

    def process_dir(self, dir_path, pid_list, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.tif'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:

            pid, _ = map(int, pattern.search(img_path).groups())

            if pid == -1:
                continue  # junk images are just ignored
            if pid in pid_list:
                pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
#            if pid == -1:
#                continue # junk images are just ignored
#            assert 0 <= pid <= 60  # pid == 0 means background
#            assert 0 <= camid <= 5
            camid -= 1 # index starts from 0
            if pid in pid_list:
                if relabel:
                    pid = pid2label[pid]
                data.append((img_path, pid, camid))

        return data



