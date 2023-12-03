# Following LVIS dataset
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import os
import time
from collections import defaultdict

import pdb
import pycocotools.mask as mask_utils
import torchvision
from PIL import Image
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.config import cfg
# from .coco import ConvertCocoPolysToMask, make_coco_transforms
from .modulated_coco import ConvertCocoPolysToMask

from .lvis import LVIS, LvisDetectionBase


class PacoDetection(LvisDetectionBase):
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, **kwargs):
        super(PacoDetection, self).__init__(img_folder, ann_file)
        self.ann_file = ann_file
        self._transforms = transforms
        self.ids = sorted(list(self.lvis.imgs.keys()))
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def categories(self):
        id2cat = {c["id"]: c for c in self.lvis.dataset["categories"]}
        all_cats = sorted(list(id2cat.keys()))
        categories = {}
        for l in list(all_cats):
            categories[l] = id2cat[l]['name']
        return categories

    def __getitem__(self, idx):
        pdb.set_trace()
        img, target = super(PacoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img = self._transforms(img)
        return img, target, idx


    def convert_dict_anno_to_box(self, annos):
        pass
    
    def get_raw_image(self, idx):
        img, target = super(PacoDetection, self).__getitem__(idx)
        return img
    
    def categories(self):
        id2cat = {c["id"]: c for c in self.lvis.dataset["categories"]}
        all_cats = sorted(list(id2cat.keys()))
        categories = {}
        for l in list(all_cats):
            categories[l] = id2cat[l]['name']
        return categories

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.lvis.imgs[img_id]
        return img_data