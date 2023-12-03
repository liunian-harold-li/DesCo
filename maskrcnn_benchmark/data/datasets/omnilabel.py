# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import os.path
import math
from PIL import Image

import random
import numpy as np

import torch
import torchvision
import torch.utils.data as data

import omnilabeltools as olt
from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
# from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
# from maskrcnn_benchmark.config import cfg
import pdb


def pil_loader(path, retry=5):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    ri = 0
    while ri < retry:
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert("RGB")
        except:
            ri += 1

def load_omnilabel_json(path_json: str, path_imgs: str):
    assert isinstance(path_json, str)

    ol = olt.OmniLabel(path_json)
    dataset_dicts = []
    for img_id in ol.image_ids:
        img_sample = ol.get_image_sample(img_id)
        dataset_dicts.append({
            "image_id": img_sample["id"],
            "file_name": os.path.join(path_imgs, img_sample["file_name"]),
            "inference_obj_descriptions": [od["text"] for od in img_sample["labelspace"]],
            "inference_obj_description_ids": [od["id"] for od in img_sample["labelspace"]],
            "tokens_positive":[od['anno_info'].get("tokens_positive", None) for od in img_sample["labelspace"]],
        })
    return dataset_dicts

class OmniLabelDataset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        img_folder (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, img_folder, ann_file, transforms=None, **kwargs):
        self.img_folder = img_folder
        self.transforms = transforms
        self.dataset_dicts = load_omnilabel_json(ann_file, img_folder)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        data_dict = self.dataset_dicts[index]
        img_id = data_dict["image_id"]
        
        path = data_dict["file_name"]
        img = pil_loader(path)

        # only support test. No box here
        target = BoxList(torch.Tensor(0,4), img.size, mode="xywh").convert("xyxy")
        target.add_field("inference_obj_descriptions", data_dict["inference_obj_descriptions"])
        target.add_field("inference_obj_description_ids", data_dict["inference_obj_description_ids"])
        target.add_field("tokens_positive", data_dict["tokens_positive"])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target, img_id

    def __len__(self):
        return len(self.dataset_dicts)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.img_folder)
        return fmt_str

    # def get_img_info(self, index):
    #     img_id = self.id_to_img_map[index]
    #     img_data = self.coco.imgs[img_id]
    #     return img_data