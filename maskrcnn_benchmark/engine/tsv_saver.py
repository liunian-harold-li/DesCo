import os
import torch
from tqdm import tqdm
from collections import defaultdict
import collections
import numpy as np
import cv2, json, base64
import pdb
from copy import deepcopy
from pprint import pprint
import os.path as op

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.data.datasets.od_to_grounding import clean_name

def ensure_file(file_name):
    # if the directory does not exist, create it
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
        ensure_file(os.path.dirname(file_name))

class TSVResultWriter(object):
    def __init__(self, tokenizer = None, max_visualize_num=-1, dataset_length=-1, threshold = -1.0, in_order = True, write_freq = 100, file_name = None):
        self.tokenizer = tokenizer
        self.max_visualize_num = max_visualize_num
        self.dataset_length = dataset_length
        self.threshold = threshold
        self.in_order = in_order
        self.file_name = file_name
        self.write_freq = write_freq
        self.predictions = []
        if not self.in_order:
            assert(0)

    @staticmethod
    def imagelist_to_b64(imgs):
        imgs = imgs.tensors.permute(0, 2, 3, 1).cpu().numpy()
        # the last dimension is BGR, convert to RGB
        imgs = ((imgs * [0.225, 0.224, 0.229] + [0.406, 0.456, 0.485]) * 255).astype(np.uint8)
        # imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
        imgs = [base64.b64encode(cv2.imencode('.jpg', image)[1]) for image in imgs]
        return imgs

    def update(self, imgs, results):
        if self.max_visualize_num > 0 and len(self.predictions) >= self.max_visualize_num:
            return

        imgs = self.imagelist_to_b64(imgs)

        for img_encoded_str, result in zip(imgs, results):
            # result: (img_id, {"scores": scores, "labels": labels, "boxes": boxes})
            annotations = result[1]
            # img_encoded_str = image #base64.b64encode(cv2.imencode('.jpg', image)[1])
            # convert boxes
            boxes = annotations["raw_boxes"] #box_cxcywh_to_xyxy(annotations["boxes"])
            pred = {}
            pred["objects"] = []

            # pred["caption"] = ""
            for s, rect, l in zip(annotations["scores"], boxes.tolist(), annotations["labels_text"]):
                pred["num_boxes"] = len(rect)
                pred["objects"].append({"rect": rect,
                                        "class": l,
                                        "conf": float(s)
                                        #"caption": captions[0]
                                        })
            if "caption" in annotations:
                pred['objects'][0]["caption"] = annotations["caption"] # record the caption in the first object; a workaround for the tsvviewer

            pred["predicates"] = []
            pred["relations"] = []
            pred = [str(result[0]), json.dumps(pred, sort_keys=False), img_encoded_str]
            self.predictions.append(pred)

        if len(self.predictions) % self.write_freq == 0 or len(self.predictions) >= self.max_visualize_num:
            self.tsv_writer(self.predictions, self.file_name)


    def update_train_data(self, imgs, targets):
        if self.max_visualize_num > 0 and len(self.predictions) >= self.max_visualize_num:
            return

        imgs = self.imagelist_to_b64(imgs)
        for img_encoded_str, target in zip(imgs, targets):
            boxes = target.bbox
            pred = {}
            pred["objects"] = []
            pred["caption"] = [target.extra_fields["caption"]]
            caption_tokenized = self.tokenizer.tokenize(target.extra_fields["caption"])
            for rect, positive_map in zip(boxes.tolist(), target.extra_fields["positive_map"]):
                pred["num_boxes"] = len(rect)
                non_zero_indexes = positive_map.nonzero().squeeze(1).tolist()
                label = [caption_tokenized[i-1] for i in non_zero_indexes]
                label = " ".join(label).replace(" ##", "")
                pred["objects"].append({"rect": rect,
                                        "class": label,
                                        "conf": 1.0,
                                        #"caption": target.extra_fields["caption"]
                                        })
            try:
                pred['objects'][0]["caption"] = target.extra_fields["caption"] # record the caption in the first object; a workaround for the tsvviewer
            except:
                pass
            pred["predicates"] = []
            pred["relations"] = []
            pred = [str(0), json.dumps(pred, sort_keys=False), img_encoded_str]
            self.predictions.append(pred)
        if len(self.predictions) % self.write_freq == 0 or len(self.predictions) >= self.max_visualize_num:
            ensure_file(self.file_name)
            self.tsv_writer(self.predictions, self.file_name)

    def update_gold_od_data(self, imgs, targets, categories):
        if self.max_visualize_num > 0 and len(self.predictions) >= self.max_visualize_num:
            return

        imgs = self.imagelist_to_b64(imgs)
        for img_encoded_str, target in zip(imgs, targets):
            boxes = target["boxes"]
            pred = {}
            pred["objects"] = []

            for rect, label in zip(boxes.tolist(), target["labels"].tolist()):
                pred["num_boxes"] = len(rect)
                cat = categories[label]
                label_text = "{}_{}".format(cat["name"], cat["frequency"])
                pred["objects"].append({"rect": rect,
                                        "class": label_text,
                                        "conf": 1.0,
                                        #"caption": target.extra_fields["caption"]
                                        })
            pred["predicates"] = []
            pred["relations"] = []
            pred = [str(0), json.dumps(pred, sort_keys=False), img_encoded_str]
            self.predictions.append(pred)

        if len(self.predictions) % self.write_freq == 0 or len(self.predictions) >= self.max_visualize_num:
            ensure_file(self.file_name)
            print("Writing to {}".format(self.file_name))
            self.tsv_writer(self.predictions, self.file_name)

    @staticmethod
    def tsv_writer(values, tsv_file, sep='\t'):
        try:
            os.makedirs(op.dirname(tsv_file))
        except:
            pass
        lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
        idx = 0
        tsv_file_tmp = tsv_file + '.tmp'
        lineidx_file_tmp = lineidx_file + '.tmp'
        with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
            assert values is not None
            for value in values:
                assert value is not None
                # this step makes sure python2 and python3 encoded img string are the same.
                # for python2 encoded image string, it is a str class starts with "/".
                # for python3 encoded image string, it is a bytes class starts with "b'/".
                # v.decode('utf-8') converts bytes to str so the content is the same.
                # v.decode('utf-8') should only be applied to bytes class type.
                value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
                v = '{0}\n'.format(sep.join(map(str, value)))
                fp.write(v)
                fpidx.write(str(idx) + '\n')
                idx = idx + len(v)
        os.rename(tsv_file_tmp, tsv_file)
        os.rename(lineidx_file_tmp, lineidx_file)
