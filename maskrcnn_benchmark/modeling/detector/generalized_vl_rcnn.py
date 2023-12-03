# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized VL R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from ..backbone import build_backbone, build_fusion_backbone
from ..rpn import build_rpn
from ..roi_heads import build_roi_heads

from ..language_backbone import build_language_backbone
from transformers import AutoTokenizer

import random
import timeit
import pdb
from copy import deepcopy


def random_word(input_ids, mask_token_id, vocabs, padding_token_id, greenlight_map):
    """
    greenlight_map, batch_size x 256 (seq_len):
        0 means this location cannot be calculated in the MLM loss
        -1 means this location cannot be masked!!
        1 means this location can be masked and can be calculated in the MLM loss
    """
    output_label = deepcopy(input_ids)
    for j in range(input_ids.size(0)):
        for i in range(input_ids.size(1)):
            prob = random.random()
            # mask token with probability
            ratio = 0.15
            if greenlight_map is not None and greenlight_map[j, i] == -1:
                output_label[j, i] = -100
                continue

            if (not input_ids[j, i] == padding_token_id) and prob < ratio:
                prob /= ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    input_ids[j, i] = mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    input_ids[j, i] = random.choice(vocabs)

            else:
                # no masking token (will be ignored by loss function later)
                output_label[j, i] = -100

            if greenlight_map is not None and greenlight_map[j, i] != 1:
                output_label[j, i] = -100  # If this location should not be masked
    return input_ids, output_label

def get_char_token_with_relaxation(tokenized, beg, end, batch_index = None):
    beg_pos = tokenized.char_to_token(batch_index, beg)
    end_pos = tokenized.char_to_token(batch_index, end - 1)
    if beg_pos is None:
        try:
            beg_pos = tokenized.char_to_token(batch_index, beg + 1)
            if beg_pos is None:
                beg_pos = tokenized.char_to_token(batch_index, beg + 2)
        except:
            beg_pos = None
    if end_pos is None:
        try:
            end_pos = tokenized.char_to_token(batch_index, end - 2)
            if end_pos is None:
                end_pos = tokenized.char_to_token(batch_index, end - 3)
        except:
            end_pos = None
    if beg_pos is None or end_pos is None:
        return None, None
    return beg_pos, end_pos + 1

class GeneralizedVLRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedVLRCNN, self).__init__()
        self.cfg = cfg
        self.fusion_in_backbone = cfg.MODEL.SWINT.VERSION == "fusion"

        # visual encoder
        backbone = build_backbone(cfg)

        # language encoder
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            # self.tokenizer = build_tokenizer("clip")
            from transformers import CLIPTokenizerFast

            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
                self.tokenizer = CLIPTokenizerFast.from_pretrained(
                    "openai/clip-vit-base-patch32", from_slow=True, mask_token="ðŁĴĳ</w>"
                )
            else:
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", from_slow=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_vocab_ids = [item for key, item in self.tokenizer_vocab.items()]

        # if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
        #     self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        # elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "roberta-base":
        #     self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        # else:
        #     raise NotImplementedError
        # self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)

        language_backbone = build_language_backbone(cfg)

        if self.fusion_in_backbone:
            self.fusion_backbone = build_fusion_backbone(
                backbone,
                language_backbone,
                cfg.MODEL.BACKBONE.FUSION_VERSION,
                add_linear_layer=cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER,
            )
        else:
            self.backbone = backbone
            self.language_backbone = language_backbone

        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.DEBUG = cfg.MODEL.DEBUG

        self.freeze_backbone = cfg.MODEL.BACKBONE.FREEZE
        self.freeze_fpn = cfg.MODEL.FPN.FREEZE
        self.freeze_rpn = cfg.MODEL.RPN.FREEZE
        self.add_linear_layer = cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER

        self.force_boxes = cfg.MODEL.RPN.FORCE_BOXES

        if cfg.MODEL.LINEAR_PROB:
            assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
            if self.fusion_in_backbone:
                if hasattr(self.fusion_backbone.backbone, "fpn"):
                    assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
            else:
                if hasattr(self.backbone, "fpn"):
                    assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
        self.linear_prob = cfg.MODEL.LINEAR_PROB
        self.freeze_cls_logits = cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # disable cls_logits
            if hasattr(self.rpn.head, "cls_logits"):
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False

        self.freeze_language_backbone = self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE
        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            if self.fusion_in_backbone:
                for p in self.fusion_backbone.language_backbone.parameters():
                    p.requires_grad = False
            else:
                for p in self.language_backbone.parameters():
                    p.requires_grad = False

        self.use_mlm_loss = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS
        self.mlm_loss_for_only_positives = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER and not self.fusion_in_backbone:
            self.tunable_linear = torch.nn.Linear(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, 1000, bias=False)
            self.tunable_linear.weight.data.fill_(0.0)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GeneralizedVLRCNN, self).train(mode)
        if self.freeze_backbone:
            if self.fusion_in_backbone:
                self.fusion_backbone.backbone.body.eval()
                for p in self.fusion_backbone.backbone.body.parameters():
                    p.requires_grad = False
            else:
                self.backbone.body.eval()
                for p in self.backbone.body.parameters():
                    p.requires_grad = False
        if self.freeze_fpn:
            if self.fusion_in_backbone:
                self.fusion_backbone.backbone.fpn.eval()
                for p in self.fusion_backbone.backbone.fpn.parameters():
                    p.requires_grad = False
            else:
                self.backbone.fpn.eval()
                for p in self.backbone.fpn.parameters():
                    p.requires_grad = False
        if self.freeze_rpn:
            if hasattr(self.rpn, "head"):
                self.rpn.head.eval()
            for p in self.rpn.parameters():
                p.requires_grad = False
        if self.linear_prob:
            if self.rpn is not None:
                for key, value in self.rpn.named_parameters():
                    if not (
                        "bbox_pred" in key
                        or "cls_logits" in key
                        or "centerness" in key
                        or "cosine_scale" in key
                        or "dot_product_projection_text" in key
                        or "head.log_scale" in key
                        or "head.bias_lang" in key
                        or "head.bias0" in key
                    ):
                        value.requires_grad = False
            if self.roi_heads is not None:
                for key, value in self.roi_heads.named_parameters():
                    if not (
                        "bbox_pred" in key
                        or "cls_logits" in key
                        or "centerness" in key
                        or "cosine_scale" in key
                        or "dot_product_projection_text" in key
                        or "head.log_scale" in key
                        or "head.bias_lang" in key
                        or "head.bias0" in key
                    ):
                        value.requires_grad = False
        if self.freeze_cls_logits:
            if hasattr(self.rpn.head, "cls_logits"):
                self.rpn.head.cls_logits.eval()
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False
        if self.add_linear_layer:
            if not self.fusion_in_backbone:
                if self.rpn is not None:
                    for key, p in self.rpn.named_parameters():
                        if "tunable_linear" in key:
                            p.requires_grad = True
            else:
                for key, p in self.fusion_backbone.named_parameters():
                    if "tunable_linear" in key:
                        p.requires_grad = True

        if self.freeze_language_backbone:
            if self.fusion_in_backbone:
                self.fusion_backbone.language_backbone.eval()
                for p in self.fusion_backbone.language_backbone.parameters():
                    p.requires_grad = False
            else:
                self.language_backbone.eval()
                for p in self.language_backbone.parameters():
                    p.requires_grad = False

    def forward(self, images, targets=None, captions=None, positive_map=None, greenlight_map=None, spans = None, span_map = None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

            mask_black_list: batch x 256, indicates whether or not a certain token is maskable or not

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        # batch_size = images.tensors.shape[0]
        device = images.tensors.device

        # if we use the advanced span prediction version, we need to do both preprocessing and postprocessing
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SPAN_VERSION is not None and self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SPAN_VERSION.startswith("v2"):
            if spans is None:
                spans = [i.extra_fields['spans'] if "spans" in i.extra_fields else [] for i in targets] # if we did not pass the spans explicitly
            assert(len(spans) == len(captions))
            new_captions = []
            mapping_batch_span_to_caption_num = {} # (batch_num, start, end) -> caption_num
            mapping_batch_to_caption_num = {}
            corrected_spans = deepcopy(spans)

            for i in range(len(captions)):
                if len(spans[i]) == 0: # if this instance does not have span
                    mapping_batch_to_caption_num[i] = len(new_captions)
                    new_captions.append(captions[i])
                    continue

                for j in range(len(spans[i])):
                    ''' spans[i][j]: 
                    [[230, 241],
                    [241, 254],
                    [254, 269],
                    [269, 298],]
                    '''
                    valid_spans = [k for k in spans[i][j] if k[0] != -1]
                    if "independent" in self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SPAN_VERSION:
                        for k, span_i_j_k in enumerate(valid_spans):
                            mapping_batch_span_to_caption_num[(i, span_i_j_k[0], span_i_j_k[1])] = len(new_captions)
                            corrected_spans[i][j][k] = (0, span_i_j_k[1] - span_i_j_k[0])
                            new_captions.append(captions[i][span_i_j_k[0]:span_i_j_k[1]])
                    else:
                        start = valid_spans[0][0]
                        end = valid_spans[-1][-1]
                        # rewrite the spans !!
                        corrected_spans[i][j] = [(k[0] - start, k[1] - start) for k in spans[i][j]]
                        
                        for k in valid_spans:
                            mapping_batch_span_to_caption_num[(i, k[0], k[1])] = len(new_captions)
                        #mapping_batch_to_caption_num[i] = len(new_captions)
                        new_captions.append(captions[i][start:end])
            captions = new_captions
            padding_method = "longest"
            #print(new_captions)
        else:
            mapping_batch_span_to_caption_num = None
            padding_method = "max_length" if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest"

        # language embedding
        language_dict_features = {}
        if captions is not None:
            # print(captions[0])
            tokenized = self.tokenizer.batch_encode_plus(
                captions,
                max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                padding=padding_method,
                return_special_tokens_mask=True,
                return_tensors="pt",
                truncation=True,
            ).to(device)
            if self.use_mlm_loss:
                if not self.mlm_loss_for_only_positives:
                    greenlight_map = None
                input_ids, mlm_labels = random_word(
                    input_ids=tokenized.input_ids,
                    mask_token_id=self.tokenizer.mask_token_id,
                    vocabs=self.tokenizer_vocab_ids,
                    padding_token_id=self.tokenizer.pad_token_id,
                    greenlight_map=greenlight_map,
                )
            else:
                input_ids = tokenized.input_ids
                mlm_labels = None

            tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}

            if not self.fusion_in_backbone:
                if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
                    with torch.no_grad():
                        language_dict_features = self.language_backbone(tokenizer_input)
                else:
                    language_dict_features = self.language_backbone(tokenizer_input)

                # ONE HOT
                if self.cfg.DATASETS.ONE_HOT:
                    new_masks = torch.zeros_like(
                        language_dict_features["masks"], device=language_dict_features["masks"].device
                    )
                    new_masks[:, : self.cfg.MODEL.DYHEAD.NUM_CLASSES] = 1
                    language_dict_features["masks"] = new_masks

                # MASK ALL SPECIAL TOKENS
                if self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL:
                    language_dict_features["masks"] = 1 - tokenized.special_tokens_mask

                language_dict_features["mlm_labels"] = mlm_labels

        if not self.fusion_in_backbone:
            # visual embedding
            swint_feature_c4 = None
            if "vl" in self.cfg.MODEL.SWINT.VERSION:
                # the backbone only updates the "hidden" field in language_dict_features
                inputs = {"img": images.tensors, "lang": language_dict_features}
                visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
            else:
                visual_features = self.backbone(images.tensors)

        else:
            visual_features, language_dict_features, swint_feature_c4 = self.fusion_backbone(tokenizer_input, images)
            language_dict_features["mlm_labels"] = mlm_labels

        # add the prompt tuning linear layer if not fusion, for fusion do it inside the backbone
        if not self.fusion_in_backbone:
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
                embedding = language_dict_features["embedded"]
                embedding = self.tunable_linear.weight[: embedding.size(1), :].unsqueeze(0) + embedding
                language_dict_features["embedded"] = embedding
                language_dict_features["hidden"] = (
                    self.tunable_linear.weight[: embedding.size(1), :].unsqueeze(0) + language_dict_features["hidden"]
                )

        # if we do span prediction
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SPAN_VERSION is not None:
            loss_version = self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SPAN_VERSION.split(".")[0]
            pooling_version = self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SPAN_VERSION.split(".")[-1]

            # will just override everything

            # Step 1. get the spans
            embedding = language_dict_features["hidden"]
            if spans is None:
                spans = [i.extra_fields['spans'] if "spans" in i.extra_fields else [] for i in targets] # if we did not pass the spans explicitly

            if mapping_batch_span_to_caption_num is not None: # need to do a remapping
                flatterned_spans = []
                for i in spans:
                    _ = []
                    for j in i:
                        _.extend(j)
                    flatterned_spans.append(_)
                spans = flatterned_spans

                # flattern corrected spans
                flatterned_corrected_spans = []
                for i in corrected_spans:
                    _ = []
                    for j in i:
                        _.extend(j)
                    flatterned_corrected_spans.append(_)
                corrected_spans = flatterned_corrected_spans

                max_span_num = max([len(i) for i in spans])

                # go over the batch, see if there is an instance without spans; if so, we override the span_num to the token_num of that instance
                for i, spans_i in enumerate(spans):
                    if len(spans_i) == 0: # no spans
                        text_length = sum(tokenized.attention_mask[mapping_batch_to_caption_num[i]])
                        max_span_num = max(text_length, max_span_num) # override
                
                # Step 2. Get the Masks
                span_masks = torch.zeros((len(spans), max_span_num), device=embedding.device, dtype=torch.long)
                for i, spans_i in enumerate(spans):
                    if len(spans_i) == 0:
                        # this would be the text masks
                        text_mask_i = tokenized.attention_mask[mapping_batch_to_caption_num[i]]
                        text_length = sum(text_mask_i)
                        span_masks[i, : text_length] = text_mask_i[:text_length]
                    else:
                        span_masks[i, : len(spans_i)] = 1

                # Step 3. get the span features
                span_features = torch.zeros((len(spans), max_span_num, embedding.size(2)), device=embedding.device, dtype=embedding.dtype)
                # the complexity is just batch x span_num; should be begign for a foor loop
                for i, spans_i in enumerate(spans):
                    if len(spans_i) == 0:
                        # directly override with the embedding
                        __len = min(max_span_num, embedding[mapping_batch_to_caption_num[i]].size(0))
                        span_features[i, :__len, :] = embedding[mapping_batch_to_caption_num[i], :__len, :]
                    else:
                        for j, span in enumerate(spans_i):
                            # first need to get the correct tokenized version
                            mapped_sentence_index = mapping_batch_span_to_caption_num[(i, span[0], span[1])] # here we use the original span

                            start, end = get_char_token_with_relaxation(tokenized, corrected_spans[i][j][0], corrected_spans[i][j][1], batch_index = mapped_sentence_index) # here use the span location after we have partitioned the sentence
                            if start is None or end is None:
                                span_masks[i, j] = 0 # mark this span as invalid

                            if pooling_version == "mean":
                                span_rep_i_j = torch.mean(embedding[mapped_sentence_index, start:end, :], dim=0)
                            elif pooling_version == "max":
                                span_rep_i_j = torch.max(embedding[mapped_sentence_index, start:end, :], dim=0)[0]
                            span_features[i, j, :] = span_rep_i_j
            else:
                assert(0)
                # max_span_num = max([len(i) for i in spans])
                
                # # Step 2. Get the Masks
                # span_masks = torch.zeros((len(spans), max_span_num), device=embedding.device, dtype=torch.long)
                # for i, spans_i in enumerate(spans):
                #     span_masks[i, : len(spans_i)] = 1

                # # Step 3. get the span features
                # span_features = torch.zeros((len(spans), max_span_num, embedding.size(2)), device=embedding.device, dtype=embedding.dtype)
                # # the complexity is just batch x span_num; should be begign for a foor loop
                # for i, spans in enumerate(spans):
                #     for j, span in enumerate(spans):
                #         # span records the char location; needs to convert to token location first
                #         start, end = get_char_token_with_relaxation(tokenized, span[0], span[1], batch_index = i)
                        
                #         if start is None or end is None:
                #             span_masks[i, j] = 0 # mark this span as invalid

                #         if pooling_version == "mean":
                #             span_rep_i_j = torch.mean(embedding[i, start:end, :], dim=0)
                #         elif pooling_version == "max":
                #             span_rep_i_j = torch.max(embedding[i, start:end, :], dim=0)[0]
                #         span_features[i, j, :] = span_rep_i_j
                
            # Step 4. Rewrite the labels (?)
            # we need to rewrite targets, positive_map, text_masks, text_embeddings
            if span_map is None:
                span_map = torch.zeros((positive_map.size(0), max_span_num), device=embedding.device, dtype=torch.float)
                _all_span_map_flattern = [] # box_num x span_num

                for target_i in targets:
                    if "span_map" in target_i.extra_fields:
                        _all_span_map_flattern.extend([j for j in target_i.extra_fields["span_map"]])
                    else:
                        # if not, create a list of empty lists
                        num_box = target_i.bbox.size(0)
                        _all_span_map_flattern.extend([[]] * num_box) # very important
                    
                assert(len(_all_span_map_flattern) == positive_map.size(0))
                for i, span_map_i in enumerate(_all_span_map_flattern):
                    if len(span_map_i) == 0:
                        seq_len = min(max_span_num, positive_map.size(1))
                        span_map[i, :seq_len] = positive_map[i, :seq_len] # use the original positive map in this case!
                    else:
                        span_map[i, :len(span_map_i)] = span_map_i

            
            # Step 5. Override
            positive_map = span_map
            language_dict_features["masks"] = span_masks
            language_dict_features["embedded"] = span_features
            language_dict_features["hidden"] = span_features
            if targets is not None:
                for i in targets:
                    if "span_map" in i.extra_fields:
                        i.extra_fields["positive_map"] = i.extra_fields["span_map"] # override if span

        # rpn force boxes
        if targets:
            targets = [target.to(device) for target in targets if target is not None]

        if self.force_boxes:
            proposals = []
            for t in targets:
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                proposals.append(tb)
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                _, proposal_losses, fused_visual_features = self.rpn(
                    images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4
                )
            elif self.training:
                null_loss = 0
                for key, param in self.rpn.named_parameters():
                    null_loss += 0.0 * param.sum()
                proposal_losses = {("rpn_null_loss", null_loss)}
        else:
            proposals, proposal_losses, fused_visual_features = self.rpn(
                images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4
            )

        if self.roi_heads:
            if not self.training:
                assert len(proposals) == 1, "Evaluation batch size per GPU should be 1!"
                if len(proposals[0]) == 0:
                    return proposals
            if self.cfg.MODEL.ROI_MASK_HEAD.PREDICTOR.startswith("VL"):
                if self.training:
                    # "Only support VL mask head right now!!"
                    assert len(targets) == 1 and len(targets[0]) == len(
                        positive_map
                    ), "shape match assert for mask head!!"
                    # Not necessary but as a safe guard:
                    # use the binary 0/1 positive map to replace the normalized positive map
                    targets[0].add_field("positive_map", positive_map)
            # TODO: make sure that this use of language_dict_features is correct!! Its content should be changed in self.rpn
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                x, result, detector_losses = self.roi_heads(
                    fused_visual_features,
                    proposals,
                    targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None,
                )
            else:
                x, result, detector_losses = self.roi_heads(
                    visual_features,
                    proposals,
                    targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None,
                )
        else:
            # RPN-only models don't have roi_heads
            x = visual_features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
