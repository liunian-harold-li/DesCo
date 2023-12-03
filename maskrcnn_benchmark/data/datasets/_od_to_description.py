# Utilities for converting object detection data into grounding data
import numpy as np
import torch
import pdb, json, random, re
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
from collections import defaultdict
from tqdm import tqdm
from maskrcnn_benchmark.data.datasets.parse_gpt import GPTOutputParser
from ._pos_rate import PosRateController, PosRateControllerLength, PosRateControllerV2
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert(counter == len(lst))

    return all_

def clean_name(name):

    def _clean_name(name):
        name = re.sub(r"\(.*\)", "", name)
        name = re.sub(r"_", " ", name)
        name = re.sub(r"  ", " ", name)
        return name

    if ":" in name:
        obj_name, part_name = name.split(":")
        obj_name = _clean_name(obj_name)
        part_name = _clean_name(part_name)
        return  part_name + " of " + obj_name
    else:
        return _clean_name(name)

def clean_string(input_string):
    # remove leading and trailing spaces
    input_string = input_string.strip()
    # remove trailing ";" and "."
    input_string = re.sub(r";$", "", input_string)
    input_string = re.sub(r"\.$", "", input_string)
    return input_string


class DetectionToGrounding():
    '''
    Convert detection data into grounding data;
    Construct prompts for training and inference;
    '''
    def __init__(self, version):
        pass


class DescriptionConverter():
    def __init__(
            self,
            description_file,
            od_to_grounding_version,
            categories,
            ind_to_class,
            similarity_file = None,):
        self.description_file = description_file
        self.od_to_grounding_version = od_to_grounding_version
        self.categories = categories
        self.name_to_def = {}
        for cat in self.categories:
            try:
                self.name_to_def[cat["name"]] = cat["def"]
            except:
                pass
        if description_file is not None:
            with open(description_file, "r") as f:
                self.description_list = json.load(f)

            self.gpt_parser = GPTOutputParser(od_to_grounding_version.split(".")[-1])
            #self.preparse_descriptions()

            self.category_name_to_description = {}
            for i in self.description_list:
                # {'object': 'aerosol_can', 'object_id': 1, 'gpt3_output': '"\n{\"type\": \"vegetable\", \n\"description\": \"cylindrical, green, smooth; could have brown and rough stems; could be sliced into round pieces; could has green leaves\", \n\"similar objects\": [\"cucumber\", \"eggplant\", \"green bean\"]}"}'}
                self.category_name_to_description[i["object"]] = i

        # stats to print warning
        self.drop_label_count = 0
        self.all_count = 0

        self.ind_to_class = ind_to_class

        if similarity_file is not None:
            with open(similarity_file, "r") as f:
                self.category_name_to_similarity = json.load(f)

        if "control_pos" in od_to_grounding_version:
            self.pos_rate_controller = PosRateControllerLength(max_length = 9, center_length=8)

        self.pos_rates = []

    def inference_od_to_grounding(self, dataset, cfg, negative_label=None, negative_index=None):
        categories = dataset.categories()

        labels = []
        label_list = []
        keys = list(categories.keys())
        keys.sort()
        if negative_label is not None:
            labels.append(negative_label)
            label_list.append(categories[negative_label])
        else:
            for i in keys:
                labels.append(i)
                label_list.append(categories[i])

        if cfg.TEST.CHUNKED_EVALUATION != -1:
            labels = chunks(labels, cfg.TEST.CHUNKED_EVALUATION)
            label_list = chunks(label_list, cfg.TEST.CHUNKED_EVALUATION)
        else:
            labels = [labels]
            label_list = [label_list]

        all_queries = []
        all_positive_map_label_to_token = []

        from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "roberta-base":
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True)
        else:
            tokenizer = None
            raise NotImplementedError

        for i in tqdm(range(len(labels))):
            labels_i = labels[i]
            label_list_i = label_list[i]
            query_i, positive_map_label_to_token_i = self._create_queries_and_maps(
                labels_i, label_list_i, additional_labels = cfg.DATASETS.SUPRESS_QUERY if cfg.DATASETS.USE_SUPRESS_QUERY else None, cfg = cfg, tokenizer = tokenizer, negative_label=negative_label, negative_index=negative_index)

            all_queries.append(query_i)
            all_positive_map_label_to_token.append(positive_map_label_to_token_i)
        print("All queries", all_queries)
        return all_queries, all_positive_map_label_to_token

    def _create_queries_and_maps(self, labels, label_list, additional_labels = None, cfg = None, tokenizer = None, negative_label=None, negative_index=None):

        label_to_positions, objects_query, label_to_spans, label_to_positive_spans = self._generate_senetence_given_labels(labels, self.ind_to_class, disable_shuffle=True, negative_label=negative_label, negative_index=negative_index)
        tokens_positive = [[label_to_positions[i]] for i in labels]
        print(objects_query)
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased" or cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "roberta-base":
            tokenized = tokenizer(objects_query, return_tensors="pt")
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            tokenized = tokenizer(objects_query,
                                max_length=cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                truncation=True,
                                return_tensors="pt")
        else:
            raise NotImplementedError
        # Create the mapping between tokenized sentence and the original label
        positive_map_token_to_label, positive_map_label_to_token = self._infer_create_positive_dict(
            tokenized,
            tokens_positive,
            labels=labels)  # from token position to original label

        # Create the spans, and the span maps
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.SPAN_VERSION is not None:
            if "sep_span" in self.od_to_grounding_version:
                all_spans = []
                for k, v in label_to_spans.items():
                    all_spans.append(v)
                all_spans = sorted(all_spans, key=lambda x: x[0][0])
                all_spans_flattered = []
                for i in all_spans:
                    all_spans_flattered += i

            else:
                all_spans = []
                for k, v in label_to_spans.items():
                    all_spans += v
                # sort the spans based on the start index
                all_spans = sorted(all_spans, key=lambda x: x[0])
                all_spans_flattered = all_spans

            span_map = self._infer_create_span_map(all_spans_flattered, label_to_positive_spans)
            positive_map_label_to_token = (positive_map_label_to_token, span_map, all_spans)

        return objects_query, positive_map_label_to_token


    def _infer_create_positive_dict(self, tokenized, tokens_positive, labels):
        """construct a dictionary such that positive_map[i] = j, iff token i is mapped to j label"""
        positive_map = defaultdict(int)

        # Additionally, have positive_map_label_to_tokens
        positive_map_label_to_token = defaultdict(list)

        for j, tok_list in enumerate(tokens_positive):
            for (beg, end) in tok_list:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
                if beg_pos is None:
                    try:
                        beg_pos = tokenized.char_to_token(beg + 1)
                        if beg_pos is None:
                            beg_pos = tokenized.char_to_token(beg + 2)
                    except:
                        beg_pos = None
                if end_pos is None:
                    try:
                        end_pos = tokenized.char_to_token(end - 2)
                        if end_pos is None:
                            end_pos = tokenized.char_to_token(end - 3)
                    except:
                        end_pos = None
                if beg_pos is None or end_pos is None:
                    continue

                assert beg_pos is not None and end_pos is not None
                for i in range(beg_pos, end_pos + 1):
                    positive_map[i] = labels[j]  # because the labels starts from 1
                    positive_map_label_to_token[labels[j]].append(i)
                # positive_map[j, beg_pos : end_pos + 1].fill_(1)
        return positive_map, positive_map_label_to_token  # / (positive_map.sum(-1)[:, None] + 1e-6)

    def _infer_create_span_map(self, all_spans, label_to_positive_spans):
        # input: boxes, num_box to spans mapping
        # output: boxes, spans, num_box to spans mapping
        index_spans = {}
        for i, span in enumerate(all_spans):
            index_spans[tuple(span)] = i

        span_map = defaultdict(list)
        for label, spans in label_to_positive_spans.items():
            span_map[label].extend([index_spans[tuple(span)] for span in spans])
        return span_map


    def train_od_to_grounding(self,
            target,
            image_id,
            ind_to_class,
            tokenizer,
            random_sample_negative=8):

        '''
        1. _random_label_selection: select which labels to include in the caption
        2. _generate_senetence_given_labels: generate a caption given the selected labels
        3. _create_new_target: create the new target (optionally drop the boxes if positive label is missing)
        '''

        separation_tokens = ". "
        max_num_labels = 8
        if "description.gpt" in self.od_to_grounding_version:
            max_num_labels = 8
        if "description.baseline" in self.od_to_grounding_version:
            max_num_labels = 8

        max_seq_length = 254
        if "sep_span" in self.od_to_grounding_version:
            max_num_labels = random_sample_negative #
            if random_sample_negative == 8:
                max_seq_length = 254 # hacky to reproduce the results
            else:
                max_seq_length = int(254 * random_sample_negative / 8) # hacky to maintain the results

        screened_label_list = self._random_label_selection(
                all_labels = list(ind_to_class.keys()),
                ind_to_class = ind_to_class,
                max_seq_length = max_seq_length,
                max_num_labels = max_num_labels,
                tokenizer = tokenizer,
                positive_label_set = set(target.extra_fields["labels"].tolist()),
            )
        label_to_positions, pheso_caption, label_to_spans, label_to_positive_spans = self._generate_senetence_given_labels(
                label_list=screened_label_list,
                ind_to_class=ind_to_class,)

        new_target, greenlight_span_for_masked_lm_objective, new_target_boxlist = self._create_new_target(target, image_id, label_to_positions, label_to_spans)
        return new_target, pheso_caption, greenlight_span_for_masked_lm_objective, label_to_positions, new_target_boxlist

    def _random_label_selection(self, all_labels, ind_to_class, max_seq_length, max_num_labels, tokenizer, positive_label_set):

        if "complete_random" in self.od_to_grounding_version:
            random_label_num = np.random.choice(max_num_labels + 1)
            shuffle_label_list = [i for i in all_labels]
            random.shuffle(shuffle_label_list)
            screened_label_list = shuffle_label_list[:random_label_num]
            return screened_label_list

        full_positive = len(positive_label_set)
        full_negative = max_num_labels - full_positive

        outer_prob = random.random()

        if "control_pos" in self.od_to_grounding_version:
            num_positives, num_negatives = self.pos_rate_controller(full_positive, len(all_labels))

        elif "allow_zero" in self.od_to_grounding_version:
            if outer_prob < 0.5:
                num_negatives = full_negative
                num_positives = full_positive
            elif outer_prob < 0.6:
                num_negatives = np.random.choice(max(1, full_negative + 1))  # mininum 1
                num_positives = full_positive
            else:
                num_positives = np.random.choice(max(1, full_positive + 1)) # mininum 1
                num_negatives = full_negative
        elif "keep_all" in self.od_to_grounding_version:
            num_positives = full_positive
            num_negatives = full_negative
        else:
            if outer_prob < 0.5:
                num_negatives = full_negative
                num_positives = full_positive
            elif outer_prob < 0.6:
                num_negatives = np.random.choice(max(1, full_negative)) + 1  # mininum 1
                num_positives = full_positive
            else:
                num_positives = np.random.choice(max(1, full_positive)) + 1  # mininum 1
                num_negatives = full_negative

        # Keep some negatives
        negative_label_list = [label for label in all_labels if label not in positive_label_set]
        random.shuffle(negative_label_list)
        negative_label_list = negative_label_list[:num_negatives]

        # Keep some positives
        positive_label_list = list(positive_label_set)
        random.shuffle(positive_label_list)
        positive_label_list = positive_label_list[:num_positives]

        selected_label_list = positive_label_list + negative_label_list
        screened_label_list = self._label_drop_with_length_limit(selected_label_list, ind_to_class, max_seq_length, tokenizer)

        # calculate the current positive rate
        _screened_label_list = set(screened_label_list)
        _pos_label_list = set(positive_label_list).intersection(_screened_label_list)
        if "control_pos" in self.od_to_grounding_version:
            self.pos_rate_controller.update_true_pos_rate(len(_pos_label_list), max(len(screened_label_list), 1.0))

        return screened_label_list

    def _generate_sentence(self, label, ind_to_class, pheso_caption = "", force_mode = None, negative_label=None, negative_index=None):
        start_index = len(pheso_caption)
        category_name = ind_to_class[label]
        clean_category_name = clean_name(category_name)

        # generate_version
        od_to_grounding_version = ".".join(self.od_to_grounding_version.split(".")[:3])
        range_version = "partial"

        if od_to_grounding_version == "description.gpt.v10":
            if negative_label is not None:
                if negative_index == 0:
                    description = self.category_name_to_description[category_name]["gpt3_output"]
                else:
                    from copy import deepcopy
                    description = deepcopy(self.category_name_to_description[category_name]["gpt3_output"])
                    try:
                        neg_desc = self.category_name_to_description[category_name]['chatgpt_negatives'].split('\n')[negative_index-1]
                    except:
                        neg_desc = self.category_name_to_description[category_name]['chatgpt_negatives'].split('\n')[-1]
                    description = json.loads(description)
                    description['description'] = neg_desc
                    description = json.dumps(description)
            else:
                description = self.category_name_to_description[category_name]["gpt3_output"]
            if "infer" in self.od_to_grounding_version:
                prob = 0.0
            else:
                prob = random.random()

            if "independent" in self.od_to_grounding_version:
                func = self.gpt_parser.form_span_independent
            else:
                func = self.gpt_parser.form_span

            if prob < 0.5:
                des_caption_i, end_index, spans, positive_spans = func(
                    noun=clean_category_name,
                    description=description,
                    type = "vanilla_span",
                    start_index = start_index,
                    positive_range = range_version,
                    od_to_grounding_version=self.od_to_grounding_version)
            else:
                des_caption_i, end_index, spans, positive_spans = func(
                    noun=clean_category_name,
                    description=description,
                    type = "remove_noun_span",
                    start_index = start_index,
                    positive_range = range_version,
                    od_to_grounding_version=self.od_to_grounding_version)
            end_index = len(pheso_caption) + end_index
            pheso_caption += des_caption_i
            return pheso_caption, (start_index, end_index), spans, positive_spans

        else:
            raise NotImplementedError


        return pheso_caption, (start_index, end_index), None, None

    def _generate_senetence_given_labels(
            self,
            label_list,
            ind_to_class,
            disable_shuffle=False,
            negative_label=None,
            negative_index=None,
        ):
        '''
        given a label list, generate a caption (with descriptions)
        also generate a label_to_positions dictionary
        '''

        label_to_positions = {}
        label_to_spans = {}
        label_to_positive_spans = {} #
        if not disable_shuffle:
            random.shuffle(label_list)

        pheso_caption = "Detect: "

        for index, label in enumerate(label_list):

            pheso_caption, (start_index, end_index), spans, positive_spans = self._generate_sentence(label, ind_to_class, pheso_caption, negative_label=negative_label, negative_index=negative_index)

            # need to record the spans

            label_to_positions[label] = (start_index, end_index)
            label_to_spans[label] = spans
            label_to_positive_spans[label] = positive_spans
        return label_to_positions, pheso_caption, label_to_spans, label_to_positive_spans

    def _create_new_target(self, target, image_id, label_to_positions, label_to_spans = None, label_to_positive_spans = None):
        new_target = []
        areas = target.area()
        #greenlight_span_for_masked_lm_objective = []
        for i in range(len(target)):
            new_target_i = {}
            new_target_i["area"] = areas[i]
            new_target_i["iscrowd"] = 0
            new_target_i["image_id"] = image_id
            new_target_i["category_id"] = target.extra_fields["labels"][i].item()
            new_target_i["id"] = None
            new_target_i['bbox'] = target.bbox[i].numpy().tolist()

            label_i = target.extra_fields["labels"][i].item()
            new_target_i["original_od_label"] = label_i

            if label_i in label_to_positions:  # NOTE: Only add labels that actually appear in the final caption
                new_target_i["tokens_positive"] = [label_to_positions[label_i]]

                if label_to_positive_spans is not None: # NOTE: Use label_to_positive_spans instead of label_to_spans; as certain spans can be negative
                    new_target_i["spans_positive"] = label_to_positive_spans[label_i]
                new_target.append(new_target_i)
                #greenlight_span_for_masked_lm_objective.append(label_to_positions[label_i])

        if "sep_span" in self.od_to_grounding_version:
            all_spans = []
            for k, v in label_to_spans.items():  # NOTE: Use the label_to_spans to get all the spans
                all_spans.append(v)
            all_spans = sorted(all_spans, key=lambda x: x[0][0])

            # max_spans_per_seq = max([len(i) for i in all_spans])
            # all_spans_tensor = torch.ones((len(all_spans), max_spans_per_seq, 2), dtype=torch.long) * -1
            # for i, spans in enumerate(all_spans):
            #     for j, span in enumerate(spans):
            #         all_spans_tensor[i, j, :] = torch.as_tensor(span)

        elif "span" in self.od_to_grounding_version:
            all_spans = []
            for k, v in label_to_spans.items():
                all_spans += v
            # sort the spans based on the start index
            all_spans = sorted(all_spans, key=lambda x: x[0])
            all_spans = torch.as_tensor(all_spans)
        else:
            all_spans = None

        # reconstruct the target
        new_target_boxlist = BoxList(torch.as_tensor([i['bbox'] for i in new_target]).reshape(-1, 4), target.size, mode="xyxy")
        new_target_boxlist.add_field("labels", torch.as_tensor([i['category_id'] for i in new_target]))
        if all_spans is not None:
            new_target_boxlist.add_field("spans", all_spans)
        greenlight_span_for_masked_lm_objective = [value for value in label_to_positions.values()]
        return new_target, greenlight_span_for_masked_lm_objective, new_target_boxlist

    def _label_drop_with_length_limit(self, label_list, ind_to_class, length_limit, tokenizer):
        screened_label_list = []
        random.shuffle(label_list) # randomly drop labels
        for label in label_list:
            pheso_caption, *_ = self._generate_sentence(label, ind_to_class, "")
            tokenized = tokenizer.tokenize(pheso_caption)

            length_limit -= len(tokenized)
            if length_limit > 0:
                screened_label_list.append(label) # keep this label
            else:
                break
        self.all_count += 1
        if len(screened_label_list) < len(label_list):
            self.drop_label_count += 1

        if self.drop_label_count / self.all_count > 0.3:
            print("Warning: {} of {} examples have dropped labels".format(self.drop_label_count, self.all_count))

        return screened_label_list
