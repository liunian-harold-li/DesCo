# Utilities for converting object detection data into grounding data
import numpy as np
import torch
import pdb, json, random, re
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
from collections import defaultdict
from tqdm import tqdm
from maskrcnn_benchmark.data.datasets.parse_gpt import GPTOutputParser

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

def sanity_check_target_after_processing(target):
    assert(len(target.bbox) == len(target.extra_fields["boxes"]))


def convert_od_to_grounding_simple(
    target, 
    image_id, 
    ind_to_class, 
    disable_shuffle=True, 
    add_detection_prompt=False, 
    separation_tokens=" ",
    caption_prompt=None):
    """
    Convert object detection data into grounding data format, on the fly.
    ind_to_class: {0: "__background__", 1 : "person" ...}, contiguous id
    """

    def generate_sentence_from_labels(positive_label_list, negative_label_list, disable_shuffle=True):
        label_to_positions = {}
        label_list = negative_label_list + positive_label_list
        if not disable_shuffle:
            random.shuffle(label_list)
            assert (caption_prompt is None), "Should not specify caption_prompt when shuffle is enabled!!"  # avoid potential bug

        if add_detection_prompt:
            pheso_caption = "object detection : "
        else:
            pheso_caption = ""
        
        for index, label in enumerate(label_list):
            if caption_prompt is not None:
                pheso_caption += caption_prompt[index]['prefix']

            start_index = len(pheso_caption)
            if caption_prompt is not None:
                pheso_caption += clean_name(caption_prompt[index]['name'])
            else:
                pheso_caption += clean_name(ind_to_class[label])  # NOTE: slight change...
            end_index = len(pheso_caption)

            if caption_prompt is not None:
                pheso_caption += caption_prompt[index]['suffix']

            # e.g.: pheso_caption = "cat dog", where cat is label 4, and dog is label 17
            # label_to_positions: {4: (0, 3), 17: (4, 7)}
            label_to_positions[label] = [start_index, end_index]

            if index != len(label_list) - 1:
                pheso_caption += separation_tokens

        return label_to_positions, pheso_caption

    label_list = list(sorted(ind_to_class.keys()))  # do not include the background
    label_to_positions, pheso_caption = generate_sentence_from_labels(
        positive_label_list=label_list,
        negative_label_list=[],
        disable_shuffle=disable_shuffle
    )

    new_target = []

    '''
    Convert into:
    {'area': 10506.0, 'iscrowd': 0, 'image_id': 571335, 'category_id': 1, 'id': 2999421, 'bbox': [221, 319, 103, 102], 'tokens_positive': [[0, 3]]} 
    tokens_positive is the char position
    '''
    areas = target.area()
    greenlight_span_for_masked_lm_objective = []
    for i in range(len(target)):
        new_target_i = {}
        new_target_i["area"] = areas[i]
        new_target_i["iscrowd"] = 0
        new_target_i["image_id"] = image_id
        new_target_i["category_id"] = target.extra_fields["labels"][i].item()
        new_target_i["id"] = None
        new_target_i['bbox'] = target.bbox[i].numpy().tolist()

        label_i = target.extra_fields["labels"][i].item()

        if label_i in label_to_positions:  # NOTE: Only add those that actually appear in the final caption
            new_target_i["tokens_positive"] = [label_to_positions[label_i]]
            new_target.append(new_target_i)
            greenlight_span_for_masked_lm_objective.append(label_to_positions[label_i])
    
    return new_target, pheso_caption, greenlight_span_for_masked_lm_objective


def check_for_positive_overflow(target, ind_to_class, tokenizer, max_seq_length=256):
    # NOTE: Only call this function for OD data; DO NOT USE IT FOR GROUNDING DATA
    # NOTE: called only in coco_dt

    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_set = set()
    for i in range(len(target)):
        label_i = target.extra_fields["labels"][i].item()
        positive_label_set.add(label_i)
    positive_label_list = list(positive_label_set)

    # random shuffule so we can sample different annotations at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(ind_to_class[label]) + ". " # "dog. "

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_seq_length:
            break
        else:
            kept_lables.append(label)
    
    ## filter boxes
    keep_box_index = []
    for i in range(len(target)):
        label_i = target.extra_fields["labels"][i].item()
        if label_i in kept_lables:
            keep_box_index.append(i)
    
    keep_box_index = torch.LongTensor(keep_box_index)

    target = target[keep_box_index] ## filter boxes

    return target, length



def _label_drop_with_length_limit(label_list, ind_to_class, length_limit, tokenizer):
    screened_label_list = []
    random.shuffle(label_list) # randomly drop labels
    for label in label_list:
        label_text = clean_name(ind_to_class[label]) + ". " # "dog. "

        tokenized = tokenizer.tokenize(label_text)
        
        length_limit -= len(tokenized)

        if length_limit > 0: 
            screened_label_list.append(label) # keep this label
        else:
            break
    return screened_label_list

def _randomv1_od_to_grounding(all_labels, ind_to_class, max_seq_length, max_num_labels, tokenizer):
    
    label_num = np.random.randint(1, max_num_labels)
    selected_label_list = np.random.choice(all_labels, label_num, replace=False)
    screened_label_list = _label_drop_with_length_limit(selected_label_list, ind_to_class, max_seq_length, tokenizer)

    return screened_label_list

def _randomv2_od_to_grounding(all_labels, ind_to_class, max_seq_length, max_num_labels, tokenizer, positive_label_set):
    
    full_positive = len(positive_label_set)
    full_negative = max_num_labels - full_positive

    outer_prob = random.random()

    if outer_prob < 0.8:
        num_negatives = full_negative
        num_positives = full_positive
    elif outer_prob < 0.9:
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
    screened_label_list = _label_drop_with_length_limit(selected_label_list, ind_to_class, max_seq_length, tokenizer)
    return screened_label_list

def od_to_grounding_optimized_streamlined(
        target,
        image_id,
        ind_to_class,
        tokenizer,
        od_to_grounding_version,
    ):

    if od_to_grounding_version == "random.v1":
        separation_tokens = ". "
        max_num_labels = 85
        max_seq_length = 254
    elif od_to_grounding_version == "random.v2":
        separation_tokens = ". "
        max_num_labels = 60
        max_seq_length = 254

    def generate_senetence_given_labels(
            label_list,
            disable_shuffle=False,
        ):
        label_to_positions = {}
        if not disable_shuffle:
            random.shuffle(label_list)

        pheso_caption = ""

        for index, label in enumerate(label_list):

            start_index = len(pheso_caption)
            pheso_caption += clean_name(ind_to_class[label])  # NOTE: slight change...
            end_index = len(pheso_caption)

            # e.g.: pheso_caption = "cat dog", where cat is label 4, and dog is label 17
            # label_to_positions: {4: (0, 3), 17: (4, 7)}
            label_to_positions[label] = [start_index, end_index]

            if index != len(label_list) - 1:
                pheso_caption += separation_tokens

        return label_to_positions, pheso_caption


    if od_to_grounding_version == "random.v1":
        # all_labels, ind_to_class, max_seq_length, max_num_labels, tokenizer
        screened_label_list = _randomv1_od_to_grounding(
            all_labels = list(ind_to_class.keys()),
            ind_to_class = ind_to_class,
            max_seq_length = max_seq_length,
            max_num_labels = max_num_labels,
            tokenizer = tokenizer,
        )
        label_to_positions, pheso_caption = generate_senetence_given_labels(
            label_list=screened_label_list, )
    elif od_to_grounding_version == "random.v2":
        screened_label_list = _randomv2_od_to_grounding(
            all_labels = list(ind_to_class.keys()),
            ind_to_class = ind_to_class,
            max_seq_length = max_seq_length,
            max_num_labels = max_num_labels,
            tokenizer = tokenizer,
            positive_label_set = set(target.extra_fields["labels"].tolist()),
        )
        label_to_positions, pheso_caption = generate_senetence_given_labels(
            label_list=screened_label_list, )
    else:
        raise NotImplementedError
    
    new_target = []

    '''
    Convert into:
    {'area': 10506.0, 'iscrowd': 0, 'image_id': 571335, 'category_id': 1, 'id': 2999421, 'bbox': [221, 319, 103, 102], 'tokens_positive': [[0, 3]]} 
    tokens_positive is the char position
    '''
    areas = target.area()
    greenlight_span_for_masked_lm_objective = []
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
            new_target.append(new_target_i)
            greenlight_span_for_masked_lm_objective.append(label_to_positions[label_i])

    # reconstruct the target 
    new_target_boxlist = BoxList(torch.as_tensor([i['bbox'] for i in new_target]).reshape(-1, 4), target.size, mode="xyxy")
    new_target_boxlist.add_field("labels", torch.as_tensor([i['category_id'] for i in new_target]))

    return new_target, pheso_caption, greenlight_span_for_masked_lm_objective, label_to_positions, new_target_boxlist



def convert_object_detection_to_grounding_optimized_for_od(
        target,
        image_id,
        ind_to_class,
        disable_shuffle,
        add_detection_prompt,
        add_detection_prompt_advanced,
        random_sample_negative,
        control_probabilities,
        restricted_negative_list=None,
        separation_tokens=" ",
        max_num_labels=-1,
        max_seq_length=256,
        tokenizer=None,
        positive_caption_length=0,
        od_to_grounding_version = "vanilla",
):
    '''
    ind_to_class: {0: "__background__", 1 : "person" ...}
    target:

    restricted_negative_list : for datasets with restricted negatives, sample only the negatives

    Convert object detection data into grounding data format, on the fly.

    Control options:
        1. add_detection_prompt: add "object detection : " to the front of the prompt
        2. num_negatives: randomly sampled negative classes
        3. num_positives: how many positives to keep (-1 means do not cut any)

    Probabilities to generate the control options:

        a. probability_one_negative: only give one negative class to mimic evaluation
        b. probability_one_positive: only give one positive class to mimic evaluation
        c. probability_full: add both all positive and all negatives
        d. other:
            randomly sample some negatives and some positives
            The below control options are independent of each other:
            - probability_random_negative: probability of randomly sample X negatives
            - probability_random_positive: probability of randomly sample some positives
    
            
    NEW: control version; we will have a few pre-defined control versions; and we only need to sepecify the version instead of all the detailed paratmeters
    '''
    def generate_senetence_given_labels(
            positive_label_list,
            negative_label_list,
            prompt_engineer_version="v2",
            disable_shuffle=False,
            positive_question_probability=0.6,
            negative_question_probability=0.8,
            full_question_probability=0.5):

        '''
        v3: with simple prompt such as "there are", "are there?"
        v4: try to merge some are there / there are together, to avoid sequence being too long
        '''

        label_to_positions = {}

        assert (prompt_engineer_version == "v2")
        num_negatives = len(negative_label_list)
        num_positives = len(positive_label_list)
        label_list = negative_label_list + positive_label_list
        if not disable_shuffle:
            random.shuffle(label_list)

        if add_detection_prompt:
            if add_detection_prompt_advanced and (num_negatives == 0 or num_positives == 0) and not disable_shuffle:
                pheso_caption = "object detection query : "
            else:
                pheso_caption = "object detection : "
        else:
            pheso_caption = ""

        for index, label in enumerate(label_list):

            start_index = len(pheso_caption)

            pheso_caption += clean_name(ind_to_class[label])  # NOTE: slight change...
            end_index = len(pheso_caption)

            # e.g.: pheso_caption = "cat dog", where cat is label 4, and dog is label 17
            # label_to_positions: {4: (0, 3), 17: (4, 7)}
            label_to_positions[label] = [start_index, end_index]

            if index != len(label_list) - 1:
                pheso_caption += separation_tokens

        return label_to_positions, pheso_caption

    
    positive_label_set = set()
    for i in range(len(target)):
        label_i = target.extra_fields["labels"][i].item()
        positive_label_set.add(label_i)

    if restricted_negative_list is None:
        valid_negative_indexes = list(ind_to_class.keys())
    else:
        valid_negative_indexes = restricted_negative_list

    all_vailable_labels = positive_label_set | set(valid_negative_indexes)

    if disable_shuffle:
        label_list = list(sorted(ind_to_class.keys()))[1:]  # do not include the background
        label_to_positions, pheso_caption = generate_senetence_given_labels(
            positive_label_list=label_list,
            negative_label_list=[],
            disable_shuffle=True)
    elif od_to_grounding_version == "random":
        # all_labels, ind_to_class, max_seq_length, max_num_labels, tokenizer
        screened_label_list = _random_od_to_grounding(
            all_labels = all_vailable_labels,
            ind_to_class = ind_to_class,
            max_seq_length = max_seq_length,
            max_num_labels = max_num_labels,
            tokenizer = tokenizer,
        )
        label_to_positions, pheso_caption = generate_senetence_given_labels(
            positive_label_list=screened_label_list)
    else:
        full_positive = len(positive_label_set)
        if max_num_labels <= 0:
            full_negative = random_sample_negative
        else:
            full_negative = max(min(max_num_labels-full_positive, random_sample_negative), 0)

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        num_negatives, num_positives = generate_control_options_given_probabilities(
            control_probabilities=control_probabilities,
            full_positive=full_positive,
            full_negative=full_negative)
        # num_positives not used

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)
            for i in np.random.choice(valid_negative_indexes, size=num_negatives, replace=False):
                # label_sets.add(i)
                if i not in positive_label_set:
                    negative_label_list.add(i)

        # Keep all positives; ignoring num_positives
        positive_label_list = list(positive_label_set)
        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)  # e.g.: [17, 1, 13] where each number is the class name
        random.shuffle(negative_label_list)

        # Do a pre-screen. If we cannot afford this many negatives, we will sample less
        negative_max_length = max_seq_length - positive_caption_length
        screened_negative_label_list = []
        for negative_label in negative_label_list:
            label_text = clean_name(ind_to_class[negative_label]) + ". " # "dog. "

            tokenized = tokenizer.tokenize(label_text)
            
            negative_max_length -= len(tokenized)

            if negative_max_length > 0: 
                screened_negative_label_list.append(negative_label) # keep this negative
            else:
                break
        negative_label_list = screened_negative_label_list

        label_to_positions, pheso_caption = generate_senetence_given_labels(
            positive_label_list=positive_label_list,
            negative_label_list=negative_label_list)

    new_target = []

    '''
    Convert into:
    {'area': 10506.0, 'iscrowd': 0, 'image_id': 571335, 'category_id': 1, 'id': 2999421, 'bbox': [221, 319, 103, 102], 'tokens_positive': [[0, 3]]} 
    tokens_positive is the char position
    '''
    areas = target.area()
    greenlight_span_for_masked_lm_objective = []
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

        if label_i in label_to_positions:  # NOTE: Only add those that actually appear in the final caption
            new_target_i["tokens_positive"] = [label_to_positions[label_i]]
            new_target.append(new_target_i)
            greenlight_span_for_masked_lm_objective.append(label_to_positions[label_i])

    return new_target, pheso_caption, greenlight_span_for_masked_lm_objective, label_to_positions


def generate_control_options_given_probabilities(
        control_probabilities,
        full_positive,
        full_negative):
    
    # The function was originally designed to perform data augmentation by randomly dropping negative and positive classes. Later, we decided to only consider dropping negative classes. So the returned 'num_positives' by this function will be ignored.

    outer_prob = random.random()

    probability_one_negative = control_probabilities[0]
    probability_one_positive = control_probabilities[1]
    probability_full = control_probabilities[2]
    probability_drop_positive = control_probabilities[3]

    assert(probability_drop_positive == 0)

    if outer_prob < probability_one_negative:
        # a. probability_one_negative: only give one negative class to mimic evaluation (10%)
        num_negatives = 1
        num_positives = 0
    elif outer_prob < probability_one_positive + probability_one_negative:
        # b. probability_one_positive: only give one positive class to mimic evaluation (10%)
        num_negatives = 0
        num_positives = 1
    elif outer_prob < probability_full + probability_one_positive + probability_one_negative:
        # c. probability_full: add both all positive and all negatives (20%)
        num_negatives = full_negative
        num_positives = full_positive
    else:
        if random.random() < 1.0:  # - probability_random_negative: probability of randomly sample X negatives (100%)
            num_negatives = np.random.choice(max(1, full_negative)) + 1  # mininum 1
        else:
            num_negatives = full_negative  # Full

        if random.random() < probability_drop_positive:  #
            num_positives = np.random.choice(max(1, full_positive)) + 1
        else:
            num_positives = full_positive  # Full

    return num_negatives, num_positives
