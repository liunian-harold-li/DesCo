# Utilities for converting object detection data into grounding data
import numpy as np
import torch
import pdb, json, random, re
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
from collections import defaultdict
import json
import json
import nltk
from collections import Counter
from tqdm import tqdm
import random
import pdb
from copy import deepcopy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from maskrcnn_benchmark.data.datasets.parse_gpt import GPTOutputParser
def find_only_noun(caption: str):
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<NN.*>+}"
    #grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == "NP":
            noun_phrases.append(" ".join(t[0] for t in subtree.leaves()))

    return noun_phrases

def find_jj_noun(caption: str):
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<JJ.*>+<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == "NP":
            noun_phrases.append(" ".join(t[0] for t in subtree.leaves()))

    return noun_phrases

def remove_stop_words(caption, stop_words):

    word_tokens = caption.split(" ")
    # converts the words in word_tokens to lower case and then checks whether
    # they are present in stop_words or not
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    # with no lower case conversion
    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    return " ".join(filtered_sentence)
def rand_element(dic):
    ind = random.randint(0, len(dic) - 1)
    return list(dic.keys())[ind]


def replace_word(w, voc):
    new_w = rand_element(voc)
    while new_w == w:
        new_w = rand_element(voc)
    return new_w

def replace_pos(tags, l, vocab):
    if len(l) == 0:
        return '', ''
    ind = random.randint(0, len(l) - 1)
    ind = l[ind]
    word, tag = tags[ind]
    new_word = replace_word(word, vocab[tag])
    return word, new_word

noun_pos = set(['NN', 'NNS', 'NNP', 'NNPS'])
verb_pos = set(['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'])
adj_pos = set(['JJ', 'JJR', 'JJS'])


class CaptionAugmentation():
    def __init__(self, caption_augmentation_version, tokenizer = None, caption_vocab_file = None):
        self.caption_augmentation_version = caption_augmentation_version
        self.tokenizer = tokenizer
        # v1 and v2 are legacy experimental versions so we remove them from the code
        if self.caption_augmentation_version.startswith("v3"):
            self.augmentation = AugmentationV3(self.caption_augmentation_version, self.tokenizer, caption_vocab_file)
        elif self.caption_augmentation_version.startswith("v4"):
            self.augmentation = AugmentationV4(self.caption_augmentation_version, self.tokenizer, caption_vocab_file)
        elif self.caption_augmentation_version.startswith("mixed"):
            # format: mixed.v4-v3.4-4-2.content.v1
            self.augmentations = []
            self.rations = []
            versions = self.caption_augmentation_version.split(".")[1]
            ratios = self.caption_augmentation_version.split(".")[2]
            suffix = ".".join(self.caption_augmentation_version.split(".")[3:])
            for version in versions.split("-"):
                self.augmentations.append(CaptionAugmentation(version + "." + suffix, self.tokenizer, caption_vocab_file))
            for ratio in ratios.split("-"):
                self.rations.append(float(ratio) * 0.1)
            print(self.rations)
            print(self.augmentations)
        else:
            raise NotImplementedError

    def __call__(self, caption, target, **kwargs):
        if self.caption_augmentation_version.startswith("mixed"):
            # do a mixed augmentation
            random_prob = random.random()
            for augmentation, ratio in zip(self.augmentations, self.rations):
                if random_prob < ratio:
                    return augmentation(caption, target, **kwargs)
                random_prob -= ratio

            return caption, target, None # this is the vanilla case

        else:
            return self.augmentation(caption, target, **kwargs)

class NegativeCaptionGenerator():
    def __init__(self, caption_augmentation_version, **kwargs):
        self.caption_augmentation_version = caption_augmentation_version
        if self.caption_augmentation_version.endswith("v1"):
            self.generator = NegativeCaptionGeneratorV1(self.caption_augmentation_version, **kwargs)
        elif self.caption_augmentation_version.endswith("v2"):
            self.generator = NegativeCaptionGeneratorV2(self.caption_augmentation_version, **kwargs)
        else:
            raise NotImplementedError

    def __call__(self, caption, **kwargs):
        return self.generator(caption, **kwargs)


class NegativeCaptionGeneratorV1():
    def __init__(self, caption_augmentation_version, caption_vocab_file=None):
        self.caption_augmentation_version = caption_augmentation_version
        self.caption_vocab_file = caption_vocab_file
        self.vocab = json.load(open('tools/data_process/image_caption/vocab.json'))
        for tag in self.vocab:
            most_common = 1000
            self.vocab[tag] = dict(Counter(self.vocab[tag]).most_common(1000))
            min_cnt = 5
            self.vocab[tag] = {x: cnt for x, cnt in self.vocab[tag].items() if cnt >= min_cnt}

    def __call__(self, caption, num_negative_caption=4):
        tokens = nltk.word_tokenize(caption)
        tags = nltk.pos_tag(tokens)
        nouns = []
        verbs = []
        adjs = []
        for ind, (word, tag) in enumerate(tags):
            if tag in noun_pos:
                nouns.append(ind)
            elif tag in verb_pos:
                verbs.append(ind)
            elif tag in adj_pos:
                adjs.append(ind)
        negative_caption = []

        for i in range(random.randint(0, num_negative_caption)):
            replace_atoms = random.choice([nouns, verbs, adjs])
            word, new_word = replace_pos(tags, replace_atoms, self.vocab)
            if word == '':
                continue
            new_caption = caption.replace(word, new_word)
            negative_caption.append(new_caption)
        return negative_caption

class NegativeCaptionGeneratorV2():
    def __init__(self, caption_augmentation_version, tokenizer = None, caption_vocab_file="tools/files/llm_10K_noun_freq_mixed.json"):
        self.caption_augmentation_version = caption_augmentation_version
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = tokenizer
        with open(caption_vocab_file, 'r') as f:
            self.vocab = json.load(f)

    def parse_info(self, noun):
        # given a noun, return the category and other info
        '''
        "chrome faucet": ["Yes. 'Chrome faucet' has a tangible appearance and is a type of plumbing fixture.\nA few things that are visually similar to 'chrome faucet' but are not 'chrome faucet' are:\tbrushed nickel faucet\tstainless steel faucet\tchrome showerhead\tchrome soap dispenser\nThere are several useful visual features to tell there is 'chrome faucet' and not similar things in a photo:\tchrome finish\ton/off handles\tspout for water flow\tsingle or double handled faucet\tmounted on a sink or countertop", 57]
        '''
        noun = remove_stop_words(noun, self.stop_words)
        if noun not in self.vocab:
            return 0, [], [], ""
        info = self.vocab[noun]

        # check the format of type of thing
        if "has a tangible appearance and is" in info[0]:
            type_of_thing = info[0].split(" has a tangible appearance and is ")[-1].split(".")[0]
        elif "has a tangible appearance" in info[0]:
            type_of_thing = info[0].split(" has a tangible appearance ")[-1].split(".")[0]
        else:
            #print(info[0], "type of thing not found")
            type_of_thing = ""

        if " are:\t" in info[0]:
            similar_things = info[0].split(" are:\t")[-1].split("\nThere are several useful visual features to tell")[0].split("\t")
            similar_things = [i for i in similar_things if i.strip() != ""]
        else:
            #print(info[0], "similar things not found")
            similar_things = []

        if " and not similar things in a photo:\t" in info[0]:
            visual_feature_descriptions = info[0].split(" and not similar things in a photo:\t")[-1].split("\t")
            visual_feature_descriptions = [i for i in visual_feature_descriptions if i.strip() != ""]
        else:
            #print(info[0], "visual feature descriptions not found")
            visual_feature_descriptions = []


        return info[1], visual_feature_descriptions, similar_things, type_of_thing

    def __call__(self, caption, num_negative_caption=4):
        nouns = set(caption.split(" ")) #find_only_noun(caption)
        negative_captions = []
        for noun in nouns:
            freq, visual_feature_descriptions, similar_things, type_of_thing = self.parse_info(noun)
            if freq > 20000:
                continue
            # print(freq, noun, visual_feature_descriptions, similar_things, type_of_thing)

            if len(visual_feature_descriptions) == 0 or len(similar_things) == 0 or type_of_thing == "Yes":
                continue # did not find the noun in the vocab

            negative_captions.append(caption.replace(noun, random.choice(similar_things)))

        return negative_captions

class AugmentationV3():
    '''
    Extract the noun entity; get descriptions and confusable entities; form the new query; throw away the original caption
    '''
    def __init__(self, caption_augmentation_version, tokenizer = None, caption_vocab_file="tools/files/llm_10K_noun_freq_mixed.json"):
        self.caption_augmentation_version = caption_augmentation_version
        self.tokenizer = tokenizer
        with open(caption_vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.vocab_keys = list(self.vocab.keys())
        self.stop_words = set(stopwords.words('english'))
        self.do_augment_prob = 1.0
        self.include_name_prob = 0.5
        self.include_only_description_prob = 0.0
        self.length_limit = 800 if "span" in caption_augmentation_version else 180
        self.gpt_parser = GPTOutputParser(caption_augmentation_version.split(".")[-1])

    def parse_info(self, noun):
        # given a noun, return the category and other info
        '''
        {'type': 'human', 'description': 'female; could have long hair; could wear dresses', 'similar objects': ['girl', 'lady', 'mother']}
        '''
        noun = remove_stop_words(noun, self.stop_words)
        if noun not in self.vocab:
            return 0, [], [], ""
        info = self.vocab[noun]
        descriptions = self.gpt_parser(info[0])

        return info[1], descriptions["description"], descriptions["similar objects"], descriptions["type"]

    def get_freq(self, noun):
        noun = remove_stop_words(noun, self.stop_words)
        if noun not in self.vocab:
            return 0
        info = self.vocab[noun]
        return info[1]

    def get_similar_things(self, noun):
        noun = remove_stop_words(noun, self.stop_words)
        if noun not in self.vocab:
            return []
        info = self.vocab[noun]
        descriptions = self.gpt_parser(info[0])
        return descriptions["similar objects"]

    def form_span(self, noun):
        noun = remove_stop_words(noun, self.stop_words)
        info = self.vocab[noun]
        description = info[0]
        if random.random() < self.include_name_prob:
            #postive_span = "{}, {}".format(noun, type_of_thing)
            #final_span = "{}, {}, {}".format(noun, type_of_thing, ", ".join(similar_visual_feature_descriptions))
            final_span, end_index, spans, *_ = self.gpt_parser.form_span(noun, description, type = "vanilla_span", positive_range = "partial")
        else:
            final_span, end_index, spans, *_ = self.gpt_parser.form_span(noun, description, type = "remove_noun_span", positive_range = "partial")
        return final_span, end_index, spans

    def __call__(self, caption, target, **kwargs):
        # 1. get the categories mentioned in the caption
        original_str_spans = []
        original_nouns = defaultdict(list)
        for box_index, box in enumerate(target):
            for start, end in box["tokens_positive"]:
                original_str_spans.append(caption[start:end])
                if "nouns" in box:
                    original_nouns[caption[start:end]] = box["nouns"]
        original_str_spans = set(original_str_spans)

        #### Important structures
        positive_text_pieces = {} # mapping from positive text pieces to the original text span
        positive_text_pieces_reverse = {}
        positive_text_pieces_center_length = {}
        all_pieces = []
        text_pieces_to_spans = {} # mapping from text pieces to the spans
        all_spans = [] # all the spans, noun_num x span_num_each_noun x 2
        #####
        length_limit = self.length_limit
        original_str_spans = list(original_str_spans)
        # shuffle
        random.shuffle(original_str_spans)

        for text_span in original_str_spans:
            if len(original_nouns[text_span]) == 0:
                nouns = text_span.split(" ") #[text_span] #find_only_noun(text_span)
            else:
                nouns = original_nouns[text_span]

            for noun in nouns:
                frequency = self.get_freq(noun)
                if frequency > 10000 or frequency == 0:
                    continue

                positive_span, centern_noun_lenghth, span_locations = self.form_span(noun)
                length_limit -= len(positive_span.split(" "))
                if length_limit < 0:
                    break
                text_pieces_to_spans[positive_span] = span_locations
                positive_text_pieces[positive_span] = text_span
                positive_text_pieces_reverse[text_span] = positive_span
                positive_text_pieces_center_length[positive_span] = centern_noun_lenghth
                all_pieces.append(positive_span)

                # do the augmentation
                if "no_similar" in self.caption_augmentation_version:
                    continue # skip the similar things

                for similar_thing in self.get_similar_things(noun):
                    frequency = self.get_freq(similar_thing)
                    if frequency > 10000 or frequency == 0:
                        continue # did not find the noun in the vocab
                    negative_span, _, span_locations = self.form_span(similar_thing)
                    length_limit -= len(negative_span.split(" "))
                    if length_limit < 0:
                        break
                    all_pieces.append(negative_span)
                    text_pieces_to_spans[negative_span] = span_locations # record the span mapping

                # randomly sample some negatives

        if len(all_pieces) == 0:
            return caption, target, None

        if random.random() > self.do_augment_prob: #
            return caption, target, None

        # if we have some space left, sample more descriptions
        while length_limit > 0:
            random_noun = random.choice(self.vocab_keys)
            frequency = self.get_freq(random_noun)
            if frequency > 10000 or frequency == 0:
                continue

            negative_span, _, span_locations = self.form_span(random_noun,)
            length_limit -= len(negative_span.split(" "))
            if length_limit < 0:
                break
            all_pieces.append(negative_span) # add the negative span
            text_pieces_to_spans[negative_span] = span_locations # record the span mapping


        # 2. randomly assemble the caption
        new_target = deepcopy(target)
        random.shuffle(all_pieces)
        final_caption = ""

        # create the mapping from "text_span" to "tokens_positive"
        text_span_to_tokens_positive = {}
        for text_piece in all_pieces:
            if text_piece in positive_text_pieces:
                text_span_to_tokens_positive[positive_text_pieces[text_piece]] = (len(final_caption), len(final_caption) + positive_text_pieces_center_length[text_piece]) # only mark the centern noun as positive

            # update the spans
            cur_length = len(final_caption)

            for span in text_pieces_to_spans[text_piece]:
                span[0] = span[0] + cur_length
                span[1] = span[1] + cur_length

            final_caption += text_piece

        # update the target
        new_target = []
        for box in target:
            new_tokens_positive = []
            new_spans = []
            for start, end in box["tokens_positive"]:
                if caption[start:end] in text_span_to_tokens_positive:
                    new_tokens_positive.append(text_span_to_tokens_positive[caption[start:end]])
                    new_spans.extend(text_pieces_to_spans[positive_text_pieces_reverse[caption[start:end]]])
            if len(new_tokens_positive) != 0:
                _box = deepcopy(box)
                _box["tokens_positive"] = new_tokens_positive
                _box["spans_positive"] = new_spans
                new_target.append(_box)

        '''
        For using span representation, all that needs done is to give: spans, and spans_positive for each box
        '''

        all_spans = list(text_pieces_to_spans.values())
        all_spans = sorted(all_spans, key=lambda x: x[0][0])


        #print("V3 Augmented caption: ", final_caption)
        # Need to provide the spans
        return final_caption, new_target, all_spans

class AugmentationV4():
    def __init__(self, caption_augmentation_version, tokenizer, caption_vocab_file):
        self.caption_augmentation_version = caption_augmentation_version
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = tokenizer
        self.do_augment_prob = 0.9
        self.include_name_prob = 0.5
        self.include_only_description_prob = 0.0
        self.length_limit = 800 if "span" in caption_augmentation_version else 180
        self.gpt_parser = GPTOutputParser(caption_augmentation_version.split(".")[-1])

        with open(caption_vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.vocab_keys = list(self.vocab.keys())
        self.include_v3_augmentation = "include_v3" in caption_augmentation_version

        # do a stat
        from ._pos_rate import PosRateController, PosRateControllerLength, PosRateControllerV2
        self.pos_rate_controller = PosRateControllerV2(max_length=35, center_length = 20)

    def parse_info(self, noun):
        # given a noun, return the category and other info
        '''
        {'type': 'human', 'description': 'female; could have long hair; could wear dresses', 'similar objects': ['girl', 'lady', 'mother']}
        '''
        noun = remove_stop_words(noun, self.stop_words)
        if noun not in self.vocab:
            return 0, [], [], ""
        info = self.vocab[noun]
        descriptions = self.gpt_parser(info[0])

        return info[1], descriptions["description"], descriptions["similar objects"], descriptions["type"]

    def get_freq(self, noun):
        noun = remove_stop_words(noun, self.stop_words)
        if noun not in self.vocab:
            return 0
        info = self.vocab[noun]
        return info[1]

    def get_similar_things(self, noun):
        noun = remove_stop_words(noun, self.stop_words)
        if noun not in self.vocab:
            return []
        info = self.vocab[noun]
        descriptions = self.gpt_parser(info[0])
        return descriptions["similar objects"]

    def form_span(self, noun):
        noun = remove_stop_words(noun, self.stop_words)
        info = self.vocab[noun]
        description = info[0]
        if random.random() < self.include_name_prob:
            #postive_span = "{}, {}".format(noun, type_of_thing)
            #final_span = "{}, {}, {}".format(noun, type_of_thing, ", ".join(similar_visual_feature_descriptions))
            final_span, end_index, spans, *_ = self.gpt_parser.form_span(noun, description, type = "vanilla_span")
        else:
            final_span, end_index, spans, *_ = self.gpt_parser.form_span(noun, description, type = "remove_noun_span")
        return final_span, end_index, spans

    def simple_gpt_parser(self, gpt_output):
        '''
        Visually concrete phrases and their visual descriptions: {"beans": "a kind of vegetable, small, round, usually greeen"}
        Negative visual phrases and their visual descriptions: {"coffee beans": "a kind of vegetable, small, round, brown and dark", "beeds": "a kind of decorations, small, round, colorful"}
        Negative captions: ["the beans in the green silver cup.", "the apples in the red silicone cup.", "the beans in the red porcelain cup."]
        '''
        try:
            if "\n" not in gpt_output:
                pos_description = gpt_output[gpt_output.find("1. Visually concrete objects and descriptions:") : gpt_output.find(" 2. Objects that can be confused with the above objects:")].replace("1. Visually concrete objects and descriptions:", "").strip()
                pos_description = json.loads(pos_description)
                neg_description = gpt_output[gpt_output.find(" 2. Objects that can be confused with the above objects:") : gpt_output.find(" 3. Negative captions:")].replace(" 2. Objects that can be confused with the above objects:", "").strip()
                neg_description = json.loads(neg_description)
                neg_captions = gpt_output[gpt_output.find(" 3. Negative captions:") : ].replace(" 3. Negative captions:", "").strip().replace("</s>", "").replace("<unk>", "")
                neg_captions = json.loads(neg_captions)
            else:
                pos_description = gpt_output.split("\n")[0].split("descriptions: ")[1].strip()
                pos_description = json.loads(pos_description)

                try:
                    neg_description = gpt_output.split("\n")[1].split("descriptions: ")[1].strip()
                    neg_description = json.loads(neg_description)
                except:
                    neg_description = gpt_output.split("\n")[1].split("objects: ")[1].strip()
                    neg_description = json.loads(neg_description)

                neg_captions = gpt_output.split("\n")[2].split("captions: ")[1].strip()
                neg_captions = json.loads(neg_captions)

            return {
                "pos_description": pos_description,
                "neg_description": neg_description,
                "neg_captions": neg_captions
            }
        except:
            return {
                "pos_description": {},
                "neg_description": {},
                "neg_captions": []
            }

    @staticmethod
    def randomly_assemble_pieces_while_maintaining_spans_locations(
        caption, # the original caption
        all_pieces, # a list of text strings that will form the final caption
        positive_text_pieces, # a mapping from the positive text pieces to the original text piece
        positive_text_pieces_reverse, # reversed mapping
        positive_text_pieces_center_length, # the length of the center noun
        text_pieces_to_spans, # record the mapping from text pieces to their spans
        target, # a list of boxes, each box has a "tokens_positive" field
    ):
        final_caption = ""

        # create the mapping from "text_span" to "tokens_positive"
        text_span_to_tokens_positive = {}
        for text_piece in all_pieces:
            if text_piece in positive_text_pieces:
                text_span_to_tokens_positive[positive_text_pieces[text_piece]] = (len(final_caption), len(final_caption) + positive_text_pieces_center_length[text_piece]) # only mark the centern noun as positive

            # update the spans
            cur_length = len(final_caption)

            for span in text_pieces_to_spans[text_piece]:
                span[0] = span[0] + cur_length
                span[1] = span[1] + cur_length

            final_caption += text_piece

        # update the target
        new_target = []
        for box in target:
            new_tokens_positive = []
            new_spans = []
            for start, end in box["tokens_positive"]:
                if caption[start:end] in text_span_to_tokens_positive:
                    new_tokens_positive.append(text_span_to_tokens_positive[caption[start:end]])
                    new_spans.extend(text_pieces_to_spans[positive_text_pieces_reverse[caption[start:end]]])
            if len(new_tokens_positive) != 0:
                _box = deepcopy(box)
                _box["tokens_positive"] = new_tokens_positive
                _box["spans_positive"] = new_spans
                new_target.append(_box)

        '''
        For using span representation, all that needs done is to give: spans, and spans_positive for each box
        '''

        all_spans = list(text_pieces_to_spans.values())
        all_spans = sorted(all_spans, key=lambda x: x[0][0])
        return final_caption, new_target, all_spans


    def merge_token_posivie(self, tokens_positive):
        previous_end = -5
        current_start = -5
        new_tokens_positive = []
        for token_positive in tokens_positive:
            # try to merge tokens positive if they are continuous
            if current_start == -5: # this is the start
                current_start = token_positive[0]
                previous_end = token_positive[1]
                continue

            if token_positive[0] == previous_end + 1: # continus
                previous_end = token_positive[1]
            else:
                new_tokens_positive.append((current_start, previous_end))
                current_start = token_positive[0]
                previous_end = token_positive[1]
        new_tokens_positive.append((current_start, previous_end))
        return new_tokens_positive

    def _change_target(self, start_original_span, end_original_span, description, target, caption, centern_noun_lenghth):
        subcaptions = []
        # find if there is a match
        matched_i = False
        for box_index, box in enumerate(target):
            for start, end in box["tokens_positive"]:
                # if the tokens_positive is within the span or it contains the span
                if (start_original_span <= start and end <= end_original_span) or (start <= start_original_span and end_original_span <= end):
                    # add the description to the positive_text_pieces
                    # mark the matching between this box and this new subcaption # need to think later
                    # TODO: support partial match
                    box['tokens_positive'].append((len(caption), len(caption) + centern_noun_lenghth))
                    matched_i = True

        if matched_i:
            # add the description to the caption
            caption += description
            subcaptions.append(description)
            #negative_captions.extend(list(gpt_result["neg_description"].values()))
        return caption, subcaptions, target

    def __call__(self, caption, target, gpt3_outputs = None,):
        if gpt3_outputs is None:
            return caption, target, None # skip this augmentation

        ####
        negative_captions = []
        subcaptions = []
        original_subcaptions = []
        grouping_subcaptions = defaultdict(list)
        ####
        probablity = random.random()
        # 40% chance to only include original subcaptions and neg captions
        # 20% chance to include only v3 captions
        # 10% chance to include only v4 descriptions
        # 20% chance to include all kinds of stuff
        # 10% chance to return original
        if probablity < 0.2:
            include_v3 = False
            include_v4_des = False
            include_original = True
        elif probablity < 1.0:
            include_v3 = False
            include_v4_des = True
            include_original = True
        else:
            return caption, target, None

        # 1. do somme preprocessing; extract the subcaptions
        original_caption = deepcopy(caption)
        original_target = deepcopy(target)
        # parse the GPT outputstart_index = 0
        start_index = 0
        for i in range(len(caption)):
            if caption[i] == "." or caption[i] == "?":
                subcaption_i = caption[start_index:i+1]
                subcaptions.append(subcaption_i)
                start_index = i + 1
        if start_index != len(caption):
            # some remaining stuff
            subcaption_i = caption[start_index:]
            if subcaption_i.strip() != "":
                subcaptions.append(subcaption_i)

        original_subcaptions = deepcopy(subcaptions) # keep a copy of the original subcaptions
        for box in target:
            box['tokens_positive'] = self.merge_token_posivie(box['tokens_positive']) # merge the tokens_positive if they happen to be continuous

        if self.include_v3_augmentation and include_v3:
            # 1. get the categories mentioned in the caption
            all_nouns = []
            for box_index, box in enumerate(target):
                for start, end in box["tokens_positive"]:
                    if "nouns" in box:
                        all_nouns.extend(box["nouns"]) # if we pre-extract the nouns, we can use them
                    else:
                        all_nouns.extend(caption[start:end].split(" ")) # otherwise, we just use the tokens_positive and do a split by " "
            all_nouns = list(set(all_nouns))

            #####
            # shuffle
            random.shuffle(all_nouns)

            for noun in all_nouns:
                frequency = self.get_freq(noun)
                if frequency > 10000 or frequency == 0:
                    continue

                positive_span, centern_noun_lenghth, span_locations = self.form_span(noun)

                # find the noun in the caption
                start_i = original_caption.find(noun)
                end_i = start_i + len(noun)

                # add the positive span to the caption
                caption, subcaptions_noun, target = self._change_target(
                    start_original_span = start_i,
                    end_original_span = end_i,
                    description = positive_span,
                    target = target,
                    caption = caption,
                    centern_noun_lenghth=centern_noun_lenghth)

                if len(subcaptions_noun) != 0:
                    subcaptions.extend(subcaptions_noun)
                    # do the augmentation
                    _tmp_negs = []
                    for similar_thing in self.get_similar_things(noun):
                        frequency = self.get_freq(similar_thing)
                        if frequency > 10000 or frequency == 0:
                            continue # did not find the noun in the vocab
                        negative_span, _, span_locations = self.form_span(similar_thing)
                        negative_captions.append(negative_span)
                        _tmp_negs.append(negative_span)

                    grouping_subcaptions["v3"].append((positive_span, _tmp_negs))

        if gpt3_outputs is None:
            gpt3_outputs = {}

        ban_list = ['man', "woman", "child", "men", "women", "children", "people", "person"]
        for key, value in gpt3_outputs.items():
            try:
                gpt_result = self.simple_gpt_parser(value)
                for key_phrase, description_i in gpt_result["pos_description"].items():
                    # find the location of the span
                    start_i = caption.find(key_phrase)
                    end_i = start_i + len(key_phrase)
                    description_i = description_i + ". " if description_i[-1] != "." else description_i
                    if random.random() < 0.5:
                        description_i = key_phrase + ", " + description_i
                        center_ = 2
                    else:
                        center_ = 1
                    # find the center noun
                    center_length = len(",".join(description_i.split(",")[:center_]))
                    # else:
                    #     center_length = len(description_i)

                    # find if there is a match
                    matched_i = False
                    skip_i = False
                    for ban_noun in ban_list:
                        if ban_noun in key_phrase:
                            skip_i = True
                            break
                    if skip_i:
                        continue

                    for box_index, box in enumerate(target):
                        for start, end in box["tokens_positive"]:
                            # if the tokens_positive is within the span or it contains the span
                            if (start_i <= start and end <= end_i) or (start <= start_i and end_i <= end):
                                # add the description to the positive_text_pieces
                                # mark the matching between this box and this new subcaption # need to think later
                                # TODO: support partial match
                                box['tokens_positive'].append((len(caption), len(caption) + center_length))
                                matched_i = True

                    if matched_i and include_v4_des:
                        # add the description to the caption
                        caption += description_i
                        subcaptions.append(description_i)
                        negative_captions.extend(list(gpt_result["neg_description"].values()))
                        grouping_subcaptions["v4_des"].append((description_i, list(gpt_result["neg_description"].values())))

                # the rest are negative captions
                negative_captions.extend(gpt_result["neg_captions"])
                grouping_subcaptions["original"].append((key, gpt_result["neg_captions"]))
            except:
                pass

        for i in range(len(negative_captions)):
            if negative_captions[i].endswith(".") or negative_captions[i].endswith("?"):
                negative_captions[i] = negative_captions[i] + " "
            elif negative_captions[i].endswith(". ") or negative_captions[i].endswith("? "):
                pass
            else:
                negative_captions[i] = negative_captions[i] + ". "
        for value in grouping_subcaptions.values():
            for caps in value:
                for index in range(len(caps[1])):
                    if caps[1][index].endswith(".") or caps[1][index].endswith("?"):
                        caps[1][index] = caps[1][index] + " "
                    elif caps[1][index].endswith(". ") or caps[1][index].endswith("? "):
                        pass
                    else:
                        caps[1][index] = caps[1][index] + ". "

        if "drop_positive" in self.caption_augmentation_version:
            drop_positive_rate = 0.5
            if random.random() < 0.1: # 10% drop all the positive
                drop_positive_rate = 1.0
            drop_negative_rate = 0.0
        else:
            drop_positive_rate = 0.0
            drop_negative_rate = 0.0

        if len(subcaptions) == 0 and len(negative_captions) == 0:
            return original_caption, original_target, None

        if "control_pos" in self.caption_augmentation_version:
            # calculate on average how many captions we can afford here
            sub_cap_mean_length = np.mean([len(i.split(" ")) for i in subcaptions])
            neg_cap_mean_length = np.mean([len(i.split(" ")) for i in negative_captions])
            mean_length = (sub_cap_mean_length * len(subcaptions) + neg_cap_mean_length * len(negative_captions)) / (len(subcaptions) + len(negative_captions))
            if sub_cap_mean_length * len(subcaptions) + neg_cap_mean_length * len(negative_captions) > 200:
                # need to drop some of the captions
                max_cap_num = 180 // mean_length
            else:
                max_cap_num = -1
            if "grouping" in self.caption_augmentation_version:
                # dynamically determine the number of positive and negative captions
                final_included_groups = []
                if include_v3:
                    final_included_groups.extend(grouping_subcaptions["v3"])
                if include_v4_des:
                    final_included_groups.extend(grouping_subcaptions["v4_des"])
                if include_original:
                    final_included_groups.extend(grouping_subcaptions["original"])
                # negative captions
                grouped_positive_num = len(final_included_groups)
                grouped_negative_num = sum([len(i[1]) for i in final_included_groups])
            else:
                grouped_positive_num = len(subcaptions)
                grouped_negative_num = len(negative_captions)


            # prefered captions
            pos_num, neg_num = self.pos_rate_controller(grouped_positive_num, grouped_negative_num, max_cap_num=max_cap_num)

            if "grouping" in self.caption_augmentation_version:
                # do the preselection
                preselected_captions = set()
                preselected_captions_neg = set()
                neg_counter = 0
                # let's see if we need to drop some negative; do a preselection of negative captions
                random.shuffle(final_included_groups)
                for i in range(pos_num):
                    preselected_captions.add(final_included_groups[i][0])
                    if neg_counter < neg_num:
                        _tmp = random.randint(0, len(final_included_groups[i][1]))
                        preselected_captions_neg.update(final_included_groups[i][1][:_tmp])
                        neg_counter += _tmp

                if neg_counter < neg_num:
                    random.shuffle(negative_captions)
                    preselected_captions_neg.update(negative_captions[:neg_num - neg_counter])

                # print(include_v3, include_v4_des, include_original)
                # print(preselected_captions)
                # print(preselected_captions_neg)
                # print(pos_num, neg_num)
                # print("grouped", grouped_positive_num, grouped_negative_num)
                # print("original", len(subcaptions), len(negative_captions))

                preselected_captions.update(preselected_captions_neg)
            else:
                preselected_captions = None

        augmented_caption, location_mapping, final_pos_num, final_neg_num = random_resemble_captions( subcaptions, negative_captions, pos_num, neg_num, tokenizer = self.tokenizer, preselected_captions= preselected_captions)

        self.pos_rate_controller.update_true_pos_rate(final_pos_num, final_pos_num + final_neg_num)

        # update the target
        new_target = []
        for box in target:
            new_tokens_positive = []
            for start, end in box["tokens_positive"]:
                if start in location_mapping and end - 1 in location_mapping:
                    new_tokens_positive.append([location_mapping[start], location_mapping[end - 1] + 1]) # location of the character in the new string

            if len(new_tokens_positive) > 0: # possible the caption was dropped
                _box = deepcopy(box)
                _box["tokens_positive"] = new_tokens_positive
                new_target.append(_box)

        original_spans = []
        for box in target:
            for start, end in box["tokens_positive"]:
                original_spans.append(caption[start:end])

        augmented_spans = []
        for box in new_target:
            for start, end in box["tokens_positive"]:
                augmented_spans.append(augmented_caption[start:end])

        if len(augmented_caption) == 0:
            return original_caption, original_target, None
        return augmented_caption, new_target, None


def find_noun_phrases(caption: str):
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == "NP":
            noun_phrases.append(" ".join(t[0] for t in subtree.leaves()))

    return noun_phrases

def random_resemble_captions(
        captions, additional_captions, sub_sample_pos_num = -1, sub_sample_neg_num = -1, preselected_captions = None, tokenizer=None):
    location_mapping = {}
    indexes = list(range(len(captions) + len(additional_captions)))
    all_captions = captions + additional_captions
    random.shuffle(indexes)
    # create a mapping between the original location and the new location

    # 1. create a mapping from original index to their character location
    original_index_to_location = defaultdict(list)
    current_caption = ''
    for i, caption in enumerate(captions):
        current_len = len(current_caption)
        for j in range(len(caption)):
            original_index_to_location[i].append(current_len + j) # location of the character in the original string
        current_caption += caption
        #current_caption += '. '

    # determind the kept indexes
    if sub_sample_pos_num != -1:
        pos_indexes = list(range(len(captions)))
        if preselected_captions is not None:
            pos_indexes = [i for i in pos_indexes if all_captions[i] in preselected_captions]

        random.shuffle(pos_indexes)
        kept_pos_indexes = set(pos_indexes[:sub_sample_pos_num])
    else:
        kept_pos_indexes = set(range(len(captions)))

    if sub_sample_neg_num != -1:
        neg_indexes = list(range(len(captions), len(captions) + len(additional_captions)))
        if preselected_captions is not None:
            neg_indexes = [i for i in neg_indexes if all_captions[i] in preselected_captions]
        random.shuffle(neg_indexes)
        kept_neg_indexes = set(neg_indexes[:sub_sample_neg_num])
    else:
        kept_neg_indexes = set(range(len(captions), len(captions) + len(additional_captions)))

    kep_indexes = kept_pos_indexes | kept_neg_indexes


    final_kept_positive = []
    final_kept_negative = []

    final_kep_indexes = []
    # 2. create a mapping from original locations
    length_limit = 254
    current_caption = ""
    # need to avoid calling the tokenizer too many times

    for i in range(len(indexes)):
        caption = all_captions[indexes[i]]

        if indexes[i] not in kep_indexes: # will not be kept
            continue

        tokenized = tokenizer.tokenize(caption)
        #tokenized = caption.split(" ")

        length_limit -= len(tokenized)
        if length_limit < 0:
            break # we have reached the length limit

        # if not caption.startswith(" "):
        #     current_caption += " "

        current_len = len(current_caption)
        if indexes[i] < len(captions): # means it is one of the original caption and we need to record location
            for j in range(len(caption)):
                location_mapping[ original_index_to_location[indexes[i]][j] ] = current_len + j # location of the character in the new string

        current_caption += caption

        if current_caption.endswith("."):
            current_caption += ' '
        elif current_caption.endswith("?"):
            current_caption += ' '
        elif current_caption.endswith(". ") or current_caption.endswith("? "):
            pass
        else:
            current_caption += '. '

        if indexes[i] in kept_pos_indexes:
            final_kept_positive.append(caption)
        else:
            final_kept_negative.append(caption)

    return current_caption, location_mapping, len(final_kept_positive), len(final_kept_negative)
