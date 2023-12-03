import json
import re
from copy import deepcopy
import random
def clean_string(input_string):
    # remove leading and trailing spaces
    input_string = input_string.strip()
    # remove trailing ";" and "."
    input_string = re.sub(r";$", "", input_string)
    input_string = re.sub(r"\.$", "", input_string)
    return input_string

class GPTOutputParser():
    def __init__(self, version):
        self.version = version

    def __call__(self, description):
        if self.version == "v1":
            try:
                description = json.loads(description.strip("\n"))
                description['description'] = description['description'].split("; ")
                description['type'] = "a kind of {}".format(description['type'])
            except:
                description = {
                    "type": "object",
                    "description": [],
                    "similar objects": []
                }
            return description
        if self.version == "v5":
            info = description
            # check the format of type of thing
            if "has a tangible appearance and is" in info[0]:
                type_of_thing = info[0].split(" has a tangible appearance and is ")[-1].split(".")[0]
            elif "has a tangible appearance" in info[0]:
                type_of_thing = info[0].split(" has a tangible appearance ")[-1].split(".")[0]
                if "and refers to" in type_of_thing:
                    type_of_thing = type_of_thing.split("and refers to ")[-1]
            else:
                #print(info[0], "type of thing not found")
                type_of_thing = ""
            
            if " are:\t" in info[0]:
                similar_things = info[0].split(" are:\t")[-1].split("\nThere are several useful visual features to tell")[0].split("\t")
                similar_things = [i for i in similar_things if i.strip() != ""] # remove empty strings
            else:
                #print(info[0], "similar things not found")
                similar_things = []
            
            if " and not similar things in a photo:\t" in info[0]:
                visual_feature_descriptions = info[0].split(" and not similar things in a photo:\t")[-1].split("\t")
                visual_feature_descriptions = [i for i in visual_feature_descriptions if i.strip() != ""] # remove empty strings
            else:
                #print(info[0], "visual feature descriptions not found")
                visual_feature_descriptions = []
            return {
                "type": type_of_thing,
                "description": "; ".join(visual_feature_descriptions),
                "similar_things": similar_things
            }
        if self.version == "v6":
            description = description.lower()
            type_ = re.findall(r"type: (.*)", description)[0]
            # description
            
            visual_description = re.findall(r"visual description.*:(.*)similar objects", description, re.DOTALL)[0]
        
            # fine substrings with leading 1. 2. 
            visual_description = re.findall(r"[(\d\.)(-\.)]\ (.*)", visual_description)

            # similar objects
            similar_objects = re.findall(r"similar objects:(.*)", description, re.DOTALL)[0]
            similar_objects = re.findall(r"[(\d\.)(-\.)]\ (.*)", similar_objects)

            visual_description = [clean_string(i) for i in visual_description]
            visual_description = [i for i in visual_description if i != ""]

            similar_objects = [clean_string(i) for i in similar_objects]
            similar_objects = [i for i in similar_objects if i != ""]

            final_description = {
                "type": type_,
                "description": "; ".join(visual_description),
                "similar objects": similar_objects,
            }
            return final_description
            # except:
            #     print(description_dict)
            #     pdb.set_trace()
        if self.version == "v7":
            '''
            "- plumbing fixture\n- white or off-white\n- a bowl-shaped basin\n- a drain at the bottom\n- a water supply line\n- a flush handle or button\n- a splash guard\n- a wall-mounted or floor-mounted design"
            '''
            description = description.lower()
            description = [des.replace("- ", "") for des in description.split("\n")]
            final_description = {
                "type": description[0],
                "description": description[1:],
                "similar objects": [],
            }
            return final_description

        assert False, "version not supported"
    
    def form_span(self, 
                  noun, 
                  description, 
                  type = "vanilla", 
                  positive_range = "partial",
                  start_index = 0,
                  od_to_grounding_version = ''):
        '''
        Given the parsed description, form the span
        '''

        if "random_origin" in od_to_grounding_version and random.random() < 0.1:
            # directly use the noun it self
            return noun, len(noun), None, None # forget about span

        description = self(description)
        type_of_thing = description['type']
        
        pheso_caption = ""
        spans = []

        start_index_rolling = start_index
        # the first substring is the name
        if "remove_noun" in type:
            pass
        else:
            sub_descripion = "{}, ".format(noun)
            pheso_caption += sub_descripion
            spans.append([start_index_rolling, len(pheso_caption) + start_index])
        
        start_index_rolling = len(pheso_caption) + start_index
        if "skip_des" in type:
            descriptions = [type_of_thing] 
        else:
            descriptions = [type_of_thing] + description['description']

        for index, description_i in enumerate(descriptions):
            if index != len(descriptions) - 1:
                _suffix = ", "
            else:
                _suffix = ". "

            sub_descripion = "{}{}".format(description_i, _suffix)
            pheso_caption += sub_descripion
            spans.append([start_index_rolling, len(pheso_caption) + start_index])
            start_index_rolling = len(pheso_caption) + start_index

            if index == 0: # pheso_caption: cat, a kind of animal
                if positive_range == "partial": # when 
                    end_index = len(pheso_caption)
                    positive_spans = deepcopy(spans)

        if positive_range == "all":
            end_index = len(pheso_caption)
            positive_spans = deepcopy(spans)
        return pheso_caption, end_index, spans, positive_spans

    def form_span_independent(self, 
                  noun, 
                  description, 
                  type = "vanilla", 
                  positive_range = "partial",
                  start_index = 0,
                  od_to_grounding_version = None):
        '''
        Given the parsed description, form the span
        '''
        if "infer" in od_to_grounding_version:
            use_random = False
        else:
            use_random = True
        
        description = self(description)
        type_of_thing = description['type']
        
        pheso_caption = ""
        spans = []

        start_index_rolling = start_index
        # the first substring is the name
        if use_random and random.random() < 0.5:
                drop_noun = True
        else:
            drop_noun = False
        
        if not drop_noun:
            sub_descripion = "{}, {}. ".format(noun, type_of_thing)
            pheso_caption += sub_descripion
            spans.append([start_index_rolling, len(pheso_caption) + start_index])
        
        start_index_rolling = len(pheso_caption) + start_index
        descriptions = description['description']

        for index, description_i in enumerate(descriptions):
            _suffix = ". "
            if use_random and random.random() < 0.5:
                leading_word = type_of_thing
            else:
                leading_word = noun
            
            sub_descripion = "{}, {}{}".format(leading_word, description_i, _suffix)
            pheso_caption += sub_descripion
            spans.append([start_index_rolling, len(pheso_caption) + start_index])
            start_index_rolling = len(pheso_caption) + start_index

            if index == 0: # pheso_caption: cat, a kind of animal
                if positive_range == "partial": # when 
                    end_index = len(pheso_caption)
                    positive_spans = deepcopy(spans)

        if positive_range == "all":
            end_index = len(pheso_caption)
            positive_spans = deepcopy(spans)
        return pheso_caption, end_index, spans, positive_spans