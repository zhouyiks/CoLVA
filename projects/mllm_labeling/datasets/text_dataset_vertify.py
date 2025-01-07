import os
from mmengine.dist import master_only
from vlm.datasets.evaluation.base_eval_dataset import BaseEvalDataset
import json

class SAM2TextDataset_Vertify(BaseEvalDataset):
    METAINFO: dict = dict(name='sam2 text dataset')
    def __init__(
            self,
            json_folder,
            bs=8,
    ):
        super().__init__()
        self.json_folder = json_folder
        json_files = os.listdir(json_folder)
        self.json_files = []
        self.data = []
        for _file in json_files:
            path = os.path.join(self.json_folder, _file)
            with open(path, 'r') as f:
                self.data.extend(json.load(f))

        self.meta_infos = []

        self.data_dict = {}
        for _data in self.data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']
            cap_type = _data['type']

            if video_id not in self.data_dict.keys():
                self.data_dict[video_id] = {}
            if obj_id not in self.data_dict[video_id].keys():
                self.meta_infos.append([video_id, obj_id])
                self.data_dict[video_id][obj_id] = {}
            self.data_dict[video_id][obj_id][cap_type] = _data
        self.bs = bs

    def __len__(self):
        if len(self.meta_infos) % self.bs ==0:
            return len(self.meta_infos) // self.bs
        else:
            return len(self.meta_infos) // self.bs + 1

    def _get_data(self, idx):
        video_id, obj_id = self.meta_infos[idx]
        data = self.data_dict[video_id][obj_id]
        captions = [data[key]['caption'] for key in data.keys()]

        other_infos = {}
        other_infos['video_id'] = video_id
        other_infos['obj_id'] = obj_id
        question = self.get_question(captions)
        other_infos.update({'text_prompt': question})
        return other_infos


    def get_question(self, captions):

        ret = 'Here are some detailed descriptions of an object, encompassing its colour, shape, state, relationship with surrounding objects, and so on. This object could be a part-level object or an entity-level object. These descriptions come from several multi-modal language models and may contain conflicts.\nIf significant conflicts exist in the object\'s category and colour, please generate validation questions to ask the multi-modal large language model. For example, if the object is a shirt and there are colour conflicts in the descriptions, create a validation question like: "What is the color of the shirt?" If the object\'s category is inconsistent across different descriptions, such as clothing and person, create a validation question like: "What is the object category?" Some information appears only in one description and is absent from others; this is not a conflict. Please ignore minor conflicts, such as "dark color" and "black color." If there are no significant conflicts, please only output "No conflict."'
        ret += 'The descriptions:\n'
        for i, caption in enumerate(captions):
            ret += f'Description {i}: "{caption}"'
            ret += '\n'
        ret += 'If has significant conflict, please only output the validation question without any other words, per question in a line.'
        # ret = ''
        # ret += 'Here is a detailed description of an object, encompassing its color, shape, position within the image, state, purpose, properties, and its relationship with surrounding objects. This object could be an inanimate item like a chair or a car, it could be a person, or it could be a part of an object such as hair, shoes, or a symbol on a wall. Based on this detailed object description, please provide some brief captions that describe the object. Here are some sample captions for reference: \"a man wearing blue clothes", \"shoes worn by a man who is running", "A red symbol on the wall".\n'
        # ret += 'Here is the detailed description of the object:\n'
        # ret += caption
        # ret += '\n'
        # ret += 'All brief captions should be derived from the given detailed description. Please refrain from making associations or creating non-existent descriptions. The caption should be a phrase, and try to avoid including ":" in the caption.\n'
        # ret += 'Please provide brief captions for the object based on the detailed description above.'
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.meta_infos))

        data_dicts = []
        for _idx in range(start, end):
            object_dict = self._get_data(_idx)
            data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'text'}

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}


class SAM2TextDataset_Vertify_whole(BaseEvalDataset):
    METAINFO: dict = dict(name='sam2 text dataset')
    def __init__(
            self,
            json_folders,
            bs=8,
    ):
        super().__init__()
        self.json_folders = json_folders

        self.json_files = []
        self.data = []
        for json_folder in self.json_folders:
            json_files = os.listdir(json_folder)
            for _file in json_files:
                path = os.path.join(json_folder, _file)
                with open(path, 'r') as f:
                    self.data.extend(json.load(f))

        self.meta_infos = []

        self.data_dict = {}
        for _data in self.data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']
            cap_type = _data['type']

            if video_id not in self.data_dict.keys():
                self.data_dict[video_id] = {}
            if obj_id not in self.data_dict[video_id].keys():
                self.meta_infos.append([video_id, obj_id])
                self.data_dict[video_id][obj_id] = {}
            self.data_dict[video_id][obj_id][cap_type] = _data
        self.bs = bs

    def __len__(self):
        if len(self.meta_infos) % self.bs ==0:
            return len(self.meta_infos) // self.bs
        else:
            return len(self.meta_infos) // self.bs + 1

    def _get_data(self, idx):
        video_id, obj_id = self.meta_infos[idx]
        data = self.data_dict[video_id][obj_id]
        captions = [data[key]['caption'] for key in data.keys()]

        other_infos = {}
        other_infos['video_id'] = video_id
        other_infos['obj_id'] = obj_id
        other_infos['ignore'] = len(captions) < 2
        other_infos['ori_captions'] = captions
        question = self.get_question(captions)
        other_infos.update({'text_prompt': question})
        return other_infos


    def get_question(self, captions):

        ret = 'Here are some detailed descriptions of an object, encompassing its colour, shape, state, relationship with surrounding objects, and so on. This object could be a part-level object or an entity-level object. These descriptions come from several multi-modal language models and may contain conflicts.\nIf significant conflicts exist in the object\'s category and colour, please generate validation questions to ask the multi-modal large language model. For example, if the object is a shirt and there are colour conflicts in the descriptions, create a validation question like: "What is the color of the shirt?" If the object\'s category is inconsistent across different descriptions, such as clothing and person, create a validation question like: "What is the object category?" Some information appears only in one description and is absent from others; this is not a conflict. Please ignore minor conflicts, such as "dark color" and "black color." If there are no significant conflicts, please only output "No conflict."'
        ret += 'The descriptions:\n'
        for i, caption in enumerate(captions):
            ret += f'Description {i}: "{caption}"'
            ret += '\n'
        ret += 'If has significant conflict, please only output the validation question without any other words, per question in a line.'
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.meta_infos))

        data_dicts = []
        for _idx in range(start, end):
            object_dict = self._get_data(_idx)
            data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'text', 'task': 'vertify'}

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}


class SAM2TextDataset_Summerize_whole(BaseEvalDataset):
    METAINFO: dict = dict(name='sam2 text dataset')
    def __init__(
            self,
            json_folders,
            bs=8,
    ):
        super().__init__()
        self.json_folders = json_folders

        self.json_files = []
        self.data = []
        for json_folder in self.json_folders:
            json_files = os.listdir(json_folder)
            for _file in json_files:
                path = os.path.join(json_folder, _file)
                with open(path, 'r') as f:
                    self.data.extend(json.load(f))

        self.meta_infos = []

        self.data_dict = {}
        for _data in self.data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']

            if video_id not in self.data_dict.keys():
                self.data_dict[video_id] = {}
            if obj_id not in self.data_dict[video_id].keys():
                self.meta_infos.append([video_id, obj_id])
                self.data_dict[video_id][obj_id] = {}
            self.data_dict[video_id][obj_id] = _data
        self.bs = bs

    def __len__(self):
        if len(self.meta_infos) % self.bs ==0:
            return len(self.meta_infos) // self.bs
        else:
            return len(self.meta_infos) // self.bs + 1

    def _get_data(self, idx):
        video_id, obj_id = self.meta_infos[idx]
        data = self.data_dict[video_id][obj_id]
        captions = data['ori_captions']

        other_infos = {}
        other_infos['video_id'] = video_id
        other_infos['obj_id'] = obj_id

        other_infos['ori_captions'] = captions
        question = self.get_question(captions)
        other_infos.update({'text_prompt': question})
        return other_infos


    def get_question(self, captions):
        ret = 'Here are some detailed descriptions of an object, encompassing its colour, shape, state, relationship with surrounding objects, and so on. This object could be a part-level object or an entity-level object. The descriptions are:\n'
        for i, caption in enumerate(captions):
            ret += f'Description {i}: "{caption}"'
            ret += '\n'
        ret += 'Please summarize the characteristics of this object in one sentence, ensuring that the features include the core attributes from the descriptions.\n'
        ret += 'The description 1 comes from a more detailed observation of the object, while the description 2 comes from a more comprehensive observation of the scene. Therefore, if the object described in description 1 is a part of description 2, please refer to description 1. For detailed attributes of the object, such as colour, also refer to description 1. Regarding the relationship between the object and the surrounding scene, please refer to description 2. For the complementary information in the two descriptions, please summarize and retain it.\n'
        ret += 'There are some examples:\n Description 1: The object is the upper body of a person wearing a green coat with pink embellishments.\nDescription 2: The object is a person wearing a teal coat who is talking to others.\nThe summarized description should be: The upper body of the person wearing a green coat with pink embellishments and who is talking to those around them.\n'
        ret += "please give the summarized description in one sentence."
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.meta_infos))

        data_dicts = []
        for _idx in range(start, end):
            object_dict = self._get_data(_idx)
            data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'text', 'task': 'summarize'}

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}

class SAM2TextDataset_Formatting_whole(BaseEvalDataset):
    METAINFO: dict = dict(name='sam2 text dataset')
    def __init__(
            self,
            json_folders,
            bs=8,
    ):
        super().__init__()
        self.json_folders = json_folders

        self.json_files = []
        self.data = []
        for json_folder in self.json_folders:
            json_files = os.listdir(json_folder)
            for _file in json_files:
                path = os.path.join(json_folder, _file)
                with open(path, 'r') as f:
                    self.data.extend(json.load(f))

        self.meta_infos = []

        self.data_dict = {}
        for _data in self.data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']

            if video_id not in self.data_dict.keys():
                self.data_dict[video_id] = {}
            if obj_id not in self.data_dict[video_id].keys():
                self.meta_infos.append([video_id, obj_id])
                self.data_dict[video_id][obj_id] = {}
            self.data_dict[video_id][obj_id] = _data
        self.bs = bs

    def __len__(self):
        if len(self.meta_infos) % self.bs ==0:
            return len(self.meta_infos) // self.bs
        else:
            return len(self.meta_infos) // self.bs + 1

    def _get_data(self, idx):
        video_id, obj_id = self.meta_infos[idx]
        data = self.data_dict[video_id][obj_id]
        captions = data['ori_captions']
        summarize_caption = data['summarized']

        other_infos = {}
        other_infos['video_id'] = video_id
        other_infos['obj_id'] = obj_id

        other_infos['ori_captions'] = captions
        other_infos['summarized'] = summarize_caption

        question = self.get_question(summarize_caption)
        other_infos.update({'text_prompt': question})
        return other_infos


    def get_question(self, caption):
        caption = caption.split('\n')[0].split(':')[-1]
        ret = 'We will give you an object caption. If phrases like "the highlighted region" or "region highlighted in yellow" or "The object is" appear in the caption, please remove them. The final caption should be formatted with the object type as the subject. Here are some examples:\nFor the given caption, "The highlighted region depicts a person\'s legs, clad in dark-colored pants and white," your answer should be "The person\'s legs, clad in dark-colored pants and white."\nFor the caption "The object is a person, observable from the upper body down to the upper legs," you should respond with "A person, observable from the upper body down to the upper legs."\n'
        ret += f'The caption is "{caption}"\n'
        ret += 'Please give the answer. If not contains the above phrases, please directly repeat the caption.'
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.meta_infos))

        data_dicts = []
        for _idx in range(start, end):
            object_dict = self._get_data(_idx)
            data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'text', 'task': 'formatting'}

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}


class SAM2TextDataset_ShortCap_whole(BaseEvalDataset):
    METAINFO: dict = dict(name='sam2 text dataset')
    def __init__(
            self,
            json_file,
            bs=8,
    ):
        super().__init__()
        self.json_file = json_file

        with open(self.json_file, 'r') as f:
            self.data = json.load(f)

        self.meta_infos = []

        for video_id in self.data.keys():
            for obj_id in self.data[video_id]['objects'].keys():
                self.meta_infos.append([video_id, obj_id, self.data[video_id]['objects'][obj_id]['formated']])
        self.bs = bs

    def __len__(self):
        if len(self.meta_infos) % self.bs ==0:
            return len(self.meta_infos) // self.bs
        else:
            return len(self.meta_infos) // self.bs + 1

    def _get_data(self, idx):
        video_id, obj_id, detailed_caption = self.meta_infos[idx]

        other_infos = {}
        other_infos['video_id'] = video_id
        other_infos['obj_id'] = obj_id
        other_infos['formated'] = detailed_caption

        question = self.get_question(detailed_caption)
        other_infos.update({'text_prompt': question})
        return other_infos


    def get_question(self, caption):
        ret = "Please generate some short captions based on the detailed object description. The short captions should describe the same object as the detailed caption, such as “the person,” “the dog,” “the shoes,” and so on, while ignoring the surrounding environment and atmosphere descriptions.\n"
        ret += "Please be careful not to change the subject of the description. The object may be a part-level thing, such as \"A dog's brown and white paw\", where the main object is \"the paw of the dog\" and not \"the dog with a brown and white paw.\" For example, in \"A pair of white platform shoes with a distinctive, chunky sole, worn by a person walking in an indoor market,\" the main object is \"the shoes\" and not \"the person.\" Similarly, in \"the lower body of a person,\" the main object is \"the lower body of the person\" and not \"the person.\"\n"
        ret += "For example, given the detailed description: \"A small, light tan pug with distinct dark eye markings, small folded ears, and a compact, stout body, sitting on a dark surface likely indoors, with a human leg nearby, creating a warm and cozy atmosphere that accentuates the dog's unique features.\"\n"
        ret += "The short captions should be:\n"
        ret += "\"A light tan pug with distinct dark eye markings\"\n"
        ret += "\"A small dog sitting on a dark surface\""
        ret += "\"A dog with small folded ears nearby a human leg\""
        ret += f"The given detailed object description is \"{caption}\"."
        ret += "Please directly provide the short captions of the object, one per line."
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.meta_infos))

        data_dicts = []
        for _idx in range(start, end):
            object_dict = self._get_data(_idx)
            data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'text', 'task': 'short_cap'}

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}

class SAM2TextDataset_Filter_Unindentified(BaseEvalDataset):
    METAINFO: dict = dict(name='sam2 text dataset')
    def __init__(
            self,
            json_folder,
            bs=8,
    ):
        super().__init__()
        self.json_folder = json_folder
        json_files = os.listdir(json_folder)
        self.json_files = []
        self.data = []
        for _file in json_files:
            path = os.path.join(self.json_folder, _file)
            with open(path, 'r') as f:
                self.data.extend(json.load(f))

        self.meta_infos = []

        self.data_dict = {}
        for _data in self.data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']
            cap_type = _data['type']

            if video_id not in self.data_dict.keys():
                self.data_dict[video_id] = {}
            if obj_id not in self.data_dict[video_id].keys():
                self.meta_infos.append([video_id, obj_id])
                self.data_dict[video_id][obj_id] = {}
            self.data_dict[video_id][obj_id][cap_type] = _data
        self.bs = bs

    def __len__(self):
        if len(self.meta_infos) % self.bs ==0:
            return len(self.meta_infos) // self.bs
        else:
            return len(self.meta_infos) // self.bs + 1

    def _get_data(self, idx):
        video_id, obj_id = self.meta_infos[idx]
        data = self.data_dict[video_id][obj_id]
        ret = []

        for cap_type in data.keys():
            other_infos = {}
            caption = data[cap_type]['caption']
            other_infos['video_id'] = video_id
            other_infos['obj_id'] = obj_id
            other_infos['type'] = cap_type
            other_infos['caption'] = caption
            question = self.get_question(caption)
            other_infos.update({'text_prompt': question})
            ret.append(other_infos)
        return ret


    def get_question(self, caption):
        ret = f"I will provide you with a description of an object/objects."
        ret += "Please help me determine whether this description clearly states what the object/objects is. If so, please summarize the type of object in a short phrase; if not, please respond with \"Unidentified.\"\n"
        ret += "Here are some examples:\n"
        ret += "  Description: The image depicts a single, elongated object with a slightly curved, tapered shape. The object has a smooth surface and is primarily dark in color, with a gradient transitioning from a darker shade at the top to a lighter shade towards the bottom.\n"
        ret += "  Answer: Unidentified\n"
        ret += "  Description: The image depicts a small, light-colored dog with a fluffy coat. The dog has pointed ears, a short snout, and is shown in a standing position with its mouth slightly open.\n"
        ret += "  Answer: A small, light-colored dog.\n"
        ret += "  Description: The image shows a close-up of a person's hand holding a small, round object. The object appears to be a piece of food, possibly a nut or a small fruit, based on its size and texture. The hand is positioned with the fingers slightly curled around the object, suggesting a gentle grip.\n"
        ret += "  Answer: A person's hand.\n"
        ret += "  Description: The image shows a red car door. The door is partially open, revealing the interior handle and a portion of the window. The exterior of the door features a side mirror, a black rubber strip, and a handle. The car's body appears to be a solid red color with no visible damage or distinct markings.\n"
        ret += "  Answer: A red door of the car.\n"
        ret += "  Description: The image shows a person riding a red electric scooter. The scooter has black wheels and handlebars, and the rider is wearing a dark jacket and a helmet.\n"
        ret += "  Answer: A person and a red electric scooter.\n"
        ret += f"The description is: {caption}. Please give the answer."
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.meta_infos))

        data_dicts = []
        for _idx in range(start, end):
            object_dict = self._get_data(_idx)
            data_dicts.extend(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'text', 'task': 'filter_unindentified'}

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}

class SAM2TextDataset_Consistency(BaseEvalDataset):
    METAINFO: dict = dict(name='sam2 text dataset')
    def __init__(
            self,
            json_folder,
            bs=8,
    ):
        super().__init__()
        self.json_folder = json_folder
        json_files = os.listdir(json_folder)
        self.json_files = []
        self.data = []
        for _file in json_files:
            path = os.path.join(self.json_folder, _file)
            with open(path, 'r') as f:
                self.data.extend(json.load(f))

        self.meta_infos = []

        self.data_dict = {}
        for _data in self.data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']
            cap_type = _data['type']

            if video_id not in self.data_dict.keys():
                self.data_dict[video_id] = {}
            if obj_id not in self.data_dict[video_id].keys():
                self.meta_infos.append([video_id, obj_id])
                self.data_dict[video_id][obj_id] = {}
            self.data_dict[video_id][obj_id][cap_type] = _data

        _meta_infos = []
        for _info in self.meta_infos:
            video_id, obj_id = _info
            if len(self.data_dict[video_id][obj_id]) < 2:
                pass
            else:
                _meta_infos.append(_info)
        print(f"Drop {len(self.meta_infos) - len(_meta_infos)} unID items !!!")
        self.meta_infos = _meta_infos

        self.bs = bs

    def __len__(self):
        if len(self.meta_infos) % self.bs ==0:
            return len(self.meta_infos) // self.bs
        else:
            return len(self.meta_infos) // self.bs + 1

    def _get_data(self, idx):
        video_id, obj_id = self.meta_infos[idx]
        data = self.data_dict[video_id][obj_id]
        captions = [data[cap_type]['caption'] for cap_type in data.keys()]
        other_infos = {}
        other_infos['video_id'] = video_id
        other_infos['obj_id'] = obj_id
        other_infos['caption'] = data['crop']['caption']
        other_infos['category'] = data['crop']['category']
        question = self.get_question(captions)
        other_infos.update({'text_prompt': question})
        return other_infos

    def get_question(self, captions):
        ret = "I will give you two object descriptions. Please determine whether these two descriptions refer to the same object. If they do, please answer \"Yes.\" If they do not, please explain the reason.\n"
        ret += f"Description 1: {captions[0]}\n"
        ret += f"Description 2: {captions[1]}\n"
        ret += "Please give the answer."
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.meta_infos))

        data_dicts = []
        for _idx in range(start, end):
            object_dict = self._get_data(_idx)
            data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'text', 'task': 'consistency'}

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}

class SAM2TextDataset_ReConsistency(BaseEvalDataset):
    METAINFO: dict = dict(name='sam2 text dataset')
    def __init__(
            self,
            json_folder,
            bs=8,
    ):
        super().__init__()
        self.json_folder = json_folder
        json_files = os.listdir(json_folder)
        self.json_files = []
        self.data = []
        for _file in json_files:
            path = os.path.join(self.json_folder, _file)
            with open(path, 'r') as f:
                self.data.extend(json.load(f))

        self.meta_infos = []

        self.data_dict = {}
        for _data in self.data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']

            if video_id not in self.data_dict.keys():
                self.data_dict[video_id] = {}
            if obj_id not in self.data_dict[video_id].keys():
                self.meta_infos.append([video_id, obj_id])
                self.data_dict[video_id][obj_id] = {}
            self.data_dict[video_id][obj_id] = _data

        self.bs = bs

    def __len__(self):
        if len(self.meta_infos) % self.bs == 0:
            return len(self.meta_infos) // self.bs
        else:
            return len(self.meta_infos) // self.bs + 1

    def _get_data(self, idx):
        video_id, obj_id = self.meta_infos[idx]
        data = self.data_dict[video_id][obj_id]
        crop_category = data['crop_category']
        video_caption = data['video_caption']

        other_infos = {}
        other_infos['video_id'] = video_id
        other_infos['obj_id'] = obj_id
        other_infos['crop_caption'] = data['crop_caption']
        other_infos['image_caption'] = data['image_caption']
        other_infos['video_caption'] = data['video_caption']
        other_infos['crop_category'] = data['crop_category']

        question = self.get_question(crop_category, video_caption)
        other_infos.update({'text_prompt': question})
        return other_infos

    def get_question(self, crop_category, video_caption):
        ret = "I will give you a short description and a detailed description. Please determine whether the object described in the detailed description matches the one in the short description, including its type and action. If they do, please answer \"Yes.\" If they do not, please explain the reason.\n"
        ret += f"Short description: {crop_category}\n"
        ret += f"Detailed description: {video_caption}\n"
        ret += "Please give the answer."
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.meta_infos))

        data_dicts = []
        for _idx in range(start, end):
            object_dict = self._get_data(_idx)
            data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'text', 'task': 're_consistency'}

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}

class SAM2TextDataset_ChangeStyle(BaseEvalDataset):
    METAINFO: dict = dict(name='sam2 text dataset')
    def __init__(
            self,
            json_folder,
            bs=8,
    ):
        super().__init__()
        self.json_folder = json_folder
        json_files = os.listdir(json_folder)
        self.json_files = []
        self.data = []
        for _file in json_files:
            path = os.path.join(self.json_folder, _file)
            with open(path, 'r') as f:
                self.data.extend(json.load(f))

        self.meta_infos = []

        self.data_dict = {}
        for _data in self.data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']

            if video_id not in self.data_dict.keys():
                self.data_dict[video_id] = {}
            if obj_id not in self.data_dict[video_id].keys():
                self.meta_infos.append([video_id, obj_id])
                self.data_dict[video_id][obj_id] = {}
            self.data_dict[video_id][obj_id] = _data

        self.bs = bs

    def __len__(self):
        if len(self.meta_infos) % self.bs == 0:
            return len(self.meta_infos) // self.bs
        else:
            return len(self.meta_infos) // self.bs + 1

    def _get_data(self, idx):
        video_id, obj_id = self.meta_infos[idx]
        data = self.data_dict[video_id][obj_id]
        video_caption = data['video_caption']

        other_infos = {}
        other_infos['video_id'] = video_id
        other_infos['obj_id'] = obj_id
        other_infos['crop_caption'] = data['crop_caption']
        other_infos['image_caption'] = data['image_caption']
        other_infos['video_caption'] = data['video_caption']
        other_infos['crop_category'] = data['crop_category']

        question = self.get_question(video_caption)
        other_infos.update({'text_prompt': question})
        return other_infos

    def get_question(self, video_caption):
        ret = "I will give you a description of an object. This object could be an entity-level object, a part-level object, or even a multi-object. Please help me compress this description by removing unnecessary background details, the overall atmosphere of the image, and eliminating descriptions like 'highlighted object,' 'highlighted by yellow edge,' 'yellow edge,' and so on. Try to retain as much information about the object, such as its appearance, motion, action and movement. Your response should start with 'The object is.'\n"
        ret += f"The description: {video_caption}\n"
        ret += "Please provide your answer."
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.meta_infos))

        data_dicts = []
        for _idx in range(start, end):
            object_dict = self._get_data(_idx)
            data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'text', 'task': 'change_style'}

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}

class SAM2TextDataset_Translation(BaseEvalDataset):
    METAINFO: dict = dict(name='sam2 text dataset')
    def __init__(
            self,
            json_folder,
            bs=8,
    ):
        super().__init__()
        self.json_folder = json_folder
        json_files = os.listdir(json_folder)
        self.json_files = []
        self.data = []
        for _file in json_files:
            path = os.path.join(self.json_folder, _file)
            with open(path, 'r') as f:
                self.data.extend(json.load(f))

        self.meta_infos = []

        self.data_dict = {}
        for _data in self.data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']

            if video_id not in self.data_dict.keys():
                self.data_dict[video_id] = {}
            if obj_id not in self.data_dict[video_id].keys():
                self.meta_infos.append([video_id, obj_id])
                self.data_dict[video_id][obj_id] = {}
            self.data_dict[video_id][obj_id] = _data

        self.bs = bs

    def __len__(self):
        if len(self.meta_infos) % self.bs == 0:
            return len(self.meta_infos) // self.bs
        else:
            return len(self.meta_infos) // self.bs + 1

    def _get_data(self, idx):
        video_id, obj_id = self.meta_infos[idx]
        data = self.data_dict[video_id][obj_id]
        final_caption = data['final_caption']

        other_infos = {}
        other_infos['video_id'] = video_id
        other_infos['obj_id'] = obj_id
        other_infos['crop_caption'] = data['crop_caption']
        other_infos['image_caption'] = data['image_caption']
        other_infos['video_caption'] = data['video_caption']
        other_infos['crop_category'] = data['crop_category']
        other_infos['final_caption'] = data['final_caption']

        question = self.get_question(final_caption)
        other_infos.update({'text_prompt': question})
        return other_infos

    def get_question(self, video_caption):
        ret = f"Please translate this sentence into Chinese:\"{video_caption}\""
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.meta_infos))

        data_dicts = []
        for _idx in range(start, end):
            object_dict = self._get_data(_idx)
            data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'text', 'task': 'translation'}

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}