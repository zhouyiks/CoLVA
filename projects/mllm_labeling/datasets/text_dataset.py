import os
from mmengine.dist import master_only
from vlm.datasets.evaluation.base_eval_dataset import BaseEvalDataset
import json

class SAM2TextDataset(BaseEvalDataset):
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

        ret = 'Here are some detailed descriptions of an object, encompassing its colour, shape, position within the image, state, purpose, properties, and relationship with surrounding objects. This object could be an inanimate item like a chair or a car, a person, or a part of an object such as hair, shoes, or a symbol on a wall. These descriptions come from several multi-modal language models and may contain errors. Please summarize the characteristics of this object in one sentence, ensuring that the features include the core attributes from the descriptions. Disregard conflicting information in the descriptions, such as inconsistent colours, and ignore "yellow edges," as they do not pertain to the object\'s characteristics.\n'
        ret += 'The descriptions:\n'
        for i, caption in enumerate(captions):
            ret += f'Description {i}: "{caption}"'
            ret += '\n'
        ret += 'Please provide the summarized object\'s description.'
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