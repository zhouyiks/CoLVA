import json
import logging
import os 
import copy
from typing import Any

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk

from mmengine import print_log
from mmengine.config import Config, ConfigDict

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset

from projects.lisa.multiprocess_eval_refcoco import template


class OspreyRegionCaptionDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 num_proc=32,
                 lazy=False,
                 repeats=1):
        super().__init__()

        assert offline_processed_text_folder or (data_path and tokenizer)
        self.lazy = lazy

        self.max_length = max_length
        self.dataset_map_fn = dataset_map_fn
        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn.pop('type')
            self.template_map_fn = _type(**self.template_map_fn)
        
        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)
        
        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            json_data = self.json_file_preprocess(data_path)
            self.json_data = json_data
            json_data = self.filter_hf_require_infos(json_data)
            # hf_json_data = DatasetDict({"train": HFDataset.from_list(json_data)})
            if self.lazy:
                self.text_data = build_origin_dataset(json_data, 'train')
            else:
                raise NotImplementedError
        
        self.image_folder = image_folder
        size = image_processor.crop_size
        if isinstance(size, int):
            self.image_h, self.image_w = size, size
        else:
            self.image_w, self.image_h = size
        
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square
        self.down_ratio = 1
        self.repeats = repeats
        self.tokenizer = tokenizer
        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size))
        ])

    
    def filter_hf_require_infos(self, dataset_infos):
        ret = {}
        for dataset_info in dataset_infos:
            description = dataset_info["description"]
            image = dataset_info["file_name"]
            required_info = {"image": image, "description": description}
            ret.append(required_info)
        return ret

    def json_file_preprocess(self, data_path):
        with open(data_path, 'r') as f:
            json_file = json.load(f)
        
        ret = []
        for item in json_file:
            item.update({'image': item['file_name']})
            if len(item["description"]) != len(item["annotation"]):
                print("The number of description is not equal to seg !!!")
            else:
                ret.append(item)
            
        return ret
    
    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            if self.lazy:
                cur_len = 100
            else:
                cur_len = len(data_dict['input_ids'])
                if data_dict.get('image', None) is None:
                    cur_len = -cur_len
            length_list.append(cur_len)
        return length_list
    
    def __len__(self):
        return len(self.text_data) * self.repeats

    def real_len(self):
        return len(self.text_data)
    
    def multi_modal_get_item(self, data_item):
        # Build transformtion function
        return


    def __getitem__(self, index) -> Any:
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.json_data[index])
        data_dict.update(self.text_data[index])

        if self.lazy:
            result = self.dataset_map_fn(data_dict)
            data_dict.update(result)
            assert 'image' in data_dict.keys()
            if data_dict.get('image', None) is not None:
                image_file = data_dict['image']




    


