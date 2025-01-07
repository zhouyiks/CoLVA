# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset


from xtuner.registry import BUILDER
from xtuner.dataset.utils import expand2square, encode_fn
from xtuner.dataset.llava import load_jsonl

from xtuner.dataset.huggingface import build_origin_dataset

class LLaVALazyDataset(Dataset):

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
                 lazy=False,
                 ):
        super().__init__()

        assert offline_processed_text_folder or (data_path and tokenizer)
        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            self.text_data = load_from_disk(offline_processed_text_folder)
        else:
            if data_path.endswith('.json'):
                json_data = json.load(open(data_path))
            elif data_path.endswith('.jsonl'):
                json_data = load_jsonl(data_path)
            else:
                raise NotImplementedError

            for idx in range(len(json_data)):
                if "id" in json_data[idx].keys() and isinstance(json_data[idx]['id'], int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])
            json_data = DatasetDict({'train': HFDataset.from_list(json_data)})

            assert max_dataset_length is None, "max_dataset_length is not supported in Lazy mode"
            self.text_data = build_origin_dataset(json_data, 'train')

        self.image_folder = image_folder
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square

        # is_lazy = True
        if lazy:
            self.tokenizer = tokenizer
            if isinstance(self.tokenizer, dict) or isinstance(self.tokenizer, Config) or isinstance(self.tokenizer, ConfigDict):
                self.tokenizer = BUILDER.build(self.tokenizer)
            self.max_length = max_length

            self.dataset_map_fn = dataset_map_fn
            if isinstance(template_map_fn, dict) or isinstance(template_map_fn, Config) or isinstance(
                template_map_fn, ConfigDict):
                template_map_fn = BUILDER.build(template_map_fn)
            self.template_map_fn = template_map_fn
    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values'] = image
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
        result = self.dataset_map_fn(data_dict)
        data_dict.update(result)

        result = self.template_map_fn(data_dict)
        data_dict.update(result)

        result = encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length, with_image_token=True)
        data_dict.update(result)
        return data_dict
