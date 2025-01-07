# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os
from math import ceil
import copy

import torch
import tqdm
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import Resize


from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
from xtuner.dataset.utils import encode_fn
from .utils import expand2square


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


class LLaVADataset(Dataset):

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
                 exhibit_special_tokens=False,
                 encode_fn=None,
                 patch_size=None,
                 resize_long=False,
                 ):
        super().__init__()

        self.lazy = lazy
        self.max_length = max_length
        self.dataset_map_fn = dataset_map_fn
        self.template_map_fn = template_map_fn

        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        assert offline_processed_text_folder or (data_path and tokenizer)

        self.tokenizer = tokenizer
        if isinstance(tokenizer, dict) or isinstance(
                tokenizer, Config) or isinstance(tokenizer, ConfigDict):
            tokenizer_type = self.tokenizer['type']
            del self.tokenizer['type']
            self.tokenizer = tokenizer_type(**self.tokenizer)
            if not exhibit_special_tokens:
                self._add_special_tokens()

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
            try:
                if data_path.endswith('.json'):
                    json_data = json.load(open(data_path))
                elif data_path.endswith('.jsonl'):
                    json_data = load_jsonl(data_path)
            except:
                json_data = []
                with open(data_path, 'r', encoding="utf-8") as f:
                    for line in tqdm.tqdm(f):
                        _data = json.loads(line)
                        json_data.append(_data)
            for idx in range(len(json_data)):
                if "id" in json_data[idx].keys() and isinstance(json_data[idx]['id'], int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])

            json_data = DatasetDict({'train': HFDataset.from_list(json_data)})

            if self.lazy:
                self.text_data = build_origin_dataset(json_data, 'train')
            else:
                self.text_data = process_hf_dataset(
                    dataset=json_data,
                    tokenizer=self.tokenizer,
                    max_length=max_length,
                    dataset_map_fn=dataset_map_fn,
                    template_map_fn=template_map_fn,
                    split='train',
                    max_dataset_length=max_dataset_length,
                    remove_unused_columns=False,
                    pack_to_max_length=False,
                    with_image_token=True,
                    map_num_proc=32,  # because limited mem
                )

        self.image_folder = image_folder
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square
        self.resize_long = resize_long
        self.encode_fn = encode_fn
        self.patch_size = patch_size

    @property
    def modality_length(self):
        length_list = []
        if len(self.text_data) > 1000000:
            length_list = [100] * len(self.text_data)
            return length_list
        for data_dict in tqdm.tqdm(self.text_data):
            if self.lazy:
                cur_len = 100
            else:
                cur_len = len(data_dict['input_ids'])
                if data_dict.get('image', None) is None:
                    cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = copy.deepcopy(self.text_data[index])

        if self.lazy:
            result = self.dataset_map_fn(data_dict)
            data_dict.update(result)

            result = self.template_map_fn(data_dict)
            data_dict.update(result)

            if self.encode_fn is not None:
                result = self.encode_fn(
                    data_dict, tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    with_image_token=True)
            else:
                result = encode_fn(
                    data_dict, tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    with_image_token=True)
            data_dict.update(result)

        # huggingface datasets add image key items to the dict.
        assert 'image' in data_dict.keys()

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

            if self.resize_long:
                _, height, width = image.size()
                height, width = get_resize_output_image_size_long((height, width))
                image = Resize(size=(height, width))(image)

            data_dict['pixel_values'] = image
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            # modify here for pure text inputs.
            # data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
            #                                         crop_size['width'])
            data_dict['pixel_values'] = None

        if self.patch_size is not None and data_dict['pixel_values'] is not None:
            data_dict['pixel_values'] = self.process_patch(
                data_dict['pixel_values'], self.patch_size
            )
        return data_dict

    def process_patch(self, image, patch_size):
        pixel_values = image.unsqueeze(0)
        # print(torch.max(pixel_values), '   ', torch.min(pixel_values))
        h, w = pixel_values.shape[-2:]
        if max(h, w) > 1024:
            if h > w:
                h_new = 1024
                w_new = int(w * h_new / h)
                w_new = pad_patch(w_new, patch_size=patch_size)
            else:
                w_new = 1024
                h_new = int(h * w_new / w)
                h_new = pad_patch(h_new, patch_size=patch_size)
        else:
            h_new = pad_patch(h, patch_size=patch_size)
            w_new = pad_patch(w, patch_size=patch_size)
        dtype = pixel_values.dtype
        pixel_values = F.interpolate(pixel_values.to(torch.float32),
                                     size=(h_new, w_new), mode='bilinear',
                                     align_corners=False).to(dtype)
        return pixel_values[0]

    def _add_special_tokens(self):
        assert hasattr(self, "tokenizer")
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        # add for visual prompt
        region_tokens = ['<region>']
        point_tokens = ['<mark>']
        special_tokens = segmentation_tokens + phrase_tokens + region_tokens + point_tokens
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        return

def pad_patch(val, patch_size=32):
    if val % patch_size == 0:
        return val
    else:
        return (val // patch_size + 1) * patch_size


def get_resize_output_image_size_long(
    image_size, PATCH_SIZE=32, MAX_RESOLUTION = 1024, MIN_RESOLUTION = 448,
) -> tuple:
    l1, l2 = image_size  # 540, 32
    short, long = (l2, l1) if l2 <= l1 else (l1, l2)

    # set the nearest multiple of PATCH_SIZE for `long`
    requested_new_long = min(
        [
            ceil(long / PATCH_SIZE) * PATCH_SIZE,
            MAX_RESOLUTION,
        ]
    )

    requested_new_long = max(requested_new_long, MIN_RESOLUTION)

    new_long, new_short = requested_new_long, int(requested_new_long * short / long)
    # Find the nearest multiple of 64 for new_short
    new_short = ceil(new_short / PATCH_SIZE) * PATCH_SIZE
    return (new_long, new_short) if l2 <= l1 else (new_short, new_long)
