import copy
import io
import json
import os
import random
import warnings
import logging
from typing import Any
from copy import deepcopy
from distinctipy import distinctipy

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from transformers import AutoConfig, AutoTokenizer
from pycocotools import mask

from mmengine import print_log
from mmengine.config import Config, ConfigDict

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
from xtuner.utils import DEFAULT_IMAGE_TOKEN

from .process_functions import (dynamic_preprocess, preprocess_internlm, 
                                preprocess_mpt, preprocess_phi3, preprocess)
from .utils import expand2square, expand2square_mask


class InternVLDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 model_path,
                 data_path=None,
                 image_folder=None,
                 dataset_map_fn=None,
                 annotation_load_fn=None,
                 dynamic_image_size=False,
                 pad_image_to_square=False,
                 repeat_time=1,
                 max_length=8192,
                 lazy_load=True,
                 group_by_length=False,):
        super().__init__()

        self.max_length = max_length
        self.dataset_map_fn = dataset_map_fn
        self.annotation_load_fn = annotation_load_fn
        self.lazy_load = lazy_load
        self.dynamic_image_size = dynamic_image_size
        self.pad_image_to_square = pad_image_to_square
        self.group_by_length = group_by_length

        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        self.template = self.cfg.template
        self.min_dynamic_patch = self.cfg.min_dynamic_patch
        self.max_dynamic_patch = self.cfg.max_dynamic_patch
        self.downsample_ratio = self.cfg.downsample_ratio
        self.image_size = self.cfg.force_image_size
        self.use_thumbnail = self.cfg.use_thumbnail
        patch_size = self.cfg.vision_config.patch_size
        self.patch_token = int((self.image_size // patch_size)**2 * (self.downsample_ratio**2))
        self.tokenizer =  AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        self.vprompt_transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST_EXACT),
        ])

        json_data, hf_json_data = self.annotation_load_fn(data_path, repeat_time)
        
        self.json_data = json_data
        hf_json_data = DatasetDict({'train': HFDataset.from_list(hf_json_data)})
        if self.lazy_load:
            self.text_data = build_origin_dataset(hf_json_data, 'train')
        else:
            raise NotImplementedError
        
        self.image_folder = image_folder
        self._max_refetch = 1000
        self.tcs_loader = None
    
    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            if self.lazy_load:
                cur_len = 100
            else:
                cur_len = len(data_dict['input_ids'])
                if data_dict.get('image', None) is None:
                    cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, index) -> Any:
        data_dict = copy.deepcopy(self.json_data[index])
        data_dict.update(self.text_data[index])

        if self.lazy_load:
            result = self.dataset_map_fn(data_dict)
            data_dict.update(result)

            if type(data_dict['image']) == list:
                ret = self.multi_modal_multi_image_get_item(data_dict)
            else:
                ret = self.multi_modal_get_item(data_dict)
            
            return ret
        else:
            raise NotImplementedError
    
    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template == "Hermes-2":
            preprocess_function = preprocess_mpt
        elif self.template == "internlm2-chat":
            preprocess_function = preprocess_internlm
        elif self.template == "phi3-chat":
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')
    
    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in object_mask:
                rles = mask.frPyObjects([seg], ori_height, ori_width)
                m = mask.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()
            
            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        if self.pad_image_to_square:
            masks = expand2square_mask(masks)
        # masks = torch.from_numpy(masks)
        return masks

    def multi_modal_get_item(self, data_item):
        # Ensure the first conversation contains an image placeholder
        if DEFAULT_IMAGE_TOKEN not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + data_item['conversations'][0]['value']
        
        # Merge the image path
        image_path = os.path.join(self.image_folder, data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)
        ori_width, ori_height = image.size

        # process and get masks
        annotations = data_item['annotation']
        sampled_inds = data_item.get('sampled_inds', list(range(len(annotations))))

        annotations = [annotations[idx]['segmentation'] for idx in sampled_inds]
        _regions = self.decode_mask(annotations, ori_height=ori_height, ori_width=ori_width)  # n, h, w

        # merge all visual prompts into one canvas
        colors = distinctipy.get_colors(_regions.shape[0])
        merged_visual_prompts = np.zeros((ori_height, ori_width, 3), dtype=np.uint8)

        for i, _region in enumerate(_regions):
            merged_visual_prompts[:, :, 0][_region > 0] = int(colors[i][0] * 255)
            merged_visual_prompts[:, :, 1][_region > 0] = int(colors[i][1] * 255)
            merged_visual_prompts[:, :, 2][_region > 0] = int(colors[i][2] * 255)
        merged_visual_prompts = Image.fromarray(merged_visual_prompts)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images, regions, merged_regions = dynamic_preprocess(image, _regions, merged_visual_prompts,
                                                 min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                                 image_size=self.image_size, use_thumbnail=self.use_thumbnail) 
        elif self.pad_image_to_square:
            image = expand2square(
                image,
                tuple(int(x * 255) for x in self.IMAGENET_MEAN))
            images = [image]
            regions = [region for region in _regions]
            merged_visual_prompts = expand2square(
                merged_visual_prompts,
                (0, 0, 0)
            )
            merged_regions = [merged_visual_prompts]
        else:
            images = [image]
            regions = [region for region in _regions]
            merged_regions = [merged_visual_prompts]
        assert all([len(images) == len(region) for region in regions]), f"image patches: {len(images)}, region patches: {[len(region) for region in regions]}, num regions: {_regions.shape[0]}"

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        merged_visual_prompts = [self.transform(merged_region) for merged_region in merged_regions]
        merged_visual_prompts = torch.stack(merged_visual_prompts)
        
        visual_prompts = [torch.stack([self.vprompt_transform(_region).squeeze(0) for _region in region])
                          for region in regions]
        visual_prompts = torch.stack(visual_prompts)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        
        # Selcet the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.patch_token * num_patches],
                                  group_by_length=self.group_by_length, ds_name="XXX")

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            visual_prompts=visual_prompts,
            merged_visual_prompts=merged_visual_prompts,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_patches=torch.tensor([num_patches,], dtype=torch.long),
            num_vprompts=torch.tensor([visual_prompts.shape[0],], dtype=torch.long),
            sampled_mark_token_ids=torch.tensor(data_item['sampled_mark_token_ids'], dtype=torch.long),
        )

        return ret
    
    def multi_modal_multi_image_get_item(self, data_item):
        pass




        

    




    





        
