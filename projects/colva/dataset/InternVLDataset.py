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
from PIL import Image, ImageDraw
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
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
                                preprocess_mpt, preprocess_phi3, preprocess,
                                vcr_decode_mask_fn, preprocess_phi3_debug)
from .utils import (expand2square, expand2square_mask, DEFAULT_VISION_PROMPT_TOKEN,
                    VPT_CONTEXT_TOKEN, VPT_START_TOKEN, VPT_END_TOKEN, RGB_NAME)
from .process_functions import (point_rendering, box_rendering, image_blending, contour_rendering)


class InternVLDataset(Dataset):
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
                 num_dynamic_patch=None,
                 lazy_load=True,
                 group_by_length=False,
                 tokenizer=None,
                 support_prompt_types=["rectangle"],
                 pseudo_two_images_mode=False,
                 ot_image_processor=None,
                 vfm_name="RADIO",):
        super().__init__()

        self.max_length = max_length
        self.dataset_map_fn = dataset_map_fn
        self.annotation_load_fn = annotation_load_fn
        self.lazy_load = lazy_load
        self.dynamic_image_size = dynamic_image_size
        self.pad_image_to_square = pad_image_to_square
        self.group_by_length = group_by_length
        self.support_prompt_types = support_prompt_types
        self.pseudo_two_images_mode = pseudo_two_images_mode
        self.ot_image_processor = ot_image_processor
        self.vfm_name = vfm_name
        if vfm_name in ['DINOv2', 'ConvNext']:
            self.ot_image_processor.do_center_crop=False
            self.ot_image_processor.do_resize=False

        
        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        self.template = self.cfg.template
        if num_dynamic_patch is not None and len(num_dynamic_patch) == 2:
            self.min_dynamic_patch = num_dynamic_patch[0]
            self.max_dynamic_patch = num_dynamic_patch[1]
        else:
            self.min_dynamic_patch = self.cfg.min_dynamic_patch
            self.max_dynamic_patch = self.cfg.max_dynamic_patch
        self.downsample_ratio = self.cfg.downsample_ratio
        self.image_size = self.cfg.force_image_size
        self.use_thumbnail = self.cfg.use_thumbnail
        patch_size = self.cfg.vision_config.patch_size
        self.patch_token = int((self.image_size // patch_size)**2 * (self.downsample_ratio**2))
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer =  AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)
            self._add_special_tokens()
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        json_data, hf_json_data = self.annotation_load_fn(data_path, repeat_time, image_folder=image_folder)
        
        if json_data is not None:
            self.json_data = json_data
        hf_json_data = DatasetDict({'train': HFDataset.from_list(hf_json_data)})
        if self.lazy_load:
            self.text_data = build_origin_dataset(hf_json_data, 'train')
        else:
            raise NotImplementedError
        
        self.image_folder = image_folder
        self._max_refetch = 1000
        self.tcs_loader = None
    
    def _add_special_tokens(self):
        special_tokens = [VPT_CONTEXT_TOKEN,]
        num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)

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
    
    def _rand_another(self):
        return np.random.randint(0, len(self.text_data))

    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, index) -> Any:
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data
    
    def prepare_data(self, index):
        if hasattr(self, 'json_data'):
            data_dict = copy.deepcopy(self.json_data[index])
            data_dict.update(self.text_data[index])
        else:
            data_dict = copy.deepcopy(self.text_data[index])
        
        if self.lazy_load:
            result = self.dataset_map_fn(data_dict)
            if result is None:
                return None
            data_dict.update(result)
            
            if 'image' in data_dict and data_dict['image'] is not None and len(data_dict['image']) != 0:
                if type(data_dict['image']) == list or self.pseudo_two_images_mode:
                    ret = self.multi_modal_multi_image_get_item(data_dict)
                else:
                    ret = self.multi_modal_get_item(data_dict)
            elif 'video' in data_dict and data_dict['video'] is not None and data_dict['video'] != '':
                ret = self.video_get_item(data_dict)
            else:
                ret = self.pure_text_get_item(data_dict)
            return ret
        else:
            raise NotImplementedError
    
    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template == "Hermes-2":
            preprocess_function = preprocess_mpt
        elif self.template == "internlm2-chat" or "internvl2_5":
            preprocess_function = preprocess_internlm
            self.template = "internlm2-chat"
        elif self.template == "phi3-chat":
            preprocess_function = preprocess_phi3_debug #preprocess_phi3
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
            if isinstance(object_mask, dict):
                if isinstance(object_mask["counts"], list):
                    # convert to compressed RLE
                    object_mask = mask.frPyObjects(object_mask, ori_height, ori_width)
                m = mask.decode(object_mask)
                m = m.astype(np.uint8).squeeze()
            elif object_mask:
                rles = mask.frPyObjects(object_mask, ori_height, ori_width)
                rle = mask.merge(rles)
                m = mask.decode(rle).astype(np.uint8).squeeze()
            else:
                m = np.zeros((ori_height, ori_width), dtype=np.uint8)
            binary_masks.append(m)
        if len(binary_masks) == 0:
            binary_masks.append(np.zeros((ori_height, ori_width), dtype=np.uint8))
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
        try:
            image = self.load_image(image_path)
        except Exception as e:
            print(f'Error: {e}', flush=True)
            print_log(f'Error: {e}', logger='current')
            return None
        if image is None:
            return None
        ori_width, ori_height = image.size
        if ori_width < 10 or ori_height < 10:
            return None
        
        # image_name = image_path[-10:-4]

        # process and get masks/points/bbox
        merged_visual_prompts = cv2.imread(image_path)
        if merged_visual_prompts is None:
            merged_visual_prompts = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        regions = np.zeros(shape=(1, ori_height, ori_width), dtype=np.uint8)
        has_visual_prompts = False
        if 'annotation' in data_item and data_item['annotation'] is not None and len(data_item['annotation']) > 0:
            annotations = data_item['annotation']
            sampled_inds = data_item.get('sampled_inds', list(range(len(annotations))))
            if self.annotation_load_fn.__name__ == 'RegionShortConversationVCRDataset_load_fn':
                bboxes = [annotations[idx]['bbox'] for idx in sampled_inds]
                segms = [annotations[idx]['segmentation'] for idx in sampled_inds]
                regions = vcr_decode_mask_fn(bboxes, segms, ori_height, ori_width)
                regions = (regions > 0.0).astype(np.uint8)
            elif self.annotation_load_fn.__name__ == 'MDPVBoxOCRDataset_load_fn':
                bboxes = [annotations[idx]['bbox'] for idx in sampled_inds]
                regions = np.zeros(shape=(len(bboxes), ori_height, ori_width), dtype=np.uint8)
                for bidx, bbox in enumerate(bboxes):
                    x0, y0, x1, y1 = bbox
                    regions[bidx, y0:y1, x0:x1] = 1
            else:
                segms = [annotations[idx]['segmentation'] for idx in sampled_inds]
                regions = self.decode_mask(segms, ori_height=ori_height, ori_width=ori_width)  # n, h, w
            try:
                contour_rendering(merged_visual_prompts, regions)
            except Exception as e:
                pass
            has_visual_prompts = True
        merged_visual_prompts = Image.fromarray(cv2.cvtColor(merged_visual_prompts, cv2.COLOR_BGR2RGB))

        # image.save(f'/mnt/bn/xiangtai-training-data/project/xiangtai-windows/internvl/internvl_debug_out/ori_image_{image_name}.jpg')
        # merged_visual_prompts.save(f'/mnt/bn/xiangtai-training-data/project/xiangtai-windows/internvl/internvl_debug_out/merged_vprompts_{image_name}.jpg')
        # print(f"{image_name}: ", data_item['conversations'])
        # exit(0)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            try:
                images, _regions, merged_regions = dynamic_preprocess(
                    image, regions, merged_visual_prompts, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch, 
                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)
            except AssertionError as e:
                return None
        elif self.pad_image_to_square:
            image = expand2square(
                image,
                tuple(int(x * 255) for x in self.IMAGENET_MEAN))
            images = [image]
            merged_visual_prompts = expand2square(
                merged_visual_prompts,
                tuple(int(x * 255) for x in self.IMAGENET_MEAN))
            merged_regions = [merged_visual_prompts]
        else:
            images = [image]
            merged_regions = [merged_visual_prompts]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)  # num_patch, channels, h, w

        merged_visual_prompts = [self.transform(merged_region) for merged_region in merged_regions]
        merged_visual_prompts = torch.stack(merged_visual_prompts)

        transformed_visual_prompts = []
        for region in _regions:
            transformed_regions = []
            for _region in region:
                resized_region = cv2.resize(
                    _region[:, :, np.newaxis], dsize=(self.image_size, self.image_size), 
                    interpolation=cv2.INTER_NEAREST_EXACT)
                transformed_regions.append(torch.from_numpy(resized_region).squeeze(-1))
            transformed_visual_prompts.append(torch.stack(transformed_regions))
        visual_prompts = torch.stack(transformed_visual_prompts) # num_prompts, num_patch, h, w

        assert merged_visual_prompts.shape[:2] == pixel_values.shape[:2]
        
        if self.vfm_name == "DINOv2":
            OT_FORCE_IMAGE_SIZE = 512
        elif self.vfm_name in ["RADIO", "ConvNext"]:
            OT_FORCE_IMAGE_SIZE = 1024
        else:
            raise NotImplementedError
        
        image = self.load_image(image_path)
        w, h = image.size
        if w > h:
            target_size = (OT_FORCE_IMAGE_SIZE, int(h/w*OT_FORCE_IMAGE_SIZE))
        else:
            target_size = (int(w/h*OT_FORCE_IMAGE_SIZE), OT_FORCE_IMAGE_SIZE)
        resized_image = image.resize(target_size)
        cur_w, cur_h = resized_image.size
        padded_image = np.zeros(shape=(OT_FORCE_IMAGE_SIZE, OT_FORCE_IMAGE_SIZE, 3), dtype=np.uint8) * 255
        padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
        ot_pixel_values = self.ot_image_processor(images=padded_image, return_tensors='pt').pixel_values

        ot_visual_prompts = torch.tensor(regions).\
            to(ot_pixel_values.dtype).to(ot_pixel_values.device)  # num_prompts, h, w
        h, w = ot_visual_prompts.shape[-2:]
        if h > w:
            target_size = (OT_FORCE_IMAGE_SIZE, int(w/h*OT_FORCE_IMAGE_SIZE))
        else:
            target_size = (int(h/w*OT_FORCE_IMAGE_SIZE), OT_FORCE_IMAGE_SIZE)
        resized_ot_visual_prompts = F.interpolate(ot_visual_prompts.unsqueeze(1), size=target_size, mode="bilinear").squeeze(1)
        resized_padded_ot_visual_prompts = resized_ot_visual_prompts.new_zeros((resized_ot_visual_prompts.shape[0], OT_FORCE_IMAGE_SIZE, OT_FORCE_IMAGE_SIZE))
        resized_padded_ot_visual_prompts[:, :target_size[0], :target_size[1]] = resized_ot_visual_prompts

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        
        # Selcet the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        if has_visual_prompts:
            region_ids = [[region_id+1 for region_id in range(ot_visual_prompts.shape[0])],]
            object_tokens_str = ""
            for fidx, object_ids_fidx in enumerate(region_ids):
                object_tokens_str = object_tokens_str + f"Regions in the image: "
                for object_id in object_ids_fidx:
                    object_tokens_str = object_tokens_str + f"<region-{object_id}>{VPT_CONTEXT_TOKEN}, "
                object_tokens_str = object_tokens_str[:-1] + ".\n"
        else:
            object_tokens_str = ""

        ret = preprocess_function(self.template, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.patch_token * num_patches],
                                  group_by_length=self.group_by_length, ds_name="XXX",
                                  num_image=1, object_tokens_str=object_tokens_str)
    
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=merged_visual_prompts, #pixel_values,
            merged_visual_prompts=pixel_values, #merged_visual_prompts,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_patches=[num_patches,],
            visual_prompts=visual_prompts.flatten(0, 1),
            num_vprompts=[visual_prompts.shape[0],],
            vprompt_flags=[[1]*visual_prompts.shape[0], ] if has_visual_prompts else [[0]*visual_prompts.shape[0],],
            num_images=1,
            ot_pixel_values=ot_pixel_values,
            ot_visual_prompts=resized_padded_ot_visual_prompts,
            region_ids=[[region_id+1 for region_id in range(visual_prompts.shape[0])],],
        )

        return ret
    
    def multi_modal_multi_image_get_item(self, data_item):
        image_name_list = data_item['image']
        
        image_path_list = [os.path.join(self.image_folder, image_name) for image_name in image_name_list]
        images = [self.load_image(image_path) for image_path in image_path_list]
        if any([item is None for item in images]):
            return None
        merged_visual_prompts = [cv2.imread(image_path) for image_path in image_path_list]
        for idx, item in enumerate(merged_visual_prompts):
            if item is not None:
                continue
            merged_visual_prompts[idx] = cv2.cvtColor(np.asarray(image_path_list[idx]), cv2.COLOR_RGB2BGR)

        # image_name = image_path_list[0][-8:-4]
        gt_region_id = -1
        visual_prompts_list, object_ids = [], []
        if 'pos_annotations' in data_item and data_item['pos_annotations'] is not None and len(data_item['pos_annotations']) > 0:
            pos_annotations = data_item['pos_annotations']
            neg_annotations = data_item['neg_annotations']

            name_rgb = random.choice(RGB_NAME)
            color_name, color = [], []
            for k, v in name_rgb.items():
                color_name.append(str(k))
                color.append(v)
            color = color[0]
            color_name = color_name[0]
            color_anno_i = (color[2], color[1], color[0])
            for fidx in range(len(pos_annotations)-1):
                ori_width, ori_height = images[fidx].size
                regions = self.decode_mask(pos_annotations[fidx], ori_height=ori_height, ori_width=ori_width)
                visual_prompts_list.append(regions)
                object_ids.append([])
                for region in regions:
                    contours, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(merged_visual_prompts[fidx], contours, -1, color=color_anno_i, thickness=2)
                    object_ids[fidx].append(1)
            ori_width, ori_height = images[-1].size
            pos_neg_segms = pos_annotations[-1] + neg_annotations[-1]
            regions = self.decode_mask(pos_neg_segms, ori_height=ori_height, ori_width=ori_width)
            visual_prompts_list.append(regions)
            
            random_id = list(range(1, len(regions)+1))
            random.shuffle(random_id)
            try:
                contour_rendering(merged_visual_prompts[-1], regions, random_id)
                object_ids.append([_id for _id in random_id])

                choice_names = [f"{chr(i)}" for i in range(65,91)]
                if len(regions) > len(choice_names) - 1:
                    valid_num = len(choice_names) - 1
                else:
                    valid_num = len(regions)
                region_ids = random_id[:valid_num]
                choice_names = choice_names[:valid_num+1]
                gt_region_id = region_ids[0] if not data_item['is_disappear'] else -1

                region_ids.sort()
                multi_choices_str = ""
                gt_choice_str = ""
                for choice_name, region_id in zip(choice_names[:-1], region_ids):
                    multi_choices_str = multi_choices_str + f"{choice_name}. {region_id}\n"
                    if region_id == gt_region_id:
                        assert gt_choice_str == ""
                        gt_choice_str = gt_choice_str + f"{choice_name}"

                multi_choices_str = multi_choices_str + f"{choice_names[-1]}. None of the above choices are correct\n"
                if gt_choice_str == "" or data_item['is_disappear'] or len(pos_annotations[-1]) == 0:
                    gt_choice_str = f"{choice_names[-1]}"
                
                conversations = data_item['conversations']
                for i, conversation in enumerate(conversations):
                    conversation_value = conversation['value']
                    conversation_value = conversation_value.format(color=color_name, choices=multi_choices_str, answer=gt_choice_str)
                    conversation['value'] = conversation_value
                data_item['conversations'] = conversations
            except Exception as e:
                pass
        else:
            pass

        merged_visual_prompts = [Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB)) for item in merged_visual_prompts]
        
        # for fidx in range(len(images)):
        #     images[fidx].save(f'/mnt/bn/zhangtao99-2/internvl/internvl_debug_out/ori_image_{image_name}_f{fidx+1}.jpg')
        #     merged_visual_prompts[fidx].save(f'/mnt/bn/zhangtao99-2/internvl/internvl_debug_out/merged_vprompts_{image_name}_f{fidx+1}.jpg')
        # print(f"{image_name}: ", data_item['conversations'])
        # exit(0)
        
        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            num_patches_list, images_list, merged_regions_list, crop_regions_list, num_vprompts_list = [], [], [], [], []
            for image, visual_prompts, merged_visual_prompt in zip(images, visual_prompts_list, merged_visual_prompts):
                try:
                    _images, regions, merged_regions = dynamic_preprocess(
                        image, visual_prompts, merged_visual_prompt, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch, 
                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                except AssertionError as e:
                    return None
                images_list.extend(_images)
                merged_regions_list.extend(merged_regions)
                crop_regions_list.extend(regions)
                num_patches_list.append(len(_images))
                num_vprompts_list.append(len(regions))
        else:
            raise NotImplementedError

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [self.transform(image) for image in images_list]
        pixel_values = torch.stack(pixel_values)  # num_patch, channels, h, w

        merged_visual_prompts = [self.transform(merged_region) for merged_region in merged_regions_list]
        merged_visual_prompts = torch.stack(merged_visual_prompts)

        transformed_visual_prompts = []
        for region in crop_regions_list:
            transformed_regions = []
            for _region in region:
                resized_region = cv2.resize(
                    _region[:, :, np.newaxis], dsize=(self.image_size, self.image_size), 
                    interpolation=cv2.INTER_NEAREST_EXACT)
                transformed_regions.append(torch.from_numpy(resized_region).squeeze(-1))
            transformed_visual_prompts.append(torch.stack(transformed_regions))
        visual_prompts = torch.stack(transformed_visual_prompts) # num_prompts, num_patch, h, w

        assert merged_visual_prompts.shape[:2] == pixel_values.shape[:2]
        
        if self.vfm_name == "DINOv2":
            OT_FORCE_IMAGE_SIZE = 512
        elif self.vfm_name in ["RADIO", "ConvNext"]:
            OT_FORCE_IMAGE_SIZE = 1024
        else:
            raise NotImplementedError

        ot_pixel_values = []
        for fi, image in enumerate(images):
            w, h = image.size
            if w > h:
                target_size = (OT_FORCE_IMAGE_SIZE, int(h/w*OT_FORCE_IMAGE_SIZE))
            else:
                target_size = (int(w/h*OT_FORCE_IMAGE_SIZE), OT_FORCE_IMAGE_SIZE)
            resized_image = image.resize(target_size)
            cur_w, cur_h = resized_image.size
            padded_image = np.ones(shape=(OT_FORCE_IMAGE_SIZE, OT_FORCE_IMAGE_SIZE, 3), dtype=np.uint8) * 255
            padded_image[:cur_h, :cur_w, :] = np.array(resized_image)

            ot_pixel_values.append(self.ot_image_processor(images=Image.fromarray(padded_image), return_tensors='pt').pixel_values)
        # ot_pixel_values = [self.ot_image_processor(images=image, return_tensors='pt').pixel_values for image in images]
        ot_pixel_values = torch.cat(ot_pixel_values)

        ot_visual_prompts = torch.from_numpy(np.concatenate(visual_prompts_list, axis=0)).\
            to(ot_pixel_values.dtype).to(ot_pixel_values.device)  # num_prompts, h, w
        h, w = ot_visual_prompts.shape[-2:]
        if h > w:
            target_size = (OT_FORCE_IMAGE_SIZE, int(w/h*OT_FORCE_IMAGE_SIZE))
        else:
            target_size = (int(h/w*OT_FORCE_IMAGE_SIZE), OT_FORCE_IMAGE_SIZE)
        resized_ot_visual_prompts = F.interpolate(ot_visual_prompts.unsqueeze(1), size=target_size, mode="bilinear").squeeze(1)
        resized_padded_ot_visual_prompts = resized_ot_visual_prompts.new_zeros((resized_ot_visual_prompts.shape[0], OT_FORCE_IMAGE_SIZE, OT_FORCE_IMAGE_SIZE))
        resized_padded_ot_visual_prompts[:, :target_size[0], :target_size[1]] = resized_ot_visual_prompts

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        
        # Selcet the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        if gt_region_id != object_ids[-1][0] and gt_region_id != -1:
            print("query object id doesn't match with the candidate ids.")
            return None
        region_ids = [[gt_region_id for _ in object_ids[fidx]] 
                      for fidx in range(len(num_vprompts_list)-1)] #+ [object_ids[-1],]
        object_tokens_str = ""
        for fidx, object_ids_fidx in enumerate(region_ids):
            object_tokens_str = object_tokens_str + f"Objects in Image-{fidx+1}: "
            for object_id in range(1, len(object_ids_fidx)+1):
                object_tokens_str = object_tokens_str + f"<query object>{VPT_CONTEXT_TOKEN}, "
            object_tokens_str = object_tokens_str[:-2] + ".\n"
        sorted_indices = sorted(range(len(object_ids[-1])), key=lambda k: object_ids[-1][k])
        sorted_cand_object_ids = []
        object_tokens_str = object_tokens_str + f"Objects in Image-{len(object_ids)}: "
        for sorted_idx in sorted_indices:
            object_id = object_ids[-1][sorted_idx]
            object_tokens_str = object_tokens_str + f"<object-{object_id}>{VPT_CONTEXT_TOKEN}, "
            sorted_cand_object_ids.append(object_id)
        object_tokens_str = object_tokens_str[:-2] + ".\n"
        region_ids = region_ids + [sorted_cand_object_ids, ]

        total_vprompts = resized_padded_ot_visual_prompts.shape[0]
        cand_visual_prompts = resized_padded_ot_visual_prompts[total_vprompts-num_vprompts_list[-1]:]
        sorted_cand_visual_prompts = []
        for sorted_idx in sorted_indices:
            sorted_cand_visual_prompts.append(cand_visual_prompts[sorted_idx])
        sorted_cand_visual_prompts = torch.stack(sorted_cand_visual_prompts)
        resized_padded_ot_visual_prompts = torch.cat(
            [resized_padded_ot_visual_prompts[:total_vprompts-num_vprompts_list[-1]], sorted_cand_visual_prompts])
        
        # # ABLATION
        # internvl_cand_visual_prompts = visual_prompts[total_vprompts-num_vprompts_list[-1]:]
        # sorted_internvl_cand_visual_prompts = []
        # for sorted_idx in sorted_indices:
        #     sorted_internvl_cand_visual_prompts.append(internvl_cand_visual_prompts[sorted_idx])
        # sorted_internvl_cand_visual_prompts = torch.stack(sorted_internvl_cand_visual_prompts)
        # visual_prompts = torch.cat([
        #     visual_prompts[:total_vprompts-num_vprompts_list[-1]], sorted_internvl_cand_visual_prompts
        # ])
        
        ret = preprocess_function(self.template, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.patch_token * num_patch for num_patch in num_patches_list],
                                  group_by_length=self.group_by_length, ds_name="XXX",
                                  num_image=len(num_patches_list), object_tokens_str=object_tokens_str,)
        
        roi_version = self.dataset_map_fn.__name__ == "match_reasoning_map_fn_roi"

        # print("roi_version: ", roi_version)
        # exit(0)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values if not roi_version else merged_visual_prompts,
            merged_visual_prompts=merged_visual_prompts if not roi_version else pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_patches=num_patches_list,
            visual_prompts=visual_prompts.flatten(0, 1),
            num_vprompts=num_vprompts_list,
            vprompt_flags=[[1 for _ in range(nvp)] for nvp in num_vprompts_list],
            num_images=len(num_vprompts_list),
            ot_pixel_values=ot_pixel_values,
            ot_visual_prompts=resized_padded_ot_visual_prompts,
            region_ids=region_ids,
        )

        return ret

    def video_get_item(self, data_item):
        raise NotImplementedError
    
    def pure_text_get_item(self, data_item):
        ori_height = ori_width = 448
        image = Image.new('RGB', (ori_height, ori_width), (255, 255, 255))
        merged_visual_prompts = np.zeros((ori_height, ori_width, 3), dtype=np.uint8)
        merged_visual_prompts = Image.fromarray(merged_visual_prompts)
        
        # pad to square
        image = expand2square(
                image,
                tuple(int(x * 255) for x in self.IMAGENET_MEAN))
        images = [image]
        merged_visual_prompts = expand2square(
            merged_visual_prompts,
            (0, 0, 0)
        )
        merged_regions = [merged_visual_prompts]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)  # num_patch, channels, h, w

        merged_visual_prompts = [self.transform(merged_region) for merged_region in merged_regions]
        merged_visual_prompts = torch.stack(merged_visual_prompts)

        visual_prompts = torch.zeros(size=(
            merged_visual_prompts.shape[0], merged_visual_prompts.shape[-2], merged_visual_prompts.shape[-1]), 
            dtype=torch.long).to(merged_visual_prompts.device)
        
        if self.vfm_name == "DINOv2":
            OT_FORCE_IMAGE_SIZE = 512
        elif self.vfm_name in ["RADIO", "ConvNext"]:
            OT_FORCE_IMAGE_SIZE = 1024
        else:
            raise NotImplementedError

        image = Image.new('RGB', (OT_FORCE_IMAGE_SIZE, OT_FORCE_IMAGE_SIZE), (255, 255, 255))
        ot_pixel_values = self.ot_image_processor(images=image, return_tensors='pt').pixel_values
        ot_visual_prompts = torch.zeros((1, OT_FORCE_IMAGE_SIZE, OT_FORCE_IMAGE_SIZE)).\
            to(ot_pixel_values.dtype).to(ot_pixel_values.device)  # num_prompts, h, w
        # assert ot_pixel_values.shape[-2:] == ot_visual_prompts.shape[-2:], f"ot_pixel_values.shape: {ot_pixel_values.shape[-2:]}, ot_visual_prompts.shape: {ot_visual_prompts.shape[-2:]}"

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Selcet the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.patch_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, ds_name="XXX",
                                  num_image=0)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            merged_visual_prompts=merged_visual_prompts,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            num_patches=[num_patches, ],
            visual_prompts=visual_prompts,
            num_vprompts=[1, ],
            vprompt_flags=[[0,],],
            num_images=1,
            ot_pixel_values=ot_pixel_values,
            ot_visual_prompts=ot_visual_prompts,
            region_ids=[[1,],],
        )

        return ret



