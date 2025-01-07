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
from xtuner.utils import IGNORE_INDEX
IGNORE_TOKEN_ID = IGNORE_INDEX

from .process_functions import (dynamic_preprocess, preprocess_internlm, 
                                preprocess_mpt, preprocess_phi3, preprocess,
                                vcr_decode_mask_fn, preprocess_phi3_debug, preprocess_qwen2vl)
from .utils import (expand2square, expand2square_mask, DEFAULT_VISION_PROMPT_TOKEN,
                    VPT_CONTEXT_TOKEN, VPT_START_TOKEN, VPT_END_TOKEN, RGB_NAME)
from .process_functions import (point_rendering, box_rendering, image_blending, contour_rendering)

from transformers.processing_utils import ProcessingKwargs

class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class Qwen2VLDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 image_folder=None,
                 dataset_map_fn=None,
                 annotation_load_fn=None,
                 repeat_time=1,
                 lazy_load=True,
                 qwen2_processor=None,
                 ot_image_processor=None):
        super().__init__()

        self.dataset_map_fn = dataset_map_fn
        self.annotation_load_fn = annotation_load_fn
        self.lazy_load = lazy_load
        self.ot_image_processor = ot_image_processor
        self.qwen2_processor = qwen2_processor
      
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
        num_new_tokens = self.qwen2_processor.tokenizer.add_tokens(special_tokens, special_tokens=True)

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
                if type(data_dict['image']) == list:
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
        return masks
    
    def prepare_inputs(self, images, text, **kwargs):
        output_kwargs = self.qwen2_processor._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.qwen2_processor.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.qwen2_processor.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None
        
        if not isinstance(text, list):
            text = [text]

        for i in range(len(text)):
            if text[i][-len('<|im_start|>assistant\n'):] == '<|im_start|>assistant\n':
                text[i] = text[i][:-len('<|im_start|>assistant\n')]
        
        image_token = "<|image_pad|>" if not hasattr(self.qwen2_processor.tokenizer, "image_token") else self.qwen2_processor.tokenizer.image_token
        if image_grid_thw is not None:
            merge_length = self.qwen2_processor.image_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while image_token in text[i]:
                    text[i] = text[i].replace(
                        image_token, "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", image_token)

        batch_input_ids, batch_labels = [], []
        for i in range(len(text)):
            num_turns = text[i].count('<|im_start|>assistant\n')
            left_text = text[i]
            input_ids, labels = [], []
            for turn_idx in range(num_turns):
                input_text, left_text = left_text.split('<|im_start|>assistant\n', 1)
                input_text = input_text + '<|im_start|>assistant\n'

                output_text, left_text = left_text.split('<|im_end|>\n', 1)
                output_text = output_text + '<|im_end|>\n'

                input_encode = self.qwen2_processor.tokenizer.encode(input_text)
                input_ids += input_encode
                labels += [IGNORE_INDEX] * len(input_encode)

                output_encode = self.qwen2_processor.tokenizer.encode(output_text, add_special_tokens=False)
                input_ids += output_encode
                labels += copy.deepcopy(output_encode)

            #     print(f"turn#{turn_idx+1} input_text: ", input_text)
            #     print(f"turn#{turn_idx+1} output_text: ", output_text)
            # exit(0)


            if len(input_ids) > self.qwen2_processor.tokenizer.model_max_length:
                input_ids = input_ids[:self.qwen2_processor.tokenizer.model_max_length]
                labels = labels[:self.qwen2_processor.tokenizer.model_max_length]
                print(
                    f"Warning: input_ids length({len(input_ids)})"
                    f"is longer than max_length, cut to {self.qwen2_processor.tokenizer.model_max_length}"
                )
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        labels = torch.tensor(batch_labels, dtype=torch.long)
        attention_mask = input_ids.ne(self.qwen2_processor.tokenizer.pad_token_id)
        ret = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': image_inputs['pixel_values'],
            'image_grid_thw': image_inputs['image_grid_thw'],
        }
        return ret
    
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
        
        # Preprocess the conversations and generate the return dictionary
        if has_visual_prompts:
            region_ids = [[region_id+1 for region_id in range(regions.shape[0])],]
            object_tokens_str = ""
            for fidx, object_ids_fidx in enumerate(region_ids):
                object_tokens_str = object_tokens_str + f"Regions in the image: "
                for object_id in object_ids_fidx:
                    object_tokens_str = object_tokens_str + f"<region-{object_id}>{VPT_CONTEXT_TOKEN}, "
                object_tokens_str = object_tokens_str[:-1] + ".\n"
        else:
            region_ids = [[1]]
            object_tokens_str = ""

        templated_conversation = preprocess_qwen2vl(deepcopy(data_item['conversations']), object_tokens_str, num_images=1)

        if templated_conversation is None:
            return None

        text_prompt = self.qwen2_processor.apply_chat_template(templated_conversation, add_generation_prompt=True)
        
        # for reducing the sequence length
        w, h = image.size
        if w > h:
            target_size = (1024, int(h/w*1024))
        else:
            target_size = (int(w/h*1024), 1024)
        resized_image = image.resize(target_size)
        cur_w, cur_h = resized_image.size
        padded_image = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8) * 255
        padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
        padded_image = Image.fromarray(padded_image)
        inputs = self.prepare_inputs(text=[text_prompt], images=[padded_image], padding=True, return_tensors="pt")

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        
        # print("input_ids: ", input_ids.shape)
        # print("labels: ", labels.shape)
        # print("attention_mask: ", attention_mask.shape)
        # print("pixel_values: ", pixel_values.shape)
        # print("image_grid_thw: ", image_grid_thw)
        # # input_ids:  torch.Size([1, 985])                                                                                                                                                                
        # # labels:  torch.Size([1, 985])                                                                                                                                                                   
        # # attention_mask:  torch.Size([1, 985])                                                                                                                                                           
        # # pixel_values:  torch.Size([1380, 1176])                                                                                                                                                         
        # # image_grid_thw:  tensor([[ 1, 30, 46]])
        # exit(0)
        
        merge_size = self.qwen2_processor.image_processor.merge_size

        visual_prompts = torch.from_numpy(regions)
        resized_visual_prompts = F.interpolate(
            visual_prompts.unsqueeze(0),
            size=(image_grid_thw[0][1]//merge_size, image_grid_thw[0][2]//merge_size),
            mode='bilinear'
        )
        resized_visual_prompts = resized_visual_prompts.squeeze(0) > 0.5
        resized_visual_prompts = resized_visual_prompts.to(torch.long)
        # we assert the resized_visual_prompts are not used.
        
        w, h = merged_visual_prompts.size
        if w > h:
            target_size = (1024, int(h/w*1024))
        else:
            target_size = (int(w/h*1024), 1024)
        resized_merged_visual_prompts = merged_visual_prompts.resize(target_size)
        cur_w, cur_h = resized_merged_visual_prompts.size
        padded_merged_visual_prompts = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8) * 255
        padded_merged_visual_prompts[:cur_h, :cur_w, :] = np.array(resized_merged_visual_prompts)
        padded_merged_visual_prompts = Image.fromarray(padded_merged_visual_prompts)
        inputs_vp = self.qwen2_processor(text=[text_prompt], images=[padded_merged_visual_prompts], padding=True, return_tensors="pt")
        merged_visual_prompts = inputs_vp.pixel_values
        vp_grid_thw = inputs_vp.image_grid_thw

        if torch.any(image_grid_thw != vp_grid_thw):
            return None

        image = self.load_image(image_path)
        w, h = image.size
        if w > h:
            target_size = (1024, int(h/w*1024))
        else:
            target_size = (int(w/h*1024), 1024)
        resized_image = image.resize(target_size)
        cur_w, cur_h = resized_image.size
        padded_image = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8) * 255
        padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
        ot_pixel_values = self.ot_image_processor(images=padded_image, return_tensors='pt').pixel_values

        ot_visual_prompts = torch.tensor(regions).\
            to(ot_pixel_values.dtype).to(ot_pixel_values.device)  # num_prompts, h, w
        h, w = ot_visual_prompts.shape[-2:]
        if h > w:
            target_size = (1024, int(w/h*1024))
        else:
            target_size = (int(h/w*1024), 1024)
        resized_ot_visual_prompts = F.interpolate(ot_visual_prompts.unsqueeze(1), size=target_size, mode="bilinear").squeeze(1)
        resized_padded_ot_visual_prompts = resized_ot_visual_prompts.new_zeros((resized_ot_visual_prompts.shape[0], 1024, 1024))
        resized_padded_ot_visual_prompts[:, :target_size[0], :target_size[1]] = resized_ot_visual_prompts
        
        merge_length = self.qwen2_processor.image_processor.merge_size ** 2

        ret = dict(
            input_ids=input_ids[0],
            labels=labels[0],
            attention_mask=attention_mask[0],
            pixel_values=merged_visual_prompts,
            merged_visual_prompts=pixel_values,
            image_flags=torch.tensor([1] * (image_grid_thw[0][1]*image_grid_thw[0][2] // merge_length), dtype=torch.long),
            visual_prompts=resized_visual_prompts,
            num_vprompts=[visual_prompts.shape[0], ],
            vprompt_flags=[[1] * visual_prompts.shape[0]] if has_visual_prompts else [[0]*visual_prompts.shape[0]],
            num_images=1,
            ot_pixel_values=ot_pixel_values,
            ot_visual_prompts=resized_padded_ot_visual_prompts,
            region_ids=region_ids,
            image_grid_thw=image_grid_thw,
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
        
        num_vprompts_list = [vp.shape[0] for vp in visual_prompts_list]
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

        templated_conversation = preprocess_qwen2vl(deepcopy(data_item['conversations']), object_tokens_str, num_images=len(image_path_list))
        if templated_conversation is None:
            return None

        text_prompt = self.qwen2_processor.apply_chat_template(templated_conversation, add_generation_prompt=True)
        
        resized_padded_images = []
        for image in images:
            w, h = image.size
            if w > h:
                target_size = (1024, int(h/w*1024))
            else:
                target_size = (int(w/h*1024), 1024)
            resized_image = image.resize(target_size)
            cur_w, cur_h = resized_image.size
            padded_image = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8) * 255
            padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
            padded_image = Image.fromarray(padded_image)
            resized_padded_images.append(padded_image)
        inputs = self.prepare_inputs(text=[text_prompt], images=resized_padded_images, padding=True, return_tensors="pt")
        # inputs = self.prepare_inputs(text=[text_prompt], images=images, padding=True, return_tensors="pt")
        
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        
        merge_size = self.qwen2_processor.image_processor.merge_size

        resized_visual_prompts = []
        for image_grid_thw_fi, visual_prompts_fi in zip(image_grid_thw, visual_prompts_list):
            visual_prompts_fi = torch.from_numpy(visual_prompts_fi)
            resized_visual_prompts_fi = F.interpolate(
                visual_prompts_fi.unsqueeze(0), 
                size=(image_grid_thw_fi[1]//merge_size, image_grid_thw_fi[2]//merge_size),
                mode='bilinear'
            )
            resized_visual_prompts_fi = resized_visual_prompts_fi.squeeze(0) > 0.5
            resized_visual_prompts.append(resized_visual_prompts_fi.to(torch.long))
        resized_visual_prompts = torch.cat(resized_visual_prompts, dim=0)
        
        resized_padded_merged_visual_prompts = []
        for image in merged_visual_prompts:
            w, h = image.size
            if w > h:
                target_size = (1024, int(h/w*1024))
            else:
                target_size = (int(w/h*1024), 1024)
            resized_image = image.resize(target_size)
            cur_w, cur_h = resized_image.size
            padded_image = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8) * 255
            padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
            padded_image = Image.fromarray(padded_image)
            resized_padded_merged_visual_prompts.append(padded_image)
        inputs_vp = self.qwen2_processor(text=[text_prompt], images=resized_padded_merged_visual_prompts, padding=True, return_tensors="pt")
        # inputs_vp = self.qwen2_processor(text=[text_prompt], images=merged_visual_prompts, padding=True, return_tensors="pt")
        merged_visual_prompts = inputs_vp.pixel_values
        vp_grid_thw = inputs_vp.image_grid_thw
        
        if torch.any(image_grid_thw != vp_grid_thw):
            return None
    
        # print("input_ids: ", input_ids.shape)
        # print("labels: ", labels.shape)
        # print("attention_mask: ", attention_mask.shape)
        # print("pixel_values: ", pixel_values.shape)
        # print("image_grid_thw: ", image_grid_thw)
        # # input_ids:  torch.Size([1, 2653])                                                                                                                                         
        # # attention_mask:  torch.Size([1, 2653])                                                                                                                                    
        # # pixel_values:  torch.Size([9360, 1176])                                                                                                                                   
        # # image_grid_thw:  tensor([[ 1, 60, 78],[ 1, 60, 78]])
        # exit(0)
        
        ot_pixel_values = []
        for fi, image in enumerate(images):
            w, h = image.size
            if w > h:
                target_size = (1024, int(h/w*1024))
            else:
                target_size = (int(w/h*1024), 1024)
            resized_image = image.resize(target_size)
            cur_w, cur_h = resized_image.size
            padded_image = np.ones(shape=(1024, 1024, 3), dtype=np.uint8) * 255
            padded_image[:cur_h, :cur_w, :] = np.array(resized_image)

            ot_pixel_values.append(self.ot_image_processor(images=Image.fromarray(padded_image), return_tensors='pt').pixel_values)
        # ot_pixel_values = [self.ot_image_processor(images=image, return_tensors='pt').pixel_values for image in images]
        ot_pixel_values = torch.cat(ot_pixel_values)

        ot_visual_prompts = torch.from_numpy(np.concatenate(visual_prompts_list, axis=0)).\
            to(ot_pixel_values.dtype).to(ot_pixel_values.device)  # num_prompts, h, w
        h, w = ot_visual_prompts.shape[-2:]
        if h > w:
            target_size = (1024, int(w/h*1024))
        else:
            target_size = (int(h/w*1024), 1024)
        resized_ot_visual_prompts = F.interpolate(ot_visual_prompts.unsqueeze(1), size=target_size, mode="bilinear").squeeze(1)
        resized_padded_ot_visual_prompts = resized_ot_visual_prompts.new_zeros((resized_ot_visual_prompts.shape[0], 1024, 1024))
        resized_padded_ot_visual_prompts[:, :target_size[0], :target_size[1]] = resized_ot_visual_prompts

        total_vprompts = resized_padded_ot_visual_prompts.shape[0]
        cand_visual_prompts = resized_padded_ot_visual_prompts[total_vprompts-num_vprompts_list[-1]:]
        sorted_cand_visual_prompts = []
        for sorted_idx in sorted_indices:
            sorted_cand_visual_prompts.append(cand_visual_prompts[sorted_idx])
        sorted_cand_visual_prompts = torch.stack(sorted_cand_visual_prompts)
        resized_padded_ot_visual_prompts = torch.cat(
            [resized_padded_ot_visual_prompts[:total_vprompts-num_vprompts_list[-1]], sorted_cand_visual_prompts])
        
        roi_version = self.dataset_map_fn.__name__ == "match_reasoning_map_fn_roi"
        
        merge_length = self.qwen2_processor.image_processor.merge_size ** 2

        ret = dict(
            input_ids=input_ids[0],
            labels=labels[0],
            attention_mask=attention_mask[0],
            pixel_values=pixel_values if not roi_version else merged_visual_prompts,
            merged_visual_prompts=merged_visual_prompts if not roi_version else pixel_values,
            image_flags=torch.tensor([1] * sum([image_grid_thw_fi[1]*image_grid_thw_fi[2]//merge_length for image_grid_thw_fi in image_grid_thw]), dtype=torch.long),
            visual_prompts=resized_visual_prompts,
            num_vprompts=num_vprompts_list,
            vprompt_flags=[[1 for _ in range(nvp)] for nvp in num_vprompts_list],
            num_images=len(num_vprompts_list),
            ot_pixel_values=ot_pixel_values,
            ot_visual_prompts=resized_padded_ot_visual_prompts,
            region_ids=region_ids,
            image_grid_thw=image_grid_thw,
        )

        return ret

    def video_get_item(self, data_item):
        raise NotImplementedError
    
    def pure_text_get_item(self, data_item):
        ori_height = ori_width = 448
        image = Image.new('RGB', (ori_height, ori_width), (255, 255, 255))
        merged_visual_prompts = np.zeros((ori_height, ori_width, 3), dtype=np.uint8)
        merged_visual_prompts = Image.fromarray(merged_visual_prompts)

        regions = np.zeros(shape=(1, ori_height, ori_width), dtype=np.uint8)
        has_visual_prompts = False
        region_ids = [[1]]

        templated_conversation = preprocess_qwen2vl(deepcopy(data_item['conversations']), '', num_images=0)
        if templated_conversation is None:
            return None

        text_prompt = self.qwen2_processor.apply_chat_template(templated_conversation, add_generation_prompt=True)
        
        # for reducing the sequence length
        w, h = image.size
        if w > h:
            target_size = (1024, int(h/w*1024))
        else:
            target_size = (int(w/h*1024), 1024)
        resized_image = image.resize(target_size)
        cur_w, cur_h = resized_image.size
        padded_image = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8) * 255
        padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
        padded_image = Image.fromarray(padded_image)
        inputs = self.prepare_inputs(text=[text_prompt], images=[padded_image], padding=True, return_tensors="pt")
        
        # inputs = self.prepare_inputs(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        
        merge_size = self.qwen2_processor.image_processor.merge_size

        visual_prompts = torch.from_numpy(regions)
        resized_visual_prompts = F.interpolate(
            visual_prompts.unsqueeze(0),
            size=(image_grid_thw[0][1]//merge_size, image_grid_thw[0][2]//merge_size),
            mode='bilinear'
        )
        resized_visual_prompts = resized_visual_prompts.squeeze(0) > 0.5
        resized_visual_prompts = resized_visual_prompts.to(torch.long)
        
        w, h = merged_visual_prompts.size
        if w > h:
            target_size = (1024, int(h/w*1024))
        else:
            target_size = (int(w/h*1024), 1024)
        resized_merged_visual_prompts = merged_visual_prompts.resize(target_size)
        cur_w, cur_h = resized_merged_visual_prompts.size
        padded_merged_visual_prompts = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8) * 255
        padded_merged_visual_prompts[:cur_h, :cur_w, :] = np.array(resized_merged_visual_prompts)
        padded_merged_visual_prompts = Image.fromarray(padded_merged_visual_prompts)
        inputs_vp = self.qwen2_processor(text=[text_prompt], images=[padded_merged_visual_prompts], padding=True, return_tensors="pt")
        # inputs_vp = self.qwen2_processor(text=[text_prompt], images=merged_visual_prompts, padding=True, return_tensors="pt")
        merged_visual_prompts = inputs_vp.pixel_values
        vp_grid_thw = inputs_vp.image_grid_thw

        if torch.any(image_grid_thw != vp_grid_thw):
            return None

        image = Image.new('RGB', (1024, 1024), (255, 255, 255))
        ot_pixel_values = self.ot_image_processor(images=image, return_tensors='pt').pixel_values
        ot_visual_prompts = torch.zeros((1, 1024, 1024)).\
            to(ot_pixel_values.dtype).to(ot_pixel_values.device)  # num_prompts, h, w
        
        merge_length = self.qwen2_processor.image_processor.merge_size ** 2
        
        ret = dict(
            input_ids=input_ids[0],
            labels=labels[0],
            attention_mask=attention_mask[0],
            pixel_values=merged_visual_prompts,
            merged_visual_prompts=pixel_values,
            image_flags=torch.tensor([0] * (image_grid_thw[0][1]*image_grid_thw[0][2]//merge_length), dtype=torch.long),
            visual_prompts=resized_visual_prompts,
            num_vprompts=[visual_prompts.shape[0], ],
            vprompt_flags=[[0]*visual_prompts.shape[0]],
            num_images=1,
            ot_pixel_values=ot_pixel_values,
            ot_visual_prompts=ot_visual_prompts,
            region_ids=region_ids,
            image_grid_thw=image_grid_thw,
        )

        return ret
        