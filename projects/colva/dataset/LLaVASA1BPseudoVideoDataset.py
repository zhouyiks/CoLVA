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
import tqdm
import time

import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tvT
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from transformers import AutoConfig, AutoTokenizer
from pycocotools import mask

from mmdet.datasets.api_wrappers import COCO

from xtuner.utils import IGNORE_INDEX
IGNORE_TOKEN_ID = IGNORE_INDEX

from .utils import detection_utils as utils
from .utils.detectron2.data2 import transforms as T
from .utils.augmentation import build_pseudo_augmentation
from .utils import (expand2square, expand2square_mask)
from .process_functions import dynamic_preprocess

from transformers.processing_utils import ProcessingKwargs
from transformers.image_utils import get_image_size, to_numpy_array

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag

class LlavaProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "image_kwargs": {},
        "video_kwargs": {},
    }

def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


class LLaVASA1BPseudoVideoDataset(Dataset):
    def __init__(self,
                 model_path,
                 data_path=None,
                 image_folder=None,
                 dynamic_image_size=False,
                 pad_image_to_square=False,
                 num_dynamic_patch=None,
                 repeat_time=1,
                 llava_processor=None,
                 ot_image_processor=None,
                 tokenizer=None,
                 vfm_name="RADIO",
                 ):
        super().__init__()
        
        self.llava_processor = llava_processor
        self.ot_image_processor = ot_image_processor
        if vfm_name == "DINOv2":
            self.ot_image_processor.do_center_crop=False
            self.ot_image_processor.do_rescale=False
            self.ot_image_processor.do_resize=False

        with open(data_path, 'r') as f:
            data_list = json.load(f)['images']
        # self.data = data_list
        left_data_list = []
        for item in data_list:
            if item['file_name'].startswith('sa_0000'):
                continue
            left_data_list.append(item)
        self.data = left_data_list

        if vfm_name == "DINOv2":
            augs = build_pseudo_augmentation(True, force_image_size=512)
        elif vfm_name == "RADIO":
            augs = build_pseudo_augmentation(True, force_image_size=1024)
        else:
            raise NotImplementedError
        self.augmentations = T.AugmentationList(augs)
        
        self.image_folder = image_folder
        self._max_refetch = 100
    
    def parse_data_info(self, img_info: dict):
        data_info = {}
        data_info["image"] = img_info["file_name"]
        data_info["img_id"] = img_info["image_id"]
        data_info["height"] = img_info["height"]
        data_info["width"] = img_info["width"]

        anno_file = os.path.join(self.image_folder, img_info["file_name"].replace('.jpg', '.json'))
        with open(anno_file, 'r') as f:
            json_data = json.load(f)
        
        instances = []
        for i, ann in enumerate(json_data['annotations']):
            instance = {}

            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get("iscrowd", False):
                instance["ignore_flag"] = 1
            else:
                instance["ignore_flag"] = 0
            instance["bbox"] = bbox

            if ann.get("segmentation", None):
                instance["segmentation"] = ann["segmentation"]
            
            if "instance_id" in ann:
                instance["instance_id"] = ann["instance_id"]
            else:
                instance["instance_id"] = i+1
            instances.append(instance)
        data_info["annotations"] = instances
        return data_info
    
    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.data:
            cur_len = 100
            length_list.append(cur_len)
        return length_list
    
    def _rand_another(self):
        return np.random.randint(0, len(self.data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            if data is None:
                index = self._rand_another()
                continue
            return data
        
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
    
    def _convert_masks_to_pil_images(self, regions):
        ori_height, ori_width = regions.shape[-2:]
        num_pseudo_images = regions.shape[0] // 3
        if regions.shape[0] % 3 != 0:
            num_pseudo_images += 1
        pseudo_images = []
        for img_idx in range(num_pseudo_images):
            start_idx = img_idx * 3
            end_idx = start_idx + 3
            if end_idx > regions.shape[0]:
                end_idx = regions.shape[0]
            
            img_array = np.zeros(shape=(ori_height, ori_width, 3), dtype=np.uint8)
            num_regions = end_idx - start_idx
            img_array[:, :, :num_regions] = np.stack(
                [regions[idx, :, :] for idx in range(start_idx, end_idx)], axis=-1
            ) * 255
            pseudo_images.append(Image.fromarray(img_array))
        
        return pseudo_images
    
    def prepare_inputs(self, images, text, **kwargs):
        output_kwargs = self.llava_processor._merge_kwargs(
            LlavaProcessorKwargs,
            tokenizer_init_kwargs=self.llava_processor.tokenizer.init_kwargs,
            **kwargs,
        )

        return output_kwargs

    def prepare_data(self, index):
        data_dict = copy.deepcopy(self.parse_data_info(self.data[index]))

        img_annos = data_dict.pop('annotations', None)
        image_path = os.path.join(self.image_folder, data_dict['image'])

        original_image = utils.read_image(image_path, "RGB")

        sampling_frame_num = 2

        image_list = []
        annotations_list = []
        for _ in range(sampling_frame_num):
            utils.check_image_size(data_dict, original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]
            image_list.append(Image.fromarray(image))

            _img_annos = []
            for anno in img_annos:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)
            
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _img_annos
                if obj.get("iscrowd", 0) == 0
            ]
            annotations_list.append(annos)
        
        # sampled_frame_indices = random.sample(list(range(sampling_frame_num)), 2)
        sampled_frame_indices = [0, 1]

        # if random.random() < 0.2:
        #     images = [Image.open(image_path).convert('RGB'), image_list[sampled_frame_indices[0]]]
        #     annotations = [img_annos, annotations_list[sampled_frame_indices[0]]]
        # else:
        images = [image_list[sampled_frame_indices[0]], image_list[sampled_frame_indices[1]]]
        annotations = [annotations_list[sampled_frame_indices[0]], annotations_list[sampled_frame_indices[1]]]
            
        
        visual_prompts_list = []
        region_ids_list = []
        for fid, annotations_i in enumerate(annotations):
            segms = [annotations_i[idx]['segmentation'] for idx in range(len(annotations_i))]
            instance_ids = [annotations_i[idx]['instance_id'] for idx in range(len(annotations_i))]

            if isinstance(segms[0], np.ndarray):
                ori_width, ori_height = images[fid].size
                regions = np.stack(segms, axis=0)
                assert regions.shape[1] == ori_height, f"regions.shape[1]: {regions.shape[1]}, ori_height: {ori_height}"
                assert regions.shape[2] == ori_width, f"regions.shape[2]: {regions.shape[2]}, ori_width: {ori_width}"
            else:
                ori_width, ori_height = images[fid].size
                regions = self.decode_mask(segms, ori_height=ori_height, ori_width=ori_width)
            visual_prompts_list.append(regions)
            region_ids_list.append(instance_ids)
        num_vprompts_list = [vp.shape[0] for vp in visual_prompts_list]

        merged_visual_prompts = [image.copy() for image in images]
        
        image_token = self.llava_processor.tokenizer.image_token if hasattr(self.llava_processor.tokenizer, "image_token") else "<image>"
        output_kwargs = self.prepare_inputs(text=[image_token], images=merged_visual_prompts, padding=True, return_tensors="pt")
        inputs = self.llava_processor(text=[image_token], images=merged_visual_prompts, padding=True, return_tensors="pt")
        pixel_values = inputs.pixel_values

        concate_regions = np.concatenate(visual_prompts_list, axis=0)
        regions_img = self._convert_masks_to_pil_images(concate_regions)
        regions_input = self.llava_processor.image_processor(regions_img, do_rescale=False, do_normalize=False, **output_kwargs["images_kwargs"])
        resized_visual_prompts = (regions_input['pixel_values'] > 125).to(torch.long)
        if resized_visual_prompts.shape[-3:] != pixel_values.shape[-3:]:
            print("the shape of resized_visual_prompts don't match with that of pixel_values")
            return None
        resized_visual_prompts = resized_visual_prompts.flatten(0, 1)[:sum(num_vprompts_list)]
    
        ot_pixel_values = [self.ot_image_processor(images=image, return_tensors='pt').pixel_values for image in images]
        ot_pixel_values = torch.cat(ot_pixel_values)
        # vp_images_list = []
        # for _visual_prompts in visual_prompts_list:
        #     for region in _visual_prompts:
        #         region_img = np.repeat(region[:, :, np.newaxis], 3, axis=2) * 255
        #         region_img = self.ot_image_processor(images=region_img, return_tensors='pt').pixel_values
        #         vp_images_list.append(region_img)
        # ot_visual_prompts = torch.stack(vp_images_list)[:, 0, :, :]  # num_prompts, h, w

        ot_visual_prompts = torch.from_numpy(np.concatenate(visual_prompts_list, axis=0)).\
            to(ot_pixel_values.dtype).to(ot_pixel_values.device)  # num_prompts, h, w
        assert ot_pixel_values.shape[-2:] == ot_visual_prompts.shape[-2:], f"ot_pixel_values.shape: {ot_pixel_values.shape[-2:]}, ot_visual_prompts.shape: {ot_visual_prompts.shape[-2:]}"
        
        patch_size = 14
        num_image_tokens = (pixel_values.shape[-1] // patch_size) ** 2

        ret = dict(
            input_ids=[1, 1, 1],
            labels=[1, 1, 1],
            attention_mask=[1, 1, 1],
            pixel_values=pixel_values,
            merged_visual_prompts=pixel_values,
            image_flags=torch.tensor([1]*(num_image_tokens * len(num_vprompts_list)), dtype=torch.long),
            visual_prompts=resized_visual_prompts,
            num_vprompts=num_vprompts_list,
            vprompt_flags=[[1 for _ in range(nvp)] for nvp in num_vprompts_list],
            num_images=len(num_vprompts_list),
            ot_pixel_values=ot_pixel_values,
            ot_visual_prompts=ot_visual_prompts,
            region_ids=region_ids_list,
        )

        return ret