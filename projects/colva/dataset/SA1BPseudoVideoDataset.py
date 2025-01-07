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
from torchvision.transforms.functional import InterpolationMode
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from transformers import AutoConfig, AutoTokenizer
from pycocotools import mask

from mmdet.datasets.api_wrappers import COCO

from .utils import detection_utils as utils
from .utils.detectron2.data2 import transforms as T
from .utils.augmentation import build_pseudo_augmentation
from .utils import (expand2square, expand2square_mask)
from .process_functions import dynamic_preprocess

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag

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



class SA1BPseudoVideoDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 model_path,
                 data_path=None,
                 image_folder=None,
                 dynamic_image_size=False,
                 pad_image_to_square=False,
                 num_dynamic_patch=None,
                 repeat_time=1,
                 ot_image_processor=None,
                 tokenizer=None,
                 vfm_name="RADIO",
                 ):
        super().__init__()

        self.dynamic_image_size = dynamic_image_size
        self.pad_image_to_square = pad_image_to_square
        self.ot_image_processor = ot_image_processor
        if vfm_name in ["DINOv2", "ConvNext"]:
            self.ot_image_processor.do_center_crop=False
            self.ot_image_processor.do_resize=False

        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if num_dynamic_patch is not None and len(num_dynamic_patch) == 2:
            self.min_dynamic_patch = num_dynamic_patch[0]
            self.max_dynamic_patch = num_dynamic_patch[1]
        else:
            self.min_dynamic_patch = self.cfg.min_dynamic_patch
            self.max_dynamic_patch = self.cfg.max_dynamic_patch
        self.image_size = self.cfg.force_image_size
        self.use_thumbnail = self.cfg.use_thumbnail

        with open(data_path, 'r') as f:
            data_list = json.load(f)['images']
        # self.data = data_list
        left_data_list = []
        for item in data_list:
            if item['file_name'].startswith('sa_0000'):
                continue
            left_data_list.append(item)
        self.data = left_data_list

        # self.coco = COCO(data_path)
        # img_ids = self.coco.get_img_ids()
        
        # data_list = []
        # print("Loading data list...")
        # for img_id in tqdm.tqdm(img_ids):
        #     raw_img_info = self.coco.load_imgs([img_id])[0]
        #     raw_img_info["img_id"] = img_id
        #     ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        #     raw_ann_info = self.coco.load_anns(ann_ids)

        #     parsed_data_info = self.parse_data_info(
        #         {'raw_ann_info': raw_ann_info, 'raw_img_info': raw_img_info}
        #     )
        #     data_list.append(parsed_data_info)
        # self.data = data_list
        
        if vfm_name == "DINOv2":
            augs = build_pseudo_augmentation(True, force_image_size=512)
        elif vfm_name in ["RADIO", "ConvNext"]:
            augs = build_pseudo_augmentation(True, force_image_size=1024)
        else:
            raise NotImplementedError
        self.augmentations = T.AugmentationList(augs)
        
        self.transform = tvT.Compose([
            tvT.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            tvT.Resize((self.image_size, self.image_size)),
            tvT.ToTensor(),
            tvT.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

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

    # def parse_data_info(self, raw_data_info: dict):
        

    #     img_info = raw_data_info["raw_img_info"]
    #     ann_info = raw_data_info["raw_ann_info"]

    #     data_info = {}
    #     data_info["image"] = img_info["file_name"]
    #     data_info["img_id"] = img_info["img_id"]
    #     data_info["height"] = img_info["height"]
    #     data_info["width"] = img_info["width"]

    #     instances = []
    #     for i, ann in enumerate(ann_info):
    #         instance = {}

    #         if ann.get("ignore", False):
    #             continue
    #         x1, y1, w, h = ann["bbox"]
    #         inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
    #         inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
    #         if inter_w * inter_h == 0:
    #             continue
    #         if ann["area"] <= 0 or w < 1 or h < 1:
    #             continue
    #         bbox = [x1, y1, x1 + w, y1 + h]

    #         if ann.get("iscrowd", False):
    #             instance["ignore_flag"] = 1
    #         else:
    #             instance["ignore_flag"] = 0
    #         instance["bbox"] = bbox
            
    #         if ann.get("segmentation", None):
    #             instance["segmentation"] = ann["segmentation"]
            
    #         if "instance_id" in ann:
    #             instance["instance_id"] = ann["instance_id"]
    #         else:
    #             instance["instance_id"] = ann["id"]
            
    #         instances.append(instance)
    #     data_info["annotations"] = instances
    #     return data_info
    
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
        
        merged_visual_prompts = [image.copy() for image in images]
        
        if self.dynamic_image_size:
            num_patches_list, images_list, merged_regions_list, crop_regions_list, num_vprompts_list = [], [], [], [], []
            for image, visual_prompts, merged_visual_prompt in zip(images, visual_prompts_list, merged_visual_prompts):
                try:
                    _images, regions, merged_regions = dynamic_preprocess(
                        image, visual_prompts, merged_visual_prompt, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                        image_size=self.image_size, use_thumbnail=self.use_thumbnail
                    )
                except Exception as e:
                    return None
                images_list.extend(_images)
                merged_regions_list.extend(merged_regions)
                crop_regions_list.extend(regions)
                num_patches_list.append(len(_images))
                num_vprompts_list.append(len(regions))
        else:
            raise NotImplementedError
        
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
        try:
            visual_prompts = torch.stack(transformed_visual_prompts, dim=0)  # num_prompts, num_patch, h, w
        except:
            print("num regions: ", len(crop_regions_list))
            print("transformed_visual_prompts.shape: ", [ele.shape for ele in transformed_visual_prompts])
            print("pixel_values.shape: ", pixel_values.shape)
            exit(0)
        assert merged_visual_prompts.shape[:2] == pixel_values.shape[:2]
        

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

        ret = dict(
            input_ids=[1, 1, 1],
            labels=[1, 1, 1],
            attention_mask=[1, 1, 1],
            pixel_values=pixel_values,
            merged_visual_prompts=merged_visual_prompts,
            num_patches=num_patches_list,
            visual_prompts=visual_prompts.flatten(0, 1),
            num_vprompts=num_vprompts_list,
            num_images=len(num_vprompts_list),
            ot_pixel_values=ot_pixel_values,
            ot_visual_prompts=ot_visual_prompts,
            region_ids=region_ids_list,
        )

        return ret







   







