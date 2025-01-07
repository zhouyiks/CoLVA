import os

from mmengine.dist import master_only

from vlm.datasets.evaluation.base_eval_dataset import BaseEvalDataset
import json
import numpy as np
import copy
from PIL import Image
from lmdeploy.vl.constants import IMAGE_TOKEN
from pycocotools import mask as mask_utils
import torch.nn.functional as F
import torch

import cv2

class DemoImageCap(BaseEvalDataset):
    METAINFO: dict = dict(name='image dataset')
    def __init__(
            self,
            image_folder,
            bs=8,
    ):
        super().__init__()
        self.image_folder = image_folder
        image_files = []
        for file_name in os.listdir(self.image_folder):
            if 'out' not in file_name and '.jpg' in file_name:
                image_files.append(file_name)

        json_files = []
        for file_name in image_files:
            json_files.append(file_name.replace('.jpg', '_out.json'))

        self.image_files = image_files
        self.json_files = json_files

        self.data_dicts = []
        for image_file, json_file in zip(image_files, json_files):
            with open(os.path.join(image_folder, json_file), 'r') as f:
                _datas = json.load(f)
            for _data in _datas:
                self.data_dicts.append({'image_file': image_file, 'object_anno': _data})

        self.bs = bs

    def __len__(self):
        if len(self.data_dicts) % self.bs == 0:
            return len(self.data_dicts) // self.bs
        return len(self.data_dicts) // self.bs + 1

    def decode_mask(self, rle):
        m = mask_utils.decode(rle) # (h, w)
        return m

    def _get_data(self, idx):
        data = self.data_dicts[idx]
        other_infos = {}
        other_infos['image_id'] = data['image_file']

        image = Image.open(os.path.join(self.image_folder, data['image_file'])).convert('RGB')
        object_anno = data['object_anno']
        masks = data['object_anno']['segmentation']
        masks = self.decode_mask(masks)

        image_shape = image.size
        masks = torch.Tensor(masks).unsqueeze(0).unsqueeze(0)
        masks = F.interpolate(
            masks,
            size=(image_shape[1], image_shape[0]),
            mode='nearest').squeeze(0).squeeze(0)
        masks = masks.numpy()

        object_highlighted_images_relight = self.highlight_object_relight(np.array(image), masks)
        question_relight = self.get_question_relight()
        objects_images = [{
            'images': object_highlighted_images_relight, 'text_prompt': question_relight,
            'object_anno': object_anno,
        }]
        return objects_images, other_infos

    def _save_drawed_contours(self, images, video_id, obj_id, type):
        for frame_id, image in enumerate(images):
            frame_name = f'{video_id}_obj{obj_id}_frame{frame_id}_{type}.png'
            image.save(os.path.join('/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/work_dirs/object_contour_demos/', frame_name))
        return

    def get_question_relight(self, ):
        ret = f'{IMAGE_TOKEN}\n'
        ret += "I highlighted an object in the image with a yellow edge. This object could be an entity-level object. Based on the image information, the yellow-highlighted object, please generate a correct detailed description of the object and its relationship with surrounding objects.\n"
        ret += 'Please give the correct detailed description of the object highlighted by the yellow edge.'
        return ret

    def highlight_object(self, object_frames, object_masks):
        ret = []
        for frame, mask in zip(object_frames, object_masks):
            image = add_edge_color(frame, mask)
            ret.append(image)
        return ret

    def _get_crop_range(self, masks, expand_ratio=1.5):
        boxes = []
        for mask in masks:
            rows, cols = np.nonzero(mask)

            if len(rows) == 0:
                print("Warning !!! Zero mask !!!")
                continue

            x_min, x_max = cols.min(), cols.max() + 1
            y_min, y_max = rows.min(), rows.max() + 1
            boxes.append([x_min, y_min, x_max, y_max])

        h, w = masks[0].shape
        _x_min, _y_min, _x_max, _y_max = boxes[0]
        for box in boxes[1:]:
            _x_min = min(_x_min, box[0])
            _y_min = min(_y_min, box[1])
            _x_max = max(_x_max, box[2])
            _y_max = max(_y_max, box[3])

        _cx = (_x_min + _x_max) / 2.0
        _cy = (_y_min + _y_max) / 2.0

        _x_min = (_x_min - _cx) * expand_ratio + _cx
        _x_max = (_x_max - _cx) * expand_ratio + _cx
        _y_min = (_y_min - _cy) * expand_ratio + _cy
        _y_max = (_y_max - _cy) * expand_ratio + _cy

        _x_min = max(_x_min, 0)
        _y_min = max(_y_min, 0)
        _x_max = min(_x_max, w)
        _y_max = min(_y_max, h)
        return int(_x_min), int(_x_max), int(_y_min), int(_y_max)

    def highlight_object_crop(self, object_frames, object_masks, expand_ratio):
        ret = []
        _x_min, _x_max, _y_min, _y_max = self._get_crop_range(object_masks, expand_ratio=expand_ratio)
        for frame, mask in zip(object_frames, object_masks):
            frame = frame[_y_min:_y_max, _x_min:_x_max]
            mask = mask[_y_min:_y_max, _x_min:_x_max]
            # set to dark
            frame[np.logical_not(mask)] = (frame[np.logical_not(mask)].astype(np.int64) * 0 + 255).astype(np.uint8)
            image = frame.astype(np.uint8)
            image = Image.fromarray(image)
            # image = add_edge_color(frame[_y_min:_y_max, _x_min:_x_max], mask[_y_min:_y_max, _x_min:_x_max])
            ret.append(image)
        return ret

    def highlight_object_relight(self, frame, mask):
        # set to dark
        frame[np.logical_not(mask)] = (frame[np.logical_not(mask)].astype(np.int64) * 0.6).astype(np.uint8)
        frame = frame.astype(np.uint8)
        image = add_edge_color(frame, mask)
        return image

    def select_frames(self, object_masklents, nums=3):
        areas = np.array([np.sum(mask) for mask in object_masklents])
        frame_indexes = np.arange(0, len(object_masklents))

        sort_idxs = np.argsort(areas)[::-1]
        frame_indexes = frame_indexes[sort_idxs][:nums].tolist()
        frame_indexes.sort()
        return frame_indexes

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.data_dicts))

        data_dicts = []
        for _idx in range(start, end):
            objects_images, other_infos = self._get_data(_idx)
            for i, object_dict in enumerate(objects_images):
                object_dict.update(other_infos)
                # object_dict.update({'obj_id': i})
                data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'demo_imgcap'}

    @master_only
    def evaluate(self, *args, **kwargs):
        return {'Acc': 0}

def add_edge_color(image, mask, edge_color=(255, 255, 0), thickness=3):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tuple_contours = tuple([np.array(contour) for contour in contours])
    cv2.drawContours(image, tuple_contours, -1, color=edge_color, thickness=thickness)

    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    return image


class DemoImageCap_WholeCap(BaseEvalDataset):
    METAINFO: dict = dict(name='image dataset')
    def __init__(
            self,
            image_folder,
            bs=1,
    ):
        super().__init__()
        self.image_folder = image_folder
        image_files = []
        for file_name in os.listdir(self.image_folder):
            if '.jpg' in file_name or '.png' in file_name:
                image_files.append(file_name)

        txt_files = []
        for image_file in image_files:
            txt_file = image_file[:-4] + '.txt'
            txt_files.append(txt_file)

        self.image_files = image_files
        self.txt_files = txt_files

        self.bs = bs

    def __len__(self):
        if len(self.image_files) % self.bs == 0:
            return len(self.image_files) // self.bs
        return len(self.image_files) // self.bs + 1

    def _get_data(self, idx):
        image_file = self.image_files[idx]
        txt_file = self.txt_files[idx]
        other_infos = {}
        other_infos['image_id'] = image_file

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        with open(os.path.join(self.image_folder, txt_file), 'r') as f:
            text = f.read()

        question_relight = self.get_question_relight(text)
        objects_images = [{
            'images': image, 'text_prompt': question_relight,
        }]
        return objects_images, other_infos

    def _save_drawed_contours(self, images, video_id, obj_id, type):
        for frame_id, image in enumerate(images):
            frame_name = f'{video_id}_obj{obj_id}_frame{frame_id}_{type}.png'
            image.save(os.path.join('/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/work_dirs/object_contour_demos/', frame_name))
        return

    def get_question_relight(self, text):
        ret = f'{IMAGE_TOKEN}\n'
        ret += "In this image, we have drawn a colored number tag for each object and highlighted the object with the corresponding color's edge. We will provide you with detailed descriptions of these objects, and based on these descriptions, please output a very detailed overall scene image caption, including the objects' detailed characteristics, their interactions, and spatial relationships. These object captions may contain redundancy, so please ignore the captions of objects whose numbers do not appear in the image.\n"
        ret += "The objects descriptions are following the format of tag_number: description. There are the object descriptions:\n"
        ret += text
        ret += '\n Please output the very detailed overall scene image caption.'
        return ret

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, len(self.image_files))

        data_dicts = []
        for _idx in range(start, end):
            objects_images, other_infos = self._get_data(_idx)
            for i, object_dict in enumerate(objects_images):
                object_dict.update(other_infos)
                # object_dict.update({'obj_id': i})
                data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'demo_imgcap_overall'}

    @master_only
    def evaluate(self, *args, **kwargs):
        return {'Acc': 0}