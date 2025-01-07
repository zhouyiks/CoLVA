import os
from os import listdir

from mmengine.dist import master_only

from tools.sam2video_visualize.load import frames
from vlm.datasets.evaluation.base_eval_dataset import BaseEvalDataset
import json
import numpy as np
import copy
import cv2
from PIL import Image
from lmdeploy.vl.constants import IMAGE_TOKEN
import pycocotools.mask as maskUtils

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

class SAM2DatasetV3_whole(BaseEvalDataset):
    METAINFO: dict = dict(name='image dataset')
    def __init__(
            self,
            video_folder,
            json_folder,
            bs=8,
            select_frames=1,
    ):
        super().__init__()
        self.json_folder = json_folder
        self.json_files = []
        self.video_folder_idx = []
        if isinstance(json_folder, list):
            for i, _json_folder in enumerate(json_folder):
                json_files = os.listdir(_json_folder)
                for _file in json_files:
                    if 'manual.json' in _file:
                        self.json_files.append(os.path.join(_json_folder, _file))
                        self.video_folder_idx.append(i)
        else:
            json_files = os.listdir(json_folder)
            for _file in json_files:
                if 'manual.json' in _file:
                    self.json_files.append(os.path.join(json_folder, _file))

        self.video_folder = video_folder

        self.bs = bs
        self.num_select_frames = select_frames

    def __len__(self):
        return len(self.json_files) // self.bs

    def _get_data(self, idx):
        other_infos = {}
        json_name = self.json_files[idx]
        # json_path = os.path.join(self.json_folder, json_name)
        json_path = json_name
        with open(json_path, 'r') as f:
            data = json.load(f)

        other_infos['video_id'] = data['video_id']
        if isinstance(self.video_folder, list):
            video_path = os.path.join(self.video_folder[self.video_folder_idx[idx]], '{}.mp4'.format(data['video_id']))
        else:
            video_path = os.path.join(self.video_folder, '{}.mp4'.format(data['video_id']))
        if not os.path.exists(video_path):
            print(f"Not valid video !!! {video_path}")
            return None, None
        frames = get_video_frames(video_path)
        masklents = decode_masklet(data['masklet'])
        frames = frames[::4]
        if len(frames) != len(masklents):
            return None, None
        assert len(frames) == len(masklents)

        # frames [np.array(h, w, 3), ...]
        # masklents [np.array(h, w, n)]

        n_objs = masklents[0].shape[-1]

        objects_images = []
        for i in range(n_objs):
            object_masklents = [_item[:, :, i] for _item in masklents]
            select_frame_idxs = self.select_frames(object_masklents, nums=self.num_select_frames)
            object_frames = [copy.deepcopy(frames[_idx]) for _idx in select_frame_idxs]
            object_masks = [copy.deepcopy(object_masklents[_idx]) for _idx in select_frame_idxs]
            object_highlighted_images_crop, drop = self.highlight_object_crop(object_frames, object_masks, expand_ratio=1.4)
            if drop:
                continue
            # object_highlighted_images_relight = self.highlight_object_relight(object_frames, object_masks)
            object_highlighted_images_relight, _ = self.highlight_object_crop(object_frames, object_masks, expand_ratio=4.0)

            # _folder = os.path.join('./work_dirs/sam2_obj_images', 'obj_{}'.format(i))
            # os.mkdir(_folder)
            # for j, _save_iamge in enumerate(object_highlighted_images):
            #     _save_iamge.save(os.path.join(_folder, f'{j}.png'))

            question_crop = self.get_question_crop(len(object_highlighted_images_crop))
            question_relight = self.get_question_relight(len(object_highlighted_images_crop))
            # self._save_drawed_contours(object_highlighted_images_crop,
            #                            video_id=other_infos['video_id'],
            #                            obj_id=i, type='crop')
            # self._save_drawed_contours(object_highlighted_images_relight,
            #                            video_id=other_infos['video_id'],
            #                            obj_id=i, type='relight')

            objects_images.append({'images': object_highlighted_images_crop,
                                   'text_prompt': question_crop, 'type': 'crop', 'obj_id': i})
            objects_images.append(
                {'images': object_highlighted_images_relight, 'text_prompt': question_relight,
                 'type': 'relight', 'obj_id': i})
        return objects_images, other_infos

    def _save_drawed_contours(self, images, video_id, obj_id, type):
        for frame_id, image in enumerate(images):
            frame_name = f'{video_id}_obj{obj_id}_frame{frame_id}_{type}.png'
            image.save(os.path.join('/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/work_dirs/object_contour_demos/', frame_name))
        return

    # def get_question(self, num_objs):
    #     ret = ''
    #     for i in range(num_objs):
    #         ret += f'Image-{i+1}: {IMAGE_TOKEN}\n'
    #     ret += 'Here are several consecutive frames from a video. We have highlighted an object with yellow edges, meaning the object highlighted by the yellow edges in the video is the same object. We need you to provide some discriminative descriptions about this object, which can help us easily distinguish it from other similar objects in the image. The discriminative descriptions should include but are not limited to its category, color, shape, position in the image, state, purpose, properties, and its relationship with surrounding objects.\n'
    #     # ret += 'Please provide a detailed description of the object highlighted by the yellow contour, including its color, shape, position in the image, state, purpose, properties, and its relationship with surrounding objects.'
    #     ret += 'Please give the discriminative descriptions about the object.'
    #     return ret

    def get_question_crop(self, num_objs):
        ret = ''
        # print(num_objs)
        for i in range(num_objs):
            ret += f'Image-{i+1}: {IMAGE_TOKEN}\n'
        # ret += 'What is the object in the image, please answer with a phrase. If there is insufficient information to clearly identify the object, please respond with \"unidentifiable object.\"'
        ret += 'Please briefly describe the object in the image. Please only describe the category of the object and its appearance, without mentioning the white background. Additionally, focus solely on the information presented in the image without making any associations.'
        return ret

    def get_question_relight(self, num_objs):
        ret = ''
        # print(num_objs)
        for i in range(num_objs):
            ret += f'Image-{i + 1}: {IMAGE_TOKEN}\n'
        # ret += 'What is the object in the image, please answer with a phrase. If there is insufficient information to clearly identify the object, please respond with \"unidentifiable object.\"'
        ret += 'Please briefly describe the object in the image. Please only describe the category of the object and its appearance, without mentioning the white background. Additionally, focus solely on the information presented in the image without making any associations.'
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
        area = (_x_max - _x_min) * (_y_max - _y_min)
        drop = area < 180*180
        for frame, mask in zip(object_frames, object_masks):
            frame = frame[_y_min:_y_max, _x_min:_x_max]
            mask = mask[_y_min:_y_max, _x_min:_x_max]
            # set to dark
            frame[np.logical_not(mask)] = (frame[np.logical_not(mask)].astype(np.int64) * 0 + 255).astype(np.uint8)
            image = frame.astype(np.uint8)
            image = Image.fromarray(image)
            # image = add_edge_color(frame[_y_min:_y_max, _x_min:_x_max], mask[_y_min:_y_max, _x_min:_x_max])
            ret.append(image)
        return ret, drop

    def highlight_object_relight(self, object_frames, object_masks):
        ret = []
        for frame, mask in zip(object_frames, object_masks):
            # set to dark
            frame[np.logical_not(mask)] = (frame[np.logical_not(mask)].astype(np.int64) * 0 + 255).astype(np.uint8)
            # frame = frame.astype(np.uint8)
            # image = add_edge_color(frame, mask)
            image = frame.astype(np.uint8)
            image = Image.fromarray(image)
            ret.append(image)
        return ret

    def select_frames(self, object_masklents, nums=3):
        areas = np.array([np.sum(mask) for mask in object_masklents])
        frame_indexes = np.arange(0, len(object_masklents))

        sort_idxs = np.argsort(areas)[::-1]
        frame_indexes = frame_indexes[sort_idxs][:nums].tolist()
        frame_indexes.sort()
        return frame_indexes

    def __getitem__(self, idx):
        start = idx * self.bs
        end = start + self.bs

        data_dicts = []
        for _idx in range(start, end):
            objects_images, other_infos = self._get_data(_idx)
            if objects_images is None:
                continue
            for i, object_dict in enumerate(objects_images):
                object_dict.update(other_infos)
                # object_dict.update({'obj_id': i})
                data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'sam2'}

    @master_only
    def evaluate(self, *args, **kwargs):
        return {'Acc': 0}

class SAM2DatasetV3_Imagecap_whole(BaseEvalDataset):
    METAINFO: dict = dict(name='image dataset')
    def __init__(
            self,
            video_folder,
            json_folder,
            crop_cap_folder,
            bs=8,
            select_frames=1,
    ):
        super().__init__()
        self.json_folder = json_folder
        self.json_files = []
        self.video_folder_idx = []
        if isinstance(json_folder, list):
            for i, _json_folder in enumerate(json_folder):
                json_files = os.listdir(_json_folder)
                for _file in json_files:
                    if 'manual.json' in _file:
                        self.json_files.append(os.path.join(_json_folder, _file))
                        self.video_folder_idx.append(i)
        else:
            json_files = os.listdir(json_folder)
            for _file in json_files:
                if 'manual.json' in _file:
                    self.json_files.append(os.path.join(json_folder, _file))

        # crop-level caption
        self.crop_cap_data = []
        self.crop_cap_folder = crop_cap_folder
        crop_cap_json_files = os.listdir(crop_cap_folder)
        for _file in crop_cap_json_files:
            path = os.path.join(self.crop_cap_folder, _file)
            with open(path, 'r') as f:
                self.crop_cap_data.extend(json.load(f))

        self.crop_cap_data_dict = {}
        for _data in self.crop_cap_data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']

            if video_id not in self.crop_cap_data_dict.keys():
                self.crop_cap_data_dict[video_id] = {}
            if obj_id not in self.crop_cap_data_dict[video_id].keys():
                self.crop_cap_data_dict[video_id][obj_id] = {}
            self.crop_cap_data_dict[video_id][obj_id] = _data


        self.video_folder = video_folder

        self.bs = bs
        self.num_select_frames = select_frames

    def __len__(self):
        return len(self.json_files) // self.bs

    def _get_data(self, idx):
        other_infos = {}
        json_name = self.json_files[idx]
        # json_path = os.path.join(self.json_folder, json_name)
        json_path = json_name
        with open(json_path, 'r') as f:
            data = json.load(f)

        if data['video_id'] not in self.crop_cap_data_dict.keys():
            return [], {}

        other_infos['video_id'] = data['video_id']
        if isinstance(self.video_folder, list):
            video_path = os.path.join(self.video_folder[self.video_folder_idx[idx]], '{}.mp4'.format(data['video_id']))
        else:
            video_path = os.path.join(self.video_folder, '{}.mp4'.format(data['video_id']))
        frames = get_video_frames(video_path)
        masklents = decode_masklet(data['masklet'])
        frames = frames[::4]
        assert len(frames) == len(masklents)

        # frames [np.array(h, w, 3), ...]
        # masklents [np.array(h, w, n)]

        n_objs = masklents[0].shape[-1]

        objects_images = []
        for i in range(n_objs):
            if i not in self.crop_cap_data_dict[data['video_id']].keys():
                continue
            category = self.crop_cap_data_dict[data['video_id']][i]['category']
            caption = self.crop_cap_data_dict[data['video_id']][i]['caption']
            object_masklents = [_item[:, :, i] for _item in masklents]
            select_frame_idxs = self.select_frames(object_masklents, nums=self.num_select_frames)
            object_frames = [copy.deepcopy(frames[_idx]) for _idx in select_frame_idxs]
            object_masks = [copy.deepcopy(object_masklents[_idx]) for _idx in select_frame_idxs]
            # object_highlighted_images_crop = self.highlight_object_crop(object_frames, object_masks, expand_ratio=1.4)
            # object_highlighted_images_relight = self.highlight_object_relight(object_frames, object_masks)
            object_highlighted_images_relight = self.highlight_object_relight(object_frames, object_masks)

            # _folder = os.path.join('./work_dirs/sam2_obj_images', 'obj_{}'.format(i))
            # os.mkdir(_folder)
            # for j, _save_iamge in enumerate(object_highlighted_images):
            #     _save_iamge.save(os.path.join(_folder, f'{j}.png'))

            # question_crop = self.get_question_crop(len(object_highlighted_images_crop))
            question_relight = self.get_question_relight(len(object_highlighted_images_relight), caption, category)
            # self._save_drawed_contours(object_highlighted_images_relight,
            #                            video_id=other_infos['video_id'],
            #                            obj_id=i, type='relight')

            # objects_images.append({'images': object_highlighted_images_crop,
            #                        'text_prompt': question_crop, 'type': 'crop', 'obj_id': i})
            objects_images.append(
                {'images': object_highlighted_images_relight, 'text_prompt': question_relight,
                 'obj_id': i, 'crop_caption': caption, 'crop_category': category})
        return objects_images, other_infos

    def _save_drawed_contours(self, images, video_id, obj_id, type):
        for frame_id, image in enumerate(images):
            frame_name = f'{video_id}_obj{obj_id}_frame{frame_id}_{type}.png'
            image.save(os.path.join('/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/work_dirs/object_contour_demos/', frame_name))
        return

    # def get_question(self, num_objs):
    #     ret = ''
    #     for i in range(num_objs):
    #         ret += f'Image-{i+1}: {IMAGE_TOKEN}\n'
    #     ret += 'Here are several consecutive frames from a video. We have highlighted an object with yellow edges, meaning the object highlighted by the yellow edges in the video is the same object. We need you to provide some discriminative descriptions about this object, which can help us easily distinguish it from other similar objects in the image. The discriminative descriptions should include but are not limited to its category, color, shape, position in the image, state, purpose, properties, and its relationship with surrounding objects.\n'
    #     # ret += 'Please provide a detailed description of the object highlighted by the yellow contour, including its color, shape, position in the image, state, purpose, properties, and its relationship with surrounding objects.'
    #     ret += 'Please give the discriminative descriptions about the object.'
    #     return ret

    def get_question_crop(self, num_objs):
        ret = ''
        # print(num_objs)
        for i in range(num_objs):
            ret += f'Image-{i+1}: {IMAGE_TOKEN}\n'
        # ret += 'What is the object in the image, please answer with a phrase. If there is insufficient information to clearly identify the object, please respond with \"unidentifiable object.\"'
        ret += 'Please briefly describe the object in the image. Please only describe the category of the object and its appearance, without mentioning the white background. Additionally, focus solely on the information presented in the image without making any associations.'
        return ret

    def get_question_relight(self, num_objs, caption, category):
        ret = ''
        for i in range(num_objs):
            ret += f'Image-{i + 1}: {IMAGE_TOKEN}\n'
        ret += "I highlighted an object in the image with a yellow edge. This object could be an entity-level object, a part-level object, or even a multi-object. Here are some close-up observations of the object, which are reliable descriptions, but there may still be a few situations that lead to inaccuracies due to the lack of overall image information. Based on the image information, the yellow-highlighted object, and the provided descriptions, please generate a correct detailed description of the object and its relationship with surrounding objects.\n"
        ret += "The close-up observations of the object:\n"
        ret += f"  This object is {category.lower()}. {caption.replace('Image', 'Object').replace('image', 'object')}\n"
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

    def highlight_object_relight(self, object_frames, object_masks):
        ret = []
        for frame, mask in zip(object_frames, object_masks):
            # set to dark
            frame[np.logical_not(mask)] = (frame[np.logical_not(mask)].astype(np.int64) * 0.6).astype(np.uint8)
            frame = frame.astype(np.uint8)
            image = add_edge_color(frame, mask)
            # image = frame.astype(np.uint8)
            # image = Image.fromarray(image)
            ret.append(image)
        return ret

    def select_frames(self, object_masklents, nums=3):
        areas = np.array([np.sum(mask) for mask in object_masklents])
        frame_indexes = np.arange(0, len(object_masklents))

        sort_idxs = np.argsort(areas)[::-1]
        frame_indexes = frame_indexes[sort_idxs][:nums].tolist()
        frame_indexes.sort()
        return frame_indexes

    def __getitem__(self, idx):
        start = idx * self.bs
        end = start + self.bs

        data_dicts = []
        for _idx in range(start, end):
            objects_images, other_infos = self._get_data(_idx)
            for i, object_dict in enumerate(objects_images):
                object_dict.update(other_infos)
                # object_dict.update({'obj_id': i})
                data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'sam2_recap'}

    @master_only
    def evaluate(self, *args, **kwargs):
        return {'Acc': 0}

class SAM2DatasetV3_Videocap_whole(BaseEvalDataset):
    METAINFO: dict = dict(name='image dataset')
    def __init__(
            self,
            video_folder,
            json_folder,
            image_cap_folder,
            bs=8,
            select_frames=8,
    ):
        super().__init__()
        self.json_folder = json_folder
        self.json_files = []
        self.video_folder_idx = []
        if isinstance(json_folder, list):
            for i, _json_folder in enumerate(json_folder):
                json_files = os.listdir(_json_folder)
                for _file in json_files:
                    if 'manual.json' in _file:
                        self.json_files.append(os.path.join(_json_folder, _file))
                        self.video_folder_idx.append(i)
        else:
            json_files = os.listdir(json_folder)
            for _file in json_files:
                if 'manual.json' in _file:
                    self.json_files.append(os.path.join(json_folder, _file))

        # crop-level caption
        self.image_cap_data = []
        self.image_cap_folder = image_cap_folder
        image_cap_json_files = os.listdir(image_cap_folder)
        for _file in image_cap_json_files:
            path = os.path.join(self.image_cap_folder, _file)
            with open(path, 'r') as f:
                self.image_cap_data.extend(json.load(f))

        self.image_cap_data_dict = {}
        for _data in self.image_cap_data:
            video_id = _data['video_id']
            obj_id = _data['obj_id']

            if video_id not in self.image_cap_data_dict.keys():
                self.image_cap_data_dict[video_id] = {}
            if obj_id not in self.image_cap_data_dict[video_id].keys():
                self.image_cap_data_dict[video_id][obj_id] = {}
            self.image_cap_data_dict[video_id][obj_id] = _data


        self.video_folder = video_folder

        self.bs = bs
        self.num_select_frames = select_frames

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        ])

    def __len__(self):
        return len(self.json_files) // self.bs

    def _get_data(self, idx):
        other_infos = {}
        json_name = self.json_files[idx]
        # json_path = os.path.join(self.json_folder, json_name)
        json_path = json_name
        with open(json_path, 'r') as f:
            data = json.load(f)

        if data['video_id'] not in self.image_cap_data_dict.keys():
            return [], {}

        other_infos['video_id'] = data['video_id']
        if isinstance(self.video_folder, list):
            video_path = os.path.join(self.video_folder[self.video_folder_idx[idx]], '{}.mp4'.format(data['video_id']))
        else:
            video_path = os.path.join(self.video_folder, '{}.mp4'.format(data['video_id']))
        frames = get_video_frames(video_path)
        masklents = decode_masklet(data['masklet'])
        frames = frames[::4]
        assert len(frames) == len(masklents)

        # frames [np.array(h, w, 3), ...]
        # masklents [np.array(h, w, n)]

        n_objs = masklents[0].shape[-1]

        objects_images = []
        for i in range(n_objs):
            if i not in self.image_cap_data_dict[data['video_id']].keys():
                continue
            caption = self.image_cap_data_dict[data['video_id']][i]['caption']
            crop_caption = self.image_cap_data_dict[data['video_id']][i]['crop_caption']
            crop_category = self.image_cap_data_dict[data['video_id']][i]['crop_category']
            object_masklents = [_item[:, :, i] for _item in masklents]
            select_frame_idxs = self.select_frames(object_masklents, nums=self.num_select_frames)
            object_frames = [copy.deepcopy(frames[_idx]) for _idx in select_frame_idxs]
            object_masks = [copy.deepcopy(object_masklents[_idx]) for _idx in select_frame_idxs]
            # object_highlighted_images_crop = self.highlight_object_crop(object_frames, object_masks, expand_ratio=1.4)
            # object_highlighted_images_relight = self.highlight_object_relight(object_frames, object_masks)
            object_highlighted_images_relight = self.highlight_object_relight(object_frames, object_masks)

            # _folder = os.path.join('./work_dirs/sam2_obj_images', 'obj_{}'.format(i))
            # os.mkdir(_folder)
            # for j, _save_iamge in enumerate(object_highlighted_images):
            #     _save_iamge.save(os.path.join(_folder, f'{j}.png'))

            # question_crop = self.get_question_crop(len(object_highlighted_images_crop))
            question_relight = self.get_question_relight(len(object_highlighted_images_relight), caption)
            # self._save_drawed_contours(object_highlighted_images_relight,
            #                            video_id=other_infos['video_id'],
            #                            obj_id=i, type='relight')

            # objects_images.append({'images': object_highlighted_images_crop,
            #                        'text_prompt': question_crop, 'type': 'crop', 'obj_id': i})
            objects_images.append(
                {'images': object_highlighted_images_relight, 'text_prompt': question_relight,
                 'obj_id': i, 'crop_caption': crop_caption, 'crop_category': crop_category,
                 'image_caption': caption})
        return objects_images, other_infos

    def _save_drawed_contours(self, images, video_id, obj_id, type):
        for frame_id, image in enumerate(images):
            frame_name = f'{video_id}_obj{obj_id}_frame{frame_id}_{type}.png'
            image.save(os.path.join('/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/work_dirs/object_contour_demos/', frame_name))
        return

    # def get_question(self, num_objs):
    #     ret = ''
    #     for i in range(num_objs):
    #         ret += f'Image-{i+1}: {IMAGE_TOKEN}\n'
    #     ret += 'Here are several consecutive frames from a video. We have highlighted an object with yellow edges, meaning the object highlighted by the yellow edges in the video is the same object. We need you to provide some discriminative descriptions about this object, which can help us easily distinguish it from other similar objects in the image. The discriminative descriptions should include but are not limited to its category, color, shape, position in the image, state, purpose, properties, and its relationship with surrounding objects.\n'
    #     # ret += 'Please provide a detailed description of the object highlighted by the yellow contour, including its color, shape, position in the image, state, purpose, properties, and its relationship with surrounding objects.'
    #     ret += 'Please give the discriminative descriptions about the object.'
    #     return ret

    def get_question_crop(self, num_objs):
        ret = ''
        # print(num_objs)
        for i in range(num_objs):
            ret += f'Image-{i+1}: {IMAGE_TOKEN}\n'
        # ret += 'What is the object in the image, please answer with a phrase. If there is insufficient information to clearly identify the object, please respond with \"unidentifiable object.\"'
        ret += 'Please briefly describe the object in the image. Please only describe the category of the object and its appearance, without mentioning the white background. Additionally, focus solely on the information presented in the image without making any associations.'
        return ret

    def get_question_relight(self, num_objs, caption):
        ret = ''
        for i in range(num_objs):
            ret += f'Frame-{i + 1}: {IMAGE_TOKEN}\n'
        ret += "There are some detailed image-level descriptions of the object; however, due to the lack of temporal information in the video, there are inaccuracies regarding the object's movement and its actions. Based on the video information, the yellow-edge highlighted object, and the provided image-level descriptions, please generate a correct and detailed description of the object, its motion, and what it is doing.\n"
        ret += "The detailed image-level descriptions of the object:\n"
        ret += f"  {caption}\n"
        ret += 'Please give the correct detailed video-level description of the object highlighted by the yellow edge.'
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

    def highlight_object_relight(self, object_frames, object_masks):
        ret = []
        for frame, mask in zip(object_frames, object_masks):
            # set to dark
            frame[np.logical_not(mask)] = (frame[np.logical_not(mask)].astype(np.int64) * 0.6).astype(np.uint8)
            frame = frame.astype(np.uint8)
            image = add_edge_color(frame, mask)
            image = self.transformer(image)
            # image = frame.astype(np.uint8)
            # image = Image.fromarray(image)
            ret.append(image)
        return ret

    def select_frames(self, object_masklents, nums=3):
        areas = np.array([np.sum(mask) for mask in object_masklents])
        frame_indexes = np.arange(0, len(object_masklents))

        # remove none object
        frame_indexes_valid = frame_indexes[areas > 0]
        if len(frame_indexes_valid) < nums:
            start = frame_indexes_valid[0]
            start = max(0, start - (nums - len(frame_indexes_valid)) // 2)
            end = min(start + nums, len(object_masklents))
            frame_indexes = np.arange(start, end).tolist()
        else:
            stride = len(frame_indexes_valid) / (nums + 1e-4)
            frame_indexes = []
            for i in range(nums):
                frame_indexes.append(frame_indexes_valid[min(int(i * stride), len(frame_indexes_valid) - 1)])
        # print(frame_indexes, [areas[idx] for idx in frame_indexes])
        return frame_indexes

    def __getitem__(self, idx):
        start = idx * self.bs
        end = start + self.bs

        data_dicts = []
        for _idx in range(start, end):
            objects_images, other_infos = self._get_data(_idx)
            for i, object_dict in enumerate(objects_images):
                object_dict.update(other_infos)
                # object_dict.update({'obj_id': i})
                data_dicts.append(object_dict)

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'sam2_video_recap'}

    @master_only
    def evaluate(self, *args, **kwargs):
        return {'Acc': 0}

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frames = []

    frame_id = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame[:, :, ::-1])

        frame_id += 1

    cap.release()
    return frames


def images_to_video(frames, video_name, fps=6):
    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame[:, :, ::-1])

    # cv2.destroyAllWindows()
    video.release()
    return

def decode_masklet(masklet):
    masks = []
    for _rle in masklet:
        mask = maskUtils.decode(_rle)
        masks.append(mask)
    return masks

def draw_mask(image, mask):
    obj_mask = mask * 255
    obj_mask = np.stack([obj_mask * 1, obj_mask * 0, obj_mask * 0], axis=2)
    obj_mask = obj_mask * 0.5 + copy.deepcopy(image) * 0.5
    obj_mask = obj_mask.astype(np.uint8)
    return obj_mask

def add_mask2images(frames, masklets):
    show_videos = []
    for i_frames, (frame, masks) in enumerate(zip(frames, masklets)):
        if i_frames == 0:
            n_obj = masks.shape[-1]
            for i_obj in range(n_obj):
                show_videos.append([])

        n_obj = masks.shape[-1]
        for i_obj in range(n_obj):
            show_videos[i_obj].append(draw_mask(copy.deepcopy(frame), masks[:, :, i_obj]))
    return show_videos

def add_edge_color(image, mask, edge_color=(255, 255, 0), thickness=3):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tuple_contours = tuple([np.array(contour) for contour in contours])
    cv2.drawContours(image, tuple_contours, -1, color=edge_color, thickness=thickness)

    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    return image