import os
from mmengine.dist import master_only
from vlm.datasets.evaluation.base_eval_dataset import BaseEvalDataset
import json
import numpy as np
import copy
import cv2
from PIL import Image
from lmdeploy.vl.constants import IMAGE_TOKEN
import pycocotools.mask as maskUtils

class SAM2DatasetV2(BaseEvalDataset):
    METAINFO: dict = dict(name='image dataset')
    def __init__(
            self,
            video_folder,
            json_folder,
            bs=8,
            select_frames=3,
    ):
        super().__init__()
        self.json_folder = json_folder
        json_files = os.listdir(json_folder)
        self.json_files = []
        for _file in json_files:
            if 'manual.json' in _file:
                self.json_files.append(_file)

        self.video_folder = video_folder

        self.bs = bs
        self.num_select_frames = select_frames

    def __len__(self):
        return len(self.json_files) // self.bs

    def _get_data(self, idx):
        other_infos = {}
        json_name = self.json_files[idx]
        json_path = os.path.join(self.json_folder, json_name)
        with open(json_path, 'r') as f:
            data = json.load(f)

        other_infos['video_id'] = data['video_id']

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
            object_masklents = [_item[:, :, i] for _item in masklents]
            select_frame_idxs = self.select_frames(object_masklents, nums=self.num_select_frames)
            object_frames = [copy.deepcopy(frames[_idx]) for _idx in select_frame_idxs]
            object_masks = [copy.deepcopy(object_masklents[_idx]) for _idx in select_frame_idxs]
            object_highlighted_images_crop = self.highlight_object_crop(object_frames, object_masks)
            object_highlighted_images_relight = self.highlight_object_relight(object_frames, object_masks)

            # _folder = os.path.join('./work_dirs/sam2_obj_images', 'obj_{}'.format(i))
            # os.mkdir(_folder)
            # for j, _save_iamge in enumerate(object_highlighted_images):
            #     _save_iamge.save(os.path.join(_folder, f'{j}.png'))

            question_crop = self.get_question_crop(len(object_highlighted_images_crop))
            question_relight = self.get_question_relight(len(object_highlighted_images_crop))
            self._save_drawed_contours(object_highlighted_images_crop,
                                       video_id=other_infos['video_id'],
                                       obj_id=i, type='crop')
            self._save_drawed_contours(object_highlighted_images_relight,
                                       video_id=other_infos['video_id'],
                                       obj_id=i, type='relight')

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
        for i in range(num_objs):
            ret += f'Image-{i+1}: {IMAGE_TOKEN}\n'
        # ret += 'Here are several consecutive frames from a video. We have highlighted an object with yellow edges, meaning the object highlighted by the yellow edges in the video is the same. Please provide some discriminative descriptions about this object, which can help us easily distinguish it from other similar objects in the image. The discriminative descriptions should include but are not limited to its category, color, shape, state, purpose, properties, and relationship with surrounding objects.\n'
        # ret += 'There is an object highlighted with yellow edge. Please provide some discriminative descriptions about this object, which can help us easily distinguish it from other similar objects in the image. The discriminative descriptions should include but are not limited to its category, color, shape, state, purpose, properties, and relationship with surrounding objects.\n'
        # ret += 'Please provide a detailed description of the object highlighted by the yellow contour, including its color, shape, position in the image, state, purpose, properties, and its relationship with surrounding objects.'
        # ret += 'Here are several consecutive frames from a video. We have highlighted an object with a yellow edge. Please provide some discriminative descriptions about this object, which can help us easily distinguish it from other similar objects in the image. The discriminative descriptions should include its category, colour, shape, included parts, or which entity it is a part of. Please do not mention ‘yellow edge’ in your response, as it is an additional highlight rather than a characteristic of the object itself.\n'
        # ret += 'Please give the discriminative descriptions of the object.'
        # 'Here are some notes. If this region is part of an animal or human limb, such as a hand or leg (including related items like shoes, socks, or sleeves), please specify which limb it is, such as the right foot of a person or the right front leg of an animal. '
        ret += 'Here are several consecutive frames from a video. We have highlighted a region with a yellow edge. Please provide some discriminative descriptions about this region, which can help us easily distinguish it from other similar objects in the image. The discriminative descriptions should include but are not limited to its category, colour, shape, state.\n'
        ret += 'Please do not mention \'yellow edge\' in your response, as it is an additional highlight rather than a characteristic of the region.\n'
        ret += 'Please give the discriminative descriptions of the region.'
        return ret

    def get_question_relight(self, num_objs):
        ret = ''
        for i in range(num_objs):
            ret += f'Image-{i+1}: {IMAGE_TOKEN}\n'
        # ret += 'Here are several consecutive frames from a video. We have highlighted an object with yellow edges, meaning the object highlighted by the yellow edges in the video is the same. Please provide some discriminative descriptions about this object, which can help us easily distinguish it from other similar objects in the image. The discriminative descriptions should include but are not limited to its category, color, shape, position in the image, state, purpose, properties, and relationship with surrounding objects.\n'
        # ret += 'There is an object highlighted with yellow edge. Please provide some discriminative descriptions about this object, which can help us easily distinguish it from other similar objects in the image. The discriminative descriptions should include but are not limited to its category, color, shape, state, purpose, properties, position in the image and relationship with surrounding objects.\n'
        # ret += 'Please give the discriminative descriptions of the object.'

        ret += 'Here are several consecutive frames from a video. We have highlighted a region with a yellow edge. Please provide some discriminative descriptions about this region, which can help us easily distinguish it from other similar objects in the image. The discriminative descriptions should include its category, and relationship with surrounding objects.\n'
        ret += 'If there are significant features nearby that can help easily locate this object, please include them in your response. Please do not mention \'yellow edge\' in your response, as it is an additional highlight rather than a characteristic of the region.\n'
        ret += 'Please give the discriminative descriptions of the region.'
        return ret

    def highlight_object(self, object_frames, object_masks):
        ret = []
        for frame, mask in zip(object_frames, object_masks):
            image = add_edge_color(frame, mask)
            ret.append(image)
        return ret

    def _get_crop_range(self, masks, expand_ratio=2.0):
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

        _x_min = min((_x_min - _cx) * expand_ratio, -100) + _cx
        _x_max = max((_x_max - _cx) * expand_ratio, 100) + _cx
        _y_min = min((_y_min - _cy) * expand_ratio, -100) + _cy
        _y_max = max((_y_max - _cy) * expand_ratio, 100) + _cy

        _x_min = max(_x_min, 0)
        _y_min = max(_y_min, 0)
        _x_max = min(_x_max, w)
        _y_max = min(_y_max, h)
        return int(_x_min), int(_x_max), int(_y_min), int(_y_max)

    def highlight_object_crop(self, object_frames, object_masks):
        ret = []
        _x_min, _x_max, _y_min, _y_max = self._get_crop_range(object_masks)
        for frame, mask in zip(object_frames, object_masks):
            image = add_edge_color(frame[_y_min:_y_max, _x_min:_x_max], mask[_y_min:_y_max, _x_min:_x_max])
            ret.append(image)
        return ret

    def highlight_object_relight(self, object_frames, object_masks):
        ret = []
        for frame, mask in zip(object_frames, object_masks):
            frame[np.logical_not(mask)] = (frame[np.logical_not(mask)].astype(np.int64) / 2).astype(np.uint8)
            # frame = frame.astype(np.uint8)
            image = add_edge_color(frame, mask)
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

        return {'data_dicts': data_dicts, 'image_paths': None, 'type': 'sam2'}

    @master_only
    def evaluate(self, **kwargs):
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