import logging
import os
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

from xtuner.registry import BUILDER
from ..utils import expand2square, expand2square_mask
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import copy
# from xtuner.dataset.utils import encode_fn
from .utils import encode_fn
import pickle
import json
import random
from xtuner.utils import DEFAULT_IMAGE_TOKEN
import pycocotools.mask as maskUtils
import cv2

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

class VideoReVOSDataset(Dataset):
    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file,
                 image_processor,
                 tokenizer=None,
                 select_number=5,
                 sampled_frames=10,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 lazy=True,
                 repeats=1,):
        assert lazy is True
        self.tokenizer = tokenizer
        self.select_number = select_number
        self.sampled_frames = sampled_frames
        assert offline_processed_text_folder or (expression_file and tokenizer)
        self.lazy = lazy

        self.max_length = max_length

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if offline_processed_text_folder and expression_file:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            vid2metaid, metas, mask_dict = self.json_file_preprocess(expression_file, mask_file)
            self.vid2metaid = vid2metaid
            self.videos = list(self.vid2metaid.keys())
            self.mask_dict = mask_dict
            self.json_datas = metas
            json_datas = metas
            json_data = DatasetDict({'train': HFDataset.from_list(json_datas)})
            if self.lazy:
                self.text_data = build_origin_dataset(json_data, 'train')
            else:
                raise NotImplementedError

        self.image_folder = image_folder
        size = image_processor.crop_size
        if isinstance(size, int):
            self.image_h, self.image_w = size, size
        else:
            self.image_w, self.image_h = size

        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square
        self.down_ratio = 1
        self.repeats = repeats
        self.tokenizer = tokenizer

        # for visualization debug
        self.save_folder = './work_dirs/video_debug/'
        self.cur_number = 0

    def __len__(self):
        return len(self.vid2metaid) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.vid2metaid:
            cur_len = 1000
            length_list.append(cur_len)
        return length_list

    def real_len(self):
        return len(self.vid2metaid)

    def json_file_preprocess(self, expression_file, mask_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['videos']

        metas = []
        anno_count = 0  # serve as anno_id
        vid2metaid = {}
        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]

            vid_frames = sorted(vid_express_data['frames'])
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['video'] = vid_name
                meta['exp'] = exp_dict['exp']  # str
                meta['mask_anno_id'] = exp_dict['anno_id']

                if 'obj_id' in exp_dict.keys():
                    meta['obj_id'] = exp_dict['obj_id']
                else:
                    meta['obj_id'] = [0, ]  # Ref-Youtube-VOS only has one object per expression
                meta['anno_id'] = [str(anno_count), ]
                anno_count += 1
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id

                meta['length'] = vid_len
                metas.append(meta)
                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name] = []
                vid2metaid[vid_name].append(len(metas) - 1)

        # process mask annotation files
        with open(mask_file, 'rb') as f:
            mask_dict = json.load(f)

        return vid2metaid, metas, mask_dict

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def decode_mask(self, video_masks, image_size):
        ret_masks = []
        for object_masks in video_masks:
            # None object
            if len(object_masks) == 0:
                _object_masks = np.zeros(
                    (self.sampled_frames, image_size[0], image_size[1]), dtype=np.uint8)
            else:
                _object_masks = []
                for i_frame in range(len(object_masks[0])):
                    _mask = np.zeros(image_size, dtype=np.uint8)
                    for i_anno in range(len(object_masks)):
                        if object_masks[i_anno][i_frame] is None:
                            continue
                        m = maskUtils.decode(object_masks[i_anno][i_frame])
                        if m.ndim == 3:
                            m = m.sum(axis=2).astype(np.uint8)
                        else:
                            m = m.astype(np.uint8)
                        _mask = _mask | m
                    _object_masks.append(_mask)
                _object_masks = np.stack(_object_masks, axis=0)
            if self.pad_image_to_square:
                _object_masks = expand2square_mask(_object_masks)
            ret_masks.append(_object_masks)

        ret_masks = np.stack(ret_masks, axis=0)  # (n_obj, n_frames, h, w)

        ret_masks = torch.from_numpy(ret_masks)
        ret_masks = F.interpolate(ret_masks, size=(self.image_h // self.down_ratio,
                                  self.image_w // self.down_ratio), mode='nearest')
        ret_masks = ret_masks.flatten(0, 1)
        return ret_masks

    def dataset_map_fn(self, data_dict, select_k=5):
        images = []

        len_frames = len(data_dict[0]['frames'])
        for objet_info in data_dict:
            assert len_frames == len(objet_info['frames'])

        # prepare images, random select k frames
        if len_frames >= select_k:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=False)
        else:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=True)
        selected_frame_indexes.sort()

        for selected_frame_index in selected_frame_indexes:
            frame_id = data_dict[0]['frames'][selected_frame_index]
            images.append(os.path.join(data_dict[0]['video'], frame_id + '.jpg'))

        # prepare text
        expressions = [object_info['exp'] for object_info in data_dict]
        text_dict = self.prepare_text(select_k, expressions)


        # prepare masks
        video_masks = []
        for object_info in data_dict:
            anno_ids = object_info['mask_anno_id']
            # print('anno_ids: ', anno_ids)
            obj_masks = []
            for anno_id in anno_ids:
                anno_id = str(anno_id)
                frames_masks = self.mask_dict[anno_id]
                frames_masks_ = []
                for frame_idx in selected_frame_indexes:
                    frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
                obj_masks.append(frames_masks_)
            video_masks.append(obj_masks)

        ret = {'images': images, 'video_masks': video_masks, 'conversation': text_dict['conversation']}

        return ret

    def prepare_text(self, n_frames, expressions):
        questions = []
        answers = []
        for i, exp in enumerate(expressions):
            # the exp is a question
            if '?' in exp:
                questions.append(exp)
            else:
                exp = exp.replace('.', '').strip()
                question_template = random.choice(SEG_QUESTIONS)
                questions.append(question_template.format(class_name=exp.lower()))

            answers.append(random.choice(ANSWER_LIST))
        qa_list = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                frame_tokens = DEFAULT_IMAGE_TOKEN + ' '
                # frame_tokens = '=' + ' '
                frame_tokens = frame_tokens * n_frames
                frame_tokens = frame_tokens.strip() + '\n'
                qa_list.append(
                    {'from': 'human', 'value': frame_tokens + question}
                )
            else:
                qa_list.append(
                    {'from': 'human', 'value': question}
                )
            qa_list.append(
                {'from': 'gpt', 'value': answer}
            )

        input = ''
        conversation = []
        for msg in qa_list:
            if msg['from'] == 'human':
                input += msg['value']
            elif msg['from'] == 'gpt':
                conversation.append({'input': input, 'output': msg['value']})
                input = ''
            else:
                raise NotImplementedError

        return {'conversation': conversation}

    def __getitem__(self, index):
        index = index % self.real_len()
        selected_video_objects = self.vid2metaid[self.videos[index]]
        video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

        if len(video_objects_infos) > self.select_number:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]

        data_dict = self.dataset_map_fn(video_objects_infos, select_k=5)
        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length, with_image_token=True)
        data_dict.update(result)

        assert 'images' in data_dict.keys()
        pixel_values = []
        if data_dict.get('images', None) is not None:
            frames_files = data_dict['images']
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            for frame_path in frames_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size
                if self.pad_image_to_square:
                    frame_image = expand2square(
                        frame_image,
                        tuple(
                            int(x * 255) for x in self.image_processor.image_mean))
                frame_image = self.image_processor.preprocess(
                    frame_image, return_tensors='pt')['pixel_values'][0]
                pixel_values.append(frame_image)

            data_dict['pixel_values'] = pixel_values

            # process and get masks
            masks = self.decode_mask(data_dict['video_masks'], image_size=(ori_height, ori_width))
            data_dict['masks'] = masks
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
            data_dict['masks'] = None
        # # for debug
        # self.visualization_debug(data_dict)
        # if self.cur_number < 10:
        #     return self[random.randint(0, len(self))]
        return data_dict

    def visualization_debug(self, data_dict):
        save_folder = os.path.join(self.save_folder, 'sample_{}'.format(self.cur_number))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self.cur_number += 1

        # images

        show_images = []

        pixel_values = data_dict['pixel_values']
        save_folder_image = os.path.join(save_folder, 'image')
        if not os.path.exists(save_folder_image):
            os.mkdir(save_folder_image)
        for i_image, image_pixel_value in enumerate(pixel_values):
            # print(image_pixel_value.shape)
            image_pixel_value[0] = image_pixel_value[0] * 0.2686
            image_pixel_value[1] = image_pixel_value[1] * 0.2613
            image_pixel_value[2] = image_pixel_value[2] * 0.2757
            image_pixel_value[0] = image_pixel_value[0] + 0.4814
            image_pixel_value[1] = image_pixel_value[1] + 0.4578
            image_pixel_value[2] = image_pixel_value[2] + 0.4082
            image_pixel_value = image_pixel_value * 255
            image_pixel_value = image_pixel_value.permute(1, 2, 0)
            image_pixel_value = image_pixel_value.to(torch.uint8).numpy()
            # print(os.path.join(save_folder_image, '{}.jpg'.format(i_image)))
            # print(image_pixel_value.shape)
            show_images.append(image_pixel_value)
            cv2.imwrite(os.path.join(save_folder_image, '{}.jpg'.format(i_image)), image_pixel_value)

        # text
        input_text = self.tokenizer.decode(data_dict['input_ids'], skip_special_tokens=False)
        with open(os.path.join(save_folder, 'text.json'), 'w') as f:
            json.dump([input_text], f)

        # masks
        save_folder_mask = os.path.join(save_folder, 'mask')
        if not os.path.exists(save_folder_mask):
            os.mkdir(save_folder_mask)
        n_frames = len(pixel_values)
        masks = data_dict['masks']
        _, h, w = masks.shape
        masks = masks.reshape(-1, n_frames, h, w)
        for i_obj, obj_masks in enumerate(masks):
            save_folder_mask_obj_folder = os.path.join(save_folder_mask, 'obj_{}'.format(i_obj))
            if not os.path.exists(save_folder_mask_obj_folder):
                os.mkdir(save_folder_mask_obj_folder)
            for i_frame, f_mask in enumerate(obj_masks):
                f_mask = f_mask.numpy()
                f_mask = f_mask * 255
                f_mask = np.stack([f_mask * 1, f_mask * 0, f_mask * 0], axis=2)
                f_mask = show_images[i_frame] * 0.3 + 0.7 * f_mask
                f_mask = f_mask.astype(np.uint8)
                cv2.imwrite(os.path.join(save_folder_mask_obj_folder, '{}.png'.format(i_frame)), f_mask)
        return