import copy
import random
import torch

from mmengine import print_log
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX

from projects.lisa.datasets.utils import SEG_QUESTIONS, ANSWER_LIST, DEFAULT_IMAGE_TOKEN

from mmdet.datasets.refcoco import RefCocoDataset

class ReferSegmDataset(RefCocoDataset):

    def __init__(self,
                 data_root,
                 ann_file=None,
                 split_file=None,
                 processor=None,
                 extra_image_processor=None,
                 data_prefix=dict(img_path='train2014/'),
                 num_classes_per_sample=3,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=None,
            ann_file=ann_file,
            split_file=split_file,
            **kwargs,
        )
        self.begin_str = f'{DEFAULT_IMAGE_TOKEN}\n'
        if processor:
            self.processor = BUILDER.build(processor)
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.image_folder = data_root
        self.num_classes_per_sample = num_classes_per_sample
        self._max_refetch = 1000

    @property
    def modality_length(self):
        import pickle
        length_list = []
        for idx in range(len(self)):
            length_list.append(100)
        return length_list

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        masks, phrases = [], []
        instances, text = ann_info['instances'], ann_info['text']
        index = np.random.choice(range(len(instances)), min(
            len(instances), self.num_classes_per_sample))
        for idx in index:
            inst = instances[idx]
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            phrases.append(phrase)
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in inst["mask"]:
                rles = mask_utils.frPyObjects([seg], height, width)
                m = mask_utils.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()
            masks.append(binary_mask)

        conversation = []
        for i, phrase in enumerate(phrases):
            question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            conversation.append(
                {'from': 'gpt', 'value': random.choice(ANSWER_LIST)})
        masks = torch.stack([torch.from_numpy(mask) for mask in masks], dim=0)

        ann_info.update({
            'masks': masks,
            'conversations': conversation,
            'image': image_path
        })
        return ann_info

    def prepare_data(self, index):
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = self.processor(data_dict)
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('image', None) and hasattr(self, 'extra_image_processor'):
            image_file = data_dict['image']
            try:
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                return None
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(
                g_image).permute(2, 0, 1).contiguous()
            out_data_dict['g_pixel_values'] = g_pixel_values
        return out_data_dict

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data


if __name__ == '__main__':
    from transformers import CLIPImageProcessor, AutoTokenizer
    from third_parts.segment_anything.utils.transforms import ResizeLongestSide
    llm_name_or_path = 'lmsys/vicuna-7b-v1.5'

    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path)
    image_processor = dict(
        type=CLIPImageProcessor.from_pretrained,
        pretrained_model_name_or_path='openai/clip-vit-large-patch14-336')
    extra_image_processor = dict(
        type=ResizeLongestSide,
        target_length=1024,
    )
    from projects.lisa.processor.internvl_processor import InternVLProcessor
    processor = dict(
        type=InternVLProcessor,
        pretrained_model_name_or_path='OpenGVLab/InternVL2-4B'
    )
    extra_image_processor = dict(
        type=ResizeLongestSide,
        target_length=1024,
    )
    dataset = ReferSegmDataset(
        processor=processor,
        extra_image_processor=extra_image_processor,
        data_root='data/coco/',
        data_prefix=dict(img_path='train2014/'),
        ann_file='refcoco+/instances.json',
        split_file='refcoco+/refs(unc).p',
    )
    for i in range(1000):
        dataset[i]
