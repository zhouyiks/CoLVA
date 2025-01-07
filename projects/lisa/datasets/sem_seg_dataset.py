import json
import os
import random

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER

from pycocotools.coco import COCO

from projects.lisa.datasets.utils import SEG_QUESTIONS, ANSWER_LIST, DEFAULT_IMAGE_TOKEN

class ADE20kSemanticSegDataset(Dataset):

    def __init__(self,
                 data_path,
                 processor=None,
                 extra_image_processor=None,
                 image_folder=None,
                 num_classes_per_sample=3,
                 ):
        self.begin_str = f'{DEFAULT_IMAGE_TOKEN}\n'
        if processor:
            self.processor = BUILDER.build(processor)
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self.image_folder = image_folder
        self.num_classes_per_sample = num_classes_per_sample

        self.data = self._load_annotations(data_path, image_folder)
        self._max_refetch = 1000

    def decode_mask(self, label_path):
        label = np.array(Image.open(label_path))
        
        # ade20k
        label = np.where(label == 0, 255, label - 1)
        unique_labels = [lbl for lbl in np.unique(label) if lbl != 255]
        if not unique_labels:
            return None, None

        selected_labels = np.random.choice(unique_labels, min(
            len(unique_labels), self.num_classes_per_sample), replace=False)
        label = torch.from_numpy(label).long()
        masks = torch.stack(
            [label == class_id for class_id in selected_labels], dim=0)
        return masks, selected_labels

    def _load_annotations(self, data_path, image_folder=None):
        with open(data_path, 'r') as file:
            ade20k_classes = json.load(file)
        ade20k_image_dir = image_folder
        ade20k_images = [os.path.join(ade20k_image_dir, img) for img in os.listdir(
            ade20k_image_dir) if img.endswith('.jpg')]
        ade20k_labels = [img.replace(".jpg", ".png").replace(
            "images", "annotations") for img in ade20k_images]
        self.classes = np.array(ade20k_classes)

        ret = []
        for image, label in zip(ade20k_images, ade20k_labels):
            ret.append({"image": image, "label": label})
        return ret

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def __len__(self):
        return len(self.data)

    @property
    def modality_length(self):
        self.group_length = []
        for data_dict in self.data:
            self.group_length.append(100)
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length

    def _parse_annotations(self, ann_info):
        assert 'label' in ann_info
        masks, class_id = self.decode_mask(ann_info['label'])
        ann_info['masks'] = masks
        if class_id is None:
            return None
        conversation = []
        for i, c_id in enumerate(class_id):
            question = random.choice(SEG_QUESTIONS).format(
                class_name=self.classes[c_id].lower())
            answer = random.choice(ANSWER_LIST)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': answer})
        ann_info['conversations'] = conversation
        return ann_info
    
    def prepare_data(self, index):
        data_dict: dict = self.data[index]
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
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            out_data_dict['g_pixel_values'] = g_pixel_values
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))


class COCOStuffSemanticSegDataset(ADE20kSemanticSegDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 num_classes_per_sample=3,
                 processor=None,
                 extra_image_processor=None,):
         
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            num_classes_per_sample=num_classes_per_sample,
            processor=processor,
            extra_image_processor=extra_image_processor,
        )
        self.cocostuff_class2index = {c: i for i, c in enumerate(self.classes)}

    def _load_annotations(self, data_path, image_folder):
        # coco stuff
        with open(data_path, 'r') as file:
            cocostuff_classes = [line.strip().split(": ")[-1] for line in file.readlines()[1:]]
        files = os.listdir(image_folder)
        coco_stuff_images = [os.path.join('./data/coco/train2017/', img.replace('png', 'jpg')) for img in files]
        coco_stuff_labels = [os.path.join(image_folder, img) for img in files]

        self.classes = np.array(cocostuff_classes)

        ret = []
        for image, label in zip(coco_stuff_images, coco_stuff_labels):
            ret.append({"image": image, "label": label})
        return ret

    def decode_mask(self, label_path):
        label = np.array(Image.open(label_path))
        ignored_classes = [index for class_name,
                           index in self.cocostuff_class2index.items() if "-" in class_name]
        label = np.where(np.isin(label, ignored_classes), 255, label)

        unique_labels = [lbl for lbl in np.unique(label) if lbl != 255]
        if not unique_labels:
            print("No valid label !!!")
            return None, None

        selected_labels = np.random.choice(unique_labels, min(
            len(unique_labels), self.num_classes_per_sample), replace=False)

        label = torch.from_numpy(label).long()
        masks = torch.stack(
            [label == class_id for class_id in selected_labels], dim=0)
        return masks, selected_labels

class PascalPartSemanticSegDataset(ADE20kSemanticSegDataset):

    def _load_annotations(self, data_path, image_folder):
        self.coco_api = COCO(data_path)
        img_ids = self.coco_api.getImgIds()
        all_classes = self.coco_api.loadCats(self.coco_api.getCatIds())
        class_map_pascal_part = {}
        for cat in all_classes:
            cat_main, cat_part = cat["name"].strip().split(":")
            name = (cat_main, cat_part)
            class_map_pascal_part[cat["id"]] = name
        self.classes = class_map_pascal_part
        return img_ids

    def decode_mask(self, img_id):
        annotation_ids = self.coco_api.getAnnIds(imgIds=img_id)
        annotations = self.coco_api.loadAnns(annotation_ids)
        sampled_anns = np.random.choice(annotations, min(
            len(annotations), self.num_classes_per_sample), replace=False)
        masks = [self.coco_api.annToMask(ann) for ann in sampled_anns]
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks, [ann['category_id'] for ann in sampled_anns]

    def _parse_annotations(self, img_id):
        masks, class_id = self.decode_mask(img_id)
        img_info = self.coco_api.loadImgs(img_id)[0]
        ann_info = {'masks': masks, 'image': os.path.join(self.image_folder, img_info['file_name'])}
        if class_id is None:
            return None
        conversation = []
        for i, c_id in enumerate(class_id):
            sampled_cls = self.classes[c_id]
            if isinstance(sampled_cls, tuple):
                obj, part = sampled_cls
                name = f"{obj} {part}" if random.random() < 0.5 else f"the {part} of the {obj}"
            else:
                name = sampled_cls
            question = random.choice(SEG_QUESTIONS).format(class_name=name)
            answer = random.choice(ANSWER_LIST)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': answer})
        ann_info['conversations'] = conversation
        return ann_info

class PacoSemanticSegDataset(PascalPartSemanticSegDataset):
    def _load_annotations(self, data_path, image_folder):
        self.coco_api = COCO(data_path)
        all_classes = self.coco_api.loadCats(self.coco_api.getCatIds())
        class_map_paco = {}
        for cat in all_classes:
            cat_split = cat["name"].strip().split(":")
            if len(cat_split) == 1:
                name = cat_split[0].split("_(")[0]
            else:
                assert len(cat_split) == 2
                obj, part = cat_split
                obj = obj.split("_(")[0]
                part = part.split("_(")[0]
                name = (obj, part)
            class_map_paco[cat["id"]] = name
        self.classes = class_map_paco
        return self.coco_api.getImgIds()

class MapillarySemanticSegDataset(ADE20kSemanticSegDataset):
    def _load_annotations(self, data_path, image_folder=None):
        mapillary_classes = [cls["readable"].lower() for cls in json.load(open(data_path))["labels"]]
        mapillary_images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
        mapillary_labels = [img.replace(".jpg", ".png").replace("images", "v2.0/labels") for img in mapillary_images]
        self.classes = np.array(mapillary_classes)
        ret = []
        for image, label in zip(mapillary_images, mapillary_labels):
            ret.append({"image": image, "label": label})
        return ret

    def decode_mask(self, label_path):
        label = np.array(Image.open(label_path))
        unique_labels = [lbl for lbl in np.unique(label) if lbl != 255]
        if not unique_labels:
            return None, None
        selected_labels = np.random.choice(unique_labels, min(
            len(unique_labels), self.num_classes_per_sample), replace=False)
        label = torch.from_numpy(label).long()
        masks = torch.stack(
            [label == class_id for class_id in selected_labels], dim=0)
        return masks, selected_labels

class PartimagenetSemanticSegDataset(ADE20kSemanticSegDataset):
    pass

if __name__ == '__main__':
    from third_parts.segment_anything.utils.transforms import ResizeLongestSide
    from projects.lisa.processor.internvl_processor import InternVLProcessor
    processor = dict(
        type=InternVLProcessor,
        pretrained_model_name_or_path='OpenGVLab/InternVL2-4B'
    )
    extra_image_processor=dict(
        type=ResizeLongestSide,
        target_length=1024,
    )
    dataset = ADE20kSemanticSegDataset(
        data_path='projects/omg_llava/dataset/utils/ade20k_classes.json',
        image_folder='./data/ade20k/images/training/',
        extra_image_processor=extra_image_processor,
        processor=processor,
    )
    for i in range(len(dataset)):
        data = dataset[i]
