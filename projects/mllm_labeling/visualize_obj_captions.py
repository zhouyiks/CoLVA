import os
import json
import numpy as np
import copy
from PIL import Image
from lmdeploy.vl.constants import IMAGE_TOKEN
from pycocotools import mask as mask_utils
import torch.nn.functional as F
import torch

import cv2

from projects.mllm_labeling.image_blending_fn import contour_rendering

def get_masks_from_anns(annotations):
    captions = []
    masks = []
    for annotation in annotations:
        rle = annotation['object_anno']['segmentation']
        mask = mask_utils.decode(rle)
        masks.append(mask)
        captions.append(annotation['caption'])
    masks = np.stack(masks, axis=0)
    return masks, captions


image_folder = './1215_demos/mask_outs/out/'
save_dir = './1215_demos/overall_demos/'

image_files = []
for file_name in os.listdir(image_folder):
    if 'out' not in file_name and '.jpg' in file_name:
        image_files.append(file_name)

annotation_folder = '1215_demos/mllm_object_cap/'
anno_files = os.listdir(annotation_folder)
annotations = []
for anno_file in anno_files:
    with open(os.path.join(annotation_folder, anno_file), 'r') as f:
        annotations += json.load(f)

image2anno_dict = {}
for annotation in annotations:
    image_id = annotation['image_id']
    if image_id not in image2anno_dict.keys():
        image2anno_dict[image_id] = [annotation]
    else:
        image2anno_dict[image_id].append(annotation)

for i, image_name in enumerate(image2anno_dict.keys()):
    txt_strs = ''
    print('====================================================')
    image_path = os.path.join(image_folder, image_name)
    image = Image.open(image_path).convert('RGB')

    image_annotations = image2anno_dict[image_name]
    masks, captions = get_masks_from_anns(image_annotations)

    anno_ids = []

    for i_obj, caption in enumerate(captions):
        txt_strs += '\n\n' + f"{i_obj}: " + caption + '\n\n'
        anno_ids.append(i_obj)
        print(caption)
        print('+++++++++++++++++++++++++++')

    image_shape = image.size
    masks = torch.Tensor(masks).unsqueeze(0)
    masks = F.interpolate(
        masks,
        size=(image_shape[1], image_shape[0]),
        mode='nearest').squeeze(0)
    masks = masks.numpy().astype(np.uint8)

    image = np.array(image)
    contour_rendering(image, masks, mask_ids=anno_ids)

    image = Image.fromarray(image)
    image.save(os.path.join(save_dir, f"{i}.png"))
    with open(os.path.join(save_dir, f"{i}.txt"), 'w') as f:
        f.write(txt_strs)




