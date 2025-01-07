# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import json

import torch
from transformers import AutoTokenizer, AutoModel
import cv2
from pycocotools import mask as maskUtils
from PIL import Image


images_folder = '/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/qilu_folder/images/'
save_demo_dir = '/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/qilu_folder/draw_images/'
save_mask_dir = '/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/qilu_folder/masks/'

demo_items = []
image_files = os.listdir(images_folder)
gcg_question = '<image>\nCan you provide a brief description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.'
for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    demo_items.append(
        {'image_path': image_path, 'question': gcg_question}
    )

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

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

        frames.append(frame)

        frame_id += 1

    cap.release()
    return frames

def get_frames_from_video(video_path, n_frames=5):
    frames = get_video_frames(video_path)
    stride = len(frames) / (n_frames + 1e-4)
    ret = []
    for i in range(n_frames):
        idx = int(i * stride)
        frame = frames[idx]
        frame = frame[:, :, ::-1]
        frame_image = Image.fromarray(frame).convert('RGB')
        ret.append(frame_image)
    return ret


def main():
    # model_path = 'work_dirs/sa2va_8b/'
    # model_path = 'work_dirs/sa2va_4b/'
    # model = AutoModel.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     use_flash_attn=True,
    #     trust_remote_code=True,
    # ).eval().cuda()
    #
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,
    # )

    json_dir = '/mnt/bn/xiangtai-training-data/project/xiangtai-windows/glamm/groundingLMM/'

    for i, demo_item in enumerate(demo_items):
        image_path = demo_item['image_path']
        text_prompts = demo_item['question']
        ori_image = Image.open(image_path).convert('RGB')

        json_path = os.path.join(json_dir, f'0{i+1}.json')
        with open(json_path, 'r') as f:
            results = json.load(f)

        masks = results['pred_masks']
        pred_masks = [torch.Tensor(maskUtils.decode(ann)) for ann in masks]

        show_mask_pred(ori_image, pred_masks,
                           save_dir_demo=os.path.join(save_demo_dir, f'output_{i}.png'),
                           save_dir_mask=os.path.join(save_mask_dir, f"output_{i}.png")
                           )

def show_mask_pred(image, masks, save_dir_demo=None, save_dir_mask=None):
    from PIL import Image
    import numpy as np

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255)]

    masks = torch.stack(masks, dim=0).cpu().numpy()
    _mask_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
        _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
        _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]

    mask_image = np.array(_mask_image).astype(np.uint8)
    mask_image = Image.fromarray(mask_image)
    mask_image.save(save_dir_mask)

    image = np.array(image)
    image = image * 0.5 + _mask_image * 0.5
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_dir_demo)

    return

if __name__ == '__main__':
    main()
