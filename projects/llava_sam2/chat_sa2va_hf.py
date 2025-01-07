# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
from transformers import AutoTokenizer, AutoModel
import cv2
from PIL import Image


sam_prefix = '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_000/sav_train/sav_000/'
coco_prefix = './data/glamm_data/images/coco2014/train2014/'
sam_p2 = 'data/sa_eval/'

demo_items = [
    {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease describe the image.'},
    {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the person.'},
    {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the snowboard.'},
    {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the snow.'},
    {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the trees.'},
    {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nCould you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.'},
]

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
    model_path = 'work_dirs/sa2va_4b/'
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    for i, demo_item in enumerate(demo_items):
        image_path = demo_item['image_path']
        text_prompts = demo_item['question']
        ori_image = Image.open(image_path).convert('RGB')
        input_dict = {
            'image': ori_image,
            'text': text_prompts,
            'past_text': '',
            'mask_prompts': None,
            'tokenizer': tokenizer,
        }

        return_dict = model.predict_forward(**input_dict)
        print(i, ': ', return_dict['prediction'])
        if 'prediction_masks' in return_dict.keys() and return_dict['prediction_masks'] and len(return_dict['prediction_masks']) != 0:
            show_mask_pred(ori_image, return_dict['prediction_masks'], save_dir=f'./demos/output_{i}.png')

def show_mask_pred(image, masks, save_dir='./output.png'):
    from PIL import Image
    import numpy as np

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255)]

    masks = torch.stack(masks, dim=0).cpu().numpy()[:, 0]
    _mask_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
        _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
        _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]


    image = np.array(image)
    image = image * 0.5 + _mask_image * 0.5
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_dir)

    return

if __name__ == '__main__':
    main()
