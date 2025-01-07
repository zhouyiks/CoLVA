import copy
import json
import os

import numpy as np
import cv2
import pycocotools.mask as maskUtils


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


def images_to_video(frames, video_name, fps=6):
    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

    # cv2.destroyAllWindows()
    video.release()
    return

def decode_masklet(masklet):
    masks = []
    for _rle in masklet:
        mask = maskUtils.decode(_rle)
        print('mask_shape: ', mask.shape)
        masks.append(mask)
    print(len(masks))
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



demo_video_anno = '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_000/sav_train/sav_000/sav_000001_manual.json'
video_root = '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_000/sav_train/sav_000'
video_save_path = '/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/work_dirs/sam_v_demos/demo.mp4'

caption_json_path = 'work_dirs/pesudo_label_sam2_qwen72b_brief_caption/sav_000/'
cap_json_files = os.listdir(caption_json_path)
cap_json_paths = [os.path.join(caption_json_path, item) for item in cap_json_files]
caption_jsons = []
for cap_json_path in cap_json_paths:
    with open(cap_json_path, 'r') as f:
        caption_jsons.extend(json.load(f))

video_obj_cap_dict = {}
for cap_item in caption_jsons:
    video_id = cap_item['video_id']
    obj_id = cap_item['obj_id']
    if video_id not in video_obj_cap_dict.keys():
        video_obj_cap_dict[video_id] = {}
    video_obj_cap_dict[video_id].update({obj_id: cap_item})

with open(demo_video_anno, 'r') as f:
    data = json.load(f)

print(data.keys())
for key in data.keys():
    if key == 'masklet':
        continue
    print(key, ': ')
    print(data[key])

video_path = os.path.join(video_root, '{}.mp4'.format(data['video_id']))
frames = get_video_frames(video_path)
masklents = decode_masklet(data['masklet'])
frames = frames[::4]
assert len(frames) == len(masklents)
show_videos = add_mask2images(frames, masklents)

for i, show_video in enumerate(show_videos):
    text = video_obj_cap_dict[data['video_id']][i]
    print('\n\n', text, '\n\n')
    video_save_path_ = video_save_path.replace('demo.mp4', 'demo_{}.mp4'.format(i))
    images_to_video(show_video, video_save_path_)

