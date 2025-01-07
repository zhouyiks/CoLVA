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
video_save_path = 'v1_pipeline_results/visualize/demo.mp4'

caption_json_path = 'v1_pipeline_results/internvl_caps/sav_000/'
cap_json_files = os.listdir(caption_json_path)
cap_json_paths = [os.path.join(caption_json_path, item) for item in cap_json_files]
caption_jsons = []
for cap_json_path in cap_json_paths:
    with open(cap_json_path, 'r') as f:
        caption_jsons.extend(json.load(f))

caption_json_path_detail = 'v1_pipeline_results/qwen2_summ/sav_000/'
cap_json_files_detail = os.listdir(caption_json_path_detail)
cap_json_paths_detail = [os.path.join(caption_json_path_detail, item) for item in cap_json_files_detail]
caption_jsons_detail = []
for cap_json_path_detail in cap_json_paths_detail:
    with open(cap_json_path_detail, 'r') as f:
        caption_jsons_detail.extend(json.load(f))

video_obj_cap_dict = {}
for cap_item in caption_jsons:
    video_id = cap_item['video_id']
    obj_id = cap_item['obj_id']
    if video_id not in video_obj_cap_dict.keys():
        video_obj_cap_dict[video_id] = {}
    video_obj_cap_dict[video_id].update({obj_id: cap_item})

video_obj_cap_dict_detail = {}
for cap_item in caption_jsons_detail:
    video_id = cap_item['video_id']
    obj_id = cap_item['obj_id']
    if video_id not in video_obj_cap_dict_detail.keys():
        video_obj_cap_dict_detail[video_id] = {}
    video_obj_cap_dict_detail[video_id].update({obj_id: cap_item})

for video_id in video_obj_cap_dict.keys():
    video_anno_file = f'/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_000/sav_train/sav_000/{video_id}_manual.json'
    video_path = os.path.join(video_root, f'{video_id}.mp4')
    with open(video_anno_file, 'r') as f:
        data = json.load(f)
    frames = get_video_frames(video_path)
    masklents = decode_masklet(data['masklet'])
    frames = frames[::4]
    assert len(frames) == len(masklents)
    show_videos = add_mask2images(frames, masklents)

    for i, show_video in enumerate(show_videos):
        text = video_obj_cap_dict[video_id][i]['caption']
        text_detail = video_obj_cap_dict_detail[video_id][i]['caption']
        print('\n\n', text, '\n\n')
        with open(video_save_path.replace('demo.mp4', f'{video_id}_obj{i}.txt'), 'w', encoding='utf-8') as file:
            file.write(text + '\n\n' + text_detail)
        video_save_path_ = video_save_path.replace('demo.mp4', f'{video_id}_obj{i}.mp4')
        images_to_video(show_video, video_save_path_)

