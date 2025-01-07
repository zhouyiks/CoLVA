import copy
import json
import os

import numpy as np
import cv2
import pycocotools.mask as maskUtils
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder_name", type=str, )
args = parser.parse_args()


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
        # print('mask_shape: ', mask.shape)
        masks.append(mask)
    # print(len(masks))
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

ori_video_demo_folder = './manual_check_visualization'
split_num = 4

if not os.path.exists("./manual_check_visualization_1028"):
    os.mkdir("./manual_check_visualization_1028")
if not os.path.exists("./manual_check_visualization_1028/{}".format(args.folder_name)):
    os.mkdir('./manual_check_visualization_1028/{}'.format(args.folder_name))
for i in range(split_num):
    if not os.path.exists("./manual_check_visualization_1028/{}/{}".format(args.folder_name, i)):
        os.mkdir("./manual_check_visualization_1028/{}/{}".format(args.folder_name, i))

video_folder = '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full'
video_save_path = "./manual_check_visualization_1028/{}/<SPLIT>/demo.mp4".format(args.folder_name)

caption_json_path = "./whole_pesudo_cap_v3/{}/".format(args.folder_name)
cap_json_files = os.listdir(caption_json_path)
cap_json_paths = [os.path.join(caption_json_path, item) for item in cap_json_files]
caption_jsons = []
for cap_json_path in cap_json_paths:
    with open(cap_json_path, 'r') as f:
        caption_jsons.extend(json.load(f))

n_items = len(caption_jsons)

video_obj_cap_dict = {}
for cap_item in caption_jsons:
    video_id = cap_item['video_id']
    obj_id = cap_item['obj_id']
    if video_id not in video_obj_cap_dict.keys():
        video_obj_cap_dict[video_id] = {}
    video_obj_cap_dict[video_id].update({obj_id: cap_item})

video_ids = list(video_obj_cap_dict.keys())
# for video_id in video_obj_cap_dict.keys():
n_cur = 0
cur_split = 0
threthold = n_items // split_num + 1

for video_id in video_ids:
    sub_folder = video_id[:7]
    for i in video_obj_cap_dict[video_id].keys():
        print(i, '---', video_obj_cap_dict[video_id].keys())
        translation = video_obj_cap_dict[video_id][i]['translation']
        final_caption = video_obj_cap_dict[video_id][i]['final_caption']
        # print('\n\n', translation, '\n\n')
        with open(video_save_path.replace('demo.mp4', f'{video_id}_obj{i}.txt')\
                          .replace("<SPLIT>", "{}".format(cur_split)), 'w', encoding='utf-8') as file:
            file.write(translation + "\n\n" + "-------English:--------" + '\n' + final_caption)
        video_save_path_ = video_save_path.replace('demo.mp4', f'{video_id}_obj{i}.mp4').replace("<SPLIT>", "{}".format(cur_split))
        ori_video_path_ = video_save_path.replace('manual_check_visualization_1028', "manual_check_visualization").replace('demo.mp4', f'{video_id}_obj{i}.mp4').replace("<SPLIT>/", "").replace("sav_054_step6", "sav_054_step5").replace("sav_053_step6", "sav_053_step5")
        print(f"mv {ori_video_path_} {video_save_path_}")
        os.system(f"mv {ori_video_path_} {video_save_path_}")
        n_cur += 1
        if n_cur >= threthold:
            n_cur = 0
            cur_split += 1

