import copy
import os
import json
import random
from PIL import Image, ImageDraw
import numpy as np


def find_nearest(anno_list, idx):
    while anno_list[idx] is None:
        idx -= 1
    return idx


def parse_anno(path):
    with open(path, 'r') as f:
        datas = f.readlines()

    ret = []
    for data in datas:
        bbox = data.replace("\n", "").split()
        bbox = [float(_item.strip()) for _item in bbox]
        ret.append(bbox)
    return ret

# create and set the save path
if not os.path.exists('./achieved'):
    os.mkdir('./achieved')
if not os.path.exists('./achieved/images/'):
    os.mkdir('./achieved/images')

save_image_path = './achieved/images'
save_json_path = './achieved/anno.json'

final_json_data = {
    "task": "video object tracking, under water video",
    "data_source": "UTB180",
    "type": "comprehension",
    "modality": {
        "in": ["image", "text"],
        "out": ["text"]
    },
    "version": 1.0,
}

src_frames_folder = 'UTB180/UTB180/'
src_first_frame_folder = 'thumbnails/thumbnails/UTB180_1st_frames/'

_PER_NUMBER=50
_SAMPLE_FRAMES=30

with open("./under_water_attr.txt", 'r') as f:
    datas = f.readlines()
split_data_list = {'blue_water': [], "green_water": [], "yellow_water": [], "white_water": []}
split_attr_list = {'blue_water': [], "green_water": [], "yellow_water": [], "white_water": []}
for idx in range(1, 181):
    _str_idx = str(idx + 10000)[1:]
    idx_attr = datas[idx-1].lower()
    if 'blue' in idx_attr:
        split_attr_list['blue_water'].append(_str_idx)
    elif 'clear' in idx_attr or idx_attr == "":
        split_attr_list["white_water"].append(_str_idx)
    elif 'green' in idx_attr:
        split_attr_list["green_water"].append(_str_idx)
    elif 'brown' in idx_attr or 'brown' in idx_attr:
        split_attr_list["yellow_water"].append(_str_idx)

_id = 10000
for instance_name in os.listdir(src_frames_folder):
    if '.json' in instance_name or '.xlsx' in instance_name:
        continue
    _split = None
    _sub_nums = None
    for _key_str in split_attr_list['blue_water']:
        if _key_str in instance_name:
            _split = 'blue_water'
            _sub_nums = _PER_NUMBER // len(split_attr_list['blue_water']) + 1
            break
    if _split is None:
        for _key_str in split_attr_list['green_water']:
            if _key_str in instance_name:
                _split = 'green_water'
                _sub_nums = _PER_NUMBER // len(split_attr_list['green_water']) + 1
                break
    if _split is None:
        for _key_str in split_attr_list['white_water']:
            print(_key_str, '  ', instance_name)
            if _key_str in instance_name:
                _split = 'white_water'
                _sub_nums = _PER_NUMBER // len(split_attr_list['white_water']) + 1
                break
    if _split is None:
        for _key_str in split_attr_list['yellow_water']:
            if _key_str in instance_name:
                _split = 'yellow_water'
                _sub_nums = _PER_NUMBER // len(split_attr_list['yellow_water']) + 1
                break
    if _split is None:
        continue

    if len(split_data_list[_split]) >= _PER_NUMBER:
        continue

    anno_file_path = os.path.join(src_frames_folder, instance_name, "groundtruth_rect.txt")
    anno_bboxes = parse_anno(anno_file_path)
    if anno_bboxes is None:
        continue

    cur_video_folder = os.path.join(src_frames_folder, instance_name, 'imgs')
    frame_names = os.listdir(cur_video_folder)
    len_frames = len(frame_names)

    if len_frames > len(anno_bboxes):
        print(f"Wrong anno and seq, {len_frames} frames, {len(anno_bboxes)} bboxes.")
        continue

    print(instance_name)

    frame_steps = len_frames // _sub_nums
    for _sub_idx in range(_sub_nums):
        frame_start_idx = _sub_idx * frame_steps
        frame_end_idx = min((_sub_idx + 1) * frame_steps, len_frames)

        selected_frames_idxs = list(range(frame_start_idx, frame_end_idx))
        random.shuffle(selected_frames_idxs)
        selected_frames_idxs = selected_frames_idxs[:_SAMPLE_FRAMES]
        selected_frames_idxs.sort()

        if anno_bboxes[selected_frames_idxs[0]] is None:
            selected_frames_idxs.append(find_nearest(anno_bboxes, selected_frames_idxs[0]))
            selected_frames_idxs.sort()

        # copy the images
        str_id = str(_id)[1:]
        _id += 1
        drt_folder = os.path.join('./achieved/images/', str_id)
        if not os.path.exists(drt_folder):
            os.mkdir(drt_folder)
        for select_frame_idx in selected_frames_idxs:
            frame_name = frame_names[select_frame_idx]
            os.system(f"cp {os.path.join(cur_video_folder, frame_name)} {drt_folder}")

        # parse anno and generate json
        selected_anns = []
        print(len(anno_bboxes), '--', selected_frames_idxs)
        for select_frame_idx in selected_frames_idxs:
            selected_anns.append(anno_bboxes[select_frame_idx])

        _data = {"id": "vt_vot{}".format(str_id)}
        _data["input"] = {"video_folder": drt_folder.replace('/achieved', ''), "prompt": "Please tracking the object within red box in image 1."}
        _data["output"] = {"bboxes": selected_anns}

        # draw first frame
        # print(frame_names)
        # print(selected_frames_idxs)
        first_frame = Image.open(os.path.join(drt_folder, frame_names[selected_frames_idxs[0]]))
        draw = ImageDraw.Draw(first_frame)
        draw.rectangle([selected_anns[0][0], selected_anns[0][1],
                        selected_anns[0][2] + selected_anns[0][0],
                        selected_anns[0][3] + selected_anns[0][1]], outline='red', width=2)
        first_frame.save(os.path.join(drt_folder, frame_names[selected_frames_idxs[0]].replace('.jpg', '_draw.jpg')))
        split_data_list[_split].append(_data)

for _split in split_data_list.keys():
    with open(f'./achieved/{_split}.json', 'w') as f:
        print(len(split_data_list[_split]))
        _data = split_data_list[_split]
        _ret_data = copy.deepcopy(final_json_data)
        _ret_data["task"] += f", {_split}"
        _ret_data["data"] = _data
        json.dump(_ret_data, f)