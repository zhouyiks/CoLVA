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
        polygon = data.replace('\n', "").replace("m", "").split(",")
        polygon = [int(_item) for _item in polygon]
        print(len(polygon))
        print(polygon)
        if len(polygon) % 2 != 0:
            polygon = polygon[:-1]
        polygon = np.array(polygon).reshape(-1, 2)
        max_coord = np.max(polygon, axis=0)
        min_coord = np.min(polygon, axis=0)
        bbox = [int(min_coord[0]), int(min_coord[1]), int(max_coord[0] - min_coord[0]), int(max_coord[1] - min_coord[1])]
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
    "task": "video object tracking, vot video",
    "data_source": "VOT2022",
    "type": "comprehension",
    "modality": {
        "in": ["image", "text"],
        "out": ["text"]
    },
    "version": 1.0,
}

src_frames_folder = 'data/'
src_anno_folder = 'annotations/'

_PER_NUMBER=50
_SAMPLE_FRAMES=30

split_data_list = {'person': [], "animal": [], "others": []}
split_key_str = {
    'person': ['agility', 'gymnastics', "iceskater", "shaking", "girl", "graduate", "singer", "soldier", "bolt"],
    'animal': ['tiger', 'ant', 'butterfly', 'nature', 'lamb', 'rabbit', 'zebrafish', 'crabs', 'fish', 'bird', 'dinosaur', 'flamingo', 'monkey'],
}

for instance_name in os.listdir(src_frames_folder):
    if '.zip' in instance_name and not os.path.exists(os.path.join(src_frames_folder, instance_name.replace(".zip", ""))):
        folder_name = instance_name.replace(".zip", "")
        os.mkdir(os.path.join(src_frames_folder, instance_name.replace(".zip", "")))
        os.system(f"unzip -d {os.path.join(src_frames_folder, folder_name)} {os.path.join(src_frames_folder, instance_name)}")
for instance_name in os.listdir(src_anno_folder):
    if '.zip' in instance_name and not os.path.exists(os.path.join(src_anno_folder, instance_name.replace(".zip", ""))):
    # if '.zip' in instance_name:
        folder_name = instance_name.replace(".zip", "")
        os.system(f"unzip -d {os.path.join(src_anno_folder, folder_name)} {os.path.join(src_anno_folder, instance_name)}")

_id = 10000
for instance_name in os.listdir(src_frames_folder):
    if '.zip' in instance_name:
        continue
    _split = None
    _sub_nums = None
    for _key_str in split_key_str['person']:
        if _key_str in instance_name:
            _split = 'person'
            _sub_nums = 6
            break
    if _split is None:
        for _key_str in split_key_str['animal']:
            if _key_str in instance_name:
                _split = 'animal'
                _sub_nums = 5
                break
    if _split is None:
        _split = "others"
        _sub_nums = 3

    if len(split_data_list[_split]) >= _PER_NUMBER:
        continue

    anno_file_path = os.path.join(src_anno_folder, instance_name, "groundtruth.txt")
    anno_bboxes = parse_anno(anno_file_path)
    if anno_bboxes is None:
        continue

    cur_video_folder = os.path.join(src_frames_folder, instance_name)
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