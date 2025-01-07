import os
import json
import random
from PIL import Image, ImageDraw
import copy


def find_nearest(anno_list, idx):
    while anno_list[idx] is None:
        idx -= 1
    return idx


# def parse_anno(path):
#     if not os.path.exists(path):
#         path = path.replace(".txt", "_1.txt")
#     with open(path, 'r') as f:
#         datas = f.readlines()
#
#     ret = []
#     for data in datas:
#         bbox = data.replace('\n', "").split(",")
#         # print(bbox)
#         if bbox[0] == "NaN":
#             # out of the image
#             ret.append(None)
#         else:
#             bbox = [int(_item) for _item in bbox]
#             ret.append(bbox)
#     return ret

def parse_anno(path):
    if not os.path.exists(path):
        return None
        paths = []
        for i in range(1, 6):
            path_ = path.replace(".txt", f"_{i}.txt")
            if os.path.exists(path_):
                paths.append(path_)
    else:
        paths = [path]

    ret = []
    for path in paths:
        with open(path, 'r') as f:
            datas = f.readlines()

        ret = []
        for data in datas:
            bbox = data.replace('\n', "").split(",")
            # print(bbox)
            if bbox[0] == "NaN":
                # out of the image
                ret.append(None)
            else:
                bbox = [int(_item) for _item in bbox]
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
    "task": "video object tracking, uav video",
    "data_source": "UAV123",
    "type": "comprehension",
    "modality": {
        "in": ["image", "text"],
        "out": ["text"]
    },
    "version": 1.0,
}

src_frames_folder = 'data_seq/UAV123_10fps/'
src_anno_folder = 'anno/UAV123_10fps/'

_PER_NUMBER=50
_SAMPLE_FRAMES=30

split_data_list = {'person': [], "car": [], "others": []}

_id = 10000
for instance_name in os.listdir(src_frames_folder):
    if 'person' in instance_name:
        _split = 'person'
        _sub_nums = 3
    elif 'car' in instance_name:
        _split = 'car'
        _sub_nums = 3
    else:
        _split = 'others'
        _sub_nums = 2

    if len(split_data_list[_split]) >= _PER_NUMBER:
        continue

    anno_file_path = os.path.join(src_anno_folder, f"{instance_name}.txt")
    anno_bboxes = parse_anno(anno_file_path)
    if anno_bboxes is None:
        continue


    cur_video_folder = os.path.join(src_frames_folder, instance_name)
    frame_names = os.listdir(cur_video_folder)
    len_frames = len(frame_names)

    if len_frames > len(anno_bboxes):
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

        _data = {"id": "vt_uav{}".format(str_id)}
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
        _data = split_data_list[_split]
        _ret_data = copy.deepcopy(final_json_data)
        _ret_data["task"] += f", {_split}"
        _ret_data["data"] = _data
        json.dump(_ret_data, f)











