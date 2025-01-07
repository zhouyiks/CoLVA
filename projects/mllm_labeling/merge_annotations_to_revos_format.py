import json
import os
import cv2
from PIL import Image

# parse revos format
# # mask_dict = '/mnt/bn/xiangtai-training-data-video/dataset/video_vlm/video_res/revos/mask_dict.json'
# exp_dict = '/mnt/bn/xiangtai-training-data-video/dataset/video_vlm/video_res/revos/meta_expressions_valid_.json'
# #
# # with open(mask_dict, 'r') as f:
# #     mask_dict = json.load(f)
# #
# # print(mask_dict.keys())
# # keys = list(mask_dict.keys())
# # print(mask_dict[keys[0]])
#
# with open(exp_dict, 'r') as f:
#     exp_dict = json.load(f)
#
# print(exp_dict['videos']['UVO/all/-fbscFfkh4M']['expressions'])
# print(exp_dict['videos']['UVO/all/-fbscFfkh4M']['vid_id'])
# print(exp_dict['videos']['UVO/all/-fbscFfkh4M']['height'])
# print(exp_dict['videos']['UVO/all/-fbscFfkh4M']['width'])
# print(exp_dict['videos']['UVO/all/-fbscFfkh4M']['frames'])
# #{'exp': 'the person who is wearing a white shirt and blue jeans.', 'obj_id': [0], 'anno_id': [3003019], 'type_id': 0}


#--------------------------------------------------------------------------------------

mini = False
checked_folder = './manual_check_visualization_1028/checked/'
short_anno_folder = './manual_check_visualization_1028/short_annotation/'
save_dir = './ref_SAV/'

auto_annotation_folders = [
    './whole_pesudo_cap_v3/sav_054_step6/',
    './whole_pesudo_cap_v3/sav_053_step6/'
]
json_files = []
for auto_annotation_folder in auto_annotation_folders:
    file_names = os.listdir(auto_annotation_folder)
    file_names = [os.path.join(auto_annotation_folder, name) for name in file_names]
    json_files.extend(file_names)
auto_json_datas = []
for file_path in json_files:
    with open(file_path, 'r') as f:
        _data = json.load(f)
        auto_json_datas.extend(_data)

auto_json_dict = {}
for _item in auto_json_datas:
    video_id = _item['video_id']
    obj_id = _item['obj_id']
    if video_id not in auto_json_dict.keys():
        auto_json_dict[video_id] = {}
    auto_json_dict[video_id][obj_id] = _item


def parse_file_name(name):
    print(name)
    name = name[:-4]
    name = name.split('_')
    folder_id = name[1]
    split_id = name[-1]
    return folder_id, split_id

def parse_txt(path):
    with open(path, 'r') as f:
        data = f.read()
    data = data.split('\n')
    data_ = []
    for line in data:
        line = line.strip()
        if line == '':
            pass
        else:
            data_.append(line)
    return data_

def parse_txt_short_anno(path):
    with open(path, 'r') as f:
        data = f.read()
    data = data.split('\n')
    short_cap = ''
    num = 0
    for _item in data:
        if 'The' in _item or 'Object' in _item or 'object' in _item or 'a' in _item:
            short_cap = _item
            num += 1
    assert num == 1, data
    short_cap = short_cap.strip()
    if short_cap[-1] != '.':
        short_cap = short_cap + '.'
    return short_cap

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

file_names = os.listdir(checked_folder)
checked_number = 0

meta_infos = []

for file_name in file_names:
    checked_path = os.path.join(checked_folder, file_name)
    folder_id, split_id = parse_file_name(file_name)
    checked_object_ids = parse_txt(checked_path)

    for _object_id in checked_object_ids:
        _info = {'id': _object_id, 'folder_id': folder_id, 'split_id': split_id}
        meta_infos.append(_info)

if mini:
    meta_infos = meta_infos[:50]


short_file_names = os.listdir(short_anno_folder)
short_meta_infos = []
for file_name in short_file_names:
    short_cap = parse_txt_short_anno(os.path.join(short_anno_folder, file_name))
    _object_id = file_name.replace('.txt', '')

    _info = {'id': _object_id, 'folder_id': '054', 'short_cap': short_cap}
    meta_infos.append(_info)

if mini:
    meta_infos = meta_infos[:100]

ret_mask_dict = {}
ret_exp_dict = {}

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(os.path.join(save_dir, 'videos')):
    os.mkdir(os.path.join(save_dir, 'videos'))

for anno_id, _info in enumerate(meta_infos):
    print(anno_id)
    _object_id = _info['id']
    folder_id = _info['folder_id']
    # split_id = _info['split_id']
    video_id, object_id = _object_id.split('_obj')
    object_id = int(object_id.strip())

    # prepare exp

    if 'short_cap' in _info.keys():
        # print("Short manual anno.")
        # print(_info['short_cap'])
        object_exp = _info['short_cap']
        _exp_dict = {
            'exp': object_exp,
            'obj_id': [object_id],
            'anno_id': [10000 + anno_id],
            'type_id': 1,
        }
    else:
        object_exp = auto_json_dict[video_id][object_id]['final_caption']
        _exp_dict = {
            'exp': object_exp,
            'obj_id': [object_id],
            'anno_id': [10000 + anno_id],
            'type_id': 0,
        }

    # prepare mask
    mask_anno_path = \
        f"/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_{folder_id}/sav_train/sav_{folder_id}/{video_id}_manual.json"
    with open(mask_anno_path, 'r') as f:
        mask_anno_data = json.load(f)
    masklents = mask_anno_data['masklet']
    object_masklent = [_all_objects[object_id] for _all_objects in masklents]

    # save and append
    ret_mask_dict[str(10000+anno_id)] = object_masklent

    if video_id not in ret_exp_dict.keys():

        if not os.path.exists(os.path.join(save_dir, f"videos/{video_id}")):
            os.mkdir(os.path.join(save_dir, f"videos/{video_id}"))

        # prepare images
        video_path = \
            f"/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_{folder_id}/sav_train/sav_{folder_id}/{video_id}.mp4"

        video_frames = get_video_frames(video_path)
        video_valid = False
        if os.path.exists(os.path.join(save_dir, f"videos/{video_id}/")):
            video_valid = True
        video_frames = video_frames[::4]
        video_frames_ = []
        video_frames_names = []
        frames_ids = []
        for i_frame, frame in enumerate(video_frames):
            frame = frame[:, :, ::-1]
            frame_image = Image.fromarray(frame).convert('RGB')
            frames_ids.append(str(100000 + i_frame * 4))
            video_frames_names.append(f"videos/{video_id}/{100000 + i_frame * 4}.jpg")
            video_frames_.append(frame_image)

        width, height = video_frames_[0].size
        ret_exp_dict[video_id] = {
            'expressions': {},
            'vid_id': video_id,
            'height': height,
            'width': width,
            'frames': frames_ids,
        }

        for _video_frame_name, _frame_image in zip(video_frames_names, video_frames_):
            _save_pth = os.path.join(save_dir, _video_frame_name)
            _frame_image.save(_save_pth)

    ret_exp_dict[video_id]['expressions'][str(object_id)] = _exp_dict

with open(os.path.join(save_dir, 'meta_expressions_valid.json'), 'w') as f:
    json.dump({'videos': ret_exp_dict}, f)
with open(os.path.join(save_dir, 'mask_dict.json'), 'w') as f:
    json.dump(ret_mask_dict, f)











