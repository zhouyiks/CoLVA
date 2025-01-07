import os
import json

ALL_ID_LIST_FILES = [
    'sav_053_1028_id_list_0.txt', 'sav_053_1028_id_list_1.txt',
    'sav_053_1028_id_list_2.txt', 'sav_053_1028_id_list_3.txt',
    'sav_054_1028_id_list_0.txt', 'sav_054_1028_id_list_1.txt',
    'sav_054_1028_id_list_2.txt', 'sav_054_1028_id_list_3.txt',
]

checked_folder = './manual_check_visualization_1028/checked/'

def parse_file_name(name):
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

file_names = os.listdir(checked_folder)
checked_number = 0
no_checked_number = 0

meta_infos = []

for file_name in file_names:
    checked_path = os.path.join(checked_folder, file_name)
    folder_id, split_id = parse_file_name(file_name)
    no_checked_path = os.path.join(
        f"./manual_check_visualization_1028/sav_{folder_id}_step6/sav_{folder_id}_1028/",
        file_name
    )
    checked_object_ids = parse_txt(checked_path)
    no_checked_object_ids = parse_txt(no_checked_path)
    print(file_name, '---', len(checked_object_ids), '/', len(no_checked_object_ids))
    checked_number += len(checked_object_ids)
    no_checked_number += len(no_checked_object_ids) - len(checked_object_ids)

    for _object_id in no_checked_object_ids:
        if _object_id not in checked_object_ids:
            _info = {'id': _object_id, 'folder_id': folder_id, 'split_id': split_id}
            meta_infos.append(_info)

print(f"{checked_number} checked items, {no_checked_number} no checked items !!!")

split_idxs = [[0, 300], [300, 600], [600, 900], [900]]

for i, _split_idx in enumerate(split_idxs):
    if len(_split_idx) == 2:
        start, end = _split_idx
        current_meta_infos = meta_infos[start:end]
    else:
        assert len(_split_idx) == 1
        start = _split_idx[0]
        current_meta_infos = meta_infos[start:]
    save_folder = f"./manual_check_visualization_1028/mannual_short_videos/{i}/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for _info in current_meta_infos:
        _object_id = _info['id']
        print(_object_id)
        folder_id = _info['folder_id']
        split_id = _info['split_id']
        src_path = f"./manual_check_visualization_1028/sav_{folder_id}_step6/{split_id}/{_object_id}.mp4"
        os.system(f"cp {src_path} {save_folder}")
        txt_path = f"{save_folder}{_object_id}.txt"
        with open(txt_path, 'w') as f:
            f.write("")


