import json
import os

save_path = './whole_pesudo_cap_v3/sam_v_final_v3.json'

caption_json_folders = [
    "./whole_pesudo_cap_v3/sav_000_007_step5/",
    "./whole_pesudo_cap_v3/sav_008_014_step5/",
    "./whole_pesudo_cap_v3/sav_015_020_step5/",
    "./whole_pesudo_cap_v3/sav_021_028_step5/",
    "./whole_pesudo_cap_v3/sav_029_035_step5/",
    "./whole_pesudo_cap_v3/sav_036_050_step5/",
    "./whole_pesudo_cap_v3/sav_052_step5/",
]

caption_json_files = []
for caption_json_folder in caption_json_folders:
    _files = os.listdir(caption_json_folder)
    _files = [os.path.join(caption_json_folder, _file) for _file in _files]
    caption_json_files.extend(_files)

caption_jsons = []
for cap_json_path in caption_json_files:
    with open(cap_json_path, 'r') as f:
        caption_jsons.extend(json.load(f))


video_obj_cap_dict = {}
n_exps = 0
for cap_item in caption_jsons:
    n_exps += 1
    video_id = cap_item['video_id']
    obj_id = cap_item['obj_id']
    sub_folder = video_id[:7]
    video_path = f'{sub_folder}/sav_train/{sub_folder}/{video_id}.mp4'
    anno_path = f'{sub_folder}/sav_train/{sub_folder}/{video_id}_manual.json'
    cap_item.update({'video_path': video_path, 'anno_path': anno_path})
    cap_item['formated'] = cap_item['final_caption']
    if video_id not in video_obj_cap_dict.keys():
        print(video_id)
        video_obj_cap_dict[video_id] = {
            'video_id': video_id, 'video_path': video_path, 'anno_path': anno_path,
            'objects': {},
        }
    video_obj_cap_dict[video_id]['objects'].update({obj_id: cap_item})

print("{} videos, {} exps !!!".format(len(video_obj_cap_dict), n_exps))

with open(save_path, 'w') as f:
    json.dump(video_obj_cap_dict, f)

