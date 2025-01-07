import json
import os

save_path = './whole_pesudo_cap/sam_v_final_v2.json'

caption_json_path = './whole_pesudo_cap/formatted/'
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
    sub_folder = video_id[:7]
    video_path = f'{sub_folder}/sav_train/{sub_folder}/{video_id}.mp4'
    anno_path = f'{sub_folder}/sav_train/{sub_folder}/{video_id}_manual.json'
    cap_item.update({'video_path': video_path, 'anno_path': anno_path})
    if video_id not in video_obj_cap_dict.keys():
        video_obj_cap_dict[video_id] = {
            'video_id': video_id, 'video_path': video_path, 'anno_path': anno_path,
            'objects': {},
        }
    video_obj_cap_dict[video_id]['objects'].update({obj_id: cap_item})

print("{} videos !!!".format(len(video_obj_cap_dict)))

with open(save_path, 'w') as f:
    json.dump(video_obj_cap_dict, f)

