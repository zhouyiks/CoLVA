import json
import os

def split_caps(caption):
    caption = caption.strip()
    caption = caption.split('\n')
    caption = [cap.replace("\"", "").strip() for cap in caption]
    return caption

save_path = './whole_pesudo_cap/sam_v_final_shortcap_v2.json'
ori_json_path = './whole_pesudo_cap/sam_v_final_v2.json'

caption_json_path = './whole_pesudo_cap/short_cap/'
cap_json_files = os.listdir(caption_json_path)
cap_json_paths = [os.path.join(caption_json_path, item) for item in cap_json_files]
caption_jsons = []
for cap_json_path in cap_json_paths:
    with open(cap_json_path, 'r') as f:
        caption_jsons.extend(json.load(f))

with open(ori_json_path, 'r') as f:
    json_data = json.load(f)

for item in caption_jsons:
    video_id = item['video_id']
    obj_id = item['obj_id']
    json_data[video_id]['objects'][obj_id].update({'short_caps': split_caps(item['short_cap'])})

print(json_data[video_id])

print("{} videos !!!".format(len(json_data)))

with open(save_path, 'w') as f:
    json.dump(json_data, f)



