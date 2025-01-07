import json

path = '/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/ref_SAV/meta_expressions_valid.json'

with open(path, 'r') as f:
    data_dict = json.load(f)
data_dict = data_dict["videos"]
len_videos = len(data_dict.keys())

n_exps = 0
n_exps_long = 0
n_exps_short = 0
for video_id in data_dict.keys():
    n_exps += len(data_dict[video_id]["expressions"].keys())
    for object_id in data_dict[video_id]["expressions"].keys():
        if data_dict[video_id]["expressions"][object_id]['type_id'] == 0:
            n_exps_long += 1
        else:
            n_exps_short += 1

print(f"{len_videos} videos, {n_exps} expressions, {n_exps_long} long and {n_exps_short} short")
