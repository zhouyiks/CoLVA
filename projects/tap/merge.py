import os
import json
import tqdm

jsons_folder = './work_dirs/tap_caption_results/'
sub_folders = os.listdir(jsons_folder)
sub_folders = [os.path.join(jsons_folder, item) for item in sub_folders]

save_path = './work_dirs/llava_tap_pesudo_captions.json'

datas = []
json_paths = []
for folder in sub_folders:
    files_names = os.listdir(folder)
    files_paths = [os.path.join(folder, item) for item in files_names]
    json_paths += files_paths

for path in tqdm.tqdm(json_paths):
    with open(path, 'r') as f:
        _dict = json.load(f)
    datas.append(_dict)

with open(save_path, 'w') as f:
    json.dump(datas, f)


