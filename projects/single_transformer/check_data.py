import json
import os
def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


json_data = "/mnt/bn/xiangtai-training-data/project/VLM/data/SOLO_SFT/all_data.jsonl"

json_data_new =  "/mnt/bn/xiangtai-training-data/project/VLM/data/all_data_new.jsonl"

image_data_path = "/mnt/bn/xiangtai-training-data/project/VLM/data/SOLO_SFT/images"

new_json_data = []
a = load_jsonl(json_data)


for index, i in enumerate(a):

    conversations = i['conversations']

    if 'image' in i.keys() and not os.path.exists(os.path.join(image_data_path, i['image'])):
        # print("find", i)
        # exit()
        print("Missing: ",i)
        continue

    new_json_data.append(i)

with open(json_data_new, 'w') as f:
    json.dump(new_json_data, f)

    # print(os.path.join(image_data_path, i['image']))
    # if not os.path.exists(os.path.join(image_data_path, i['image'])):
    #     print(i['images'])
    # for msg in conversations:
    #     if "role" in msg.keys():
    #         print(i)
    #         print(index)
    #         exit()
    #     elif 'from' in msg.keys():
    #         continue
    #     elif 'value' in msg.keys():
    #         continue
    #     else:
    #         print(msg.keys)
        # if msg['from'] == 'human' or msg['from'] == 'user' or msg['role'] == 'user':
        #     continue

        # elif msg['from'] == 'gpt' or msg['from'] == 'model' or msg['role'] == 'assistant':
        #     continue

    # for item in conversations:
    #     if type(item) is str:
    #         print(conversations)


