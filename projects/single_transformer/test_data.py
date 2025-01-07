import json

def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


json_data = "/mnt/bn/xiangtai-training-data/project/VLM/data/SOLO_SFT/all_data.jsonl"

image_data = "/mnt/bn/xiangtai-training-data/project/VLM/data/SOLO_SFT/images"

a = load_jsonl(json_data)



for index, i in enumerate(a):
    conversations = i['conversations']
    image_name = i['image']
    for msg in conversations:
        if "role" in msg.keys():
            print(i)
            print(index)
            exit()
        elif 'from' in msg.keys():
            continue
        elif 'value' in msg.keys():
            continue
        else:
            print(msg.keys)
        # if msg['from'] == 'human' or msg['from'] == 'user' or msg['role'] == 'user':
        #     continue

        # elif msg['from'] == 'gpt' or msg['from'] == 'model' or msg['role'] == 'assistant':
        #     continue

    # for item in conversations:
    #     if type(item) is str:
    #         print(conversations)


