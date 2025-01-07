import json
import random


def RegionCaptionDataset_load_fn(data_path, repeat_time):
    with open(data_path, 'r') as f:
        json_file = json.load(f)
    
    ret, hf_ret = [], []
    for item in json_file:
        item.update({'image': item['file_name']})
        if len(item["description"]) != len(item["annotation"]):
            print("The number of description is not equal to seg !!!")
        else:
            ret.append(item)
    
    if repeat_time < 1:
        ret = random.sample(ret, int(len(ret) * repeat_time))
    elif repeat_time > 1:
        int_repeat_time = int(repeat_time)
        remaining_repeat_time = repeat_time - int_repeat_time
        if remaining_repeat_time > 0:
            remaining_ret = random.sample(
                ret, int(len(ret) * remaining_repeat_time))
            ret = ret * int_repeat_time
            ret.extend(remaining_ret)
        else:
            ret = ret * int_repeat_time
    
    for item in ret:
        image = item["file_name"]
        description = item["description"]
        hf_required_info = {"image": image, "description": description}
        hf_ret.append(hf_required_info)

    return ret, hf_ret


def RegionConversationDataset_load_fn(data_path, repeat_time):
    with open(data_path, 'r') as f:
        json_file = json.load(f)
    
    ret, hf_ret = [], []
    for dataset_info in json_file:
        if 'annotation' not in dataset_info or len(dataset_info['annotation']) == 0:
            print("The annotation is not valid, filter out!!!")
            continue
        dataset_info.update({'image': dataset_info['file_name']})
        ret.append(dataset_info)
    
    if repeat_time < 1:
        ret = random.sample(ret, int(len(ret) * repeat_time))
    elif repeat_time > 1:
        int_repeat_time = int(repeat_time)
        remaining_repeat_time = repeat_time - int_repeat_time
        if remaining_repeat_time > 0:
            remaining_ret = random.sample(
                ret, int(len(ret) * remaining_repeat_time))
            ret = ret * int_repeat_time
            ret.extend(remaining_ret)
        else:
            ret = ret * int_repeat_time

    for dataset_info in ret:
        conversations = dataset_info["conversations"]
        image = dataset_info["file_name"]
        num_regions = len(dataset_info['annotation'])
        required_info = {'image': image, 'conversations': conversations,
                         'num_regions': num_regions}
        hf_ret.append(required_info)
    
    return ret, hf_ret