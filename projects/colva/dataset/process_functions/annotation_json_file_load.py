import json
import random
import tqdm
from pycocotools.coco import COCO
import os
import numpy as np

def ViPLLaVADataset_load_fn(data_path, repeat_time, **kwargs):
    with open(data_path, 'r') as f:
        json_data = json.load(f)
    
    ret = []
    for source in json_data:
        if 'image' not in source:
            continue
        dataset_type = source['id'].split('-')[0]
        if dataset_type not in ['refcocog', 'vcr', 'vg_rel', 'flickr30k', 'v7w', 'pointQA_twice']:
            continue
        ret.append(source)

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
    
    return ret, ret


def LLaVAInstructDataset_load_fn(data_path, repeat_time, **kwargs):
    try:
        ret = json.load(open(data_path))
    except:
        ret = []
        with open(data_path, 'r', encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                _data = json.loads(line)
                ret.append(_data)
    
    for idx in range(len(ret)):
        if "id" in ret[idx].keys() and isinstance(ret[idx]['id'], int):
            ret[idx]['id'] = str(ret[idx]['id'])

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
    
    return None, ret

    

def RegionCaptionDataset_load_fn(data_path, repeat_time, **kwargs):
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


def RegionConversationDataset_load_fn(data_path, repeat_time, **kwargs):
    with open(data_path, 'r') as f:
        json_file = json.load(f)

    if 'part_level' in data_path or 'short_form' in data_path:
        limit_str = ' Answer the question using a single word or phrase.'
    else:
        limit_str = ''
    
    ret, hf_ret = [], []
    for dataset_info in json_file:
        if 'annotation' not in dataset_info or len(dataset_info['annotation']) == 0:
            print("The annotation is not valid, filter out!!!")
            continue
        dataset_info.update({'image': dataset_info['file_name'], 'limit_str': limit_str})
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
        image = dataset_info['file_name']
        num_regions = len(dataset_info['annotation'])
        required_info = {'image': image, 'conversations': conversations,
                         'num_regions': num_regions}
        hf_ret.append(required_info)
    
    return ret, hf_ret

def RegionShortCapVGDataset_load_fn(data_path, repeat_time, **kwargs):
    coco = COCO(data_path)
    img_ids = coco.getImgIds()
    
    ret, hf_ret = [], []
    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            continue

        data_info = dict(
            image=img_info['file_name'],
            description=[],
            annotation=[]
        )
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            data_info['annotation'].append(
                {'bbox': ann['bbox'], 'segmentation': ann['segmentation']}
            )
            data_info['description'].append(ann['caption'])
        ret.append(data_info)
    
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
        image = item["image"]
        description = item["description"]
        hf_required_info = {"image": image, "description": description}
        hf_ret.append(hf_required_info)
    
    return ret, hf_ret


def CoCoRefClassificationDataset_load_fn(data_path, repeat_time, **kwargs):
    coco = COCO(data_path)
    img_ids = coco.getImgIds()

    ret, hf_ret = [], []
    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        data_info=dict(
            image=img_info['file_name'],
            categories=[],
            annotation=[],
        )

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            continue

        for ann in ann_info:
            data_info['annotation'].append(
                {'bbox': ann['bbox'], 'segmentation': ann['segmentation']}
            )
            cat = coco.loadCats(ann['category_id'])
            data_info['categories'].append(
                cat[0]['name']
            )
        ret.append(data_info)
    
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
        categories = dataset_info["categories"]
        image = dataset_info["image"]
        required_info = {'image': image, 'categories': categories}
        hf_ret.append(required_info)
    
    return ret, hf_ret


def RefCOCOShortCaptionDataset_load_fn(data_path, repeat_time, **kwargs):
    coco = COCO(data_path)
    img_ids = coco.getImgIds()

    ret, hf_ret = [], []
    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        data_info=dict(
            image=img_info['file_name'],
            description=[img_info['caption']],
            annotation=[],
        )

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            continue

        for ann in ann_info:
            data_info['annotation'].append(
                {'bbox': ann['bbox'], 'segmentation': ann['segmentation']}
            )
        ret.append(data_info)
    
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
        image = item["image"]
        description = item["description"]
        hf_required_info = {"image": image, "description": description}
        hf_ret.append(hf_required_info)

    return ret, hf_ret


def PartClassificationDataset_load_fn(data_path, repeat_time, **kwargs):
    coco = COCO(data_path)
    img_ids = coco.getImgIds()

    ret, hf_ret = [], []
    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]

        data_info = dict(
            image=img_info['file_name'],
            categories=[],
            annotation=[],
        )

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            continue

        for ann in ann_info:
            cat = coco.loadCats(ann['category_id'])
            data_info['categories'].append(cat[0]['name'])
            data_info['annotation'].append(
                {'bbox': ann['bbox'], 'segmentation': ann['segmentation']}
            )
        ret.append(data_info)
        
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
        image = item["image"]
        categories = item["categories"]
        hf_required_info = {"image": image, "categories": categories}
        hf_ret.append(hf_required_info)
    
    return ret, hf_ret


def MDPVPointConversationDataset_load_fn(data_path, repeat_time, **kwargs):
    with open(data_path, 'r') as f:
        json_data = json.load(f)

    ret = []
    for source in json_data:
        data_info = dict(
            image=source['image'].split('/')[-1],
            conversations=source['conversations'],
            annotation=[]
        )

        for point in source['points']:
            data_info['annotation'].append(
                {'point': [point]}
            )
        ret.append(data_info)
    
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
    
    hf_ret = []
    for item in ret:
        image = item['image']
        conversations = item['conversations']
        num_regions = len(item['annotation'])
        hf_required_info = {"image": image, "num_regions": num_regions, "conversations": conversations}
        hf_ret.append(hf_required_info)
    
    return ret, hf_ret


def MDPVBoxConversationDataset_load_fn(data_path, repeat_time, **kwargs):
    image_folder = kwargs['image_folder']
    json_data = []
    for source_file in data_path:
        with open(source_file, 'r') as f:
            json_data.extend(json.load(f))

    if repeat_time < 1:
        json_data = random.sample(json_data, int(len(json_data) * repeat_time))
   
    ret = []
    for source in json_data:
        data_info = dict(
            image=source['image'].split('/')[-1],
            conversations=source['conversations'],
            annotation=[]
        )
        if not os.path.exists(image_folder+data_info['image']):
            # print("skip...", image_folder+data_info['image'])
            continue

        for bbox in source['bbox']:
            x0, y0, w, h = bbox
            data_info['annotation'].append(
                {'bbox': [x0, y0, x0+w, y0+h]}
            )
        ret.append(data_info)
    
    # if repeat_time < 1:
    #     ret = random.sample(ret, int(len(ret) * repeat_time))
    if repeat_time > 1:
        int_repeat_time = int(repeat_time)
        remaining_repeat_time = repeat_time - int_repeat_time
        if remaining_repeat_time > 0:
            remaining_ret = random.sample(
                ret, int(len(ret) * remaining_repeat_time))
            ret = ret * int_repeat_time
            ret.extend(remaining_ret)
        else:
            ret = ret * int_repeat_time
    
    hf_ret = []
    for item in ret:
        image = item['image']
        conversations = item['conversations']
        num_regions = len(item['annotation'])
        hf_required_info = {"image": image, "num_regions": num_regions, "conversations": conversations}
        hf_ret.append(hf_required_info)
    
    return ret, hf_ret

def MDPVBoxOCRDataset_load_fn(data_path, repeat_time, **kwargs):
    image_folder = kwargs['image_folder']
    json_data = []
    for source_file in data_path:
        with open(source_file, 'r') as f:
            json_data.extend(json.load(f))
   
    ret = []
    for source in json_data:
        data_info = dict(
            image=source['image'],
            conversations=source['conversations'],
            annotation=[]
        )
        if not os.path.exists(image_folder+data_info['image']):
            # print("skip...", source['image'].split('/')[-1])
            continue

        for bbox in source['bbox']:
            x0, y0, w, h = bbox
            data_info['annotation'].append(
                {'bbox': [x0, y0, x0+w, y0+h]}
            )
        ret.append(data_info)
    
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
    
    hf_ret = []
    for item in ret:
        image = item['image']
        conversations = item['conversations']
        num_regions = len(item['annotation'])
        hf_required_info = {"image": image, "num_regions": num_regions, "conversations": conversations}
        hf_ret.append(hf_required_info)
    
    return ret, hf_ret


def MatchDataset_load_fn(data_path, repeat_time, **kwargs):
    with open(data_path, 'r') as f:
        json_file = json.load(f)
    
    ret, hf_ret = [], []
    for item in json_file:
        if not item['file_names'][0].startswith('./data/'):
            item['file_names'] = ['./data/'+file_name[2:] for file_name in item['file_names']]
        if 'AVA' in item['file_names'][0]:
            continue
        if 'HACS' in item['file_names'][0]:
            continue
        item.update({'image': item['file_names']})
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
        images = item["file_names"]
        if "description" in item:
            description = item["description"]
            hf_required_info = {"image": images, "description": description}
        else:
            hf_required_info = {"image": images, }
        hf_ret.append(hf_required_info)

    return ret, hf_ret