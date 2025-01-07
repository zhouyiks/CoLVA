import random
from pyexpat.errors import messages

import pycocotools.mask as maskUtils
import numpy as np

from projects.lisa.datasets.utils import DEFAULT_IMAGE_TOKEN


def region_llava_map_fn(example):
    k_regions = 6
    object_datas = example['objects']

    if len(object_datas) > k_regions:
        selected_indexes = np.random.choice(list(range(0, len(object_datas))), size=k_regions, replace=False)
    else:
        selected_indexes = np.random.choice(list(range(0, len(object_datas))), size=k_regions, replace=True)
    # selected_indexes = selected_indexes.astype(np.int64).tolist()
    object_datas = [object_datas[_idx] for _idx in selected_indexes]
    region_masks  = []
    region_captions = []
    for object_data in object_datas:
        i_cap = random.randint(0, len(object_data['captions'])-1)
        region_captions.append(object_data['captions'][i_cap])
        object_rle = object_data['segm']
        _mask = maskUtils.decode(object_rle).astype(np.uint8)
        region_masks.append(_mask)
    region_masks = np.stack(region_masks, axis=0)

    messages = []
    for _cap in region_captions:
        messages.append({'from': 'human', 'value': 'Please describe {}.'.format(DEFAULT_IMAGE_TOKEN)})
        messages.append({'from': 'gpt', 'value': _cap + '.'})

    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation, 'region_masks': region_masks}