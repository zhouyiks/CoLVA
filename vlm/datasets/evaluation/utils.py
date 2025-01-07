import copy

import numpy as np
from collections import defaultdict
import json
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from xtuner.tools.utils import is_cn_string
from xtuner.dataset.utils import expand2square
from PIL import Image
import os

def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText


def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'


def MME_rating(data):
    stats = defaultdict(dict)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        category = item['category']
        image_path = item['image_path']
        score = item['score']
        if image_path not in stats[category]:
            stats[category][image_path] = []
        stats[category][image_path].append(score)

    def acc(key, mode='normal'):
        res = stats[key]
        values = []
        for val in res.values():
            if mode == 'normal':
                values.extend(val)
            elif mode == 'plus':
                values.append(val[0] * val[1])
        return np.mean(values) * 100

    scores = {}
    for k in stats:
        scores[k] = acc(k) + acc(k, 'plus')

    super_cates = dict(
        perception=[
            'OCR', 'artwork', 'celebrity', 'color', 'count', 'existence',
            'landmark', 'position', 'posters', 'scene'
        ],
        reasoning=['code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation']
    )

    ret = {}
    for sc, cate_list in super_cates.items():
        base = 0
        for c in cate_list:
            base += scores[c]
        ret[sc] = base
    ret.update(scores)
    return ret


def Hallusion_rating(data):
    def calc_fAcc(data):
        res = defaultdict(list)
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            res[f"{line['l2-category']}_{line['set_id']}_{line['figure_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100

    def calc_qAcc(data):
        res = defaultdict(list)
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            res[f"{line['l2-category']}_{line['set_id']}_{line['question_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100

    def calc_aAcc(data):
        return np.mean(data['score']) * 100

    data['set_id'] = [x.split('_')[3] for x in data['index']]
    data['figure_id'] = [x.split('_')[4] for x in data['index']]
    data['question_id'] = [x.split('_')[5] for x in data['index']]

    res = dict(split=[], aAcc=[], fAcc=[], qAcc=[])
    res['split'].append('Overall')
    res['aAcc'].append(calc_aAcc(data))
    res['fAcc'].append(calc_fAcc(data))
    res['qAcc'].append(calc_qAcc(data))

    if 'category' in data:
        cates = list(set(data['category']))
        for c in cates:
            sub = data[data['category'] == c]
            res['split'].append(c)
            res['aAcc'].append(calc_aAcc(sub))
            res['fAcc'].append(calc_fAcc(sub))
            res['qAcc'].append(calc_qAcc(sub))

    if 'l2-category' in data:
        cates = list(set(data['l2-category']))
        for c in cates:
            sub = data[data['l2-category'] == c]
            res['split'].append(c)
            res['aAcc'].append(calc_aAcc(sub))
            res['fAcc'].append(calc_fAcc(sub))
            res['qAcc'].append(calc_qAcc(sub))
    return res


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data

def custom_data_process(self, data, return_ori_image=False):
    metainfo = self.metainfo
    data_dict = {'img_id': data['img_id']}
    # 1 prepare text, the text only contain the <image> and text prompts
    # so, please add your template in the model.predict_forward()
    if metainfo['name'] == 'multiple_choice':
        # MultipleChoiceDataset
        data_dict['index'] = data['index']
        if data['context'] is not None:
            text = data['context'] + '\n' + data['question'] + '\n' + data['options']
        else:
            text = data['question'] + '\n' + data['options']
        text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if is_cn_string(text):
            text = text + '请直接回答选项字母。'
        else:
            text = text + ("Answer with the option's letter from the " 'given choices directly.')
    elif metainfo['name'] in ['chartqa', 'gvqa']:
        # TODO prompt are different of vlmevalkit
        text = data['question'] + '\nAnswer the question using a single word or phrase.'
        text = DEFAULT_IMAGE_TOKEN + '\n' + text
    elif metainfo['name'] == 'tallyqa':
        text = data['question']
        text = text + "\nAnswer the question using a single number."
        text = DEFAULT_IMAGE_TOKEN + '\n' + text
    elif metainfo['name'] in ['hallusion', 'pope']:
        # TODO prompt are different of vlmevalkit
        text = data['question'] + '\nPlease answer the question with yes or no.'
        text = DEFAULT_IMAGE_TOKEN + '\n' + text
    else:
        text = data['question']
        if metainfo['name'] == 'mme':
            text = data['question'].replace('Please answer yes or no.',
                                            'Please answer the question only a single word yes or no.')
        text = DEFAULT_IMAGE_TOKEN + '\n' + text

    # 3 process image
    # if metainfo['name'] in ['mme', 'textvqa', 'gqa', 'tallyqa']:
    if metainfo['name'] in ['textvqa', 'gqa', 'tallyqa']:
        # MMEDataset or TextVQADataset
        image_folder = self.image_folder
        image = Image.open(os.path.join(image_folder, data['image_path'])).convert('RGB')
    else:
        image = self.get_image(data['img']).convert('RGB')
    ori_image = copy.deepcopy(image)
    ori_width, ori_height = image.size

    if self.pad_image_to_square:
        image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))

    image = self.image_processor.preprocess(
        image, return_tensors='pt')['pixel_values'][0]

    data_dict['pixel_values'] = image
    data_dict['text_prompts'] = text
    data_dict['ori_image_size'] = (ori_width, ori_height)
    if return_ori_image:
        data_dict['ori_image'] = ori_image
    return data_dict