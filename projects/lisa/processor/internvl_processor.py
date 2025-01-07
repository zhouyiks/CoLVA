
import copy
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoProcessor, AutoTokenizer

from xtuner.utils import IGNORE_INDEX

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def total_image_token(orig_size,
                      min_num=1,
                      max_num=12,
                      image_size=448,
                      use_thumbnail=True):
    orig_width, orig_height = orig_size

    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if max_num >= i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    if use_thumbnail:
        blocks += 1

    return blocks

class InternVLProcessor:

    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    SYSTEM = ''
    template = dict(
        SYSTEM='<|system|>\n{system}<|end|>\n',
        # INSTRUCTION='<|user|>\n{input}<|end|>\n<|assistant|>\n',
        INSTRUCTION='<|user|>\n{input}<|end|><|assistant|>\n',
        SUFFIX='<|end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|end|>'])
    
    def __init__(self,
                 max_length=8192, 
                 special_tokens=['[SEG]'],
                 pretrained_model_name_or_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if special_tokens:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.max_length = max_length

        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int(
            (self.image_size // patch_size)**2 * (self.downsample_ratio**2))

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', '')
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template['INSTRUCTION'].format(
                input=input, round=i + 1)

            if i == 0:
                if self.SYSTEM:
                    system = self.template['SYSTEM'].format(system=self.SYSTEM)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template['SUFFIX']
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        return {'input_ids': input_ids, 'labels': labels}
    
    def __call__(self, data_dict):
        out_data_dict = {}

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            try:
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                return None
            
            images = dynamic_preprocess(image, self.min_dynamic_patch, self.max_dynamic_patch, self.image_size, self.use_thumbnail)
            pixel_values = [self.transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            out_data_dict['pixel_values'] = pixel_values

            num_image_tokens = pixel_values.shape[0] * self.patch_token
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            token_dict = self.get_inputid_labels(data_dict['conversations'], image_token_str)
            out_data_dict.update(token_dict)
        else:
            token_dict = self.get_inputid_labels(data_dict['conversations'], None)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(1, 3, self.image_size, self.image_size)
        return out_data_dict