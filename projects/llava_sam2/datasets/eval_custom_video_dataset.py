import logging
import os
from collections import OrderedDict
import pycocotools.mask as maskUtils

import mmengine
import torch
from mmengine import print_log
import numpy as np
from mmengine.dist import master_only

from xtuner.registry import BUILDER

from vlm.datasets.evaluation.base_eval_dataset import BaseEvalDataset
from vlm.utils import VideoReader
from .encode_fn import video_lisa_encode_multi_conv_fn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode, to_pil_image

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

def decode_masklet(masklet):
    masks = []
    for _rle in masklet:
        mask = maskUtils.decode(_rle)
        masks.append(mask)
    return masks


def multi_template_fn(conversations, template_map):
    for conv in conversations:
        for i, single_turn_conversation in enumerate(conv):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = template_map.INSTRUCTION.format(input=input, round=i + 1)
            system = single_turn_conversation.get('system', '')
            if system != '' and system is not None:
                system = template_map.SYSTEM.format(system=system)
                input_text = system + input_text
            single_turn_conversation['input'] = input_text

            if template_map.get('SUFFIX', None):
                output_text = single_turn_conversation.get('output', '')
                output_text += template_map.SUFFIX
                single_turn_conversation['output'] = output_text

            # SUFFIX_AS_EOS is False ==> need_eos_token is True
            single_turn_conversation['need_eos_token'] = \
                not template_map.get('SUFFIX_AS_EOS', False)
            single_turn_conversation['sep'] = template_map.get('SEP', '')


class VideoCustomDataset(BaseEvalDataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    FAST_IMG_CONTEXT_TOKEN = '<FAST_IMG_CONTEXT>'
    FAST_IMG_START_TOKEN = '<fast_img>'
    FAST_IMG_END_TOKEN = '</fast_img>'

    METAINFO: dict = dict(name='custom')

    def __init__(self,
                 image_folder,
                 expression_file,
                 extra_image_processor=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=2048,
                 lazy=True,
                 special_tokens=None,
                 # eval settings
                 num_frames=5,
                 # eval name
                 eval_name=None,
                 # fast cfg
                 use_fast=False,
                 fast_pool_size=2,
                 n_fast_images=50,
                 fast_token_after_question=False,
    ):
        super().__init__()
        # check the config
        assert lazy is True

        self.tokenizer = BUILDER.build(tokenizer)
        self.lazy = lazy

        self.max_length = max_length

        self.template_map = template_map_fn['template']

        if offline_processed_text_folder and expression_file:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            exp_json_file = mmengine.load(expression_file)
            vid_names = mmengine.list_dir_or_file(image_folder, list_dir=False, suffix='mp4')
            vid_tags = list(map(lambda x: x.split('.')[0], vid_names))

            json_data = OrderedDict()
            for vid_tag in vid_tags:
                assert vid_tag not in json_data
                if not vid_tag in exp_json_file:
                    continue
                exp_json_current = exp_json_file[vid_tag]
                json_data[vid_tag] = {
                    'video_id': vid_tag,
                    'video_path': os.path.join(image_folder, f"{vid_tag}.mp4"),
                    'anno_path': os.path.join(image_folder, f"{vid_tag}_manual.json"),
                    'objects': exp_json_current['objects'],
                }
            self.data_infos = json_data
            self.index2key = list(self.data_infos.keys())

        self.image_folder = image_folder
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self._system = ''

        self.downsample_ratio = 0.5
        self.image_size = 448
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.num_frames = num_frames

        self.use_fast = use_fast
        self.fast_pool_size = fast_pool_size

        self.fast_token_after_question = fast_token_after_question
        self.n_fast_images = n_fast_images # Dummy

        # save to json
        if eval_name is None:
            eval_name = 'results'
        self.eval_name = eval_name

        # vid
        self.vid_interval = 4

    def __len__(self):
        return len(self.data_infos)

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.data_infos:
            cur_len = 10000
            length_list.append(cur_len)
        return length_list

    def dataset_map_fn(self, text_prompts, num_frames, num_fast_frames=0):
        text_dict = self.prepare_text(num_frames, text_prompts, num_image_tokens=self.patch_token, num_fast_frames=num_fast_frames)
        ret = {'conversation': text_dict['conversation']}
        return ret

    def prepare_text(self, n_frames, expressions, num_image_tokens=256, num_fast_frames=0):

        if self.use_fast and not self.fast_token_after_question:
            fast_frame_token_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * num_fast_frames * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}' + '\n'
        else:
            fast_frame_token_str = ''

        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'
        if self.fast_token_after_question:
            assert self.use_fast
            after_question_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * num_fast_frames * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}'

        else:
            after_question_str = ''

        questions = []
        for i, exp in enumerate(expressions):
            # the exp is a question
            if '?' in exp:
                questions.append(exp)
            else:
                exp = exp.replace('.', '').strip()
                # EVAL: Use the first question all the time.
                # question_template = random.choice(SEG_QUESTIONS)
                question_template = SEG_QUESTIONS[0]
                questions.append(question_template.format(class_name=exp.lower()))

        eval_conversation_list = []
        for i, question in enumerate(questions):
            qa_list = []
            frame_tokens = frame_token_str + '\n'
            frame_tokens = frame_tokens * n_frames
            frame_tokens = frame_tokens.strip()
            qa_list.append(
                {'from': 'human', 'value': fast_frame_token_str + frame_tokens + question + after_question_str}
            )
            qa_list.append(
                {'from': 'gpt', 'value': ''}
            )
            assert len(qa_list) == 2

            input = ''
            conversation = []
            for msg in qa_list:
                if msg['from'] == 'human':
                    input += msg['value']
                elif msg['from'] == 'gpt':
                    if msg['value'] == '':
                        conversation.append({'input': input,})
                    else:
                        conversation.append({'input': input, 'output': msg['value']})
                    input = ''
                else:
                    raise NotImplementedError

            # add system information
            conversation[0].update({'system': self._system})
            eval_conversation_list.append(conversation)
        return {'conversation': eval_conversation_list}

    def __getitem__(self, index):
        data_info = self.data_infos[self.index2key[index]]
        obj_ids = data_info['objects'].keys()

        video_path = data_info['video_path']
        vid_frames = VideoReader(video_path)[::self.vid_interval]

        mask_json_file = data_info['anno_path']
        if os.path.exists(mask_json_file):
            mask_data = mmengine.load(mask_json_file)
        else:
            mask_data = None

        gt_masks = []
        text_prompts = []
        for obj_id in obj_ids:
            # obj_id_int = int(obj_id)
            # mask_ind = mask_data['masklet_id'].index(obj_id_int)
            # masks = decode_masklet([_[mask_ind] for _ in mask_data['masklet']])
            text_prompt = data_info['objects'][obj_id]['exp']

            # gt_masks.append(masks)
            text_prompts.append(text_prompt)

        data_dict = self.dataset_map_fn(text_prompts, self.num_frames, num_fast_frames=len(vid_frames))
        multi_template_fn(data_dict['conversation'], self.template_map)
        result = video_lisa_encode_multi_conv_fn(data_dict, input_ids_with_output=False, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)

        pixel_values = []
        extra_pixel_values = []
        if self.use_fast:
            fast_pixel_values = []
        ori_width, ori_height = None, None
        for frame_idx, frame_image in enumerate(vid_frames):
            if ori_height is None:
                ori_height, ori_width = frame_image.shape[0], frame_image.shape[1]
            else:
                assert ori_height == frame_image.shape[0]
                assert ori_width == frame_image.shape[1]

            frame_image = frame_image[..., ::-1] # BGR (opencv system) to RGB (numpy system)

            if self.extra_image_processor is not None:
                g_image = np.array(frame_image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_pixel_values)
            if self.use_fast:
                img = to_pil_image(frame_image, mode='RGB')
                img = self.transformer(img)
                fast_pixel_values.append(img)

            if frame_idx < self.num_frames:
                img = to_pil_image(frame_image, mode='RGB')
                img = self.transformer(img)
                pixel_values.append(img)

        pixel_values = torch.stack(pixel_values, dim=0) # (n_f, 3, h, w)
        data_dict['pixel_values'] = pixel_values
        if self.use_fast:
            fast_pixel_values = torch.stack(fast_pixel_values, dim=0)  # (n_f, 3, h, w)
            data_dict['fast_pixel_values'] = fast_pixel_values
        if self.extra_image_processor is not None:
            data_dict['g_pixel_values'] = extra_pixel_values

        data_dict['type'] = 'video'
        data_dict['video_id'] = index
        data_dict['text_prompts'] = text_prompts
        data_dict['image_folder'] = self.image_folder
        data_dict['ori_height'] = ori_height
        data_dict['ori_width'] = ori_width

        data_dict['video_path'] = video_path

        return data_dict

    @master_only
    def evaluate(self, results, work_dir):
        return {"Dummy": 0}
