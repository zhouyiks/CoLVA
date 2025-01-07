import torch
import torch.nn as nn

from mmengine.model import BaseModel
import torch.nn.functional as F
from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict

from projects.lisa.models.lisa import LisaModel
import numpy as np
from projects.lisa.datasets.sem_seg_dataset import dynamic_preprocess

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from pycocotools import mask as _mask
from projects.lisa.datasets.utils import DEFAULT_IMAGE_TOKEN

from types import MethodType

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from transformers import GenerationConfig
from third_parts.segment_anything.utils.transforms import ResizeLongestSide
from xtuner.model.utils import guess_load_checkpoint

class VideoLisaModel(LisaModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 pretrained_pth=None,
                 ):
        super(LisaModel, self).__init__()
        self.mllm = BUILDER.build(mllm)
        self.tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens()
        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        self.grounding_encoder.mask_decoder.requires_grad_(True)
        if self.mllm.use_llm_lora:
            self.mllm.model.language_model.base_model.model.lm_head.requires_grad_(True)
            self.mllm.model.language_model.base_model.model.model.embed_tokens.requires_grad_(True)

        in_dim = self.mllm.model.config.llm_config.hidden_size
        out_dim = self.grounding_encoder.mask_decoder.transformer_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

    def _add_special_tokens(self):
        special_tokens = ['[SEG]']
        num_new_tokens = self.tokenizer.add_tokens(
            special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.mllm.model.language_model.resize_token_embeddings(len(self.tokenizer))

        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        from collections import OrderedDict

        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.mllm.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict))
        elif not self.mllm.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mlp1.' in k})
        # Step 4. mask decoder of grounding model (SAM/SAM2)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'grounding_encoder.mask_decoder.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_hidden_fcs.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.weight' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'embed_tokens.weight' in k})
        return to_return

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        output = self.mllm(data, data_samples, mode)
        if gt_masks is None:
            return {'llm_loss': output.loss}
        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"

        resize_list = [pixel.shape[-2:] for pixel in g_pixel_values]

        # print('image_shape: ', [item.shape for item in g_pixel_values])
        # print('gt_masks_shape: ', [item.shape for item in gt_masks])
        # print('frames_per_batch: ', frames_per_batch)

        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess(pixel) for pixel in g_pixel_values
        ])
        image_embeddings = self.grounding_encoder.image_encoder(g_pixel_values)

        seg_token_mask = input_ids == self.seg_token_idx

        # seg_token_mask = seg_token_mask[:, 1:]
        # # DOUBLE CHECK HERE!!! why shift 1 to the left
        # seg_token_mask = torch.cat([
        #     seg_token_mask,
        #     seg_token_mask.new_zeros(seg_token_mask.shape[0], 1)], dim=-1)

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        pred_embeddings = hidden_states[seg_token_mask]

        seg_token_counts = seg_token_mask.int().sum(-1)
        # print('seg_token_counts: ', seg_token_counts)
        # print('input_ids: ', input_ids)
        # print(self.tokenizer.decode(input_ids[0], skip_special_tokens=False), self.tokenizer.decode(input_ids[1], skip_special_tokens=False))
        # print('seg_token_idx: ', self.seg_token_idx)
        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)

        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            return {'llm_loss': output.loss}
        assert len(pred_embeddings_list_video) == len(image_embeddings)
        pred_masks = self._generate_and_postprocess_masks(
            pred_embeddings_list_video, image_embeddings, resize_list, ori_size_list)

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)

        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        for i in range(bs):
            pred_mask = pred_masks[i]
            gt_mask = gt_masks_video[i]

            if len(pred_mask) != len(gt_mask):
                print('Wrning !!! Pred and GT not equal !!!')
                _zero = pred_mask.sum() * 0.0
                loss_mask += _zero
                loss_dice += _zero
                accuracy = _zero
            else:
                sam_loss_mask = self.loss_mask(pred_mask, gt_mask)
                sam_loss_dice = self.loss_dice(pred_mask, gt_mask)
                accuracy = torch.eq((pred_mask.sigmoid() > 0.5), gt_mask).to(pred_mask).mean()
                loss_mask += sam_loss_mask
                loss_dice += sam_loss_dice

        loss_dict = {
            'loss_mask': loss_mask / bs,
            'loss_dice': loss_dice / bs,
            # 'accuracy': accuracy,
            'llm_loss': output.loss,
        }
        return loss_dict

    def genetate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        if len(pred_embeddings_list) == len(frames_per_batch):
            success = True
        else:
            success = False
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video, success

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def preparing_for_generation(self, metainfo, **kwargs):
        # set stop criteria and generation configs for model
        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
        self.torch_dtype = torch.bfloat16
        self.bot_name = 'BOT'
        if 'template' in metainfo.keys():
            template = metainfo['template']
        else:
            template = PROMPT_TEMPLATE['phi3_chat']
        self.template = template
        stop_words = []
        stop_words += template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True

        self.mllm.to(self.torch_dtype)
        self.grounding_encoder.to(self.torch_dtype)
        self.text_hidden_fcs.to(self.torch_dtype)

        # for sam image processor
        self.extra_image_processor = ResizeLongestSide(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        # change phi3 prepare for generation fuction
        self.mllm.model.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation,
                                                                                  self.mllm.model.language_model)
        return

    def predict_forward(
            self, pixel_values, text_prompts,
            ori_image_size=None, ori_image=None,
            **kwargs):
        # pixel_values: image tensor
        # text_prompts: question without template
        assert self.init_prediction_config, "Please set prediction configs using self.preparing_for_generation()"

        input_dict = {}

        # prepare images
        assert ori_image is not None, "InternVL2 only support process the image from scratch !!!"
        image = ori_image
        # for pixel segmentation tasks
        if ori_image_size is not None and 'masks' in kwargs.keys():
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            input_dict['g_pixel_values'] = g_pixel_values

        images = dynamic_preprocess(image, self.min_dynamic_patch,
                                    self.max_dynamic_patch,
                                    self.image_size, self.use_thumbnail)
        pixel_values = [self.transformer(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(self.torch_dtype)
        input_dict['pixel_values'] = pixel_values

        num_image_tokens = pixel_values.shape[0] * self.patch_token
        image_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'

        ret_predictions = []
        ret_masks = []

        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        for text_prompt in text_prompts:
            # add template for text
            text_prompt = text_prompt.replace(DEFAULT_IMAGE_TOKEN, image_token_str)
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text_prompt, round=1, bot_name=self.bot_name)

            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': input_dict['pixel_values'],
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None
            }

            generate_output = self.mllm.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            predict = self.tokenizer.decode(
                generate_output.sequences[0], skip_special_tokens=False).strip()
            print(predict)
            ret_predictions.append(predict)
            # refcoco test need debug !!!
            if ori_image_size is not None and 'masks' in kwargs.keys():
                hidden_states = generate_output.hidden_states
                last_hidden_states = [item[-1][0] for item in hidden_states]
                last_hidden_states = torch.cat(last_hidden_states, dim=0)
                seg_hidden_states = get_seg_hidden_states(
                    last_hidden_states, generate_output.sequences[0][:-1],
                    seg_id=self.seg_token_idx
                )

                if len(seg_hidden_states) == 0:
                    print("Warning, no [SEG] tokens !!!")
                    ret_masks.append(None)
                    continue
                elif len(seg_hidden_states) > 1:
                    print("Warning, {} [SEG] tokens !!!".format(len(seg_hidden_states)))
                    seg_hidden_states = seg_hidden_states[:1]
                seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess(pixel.to(self.torch_dtype).to(seg_hidden_states.device)) for
                    pixel in [input_dict['g_pixel_values']]
                ])

                ori_size_list = [(ori_image_size[-1], ori_image_size[0])]
                resize_list = [pixel.shape[-2:] for pixel in g_pixel_values]
                image_embeddings = self.grounding_encoder.image_encoder(g_pixel_values)
                pred_masks = self._generate_and_postprocess_masks(
                    [seg_hidden_states], image_embeddings, resize_list, ori_size_list)
                assert len(pred_masks) == 1
                pred_masks = pred_masks[0]

                w, h = ori_image_size
                pred_h, pred_w = pred_masks.shape[1:]
                if pred_h != h or pred_w != w:
                    print("Pred mask shape is not equal to gt !!!")
                    pred_masks = F.interpolate(pred_masks.unsqueeze(0), size=(h, w),
                                               mode='bilinear', align_corners=False).squeeze(0)
                masks = pred_masks
                masks = masks.sigmoid() > 0.5
                masks = masks.int()
                ret_masks.append(masks)

        if len(ret_predictions) == 1:
            ret_predictions = ret_predictions[0]
        if len(ret_masks) == 0:
            return {'prediction': ret_predictions}

        _ret_masks = []
        for i, ret_mask in enumerate(ret_masks):
            if ret_mask is None:
                _ret_masks.append(None)
            else:
                ret_mask = ret_mask.cpu().numpy()
                _ret_masks.append(mask_to_rle(ret_mask))

        if 'masks' not in kwargs.keys():
            gt_masks = None
        else:
            gt_masks = mask_to_rle(kwargs['masks'].cpu().numpy())
        return {
            'prediction': ret_predictions, 'prediction_masks': _ret_masks,
            'gt_masks': gt_masks,
        }

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
    return rle

from transformers.cache_utils import Cache, DynamicCache

def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values) == 0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs
