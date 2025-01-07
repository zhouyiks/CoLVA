from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import CrossEntropyLoss

from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict

from projects.lisa.datasets.utils import DEFAULT_IMAGE_TOKEN
from projects.lisa.models.lisa import LisaModel

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from transformers import GenerationConfig
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

import numpy as np

from .utils import dynamic_preprocess

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from pycocotools import mask as _mask

from types import MethodType

from xtuner.model.utils import guess_load_checkpoint

from mmcv.ops import point_sample
from mmdet.models.utils import get_uncertain_point_coords_with_randomness

class VideoLLaVASAMBaselineModel(LisaModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=4,
                 use_fast_supervision=False,
                 # for inference
                 phi3=True,
                 template=None,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 # for inference large model
                 split_model=False,
                 ):
        super(LisaModel, self).__init__()
        self.split_model = split_model
        if split_model:
            mllm.model_split = split_model
        self.special_tokens = special_tokens
        self.mllm = BUILDER.build(mllm)
        self.mllm.get_model().initialize_vision_modules(self.mllm.get_model().config)
        vision_tower = self.mllm.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)

        self.arch_type = arch_type

        self.fast_pool = fast_pool
        self.fast_pool_size = fast_pool_size

        self.tokenizer = BUILDER.build(tokenizer)
        self.mllm.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        if use_fast_supervision:
            self.loss_exists = BUILDER.build(dict(
                type=CrossEntropyLoss,
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0)
            )

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75
        self.use_fast_supervision = use_fast_supervision
        self.phi3 = phi3
        self.template = template


    def activation_checkpointing_disable(self):
        self.mllm.gradient_checkpointing_disable()


    def _add_special_tokens(self):
        special_tokens = self.special_tokens
        _num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        # if not isinstance(self.mllm.model.language_model.get_output_embeddings(), nn.Linear):
        #     print("Change the lm_head to nn.Linear !!!")
        #     transposed = False
        #     old_lm_head = self.mllm.model.language_model.get_output_embeddings()
        #     old_num_tokens, old_lm_head_dim = (
        #         old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        #     )
        #     new_lm_head_shape = (old_lm_head_dim, len(tokenizer)) if not transposed else (
        #     len(tokenizer), old_lm_head_dim)
        #     has_new_lm_head_bias = old_lm_head.bias is not None
        #     new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias).to(self.device)
        #     new_lm_head.weight = old_lm_head.weight
        #     new_lm_head.bias = old_lm_head.bias
        #     self.mllm.model.language_model.set_output_embeddings(new_lm_head)

        # this is already done in mllm
        # if num_new_tokens > 0:
        #     self.mllm.model.language_model.resize_token_embeddings(len(self.tokenizer))

        # assert isinstance(self.mllm, InternVL_Slowfast)
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    def state_dict(self, *args, **kwargs):
        state_dict = super(LisaModel, self).state_dict(*args, **kwargs)
        from collections import OrderedDict

        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
            raise NotImplementedError
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
            raise NotImplementedError
        # Step 2. LLM
        if self.mllm.use_llm_lora:
            if self.arch_type == 'intern_vl':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
            elif self.arch_type == 'qwen':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.model, state_dict=state_dict)
                )
            elif self.arch_type == 'llava':
                to_return.update(
                    get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict)
                )
        elif not self.mllm.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
            raise NotImplementedError
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mlp1.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.multi_modal_projector.' in k})

        # Step 4. mask decoder of grounding model (SAM/SAM2)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mask_decoder' in k})

        # Step 5. others (fcs)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_hidden_fcs.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_exist_fcs.' in k}
        )
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.weight' in k or 'output' in k and 'sam2_model' not in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'embed_tokens.weight' in k or 'tok_embeddings' in k})
        return to_return

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            # assert len(pred_mebeds) == len(gt_masks)
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        fast_exists = data.pop('fast_exists', None)
        # if self.arch_type == 'llava' and data.get('pixel_values', None) is not None:
        #     data['pixel_values'] = data['pixel_values'].to(self.torch_dtype)
        if self.fast_pool:
            output = self.mllm(data, data_samples, mode, fast_token_idx=self.fast_token_idx)
        else:
            output = self.mllm(data, data_samples, mode)
        if gt_masks is None:
            return {'llm_loss': output.loss, 'loss_mask': output.loss * 0.0, 'loss_dice': output.loss * 0.0}

        assert frames_per_batch, "Video Lisa require frames_per_batch !!!"
        # print('frmaes_per_batch: ', frames_per_batch)
        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        # seg_token_mask = seg_token_mask[:, 1:]
        # seg_token_mask = torch.cat([
        #     seg_token_mask,
        #     seg_token_mask.new_zeros(seg_token_mask.shape[0], 1)], dim=-1)

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        pred_embeddings = hidden_states[seg_token_mask]

        seg_token_counts = seg_token_mask.int().sum(-1)
        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        # print('pred_embeddings_list: ', [item.shape for item in pred_embeddings_list])
        pred_embeddings_list_video, success = self.genetate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        if not success:
            return {'llm_loss': output.loss, 'loss_mask': output.loss * 0.0, 'loss_dice': output.loss * 0.0}

        if self.use_fast_supervision and fast_exists is not None:
            # gt_exists = []
            # for id_x, _fast_exists in enumerate(fast_exists):
            #     num_tot = _fast_exists.shape[0]
            #     num_conv = gt_masks[id_x].shape[0] // frames_per_batch[id_x]
            #     assert num_tot % num_conv == 0
            #     gt_exists.append(_fast_exists.reshape(num_conv, num_tot // num_conv))
            fast_flag = input_ids == self.fast_token_idx
            fast_tokens = output.hidden_states[-1][fast_flag]
            exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
            gt_exists = torch.cat(fast_exists)
            loss_exists = self.loss_exists(exists_logit, gt_exists)
        else:
            loss_exists = None

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        accuracy = 0
        for i in range(bs):
            pred_mask = pred_masks[i]
            gt_mask = gt_masks_video[i]
            pred_mask = F.interpolate(pred_mask.unsqueeze(0), size=ori_size_list[i], mode='bilinear').squeeze(0)

            if len(pred_mask) != len(gt_mask):
                print('Warning !!! Pred and GT not equal !!!')
                _zero = pred_mask.sum() * 0.0
                loss_mask += _zero
                loss_dice += _zero
                accuracy += _zero
            else:
                if self.loss_sample_points:
                    sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_mask, gt_mask)
                    sam_loss_dice = self.loss_dice(
                        sampled_pred_mask,
                        sampled_gt_mask, avg_factor=(len(gt_mask) + 1e-4))
                    sam_loss_mask = self.loss_mask(
                        sampled_pred_mask.reshape(-1),
                        sampled_gt_mask.reshape(-1),
                        avg_factor=(pred_mask.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
                else:
                    sam_loss_mask = self.loss_mask(pred_mask, gt_mask)
                    sam_loss_dice = self.loss_dice(pred_mask, gt_mask)
                accuracy += torch.eq((pred_mask.sigmoid() > 0.5), gt_mask).to(pred_mask).mean()
                loss_mask += sam_loss_mask
                loss_dice += sam_loss_dice

        loss_dict = {
            'loss_mask': loss_mask / (bs + 1e-4),
            'loss_dice': loss_dice / (bs + 1e-4),
            'llm_loss': output.loss,
        }
        if loss_exists is not None:
            loss_dict['loss_exists'] = loss_exists
        return loss_dict

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def genetate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        if len(pred_embeddings_list) == len(frames_per_batch):
            success = True
        else:
            success = False
            print("len(pred_embeddings_list):{} is not equal to len(frames_per_batch):{} !!!".format(len(pred_embeddings_list), len(frames_per_batch)))
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
        self.bot_name = 'BOT'
        if 'template' in metainfo.keys():
            template = metainfo['template']
        else:
            template = PROMPT_TEMPLATE['phi3_chat']
        if self.template is None:
            self.template = template
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
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
        # self.text_hidden_fcs.to(self.torch_dtype)
        # if getattr(self, 'text_exist_fcs', None) is not None:
        #     self.text_exist_fcs.to(self.torch_dtype)

        # for sam image processor
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        if 'max_dynamic_patch' in metainfo.keys():
            self.max_dynamic_patch = metainfo['max_dynamic_patch']
        else:
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

    def predict_video(self, pixel_values, text_prompts, **kwargs):
        ori_h, ori_w = kwargs['ori_height'], kwargs['ori_width']

        _input_ids = kwargs['input_ids']

        g_pixel_values = kwargs.pop('g_pixel_values', None)
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])

        fast_pixel_values = kwargs.pop('fast_pixel_values', None)
        if fast_pixel_values is None:
            fast_token_idx = None
        else:
            fast_token_idx = self.fast_token_idx

        predictions = []
        pred_masks = []
        is_exists_list = []
        for input_ids in _input_ids:
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            pixel_values = pixel_values.to(dtype=self.torch_dtype)
            if fast_pixel_values is not None:
                fast_pixel_values = fast_pixel_values.to(dtype=self.torch_dtype)
            mm_inputs = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'fast_pixel_values': fast_pixel_values,
                'fast_token_idx': fast_token_idx,
            }
            generate_output = self.mllm.evaluate(
                images_clip=pixel_values,
                images=g_pixel_values,
                input_ids=input_ids,
                resize_list=[],
                original_size_list=[],
                max_new_tokens=512,
            )

            generate_output = self.mllm.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            predict = self.tokenizer.decode(generate_output.sequences[0], skip_special_tokens=False).strip()

            # input_text = self.tokenizer.decode(mm_inputs['input_ids'][0], skip_special_tokens=False)
            # print(input_text, generate_output.sequences[0], '\n', predict, self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0])

            predictions.append(predict)

            hidden_states = generate_output.hidden_states
            last_hidden_states = [item[-1][0] for item in hidden_states]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            seg_hidden_states = get_seg_hidden_states(
                last_hidden_states, generate_output.sequences[0][:-1],
                seg_id=self.seg_token_idx
            )

            if len(seg_hidden_states) == 0:
                print("Warning, no [SEG] tokens !!!")
                pred_masks.append(torch.zeros((g_pixel_values.shape[0], ori_h, ori_w), dtype=torch.int))
                continue
            elif len(seg_hidden_states) > 1:
                print("Warning, {} [SEG] tokens !!!".format(len(seg_hidden_states)))
                seg_hidden_states = seg_hidden_states[:1]
            seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

            seg_hidden_states = seg_hidden_states.to(dtype=torch.float32)

            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            # TODO: change 5
            if len(pixel_values) < 5:
                pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * pixel_values.shape[0])
            else:
                pred_mask = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * 5)
            pred_mask = F.interpolate(
                pred_mask,
                size=(ori_h, ori_w),
                mode='bilinear',
                align_corners=False,
            )
            pred_mask = pred_mask[:, 0]
            pred_mask = pred_mask.sigmoid() > 0.5
            pred_mask = pred_mask.int()
            # supervision
            if self.use_fast_supervision and (input_ids == self.fast_token_idx).sum() > 0:
                fast_flag = input_ids.squeeze(0) == self.fast_token_idx
                len_out = generate_output.sequences[0][:-1].shape[0]
                fast_tokens = last_hidden_states[:-len_out][fast_flag].to(dtype=torch.float32)
                exists_logit = self.text_exist_fcs(fast_tokens[self.fast_pool_size ** 2 - 1::self.fast_pool_size ** 2])
                is_exists = exists_logit.squeeze(-1).sigmoid() > 0.5
                is_exists_list.append(is_exists)
                not_exists = torch.logical_not(is_exists)
                if torch.any(not_exists):
                    pred_mask[not_exists] = pred_mask[not_exists] * 0

            pred_masks.append(pred_mask)
        assert len(pred_masks) == len(text_prompts)
        ret_dict = {
            'prediction': predictions,
            'prediction_masks': [mask_to_rle(_item.cpu().numpy()) for _item in pred_masks],
        }
        if 'id' in kwargs.keys():
            ret_dict['id'] = kwargs['id']

        if len(is_exists_list) > 0:
            ret_dict['is_exists'] = is_exists_list
        return ret_dict

    def predict_forward(
            self,
            pixel_values,
            text_prompts,
            ori_image_size=None,
            ori_image=None,
            mode='eval',
            **kwargs
    ):
        assert self.init_prediction_config, "Please set prediction configs using self.preparing_for_generation()"

        if kwargs.get('type', 'image') == 'video':
            return self.predict_video(pixel_values, text_prompts, **kwargs)
        if mode == 'demo_video':
            return self.predict_demo_video(
                pixel_values, text_prompts, ori_image_size, ori_image, **kwargs)

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
            # print(predict)
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

                if mode == 'demo':
                    all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)
                    for seg_hidden_states in all_seg_hidden_states:
                        seg_hidden_states = seg_hidden_states.unsqueeze(0)
                        g_pixel_values = torch.stack([
                            self.grounding_encoder.preprocess_image(pixel, dtype=self.torch_dtype) for pixel in
                            [input_dict['g_pixel_values']]
                        ])
                        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
                        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, [seg_hidden_states])
                        w, h = ori_image_size
                        masks = F.interpolate(pred_masks, size=(h, w),
                                              mode='bilinear', align_corners=False)
                        masks = masks[:, 0]
                        masks = masks.sigmoid() > 0.5
                        masks = masks.int()
                        ret_masks.append(masks)
                    print('Done gcg demos')
                    continue

                if len(seg_hidden_states) == 0:
                    print("Warning, no [SEG] tokens !!!")
                    ret_masks.append(None)
                    continue
                elif len(seg_hidden_states) > 1:
                    print("Warning, {} [SEG] tokens !!!".format(len(seg_hidden_states)))
                    seg_hidden_states = seg_hidden_states[:1]
                seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel, dtype=self.torch_dtype) for pixel in [input_dict['g_pixel_values']]
                ])
                sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
                pred_masks = self.grounding_encoder.inject_language_embd(sam_states, [seg_hidden_states])
                w, h = ori_image_size
                masks = F.interpolate(pred_masks, size=(h, w),
                                      mode='bilinear', align_corners=False)
                masks = masks[:, 0]
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

        if mode == 'demo':
            return {
                'prediction': ret_predictions, 'prediction_masks': ret_masks,
            }

        if 'masks' not in kwargs.keys():
            gt_masks = None
        else:
            gt_masks = mask_to_rle(kwargs['masks'].cpu().numpy())
        return {
            'prediction': ret_predictions, 'prediction_masks': _ret_masks,
            'gt_masks': gt_masks,
        }

    def predict_demo_video(
            self,
            pixel_values,
            text_prompts,
            ori_image_size=None,
            ori_image=None,
            **kwargs
    ):
        input_dict = {}

        # prepare images
        assert ori_image is not None, "InternVL2 only support process the image from scratch !!!"
        assert isinstance(ori_image, list)
        all_image_token_str = ''
        all_pixel_values = []
        for idx_img, image in enumerate(ori_image):
            images = dynamic_preprocess(image, self.min_dynamic_patch,
                                        1,
                                        self.image_size, self.use_thumbnail)
            pixel_values = [self.transformer(image) for image in images]
            all_pixel_values += pixel_values

            num_image_tokens = len(pixel_values) * self.patch_token
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            image_token_str = f"Frame-{idx_img + 1}: " + image_token_str + '\n'
            all_image_token_str += image_token_str

        all_pixel_values = torch.stack(all_pixel_values).to(self.torch_dtype)
        input_dict['pixel_values'] = all_pixel_values

        ret_predictions = []
        ret_masks = []

        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        for text_prompt in text_prompts:
            # add template for text
            text_prompt = text_prompt.replace(DEFAULT_IMAGE_TOKEN, all_image_token_str)
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
            # print(predict)
            ret_predictions.append(predict)

        return {
            'prediction': ret_predictions
        }

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle
