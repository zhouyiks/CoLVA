# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
import warnings
from collections import OrderedDict
from typing import List, Optional
import torch.nn.functional as F
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (AddedToken, AutoConfig, CLIPImageProcessor, PreTrainedModel,
                          CLIPVisionModel, LlamaForCausalLM,
                          LlamaTokenizerFast, LlavaConfig,
                          LlavaForConditionalGeneration, LlavaProcessor)
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.registry import BUILDER
from xtuner.model.modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from xtuner.model.utils import (LoadWoInit, find_all_linear_names,
                                get_peft_model_state_dict, guess_load_checkpoint,
                                make_inputs_require_grad, traverse_dict)
from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from transformers import GenerationConfig
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)


def convert_state_dict_to_hf(state_dict, mapping):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith('.inv_freq'):
            continue
        for key_to_modify, new_key in mapping.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value
    return new_state_dict


class SingleLLaVAModelSFT(BaseModel):

    def __init__(self,
                 llm,
                 visual_encoder=None,
                 tokenizer=None,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=0,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 max_position_embeddings=None,
                 add_cls_token=False,
                 template=None,
                 ):
        super().__init__()

        if tokenizer is not None:
            self.tokenizer = tokenizer
            tokenizer_type = self.tokenizer['type']
            del self.tokenizer['type']
            self.tokenizer = tokenizer_type(**self.tokenizer)

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)

            self.llm = self._build_from_cfg_or_module(llm)

            if visual_encoder is not None:
                self.visual_encoder = self._build_from_cfg_or_module(
                    visual_encoder)
            else:
                self.visual_encoder = None

        self.llm.config.use_cache = False

        # dispatch_modules(self.llm)

        self.projector_depth = projector_depth
        # projector_config = ProjectorConfig(
        #     visual_hidden_size=self.visual_encoder.config.hidden_size,
        #     llm_hidden_size=self.llm.config.hidden_size,
        #     depth=self.projector_depth)
        # self.projector = ProjectorModel(projector_config).to(
        #     self.visual_encoder.dtype)
        self.projector = None

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            if self.visual_encoder is not None:
                self.visual_encoder.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(
                visual_encoder_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}',
                      'current')

        self.visual_select_layer = visual_select_layer

        self._is_init = True

        self.is_first_iter = True

        self.add_cls_token = add_cls_token

        self.template = template

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        if self.visual_encoder is not None:
            self.visual_encoder.gradient_checkpointing_enable()
        if self.projector is not None:
            self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        if self.visual_encoder is not None:
            self.visual_encoder.gradient_checkpointing_disable()
        if self.projector is not None:
            self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        return to_return

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg,
                                           max_position_embeddings):

        orig_rope_scaling = getattr(llm_cfg, 'rope_scaling', None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {'factor': 1}

        orig_rope_scaling_factor = orig_rope_scaling[
            'factor'] if 'factor' in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(llm_cfg, 'max_position_embeddings', None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(
                    math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {
                    'type': 'linear',
                    'factor': scaling_factor
                }

        # hardcode for internlm2
        llm_cfg.attn_implementation = 'flash_attention_2'
        cfg.config = llm_cfg

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = ('LlamaConfig', 'GemmaConfig', 'MistralConfig',
                             'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig',
                             'Starcoder2Config', 'Starcoder2Config',
                             'Phi3Config')
        SUPPORT_FLASH_ATTN2 = ('InternLM2Config', 'LlamaConfig', 'GemmaConfig',
                               'MistralConfig', 'MixtralConfig', 'Qwen2Config',
                               'Qwen2MoeConfig', 'Starcoder2Config',
                               'Starcoder2Config', 'Phi3Config')

        torch_dtype = torch.bfloat16 if (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        if getattr(cfg, 'attn_implementation', None) is not None:
            # Flash Attention 2.0 only supports torch.float16 and
            # torch.bfloat16 dtypes
            if cfg.attn_implementation == 'flash_attention_2':
                cfg.torch_dtype = torch_dtype
        elif SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch_dtype
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = 'sdpa'

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_qlora_zero3(cfg):
        if (not is_deepspeed_zero3_enabled()) or (not hasattr(
                cfg, 'quantization_config')):
            return cfg

        torch_dtype = torch.bfloat16 if (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        cfg.torch_dtype = torch_dtype
        quantization_config = cfg.quantization_config
        quantization_config.bnb_4bit_compute_dtype = torch_dtype
        quantization_config.bnb_4bit_quant_storage = torch_dtype

        return cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        cfg = self._prepare_for_qlora_zero3(cfg)
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(
                cfg, llm_cfg, max_position_embeddings)
        return cfg

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):
        if self.is_first_iter:
            # hardcode for qlora DeepSpeed ZeRO3, put buffers and QuantState to
            # device
            # Only required in `LLaVAModel` .
            # We do not need this in `SupervisedFinetune` .
            self.to(data['input_ids'].device)
            self.is_first_iter = False

        # if 'pixel_values' in data:
        #     # no visual encoder
        #     # visual_outputs = self.visual_encoder(
        #     #     data['pixel_values'].to(self.visual_encoder.dtype),
        #     #     output_hidden_states=True)
        #     # pixel_values = self.projector(
        #     #     visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
        #     # data['pixel_values'] = pixel_values
        #     # only merge the image and text tokens.
        #     data = prepare_inputs_labels_for_multimodal_solo(llm=self.llm, tokenizer=self.tokenizer, **data,
        #                                                      add_CLS=self.add_cls_token)
        data = prepare_inputs_labels_for_multimodal_solo(llm=self.llm, tokenizer=self.tokenizer, **data,
                                                         add_CLS=self.add_cls_token)

        if mode == 'loss':
            loss = self.compute_loss(data, data_samples)
            if torch.isnan(loss["loss"]):
                print("loss nan here")
            return loss
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        # return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

    def to_hf(self,
              cfg,
              save_dir,
              fp32=False,
              save_pretrained_kwargs={},
              save_format='xtuner',
              **kwargs):
        if save_format == 'xtuner':
            self.to_xtuner_llava(cfg, save_dir, fp32, save_pretrained_kwargs)
        elif save_format == 'huggingface':
            self.to_huggingface_llava(cfg, save_dir, fp32,
                                      save_pretrained_kwargs)
        elif save_format == 'official':
            self.to_official_llava(cfg, save_dir, fp32, save_pretrained_kwargs)
        else:
            raise NotImplementedError

    def to_xtuner_llava(self,
                        cfg,
                        save_dir,
                        fp32=False,
                        save_pretrained_kwargs={}):
        # LLM
        self.llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            self.llm.half()
        if self.use_llm_lora:
            llm_path = osp.join(save_dir, 'llm_adapter')
            print_log(f'Saving LLM adapter to {llm_path}', 'current')
            self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
        elif not self.freeze_llm:
            llm_path = save_dir
            print_log(f'Saving LLM tokenizer to {llm_path}', 'current')
            tokenizer = BUILDER.build(cfg.tokenizer)
            tokenizer.save_pretrained(llm_path, **save_pretrained_kwargs)
            print_log(f'Saving LLM to {llm_path}', 'current')
            self.llm.save_pretrained(llm_path, **save_pretrained_kwargs)
        self.llm.config.use_cache = False

        # Visual Encoder
        if self.use_visual_encoder_lora:
            visual_encoder_path = osp.join(save_dir, 'visual_encoder_adapter')
            print_log(
                f'Saving visual_encoder adapter to {visual_encoder_path}',
                'current')
            self.visual_encoder.save_pretrained(visual_encoder_path,
                                                **save_pretrained_kwargs)
        elif not self.freeze_visual_encoder:
            visual_encoder_path = osp.join(save_dir, 'visual_encoder')
            print_log(
                'Saving visual_encoder image_processor to'
                f'{visual_encoder_path}', 'current')
            image_processor = BUILDER.build(cfg.image_processor)
            image_processor.save_pretrained(visual_encoder_path,
                                            **save_pretrained_kwargs)
            print_log(f'Saving visual_encoder to {visual_encoder_path}',
                      'current')
            self.visual_encoder.save_pretrained(visual_encoder_path,
                                                **save_pretrained_kwargs)

        # Projector
        projector_path = osp.join(save_dir, 'projector')
        print_log(f'Saving projector to {projector_path}', 'current')
        self.projector.save_pretrained(projector_path,
                                       **save_pretrained_kwargs)

    def to_huggingface_llava(self,
                             cfg,
                             save_dir,
                             fp32=False,
                             save_pretrained_kwargs={}):

        LLM_MAPPING = {
            'model': 'language_model.model',
            'lm_head': 'language_model.lm_head',
        }
        VIT_MAPPING = {
            'vision_model': 'vision_tower.vision_model',
        }
        PROJECTOR_MAPPING = {
            'model.0': 'multi_modal_projector.linear_1',
            'model.2': 'multi_modal_projector.linear_2',
        }

        assert getattr(self.llm, 'hf_quantizer', None) is None, \
            'This conversion format does not support quantized LLM.'

        # get state_dict
        llm = self.llm
        if self.use_llm_lora:
            llm = self.llm.merge_and_unload()
        llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            llm.half()

        assert isinstance(llm, LlamaForCausalLM), \
            'This conversion format only supports LlamaForCausalLM.'
        llm_state_dict = llm.state_dict()
        llm_state_dict = convert_state_dict_to_hf(llm_state_dict, LLM_MAPPING)

        need_visual_encoder = (not self.freeze_visual_encoder
                               or self.use_visual_encoder_lora)
        visual_encoder = self.visual_encoder
        if self.use_visual_encoder_lora:
            visual_encoder = self.visual_encoder.merge_and_unload()
        assert isinstance(visual_encoder, CLIPVisionModel), \
            'This conversion format only supports CLIPVisionModel.'
        if need_visual_encoder:
            visual_encoder_state_dict = visual_encoder.state_dict()
            visual_encoder_state_dict = convert_state_dict_to_hf(
                visual_encoder_state_dict, VIT_MAPPING)
        else:
            visual_encoder_state_dict = {}

        projector_state_dict = self.projector.state_dict()
        projector_state_dict = convert_state_dict_to_hf(
            projector_state_dict, PROJECTOR_MAPPING)

        state_dict = {
            **projector_state_dict,
            **llm_state_dict,
            **visual_encoder_state_dict
        }

        # init model
        text_config = llm.config
        vision_config = visual_encoder.config
        config = LlavaConfig(
            text_config=text_config,
            vision_config=vision_config,
            attn_implementation='eager')

        with init_empty_weights():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='.*non-meta.*', category=UserWarning)
                model = LlavaForConditionalGeneration(config)
        model.load_state_dict(state_dict, strict=True, assign=True)

        # processor
        cfg.tokenizer.type = LlamaTokenizerFast.from_pretrained
        tokenizer = BUILDER.build(cfg.tokenizer)

        tokenizer.add_tokens(
            AddedToken(DEFAULT_IMAGE_TOKEN, special=True, normalized=False),
            special_tokens=True)
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

        image_processor = BUILDER.build(cfg.image_processor)
        assert isinstance(image_processor, CLIPImageProcessor), \
            'This conversion format only supports CLIPImageProcessor.'

        processor = LlavaProcessor(
            tokenizer=tokenizer, image_processor=image_processor)

        # Pad to 64 for performance reasons
        pad_shape = 64

        pre_expansion_embeddings = \
            model.language_model.model.embed_tokens.weight.data
        mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T
                 @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5 * sigma)

        # We add an image token so we need to resize the model
        ori_vocab_size = config.text_config.vocab_size
        tokenizer_vocab_size = tokenizer.encode('<pad>')[-1]
        added_token = tokenizer_vocab_size - ori_vocab_size

        if added_token > 0:
            model.resize_token_embeddings(ori_vocab_size + added_token,
                                          pad_shape)
            model.language_model.model.embed_tokens.weight.data[
            ori_vocab_size:] = torch.stack(
                tuple(
                    dist.sample()
                    for _ in range(model.language_model.model.embed_tokens.
                                   weight.data[ori_vocab_size:].shape[0])),
                dim=0,
            )
            model.language_model.lm_head.weight.data[
            ori_vocab_size:] = torch.stack(
                tuple(dist.sample()
                      for _ in range(model.language_model.lm_head.weight.
                                     data[ori_vocab_size:].shape[0])),
                dim=0,
            )
        model.config.image_token_index = tokenizer.encode(
            DEFAULT_IMAGE_TOKEN)[-1]
        model.config.pad_token_id = tokenizer.encode('<pad>')[-1]

        # save
        print_log(f'Saving to {save_dir}', 'current')
        model.save_pretrained(save_dir, **save_pretrained_kwargs)
        processor.save_pretrained(save_dir, **save_pretrained_kwargs)

    def to_official_llava(self,
                          cfg,
                          save_dir,
                          fp32=False,
                          save_pretrained_kwargs={}):

        VIT_MAPPING = {
            'vision_model': 'model.vision_tower.vision_tower.vision_model',
        }
        PROJECTOR_MAPPING = {
            'model.0': 'model.mm_projector.0',
            'model.2': 'model.mm_projector.2',
        }

        try:
            from llava.model import LlavaConfig, LlavaLlamaForCausalLM
        except ImportError:
            raise ImportError(
                'Please install llava with '
                '`pip install git+https://github.com/haotian-liu/LLaVA.git '
                '--no-deps`.')

        assert getattr(self.llm, 'hf_quantizer', None) is None, \
            'This conversion format does not support quantized LLM.'

        # get state_dict
        llm = self.llm
        if self.use_llm_lora:
            llm = self.llm.merge_and_unload()
        llm.config.use_cache = True
        if not fp32:
            print_log('Convert LLM to float16', 'current')
            llm.half()

        assert isinstance(llm, LlamaForCausalLM), \
            'This conversion format only supports LlamaForCausalLM.'
        llm_state_dict = llm.state_dict()

        need_visual_encoder = (not self.freeze_visual_encoder
                               or self.use_visual_encoder_lora)
        visual_encoder = self.visual_encoder
        if self.use_visual_encoder_lora:
            visual_encoder = self.visual_encoder.merge_and_unload()
        assert isinstance(visual_encoder, CLIPVisionModel), \
            'This conversion format only supports CLIPVisionModel.'
        if need_visual_encoder:
            visual_encoder_state_dict = visual_encoder.state_dict()
            visual_encoder_state_dict = convert_state_dict_to_hf(
                visual_encoder_state_dict, VIT_MAPPING)
        else:
            visual_encoder_state_dict = {}

        projector_state_dict = self.projector.state_dict()
        projector_state_dict = convert_state_dict_to_hf(
            projector_state_dict, PROJECTOR_MAPPING)

        state_dict = {
            **projector_state_dict,
            **llm_state_dict,
            **visual_encoder_state_dict
        }

        # init model
        tokenizer = BUILDER.build(cfg.tokenizer)
        image_processor = BUILDER.build(cfg.image_processor)
        assert isinstance(image_processor, CLIPImageProcessor), \
            'This conversion format only supports CLIPImageProcessor.'

        llava_config_dict = llm.config.__dict__.copy()
        llava_config_dict.update(
            dict(
                image_aspect_ratio='pad',
                mm_hidden_size=visual_encoder.config.hidden_size,
                mm_projector_type=f'mlp{self.projector_depth}x_gelu',
                mm_use_im_patch_token=False,
                mm_use_im_start_end=False,
                mm_vision_select_feature='patch',
                mm_vision_select_layer=self.visual_select_layer,
                mm_vision_tower=visual_encoder.config.name_or_path,
                unfreeze_mm_vision_tower=need_visual_encoder,
                model_type='llava',
                use_cache=True,
                use_mm_proj=True))

        llava_config = LlavaConfig(**llava_config_dict)

        with init_empty_weights():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='.*non-meta.*', category=UserWarning)
                model = LlavaLlamaForCausalLM(llava_config)

        model.load_state_dict(state_dict, strict=True, assign=True)

        # save
        print_log(f'Saving to {save_dir}', 'current')

        model.save_pretrained(save_dir, **save_pretrained_kwargs)
        image_processor.save_pretrained(save_dir, **save_pretrained_kwargs)
        tokenizer.save_pretrained(save_dir, **save_pretrained_kwargs)

    def preparing_for_generation(self, metainfo):
        # set stop criteria and generation configs for model
        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
        self.bot_name = 'BOT'
        # template = PROMPT_TEMPLATE['mistral']
        # self.template = template
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            # keep the max tokens
            max_new_tokens=2048,
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
        return

    def predict_forward(
            self, pixel_values, text_prompts, **kwargs):
        # pixel_values: image tensor
        # text_prompts: question without template
        text_prompts = text_prompts.replace('<image>\n', '').strip()
        # print("text_prompt: ", text_prompts)
        assert self.init_prediction_config, "Please set prediction configs using self.preparing_for_generation()"
        # add template for text
        input_text = ''
        input_text += self.template['INSTRUCTION'].format(
            input=text_prompts, round=1, bot_name=self.bot_name)
        # input_text = '<image>' + input_text
        cur_encode = self.tokenizer.encode(input_text)

        # chunk_encode = []
        # for idx, chunk in enumerate(input_text.split(DEFAULT_IMAGE_TOKEN)):
        #     # if idx == 0:
        #     #     cur_encode = self.tokenizer.encode(chunk)
        #     # else:
        #     #     cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
        #     cur_encode = self.tokenizer.encode(chunk)
        #     chunk_encode.append(cur_encode)
        #
        # assert len(chunk_encode) == 2
        ids = [IMAGE_TOKEN_INDEX]
        ids.extend(cur_encode)
        # for idx, cur_chunk_encode in enumerate(chunk_encode):
        #     ids.extend(cur_chunk_encode)
        #     if idx != len(chunk_encode) - 1:
        #         ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        pixel_values = pixel_values.cuda().unsqueeze(0)
        # print(torch.max(pixel_values), '   ', torch.min(pixel_values))
        h, w = pixel_values.shape[-2:]
        if max(h, w) > 1024:
            if h > w:
                h_new = 1024
                w_new = int(w * h_new / h)
                w_new = pad_32(w_new)
            else:
                w_new = 1024
                h_new = int(h * w_new / w)
                h_new = pad_32(h_new)
        else:
            h_new = pad_32(h)
            w_new = pad_32(w)
        dtype = pixel_values.dtype
        pixel_values = F.interpolate(pixel_values.to(torch.float32),
                                     size=(h_new, w_new), mode='bilinear',
                                     align_corners=False).to(dtype)

        mm_inputs = prepare_inputs_labels_for_multimodal_solo(
            llm=self.llm,
            tokenizer=self.tokenizer,
            input_ids=ids,
            pixel_values=pixel_values)

        if 'input_ids' in mm_inputs.keys() and mm_inputs['input_ids'] is not None:
            inp_length = mm_inputs['input_ids'].shape[1]
        else:
            inp_length = 0
        # print(inp_length)
        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=False,
            return_dict_in_generate=True
        )
        # predict = self.tokenizer.decode(
        #     generate_output.sequences[0], skip_special_tokens=False)
        # print(predict)
        # print('\n\n\n', '--------------------------------')
        predict = self.tokenizer.decode(
            generate_output.sequences[0][inp_length:], skip_special_tokens=True).strip()
        print(predict)
        return {'prediction': predict}


def prepare_inputs_labels_for_multimodal_solo(
        llm: PreTrainedModel,
        tokenizer=None,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        add_CLS: bool = False,
):  # (1, 3, 1024, 1024)

    ori_input_ids = input_ids

    # no image data
    if pixel_values is None:
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': None,
            'labels': labels
        }
    # image data
    _labels = labels  # (B, N)
    _position_ids = position_ids
    _attention_mask = attention_mask

    vision_patch_indices = []
    vision_patches = []
    visual_tokens = []

    patch_size = 32
    NON_VISION_TOKEN = -1
    if isinstance(pixel_values, torch.Tensor):
        assert pixel_values.shape[0] == 1
        pixel_values = pixel_values[0]
        patches = pixel_values.unfold(1, patch_size, patch_size) \
            .unfold(2, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()  # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)

        n_rows, n_cols = patches.shape[:2]
        n_patches = n_rows * n_cols
        patches = patches.view(n_patches, -1)

        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if row_idx != 0 and col_idx == 0:  # when new row starts
                    img_tokens.append(f"<vrow_sep>")
                    cur_patch_indices.append(NON_VISION_TOKEN)
                img_tokens.append(f"<vpatch>")
                cur_patch_indices.append(row_idx * n_cols + col_idx)
        # note we use </vision> for consistency.
        img_tokens.append("</vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)

        if add_CLS:
            # add cls token to align pretrain
            img_tokens.append("<|vis_cls|>")
            cur_patch_indices.append(NON_VISION_TOKEN)

        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens, ))
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

        vision_patch_indices.append(torch.Tensor(cur_patch_indices).to(ori_input_ids))
        vision_patches.append(patches.to(pixel_values.dtype))
        visual_tokens.append(cur_tokens)

    else:
        for pixel_value in pixel_values:
            per_image_patches = pixel_value.unfold(1, patch_size, patch_size) \
                .unfold(2, patch_size, patch_size)
            per_image_patches = per_image_patches.permute(1, 2, 0, 3,
                                                          4).contiguous()  # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
            n_rows, n_cols = per_image_patches.shape[:2]
            n_patches = n_rows * n_cols
            per_image_patches = per_image_patches.view(n_patches, -1)

            img_tokens = ["<vision>"]
            cur_patch_indices = [NON_VISION_TOKEN]
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    if row_idx != 0 and col_idx == 0:  # when new row starts
                        img_tokens.append(f"<vrow_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)
                    img_tokens.append(f"<vpatch>")
                    cur_patch_indices.append(row_idx * n_cols + col_idx)

            # note we use </vision>
            img_tokens.append("</vision>")
            cur_patch_indices.append(NON_VISION_TOKEN)

            if add_CLS:
                # add cls token to align pretrain
                img_tokens.append("<|vis_cls|>")
                cur_patch_indices.append(NON_VISION_TOKEN)

            cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens, ))
            assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"

            vision_patch_indices.append(torch.Tensor(cur_patch_indices).to(ori_input_ids))
            vision_patches.append(per_image_patches.to(pixel_value.dtype))
            visual_tokens.append(cur_tokens)

    # for support multi batch
    prefix_num = 0
    for i in range(len(vision_patch_indices)):
        vision_patch_indices[i] = vision_patch_indices[i] + prefix_num
        prefix_num += len(vision_patches[i])
    vision_patches = torch.cat(vision_patches, dim=0)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [
        cur_input_ids[cur_attention_mask]
        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [
        cur_labels[cur_attention_mask]
        for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]

    new_inputs_ids = []
    new_vision_ids = []
    new_labels = []
    cur_image_idx = 0

    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            new_inputs_ids.append(cur_input_ids)
            new_labels.append(labels[batch_idx])
            new_vision_ids.append(cur_input_ids * 0 + NON_VISION_TOKEN)
            cur_image_idx += 1
            continue

        need_replace = cur_input_ids == IMAGE_TOKEN_INDEX
        num_replace = need_replace.sum()

        image_token_indices = [-1] + torch.where(
            need_replace)[0].tolist() + [
                                  cur_input_ids.shape[0]
                              ]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] +
                                                    1:image_token_indices[i +
                                                                          1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] +
                                              1:image_token_indices[i + 1]])
        cur_new_inputs_ids = []
        cur_new_labels = []
        cur_new_vision_ids = []

        for i in range(num_replace + 1):
            cur_new_inputs_ids.append(cur_input_ids_noim[i])
            cur_new_vision_ids.append(cur_input_ids_noim[i] * 0 + NON_VISION_TOKEN)
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_replace:
                # image
                cur_vision_tokens = visual_tokens[cur_image_idx].to(ori_input_ids)
                cur_new_inputs_ids.append(cur_vision_tokens)
                cur_new_vision_ids.append(vision_patch_indices[cur_image_idx])
                cur_new_labels.append(
                    torch.full((cur_vision_tokens.shape[0],),
                               IGNORE_INDEX,
                               device=cur_labels.device,
                               dtype=cur_labels.dtype))
                cur_image_idx += 1

        cur_new_inputs_ids = torch.cat(cur_new_inputs_ids)
        cur_new_vision_ids = torch.cat(cur_new_vision_ids)
        cur_new_labels = torch.cat(cur_new_labels)

        new_inputs_ids.append(cur_new_inputs_ids)
        new_vision_ids.append(cur_new_vision_ids)
        new_labels.append(cur_new_labels)

    # Combine them
    max_len = max(x.shape[0] for x in new_inputs_ids)
    batch_size = len(new_inputs_ids)

    new_inputs_ids_padded = []
    new_vision_ids_padded = []
    new_labels_padded = torch.full((batch_size, max_len),
                                   IGNORE_INDEX,
                                   dtype=new_labels[0].dtype,
                                   device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len),
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len),
                               dtype=position_ids.dtype,
                               device=position_ids.device)

    for i, (cur_new_id,
            cur_new_labels) in enumerate(zip(new_inputs_ids, new_labels)):
        # print(i, new_vision_ids)
        cur_vision_id = new_vision_ids[i]
        cur_len = cur_new_id.shape[0]
        new_inputs_ids_padded.append(
            torch.cat((cur_new_id,
                       torch.zeros((max_len - cur_len,),
                                   dtype=cur_new_id.dtype,
                                   device=cur_new_id.device)),
                      dim=0))
        new_vision_ids_padded.append(
            torch.cat((cur_vision_id,
                       torch.zeros((max_len - cur_len,),
                                   dtype=cur_new_id.dtype,
                                   device=cur_new_id.device) + NON_VISION_TOKEN),
                      dim=0))
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(
                0,
                cur_len,
                dtype=position_ids.dtype,
                device=position_ids.device)

    new_inputs_ids = torch.stack(new_inputs_ids_padded, dim=0)
    new_vision_ids = torch.stack(new_vision_ids_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    vpatch_id = tokenizer.encode("<vpatch>", add_special_tokens=False)[0]
    vpatch_indices = new_inputs_ids.clone().detach()
    vpatch_indices[vpatch_indices != vpatch_id] = NON_VISION_TOKEN
    if vision_patches is not None:
        assert vision_patches.size(0) == (vpatch_indices == vpatch_id).sum().item(), \
            f"number of vision patches is the the same as indicated in indices: {vision_patches.size(0)} vs {(vpatch_indices == vpatch_id).sum().item()}."
    vpatch_indices[vpatch_indices == vpatch_id] = torch.arange((vpatch_indices == vpatch_id).sum(),
                                                               device=vpatch_indices.device)

    return {
        'input_ids': new_inputs_ids,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'past_key_values': past_key_values,
        # 'inputs_embeds': None,
        'labels': new_labels,
        'vision_patch_indices': vpatch_indices,  # new_vision_ids,
        'vision_patches': vision_patches,  # only 1 image
    }


def pad_32(val):
    if val % 32 == 0:
        return val
    else:
        return (val // 32 + 1) * 32
