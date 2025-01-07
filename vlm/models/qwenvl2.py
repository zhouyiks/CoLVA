# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoTokenizer, BitsAndBytesConfig, GenerationConfig)
from xtuner.registry import BUILDER
from xtuner.model.utils import find_all_linear_names, get_peft_model_state_dict, guess_load_checkpoint, make_inputs_require_grad



class QwenVL2(BaseModel):

    def __init__(self,
                 model_path,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 quantization_vit=False,
                 quantization_llm=False,
                 pretrained_pth=None,
                 # Extra:
                 special_tokens=None,
                 ):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert quantization_llm and llm_lora is not None

        # config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ['mlp1']
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append('vision_model')

            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append('model')

            quantization_config = dict(
                type=BitsAndBytesConfig,
                llm_int8_skip_modules=llm_int8_skip_modules,
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4')
            quantization_clazz = quantization_config.pop('type')
            quantization = quantization_clazz(**quantization_config)

        from .qwen2_vl import Qwen2VLForConditionalGeneration
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            # config=config,
            trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.tokenizer = tokenizer
        if special_tokens is not None:
            self._add_special_tokens(special_tokens)

        if self.freeze_llm:
            self.model.model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.visual.requires_grad_(False)

        if hasattr(self.model.model, 'enable_input_require_grads'):
            self.model.model.enable_input_require_grads()
        else:
            self.model.model.get_input_embeddings(
            ).register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self._count = 0


    def _add_special_tokens(self, special_tokens):
        num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            # ! important
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _post_init(self, fast_pool_size=4, fast_pool=True):
        if fast_pool:
            self.fast_pool = nn.AdaptiveAvgPool2d((fast_pool_size, fast_pool_size))
        return

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model.model = prepare_model_for_kbit_training(
            self.model.model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.model)
            lora_config.target_modules = modules
        self.model.model = get_peft_model(self.model.model, lora_config)

    def _prepare_visual_encoder_for_lora(self, lora_config):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.vision_model)
            lora_config.target_modules = modules
        self.model.vision_model = get_peft_model(self.model.vision_model,
                                                 lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.model.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.model.model.gradient_checkpointing_disable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.vision_model, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'model.visual.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.model, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'model.model.' in k
            })
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.lm_head.' in k})
        return to_return

    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        has_image = data.get('image_grid_thw', None) is not None
        if has_image:
            pixel_values = data['pixel_values'][0]
            image_grid_thw = data['image_grid_thw'][0]
        else:
            pixel_values = None
            image_grid_thw = None
        input_ids = data['input_ids']
        # position_ids = data['position_ids']
        position_ids = None
        attention_mask = data['attention_mask']

        labels = data['labels']
        use_cache = False

        # for lora
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             pixel_values=pixel_values,
                             image_grid_thw=image_grid_thw,
                             labels=labels,
                             use_cache=use_cache,
                             output_hidden_states=True,
                             )
        return outputs

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            position_ids = None,
            image_grid_thw = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.model.device
        pixel_values = pixel_values.to(device)
        image_grid_thw = image_grid_thw.to(device)
        attention_mask = attention_mask.to(device)
        input_ids = input_ids.to(device)
        outputs = self.model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
