import os
import copy
from collections import OrderedDict
from typing import List, Optional, Tuple, Union
from types import MethodType
from enum import Enum
import torch
import torch.amp
import torch.distributed
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file
from safetensors import safe_open

import torch.utils
import torch.utils.checkpoint
from xtuner.registry import BUILDER
from xtuner.model.modules import dispatch_modules
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from transformers import AutoModel, AutoConfig, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutput, BaseModelOutputWithPooling
from transformers import ConvNextV2ForImageClassification
from .utils import (LoadWoInit, traverse_dict, make_inputs_require_grad, find_all_linear_names,
                    guess_load_checkpoint, get_peft_model_state_dict)
from ..dataset.utils import (get_conv_template, IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                             VPT_CONTEXT_TOKEN, VPT_START_TOKEN, VPT_END_TOKEN)


class WrapInternVL(BaseModel):
    def __init__(self, 
                 mllm,
                 tokenizer=None,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 freeze_connector=False,
                 freeze_ot_mlp=False,
                 unfreeze_vocab=False,
                 unfreeze_lm_head=False,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 pretrained_pth=None,
                 radio_pretrained_pth=None,
                 use_activation_checkpointing=True,
                 vocab_embeds_name="tok_embeddings",
                 lm_head_name="output",
                 contras_loss=False,
                 use_object_tokens=False,
                 object_tokenizer=None,
                 object_tokenizer_pretrain=False,
                 ):
        super().__init__()
        
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_connector = freeze_connector
        self.freeze_ot_mlp = freeze_ot_mlp
        self.unfreeze_vocab = unfreeze_vocab
        self.unfreeze_lm_head = unfreeze_lm_head
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.use_activation_checkpointing=use_activation_checkpointing
        self.vocab_embeds_name = vocab_embeds_name
        self.lm_head_name = lm_head_name
        self.contras_loss = contras_loss
        self.object_tokenizer_pretrain=object_tokenizer_pretrain

        config = AutoConfig.from_pretrained(mllm["pretrained_model_name_or_path"], trust_remote_code=True)
        self.config = config
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'

        traverse_dict(mllm)
        model_clazz = mllm.pop('type')
        mllm.update(dict(config=config))
        self.model = model_clazz(**mllm)
        self.model.language_model.config.use_cache = False
        dispatch_modules(self.model.language_model)

        if use_object_tokens:
            # 1280
            ot_config = AutoConfig.from_pretrained(object_tokenizer["pretrained_model_name_or_path"], trust_remote_code=True)
            self.ot_config = ot_config
            traverse_dict(object_tokenizer)
            ot_clazz = object_tokenizer.pop('type')
            self.object_tokenizer = ot_clazz(**object_tokenizer)
            if 'dinov2' in object_tokenizer["pretrained_model_name_or_path"]:
                ot_hidden_size = self.ot_config.hidden_size
                self.vfm_name = 'DINOv2'
            elif 'RADIO' in object_tokenizer["pretrained_model_name_or_path"]:
                ot_hidden_size = self.object_tokenizer.model.num_features
                self.vfm_name = 'RADIO'
            elif 'convnext' in object_tokenizer["pretrained_model_name_or_path"]:
                ot_hidden_size = self.ot_config.hidden_sizes[-1]
                self.vfm_name = "ConvNext"
            else:
                raise NotImplementedError
            
            self.ot_mlp1 = nn.Sequential(
                nn.LayerNorm(ot_hidden_size,),
                nn.Linear(ot_hidden_size, config.llm_config.hidden_size,),
                nn.GELU(),
                nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size)
            )
        else:
            self.object_tokenizer = None
            self.ot_mlp1 = None
            self.ot_config = None
            
        if tokenizer is not None:
            self.tokenizer = self._build_from_cfg_or_module(tokenizer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(mllm["pretrained_model_name_or_path"], trust_remote_code=True)
        img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = img_context_token_id
        self._add_special_tokens()

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_model.requires_grad_(False)
        if self.freeze_connector:
            self.model.mlp1.requires_grad_(False)
        if self.object_tokenizer is not None:
            self.object_tokenizer.requires_grad_(False)
        if self.freeze_ot_mlp and self.ot_mlp1 is not None:
            self.ot_mlp1.requires_grad_(False)

        if use_activation_checkpointing:
            # it is necessary when using gradient checkpointing
            if hasattr(self.model.language_model, 'enable_input_require_grads'):
                    self.model.language_model.enable_input_require_grads()
            else:
                self.model.language_model.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
        
        self.gradient_checkpointing_enable()
        
        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)
        # put this after llm_lora
        if self.unfreeze_vocab:
            self.model.language_model.get_input_embeddings().requires_grad_(True)
        if self.unfreeze_lm_head:
            self.model.language_model.get_output_embeddings().requires_grad_(True)
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)
        
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            # for k, v in pretrained_state_dict.items():
            #     if 'radio_model.summary_idxs' in k:
            #         self.object_tokenizer.radio_model.summary_idxs = v
            #     if 'radio_model.input_conditioner.norm_mean' in k:
            #         self.object_tokenizer.radio_model.input_conditioner.norm_mean = v
            #     if 'radio_model.input_conditioner.norm_std' in k:
            #         self.object_tokenizer.radio_model.input_conditioner.norm_std = v

            mllm_state_dict = {}
            for k, v in pretrained_state_dict.items():
                if k.startswith('model.'):
                    mllm_state_dict[k[len('model.'):]] = v
            if len(mllm_state_dict) != 0:
                self.model.load_state_dict(mllm_state_dict, strict=False)
            
            if use_object_tokens:
                ot_adapter_state_dict = {}
                if radio_pretrained_pth is not None:
                    pretrained_state_dict = torch.load(radio_pretrained_pth, map_location='cpu')
                    if 'state_dict' in pretrained_state_dict:
                        pretrained_state_dict = pretrained_state_dict['state_dict']
                for k, v in pretrained_state_dict.items():
                    if k.startswith('ot_mlp1.'):
                        ot_adapter_state_dict[k[len('ot_mlp1.'):]] = v
                if len(ot_adapter_state_dict) != 0:
                    self.ot_mlp1.load_state_dict(ot_adapter_state_dict, strict=False)
         
                for k, v in self.ot_mlp1.named_parameters():
                    assert v.equal(ot_adapter_state_dict[k])
            
            # pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            # self.load_state_dict(pretrained_state_dict, strict=False)  # TODO, check whether the internvl2 weights are loaded correctly.
            print(f"Load pretrained weight from {pretrained_pth}")
            if radio_pretrained_pth is not None:
                print(f"Load pretrained weight from {radio_pretrained_pth}")
        
        self.patch_token = int(
            (config.force_image_size // config.vision_config.patch_size)**2 * (config.downsample_ratio**2))

        self._count = 0
        print_log(self, logger="current")
        print_log('InternVL_V1_5 construction is complete', logger='current')

    def _merge_lora(self):
        # print('pre merge lora: ', self.mllm.model.language_model.base_model.model.get_input_embeddings().weight.shape)
        try:
            self.model.language_model = self.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.model.vision_model = self.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        # print('after merge lora: ', self.mllm.model.language_model.get_input_embeddings().weight.shape)
        return

    
    def _add_special_tokens(self):
        assert hasattr(self, "tokenizer")

        special_tokens = [VPT_CONTEXT_TOKEN, ]
        num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        print_log(f"Added {num_new_tokens} special tokens.")
        
        self.vpt_content_token_idx = self.tokenizer(VPT_CONTEXT_TOKEN, add_special_tokens=False).input_ids[0]

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
            lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if self.config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.config.llm_config.architectures[0] in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplementedError
        lora_config.target_modules = target_modules
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)

    def _prepare_visual_encoder_for_lora(self, lora_config):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.vision_model)
            lora_config.target_modules = modules
        if self.interaction_indexes is not None:
            modules = []
            for block in self.interaction_indexes:
                for layer_id in range(block[0], block[1]+1):
                    modules.extend([f'{layer_id}.{elem}' for elem in ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']])
            lora_config.target_modules = modules

        self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.model.language_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.model.language_model.gradient_checkpointing_disable()

    def all_state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict
    
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
                for k, v in state_dict.items() if 'model.vision_model.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.language_model, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'model.language_model.'
            })
        # Step 3. Projector
        if not self.freeze_connector:
            to_return.update(
                {k: v
                for k, v in state_dict.items() if 'model.mlp1.' in k})
            o
        # Step 4. Custom part
        if not self.freeze_ot_mlp:
            to_return.update({k: v for k, v in state_dict.items() if 'ot_mlp1.' in k})  
        # for k, v in to_return.items():
        #     print(k)
        # exit(0)

        return to_return
    
    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        # for n, p in self.named_parameters():
        #     if p.requires_grad:
        #         print(n)
        # exit(0)
        pixel_values = data['pixel_values'].to(self.model.vision_model.dtype)
        merged_visual_prompts = data['merged_visual_prompts'].to(self.model.vision_model.dtype)
        num_patches = data['num_patches']
        has_ot_input = data['ot_pixel_values'] is not None
        
        vit_embeds = self.model.extract_feature(merged_visual_prompts)
        if has_ot_input and self.object_tokenizer:
            ot_pixel_values = data['ot_pixel_values'].to(self.object_tokenizer.dtype)
            ot_h, ot_w = ot_pixel_values.shape[-2:]
            if self.vfm_name == "RADIO":
                summary, ot_embeds = self.object_tokenizer(ot_pixel_values)
                ot_num_tokens_h, ot_num_tokens_w = ot_h // self.ot_config.patch_size, ot_w // self.ot_config.patch_size
            elif self.vfm_name == "DINOv2":
                ot_outputs = self.object_tokenizer(pixel_values=ot_pixel_values)
                ot_embeds = ot_outputs.last_hidden_state[:, 1:, :]
                ot_num_tokens_h, ot_num_tokens_w = ot_h // self.ot_config.patch_size, ot_w // self.ot_config.patch_size
            elif self.vfm_name == "ConvNext":
                ot_outputs = self.object_tokenizer.convnextv2(pixel_values=ot_pixel_values, 
                                                              output_hidden_states=True, 
                                                              return_dict=True)
                ot_embeds = ot_outputs.last_hidden_state  # N, C, H, W
                ot_num_tokens_h, ot_num_tokens_w = ot_embeds.shape[-2:]
                ot_embeds = ot_embeds.flatten(2, 3).transpose(1, 2)
            with torch.amp.autocast(device_type='cuda', dtype=self.model.vision_model.dtype):
                ot_embeds = self.ot_mlp1(ot_embeds)

        
        if self.object_tokenizer_pretrain:
            region_ids = data['region_ids']

            # internvit object tokenization
            num_images = data['num_images']
            batch_size = len(num_images)
            num_vprompts = data['num_vprompts']
            num_patches = data['num_patches']
            visual_prompts = data['visual_prompts']
            split_num_vprompts = torch.split(num_vprompts, [nimg for nimg in num_images])
            split_num_patches = torch.split(num_patches, [nimg for nimg in num_images])
            vprompt_split_size = [n_vp*n_p for n_vp , n_p in zip(num_vprompts, num_patches)]
            split_visual_prompts = torch.split(visual_prompts, vprompt_split_size)
            split_vit_embeds = torch.split(vit_embeds, [np for np in num_patches])
            
            object_embeds_in_batch = []
            valid_flag_in_batch = []
            start_idx = 0
            for bidx in range(batch_size):
                num_vprompt = split_num_vprompts[bidx]
                num_patch = split_num_patches[bidx]
                visual_prompts_bi = split_visual_prompts[start_idx:start_idx+num_images[bidx]]
                split_vit_embeds_bi = split_vit_embeds[start_idx:start_idx+num_images[bidx]]
                start_idx = start_idx + num_images[bidx]
                
                object_embed_list, valid_flag_list = [], []
                for fidx, visual_prompts_fi in enumerate(visual_prompts_bi):
                    h, w = visual_prompts_fi.shape[-2:]
                    visual_prompts_fi = visual_prompts_fi.reshape(num_vprompt[fidx], num_patch[fidx], h, w)
                    patch_token_edge = int(self.patch_token ** 0.5)
                    visual_prompts_fi = F.interpolate(visual_prompts_fi.to(vit_embeds.dtype), patch_token_edge, mode='bilinear')
                    visual_prompts_fi = (visual_prompts_fi > 0.55).to(vit_embeds.dtype)
                    visual_prompts_fi = visual_prompts_fi.reshape(num_vprompt[fidx], -1)

                    num_vp_tokens = torch.sum(visual_prompts_fi, dim=-1, keepdim=False)
                    valid_flag = num_vp_tokens > 0
                    
                    vit_embeds_fi = split_vit_embeds_bi[fidx].flatten(0, 1)
                    object_embeds = (visual_prompts_fi[:, :, None] / (num_vp_tokens[:, None, None] + 1e-4) * vit_embeds_fi[None, :, :])
                    object_embeds = torch.sum(object_embeds, dim=1)

                    object_embed_list.append(object_embeds)
                    valid_flag_list.append(valid_flag)
                
                object_embeds_in_batch.append(object_embed_list)
                valid_flag_in_batch.append(valid_flag_list)
            
            # object tokenizer
            ot_visual_prompts = data['ot_visual_prompts']
            split_ot_visual_prompts = torch.split(ot_visual_prompts, [nvp for nvp in num_vprompts])
            
            ot_object_embeds_in_batch = []
            ot_valid_flag_in_batch = []
            start_idx = 0
            for bidx in range(batch_size):
                num_vprompt = split_num_vprompts[bidx]
                ot_visual_prompts_bi = split_ot_visual_prompts[start_idx:start_idx+num_images[bidx]]
                ot_embeds_bi = ot_embeds[start_idx:start_idx+num_images[bidx]]  # bs, s, c
                start_idx = start_idx + num_images[bidx]

                object_embed_list, valid_flag_list = [], []
                for fidx, ot_visual_prompts_fi in enumerate(ot_visual_prompts_bi):
                    h, w = ot_visual_prompts_fi.shape[-2:]
                    # ot_visual_prompts_fi = ot_visual_prompts_fi.reshape(num_vprompt[fidx], 1, h, w)
                    ot_visual_prompts_fi = ot_visual_prompts_fi[:, None, :, :]
                    ot_visual_prompts_fi = F.interpolate(ot_visual_prompts_fi.to(ot_embeds.dtype), (ot_num_tokens_h, ot_num_tokens_w), mode="bilinear")
                    ot_visual_prompts_fi = (ot_visual_prompts_fi > 0.55).to(ot_embeds.dtype)
                    ot_visual_prompts_fi = ot_visual_prompts_fi.reshape(num_vprompt[fidx], -1)

                    num_vp_tokens = torch.sum(ot_visual_prompts_fi, dim=-1, keepdim=False)
                    valid_flag = num_vp_tokens > 0

                    ot_embeds_fi = ot_embeds_bi[fidx]
                    object_embeds = (ot_visual_prompts_fi[:, :, None] / (num_vp_tokens[:, None, None] + 1e-4) * ot_embeds_fi[None, :, :])
                    object_embeds = torch.sum(object_embeds, dim=1)

                    object_embed_list.append(object_embeds)
                    valid_flag_list.append(valid_flag)
                ot_object_embeds_in_batch.append(object_embed_list)
                ot_valid_flag_in_batch.append(valid_flag_list)
            
            # contrastive loss
            contras_loss = torch.zeros(size=(1, ), dtype=torch.float32).cuda()
            # contras_loss += ot_embeds.sum() * 0.0
            # return {"loss": contras_loss}
            valid_contras_sample = 0
            for bidx in range(batch_size):
                region_ids_bi = region_ids[bidx]
                object_embeds_bi = object_embeds_in_batch[bidx]
                ot_object_embeds_bi = ot_object_embeds_in_batch[bidx]
                valid_flags_bi = valid_flag_in_batch[bidx]
                ot_valid_flags_bi = ot_valid_flag_in_batch[bidx]
                
                for ot_object_embeds, object_embeds, ot_region_ids, _region_ids, ot_valid_flags, valid_flags in zip(
                    ot_object_embeds_bi, object_embeds_bi[::-1],
                    region_ids_bi, region_ids_bi[::-1],
                    ot_valid_flags_bi, valid_flags_bi[::-1],
                ):
                    region_id_to_indices = {region_id: idx for idx, region_id in enumerate(_region_ids)}
                    for anchor_embed, valid_flag, region_id in zip(ot_object_embeds, ot_valid_flags, ot_region_ids):
                        if not valid_flag:
                            continue
                        anchor_embed = anchor_embed.unsqueeze(0)  # 1, C
                        pos_idx = region_id_to_indices[region_id]
                        if not valid_flags[pos_idx]:
                            continue
                        pos_embed = object_embeds[pos_idx].unsqueeze(0)  # 1, C
                        if pos_idx == 0:
                            neg_embeds = object_embeds[1:, :][valid_flags[1:]] # N, C
                        elif pos_idx == (len(object_embeds) - 1):
                            neg_embeds = object_embeds[:-1, :][valid_flags[:-1]] # N, C
                        else:
                            neg_embeds = torch.cat([
                                object_embeds[:pos_idx, :][valid_flags[:pos_idx]],
                                object_embeds[pos_idx+1:, :][valid_flags[pos_idx+1:]]
                            ], dim=0) # N, C
                        
                        pos_neg_embeds = torch.cat([pos_embed, neg_embeds], dim=0)
                        pos_neg_label = pos_neg_embeds.new_zeros((pos_neg_embeds.shape[0], ), dtype=torch.int64)
                        pos_neg_label[:1] = 1

                        # dot product
                        dot_product = torch.einsum('ac,kc->ak', [anchor_embed, pos_neg_embeds])
                        pos_neg_label = pos_neg_label.unsqueeze(0)
                        pos_inds = (pos_neg_label == 1)
                        neg_inds = (pos_neg_label == 0)
                        pred_pos = dot_product * pos_inds.float()
                        pred_neg = dot_product * neg_inds.float()
                        # use -inf to mask out unwanted elements
                        pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
                        pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

                        _pos_expand = torch.repeat_interleave(pred_pos, dot_product.shape[1], dim=1)
                        _neg_expand = pred_neg.repeat(1, dot_product.shape[1])
                        x = F.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0)
                        try:
                            contras_loss += torch.logsumexp(x, dim=1)
                            valid_contras_sample += 1
                        except Exception as e:
                            print("x: ", x.shape)
                            print("sumexp: ", torch.logsumexp(x, dim=1).shape)
                            exit(0)
            if valid_contras_sample == 0 or torch.any(torch.isnan(contras_loss)):
                loss_dict = {"loss": ot_embeds.sum() * 0.0}
            else:
                loss_dict = {"loss": contras_loss / valid_contras_sample}
            return loss_dict
        
        # ##### Abaltion #########

        # ot_object_embeds = None
        # vprompt_flags = data['vprompt_flags']
        # ot_object_embeds_in_batch = []
        # skip_this_batch = False

        # num_vprompts = data['num_vprompts']
        # num_images = data['num_images']
        # batch_size = len(num_images)
        # num_patches = data['num_patches']
        # visual_prompts = data['visual_prompts']
        # try:
        #     split_num_vprompts = torch.split(num_vprompts, [nimg for nimg in num_images])
        #     split_num_patches = torch.split(num_patches, [nimg for nimg in num_images])
        #     vprompt_split_size = [n_vp*n_p for n_vp , n_p in zip(num_vprompts, num_patches)]
        #     split_visual_prompts = torch.split(visual_prompts, vprompt_split_size)
        #     split_vit_embeds = torch.split(vit_embeds, [np for np in num_patches])

        #     start_idx = 0
        #     for bidx in range(batch_size):
        #         num_vprompt = split_num_vprompts[bidx]
        #         num_patch = split_num_patches[bidx]
        #         visual_prompts_bi = split_visual_prompts[start_idx:start_idx+num_images[bidx]]
        #         split_vit_embeds_bi = split_vit_embeds[start_idx:start_idx+num_images[bidx]]
        #         start_idx = start_idx + num_images[bidx]
                
        #         object_embed_list = []
        #         for fidx, visual_prompts_fi in enumerate(visual_prompts_bi):
        #             h, w = visual_prompts_fi.shape[-2:]
        #             visual_prompts_fi = visual_prompts_fi.reshape(num_vprompt[fidx], num_patch[fidx], h, w)
        #             patch_token_edge = int(self.patch_token ** 0.5)
        #             visual_prompts_fi = F.interpolate(visual_prompts_fi.to(vit_embeds.dtype), patch_token_edge, mode='bilinear')
        #             visual_prompts_fi = (visual_prompts_fi > 0.55).to(vit_embeds.dtype)
        #             visual_prompts_fi = visual_prompts_fi.reshape(num_vprompt[fidx], -1)

        #             num_vp_tokens = torch.sum(visual_prompts_fi, dim=-1, keepdim=False)
                    
        #             vit_embeds_fi = split_vit_embeds_bi[fidx].flatten(0, 1)
        #             object_embeds = (visual_prompts_fi[:, :, None] / (num_vp_tokens[:, None, None] + 1e-4) * vit_embeds_fi[None, :, :])
        #             object_embeds = torch.sum(object_embeds, dim=1)

        #             object_embed_list.append(object_embeds)
                
        #         ot_object_embeds_in_batch.append(object_embed_list)
        #     ot_object_embeds = []
        #     for ele in ot_object_embeds_in_batch:
        #         ot_object_embeds.extend(ele)
        #     ot_object_embeds = torch.cat(ot_object_embeds, dim=0)
        # except:
        #     skip_this_batch = True


        
        ot_object_embeds = None
        vprompt_flags = data['vprompt_flags']
        ot_object_embeds_in_batch = []
        skip_this_batch = False
        if has_ot_input and self.object_tokenizer:
            # object tokenizer
            num_vprompts = data['num_vprompts']
            num_images = data['num_images']
            batch_size = len(num_images)
            ot_visual_prompts = data['ot_visual_prompts']
            
            try:
                split_ot_visual_prompts = torch.split(ot_visual_prompts, [nvp for nvp in num_vprompts])
            except:
                nvp_list = [1 for nvp in num_vprompts]
                if ot_visual_prompts.shape[0] >= len(nvp_list):
                    split_ot_visual_prompts = torch.split(ot_visual_prompts[:len(nvp_list)], nvp_list)
                else:
                    split_ot_visual_prompts = torch.stack([ot_visual_prompts[0] for nvp in nvp_list])
                num_vprompts = torch.tensor(nvp_list).to(num_vprompts.dtype).to(num_vprompts.device)
                skip_this_batch = True
            split_num_vprompts = torch.split(num_vprompts, [nimg for nimg in num_images]) 

            start_idx = 0
            for bidx in range(batch_size):
                num_vprompt = split_num_vprompts[bidx]
                ot_visual_prompts_bi = split_ot_visual_prompts[start_idx:start_idx+num_images[bidx]]
                ot_embeds_bi = ot_embeds[start_idx:start_idx+num_images[bidx]]  # bs, s, c
                start_idx = start_idx + num_images[bidx]
                
                ot_object_embeds_list = []
                for fidx, ot_visual_prompts_fi in enumerate(ot_visual_prompts_bi):
                    h, w = ot_visual_prompts_fi.shape[-2:]
                    ot_visual_prompts_fi = ot_visual_prompts_fi.reshape(num_vprompt[fidx], 1, h, w)
                    # ot_visual_prompts_fi = ot_visual_prompts_fi[:, None, :, :]
                    ot_visual_prompts_fi = F.interpolate(ot_visual_prompts_fi.to(ot_embeds.dtype), (ot_num_tokens_h, ot_num_tokens_w), mode="bilinear")
                    ot_visual_prompts_fi = (ot_visual_prompts_fi > 0.5).to(ot_embeds.dtype)
                    ot_visual_prompts_fi = ot_visual_prompts_fi.reshape(num_vprompt[fidx], -1)

                    num_vp_tokens = torch.sum(ot_visual_prompts_fi, dim=-1, keepdim=False)
                    ot_embeds_fi = ot_embeds_bi[fidx]
                    object_embeds = (ot_visual_prompts_fi[:, :, None] / (num_vp_tokens[:, None, None] + 1e-4) * ot_embeds_fi[None, :, :])
                    object_embeds = torch.sum(object_embeds, dim=1)
                    ot_object_embeds_list.append(object_embeds)
                ot_object_embeds_in_batch.append(ot_object_embeds_list)
            ot_object_embeds = []
            for ele in ot_object_embeds_in_batch:
                ot_object_embeds.extend(ele)
            ot_object_embeds = torch.cat(ot_object_embeds, dim=0)

        if mode == "loss":
            input_ids = data['input_ids']
            position_ids = data['position_ids']
            attention_mask = data['attention_mask']
            image_flags = data['image_flags']

            labels = data['labels']
            use_cache = False

            outputs, _skip_this_case = self._llm_forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                labels=labels,
                use_cache=use_cache,
                vit_embeds=vit_embeds,
                ot_object_embeds=ot_object_embeds,
                vprompt_flags=vprompt_flags,
            )
            
            if skip_this_batch or _skip_this_case:
                print("skip this batch!")
                loss_dict = {'loss': outputs.loss * 0.0}
            else:
                loss_dict = {'loss': outputs.loss}
            if not self.contras_loss:
                return loss_dict
            
            # num_images = data['num_images']
            # batch_size = len(num_images)

            # region_ids = data['region_ids']
            # contras_loss = loss_dict['loss'].new_zeros((1,))
            # valid_contras_sample = 0
            # for bidx in range(batch_size):
            #     if num_images[bidx] < 2 or skip_this_batch:
            #         contras_loss += ot_embeds.sum() * 0.0
            #     else:
            #         assert num_images[bidx] == 2
            #         region_id_to_indices = {region_id: idx for idx, region_id in enumerate(region_ids[bidx][-1])}
            #         anchor_region_id = region_ids[bidx][0][0]

            #         if anchor_region_id == -1:
            #             contras_loss += ot_embeds.sum() * 0.0
            #             continue

            #         anchor_embeds = ot_object_embeds_in_batch[bidx][0].mean(dim=0, keepdim=True)  # 1, C
            #         pos_idx = region_id_to_indices[anchor_region_id]
            #         try:
            #             pos_embeds = ot_object_embeds_in_batch[bidx][-1][pos_idx].unsqueeze(0)
            #         except:
            #             contras_loss += ot_embeds.sum() * 0.0
            #             continue
            #         try:
            #             if pos_idx == 0:
            #                 neg_embeds = ot_object_embeds_in_batch[bidx][-1][1:, :] # N, C
            #             elif pos_idx == (len(ot_object_embeds_in_batch[bidx][-1]) - 1):
            #                 neg_embeds = ot_object_embeds_in_batch[bidx][-1][:-1, :] # N, C
            #             else:
            #                 neg_embeds = torch.cat([
            #                     ot_object_embeds_in_batch[bidx][-1][:pos_idx, :],
            #                     ot_object_embeds_in_batch[bidx][-1][pos_idx+1:, :]
            #                 ], dim=0) # N, C
            #         except:
            #             contras_loss += ot_embeds.sum() * 0.0
            #             continue
                    
            #         pos_neg_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
            #         pos_neg_label = pos_neg_embeds.new_zeros((pos_neg_embeds.shape[0], ), dtype=torch.int64)
            #         pos_neg_label[:1] = 1

            #         # dot product
            #         dot_product = torch.einsum('ac,kc->ak', [anchor_embeds, pos_neg_embeds])
            #         pos_neg_label = pos_neg_label.unsqueeze(0)
            #         pos_inds = (pos_neg_label == 1)
            #         neg_inds = (pos_neg_label == 0)
            #         pred_pos = dot_product * pos_inds.float()
            #         pred_neg = dot_product * neg_inds.float()
            #         # use -inf to mask out unwanted elements
            #         pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
            #         pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

            #         _pos_expand = torch.repeat_interleave(pred_pos, dot_product.shape[1], dim=1)
            #         _neg_expand = pred_neg.repeat(1, dot_product.shape[1])
            #         x = F.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0)
            #         try:
            #             contras_loss += torch.logsumexp(x, dim=1)
            #             valid_contras_sample += 1
            #         except Exception as e:
            #             print("x: ", x.shape)
            #             print("sumexp: ", torch.logsumexp(x, dim=1).shape)
            #             exit(0)

            # num_images = data['num_images']
            # batch_size = len(num_images)
            # num_vprompts = data['num_vprompts']
            # num_patches = data['num_patches']
            # visual_prompts = data['visual_prompts']
            # split_num_vprompts = torch.split(num_vprompts, num_images)
            # split_num_patches = torch.split(num_patches, num_images)
            # vprompt_split_size = [n_vp*n_p for n_vp , n_p in zip(num_vprompts, num_patches)]
            # split_visual_prompts = torch.split(visual_prompts, vprompt_split_size)

            # contras_loss = loss_dict['loss'].new_zeros((1,))
            # valid_contras_sample = 0
            # for bidx in range(batch_size):
            #     num_vprompt = split_num_vprompts[bidx]
            #     num_patch = split_num_patches[bidx]
            #     visual_prompts = split_visual_prompts[:num_images[bidx]]
            #     split_vit_embeds = torch.split(vit_embeds, [np for np in num_patch])

            #     if num_images[bidx] < 2:
            #         visual_prompts = visual_prompts[0]
            #         h, w = visual_prompts.shape[-2:]
            #         visual_prompts = visual_prompts.reshape(num_vprompt[0], num_patch[0], h, w)
            #         contras_loss += vit_embeds.sum() * 0.0
            #     else:
            #         assert num_images[bidx] == 2
            #         visual_prompts_1 = visual_prompts[0]
            #         visual_prompts_2 = visual_prompts[1]
            #         h, w = visual_prompts_1.shape[-2:]
            #         visual_prompts_1 = visual_prompts_1.reshape(num_vprompt[0], num_patch[0], h, w)
            #         visual_prompts_2 = visual_prompts_2.reshape(num_vprompt[1], num_patch[1], h, w)

            #         patch_token_edge = int(self.patch_token ** 0.5)
            #         visual_prompts_1 = F.interpolate(visual_prompts_1.to(vit_embeds.dtype), patch_token_edge, mode='bilinear')
            #         visual_prompts_2 = F.interpolate(visual_prompts_2.to(vit_embeds.dtype), patch_token_edge, mode='bilinear')
            #         visual_prompts_1 = (visual_prompts_1 > 0.6).to(vit_embeds.dtype)
            #         visual_prompts_2 = (visual_prompts_2 > 0.6).to(vit_embeds.dtype)
            #         visual_prompts_1 = visual_prompts_1.reshape(num_vprompt[0], -1)
            #         visual_prompts_2 = visual_prompts_2.reshape(num_vprompt[1], -1)

            #         vit_embeds_1 = split_vit_embeds[0].flatten(0, 1)
            #         vit_embeds_2 = split_vit_embeds[1].flatten(0, 1)
                    
            #         anchor_num_tokens = torch.sum(visual_prompts_1, dim=1)
            #         if anchor_num_tokens.sum() < 1:
            #             contras_loss += vit_embeds.sum() * 0.0
            #         anchor_embeds = (visual_prompts_1[:, :, None] / anchor_num_tokens[:, None, None] * vit_embeds_1[None, :, :])
            #         anchor_embeds = torch.sum(anchor_embeds, dim=1).mean(dim=0, keepdim=True)  # 1, C

            #         pos_num_tokens = torch.sum(visual_prompts_2[:1], dim=1)
            #         if pos_num_tokens.sum() < 1:
            #             contras_loss += vit_embeds.sum() * 0.0
            #         pos_embeds = (visual_prompts_2[:1, :, None] / pos_num_tokens[:, None, None] * vit_embeds_2[None, :, :])
            #         pos_embeds = torch.sum(pos_embeds, dim=1).mean(dim=0, keepdim=True)  # 1, C

            #         neg_num_tokens = torch.sum(visual_prompts_2[1:], dim=1)
            #         valid_neg_sample = neg_num_tokens > 0
            #         if len(neg_num_tokens[valid_neg_sample]) < 4:
            #             contras_loss += vit_embeds.sum() * 0.0
            #         visual_prompts_2 = visual_prompts_2[1:, :][valid_neg_sample]
            #         neg_num_tokens = neg_num_tokens[valid_neg_sample]
            #         neg_embeds = (visual_prompts_2[:, :, None] / neg_num_tokens[:, None, None] * vit_embeds_2[None, :, :])
            #         neg_embeds =torch.sum(neg_embeds, dim=1)  # N, C

            #         pos_neg_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
            #         pos_neg_label = pos_neg_embeds.new_zeros((pos_neg_embeds.shape[0], ), dtype=torch.int64)
            #         pos_neg_label[:1] = 1

            #         # dot product
            #         dot_product = torch.einsum('ac,kc->ak', [anchor_embeds, pos_neg_embeds])
            #         pos_neg_label = pos_neg_label.unsqueeze(0)
            #         pos_inds = (pos_neg_label == 1)
            #         neg_inds = (pos_neg_label == 0)
            #         pred_pos = dot_product * pos_inds.float()
            #         pred_neg = dot_product * neg_inds.float()
            #         # use -inf to mask out unwanted elements
            #         pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
            #         pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

            #         _pos_expand = torch.repeat_interleave(pred_pos, dot_product.shape[1], dim=1)
            #         _neg_expand = pred_neg.repeat(1, dot_product.shape[1])
            #         x = F.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0)
            #         try:
            #             contras_loss += torch.logsumexp(x, dim=1)
            #             valid_contras_sample += 1
            #         except Exception as e:
            #             print("x: ", x.shape)
            #             print("sumexp: ", torch.logsumexp(x, dim=1).shape)
            #             exit(0)

            # if valid_contras_sample == 0 or torch.any(torch.isnan(contras_loss)) or skip_this_batch:
            #     loss_dict.update({"contras_loss": ot_embeds.sum() * 0.0})
            # else:
            #     loss_dict.update({"contras_loss": contras_loss / valid_contras_sample})
            # return loss_dict
        elif mode == "predict":
            pass
        elif mode == "tensor":
            pass
        else:
            raise NotImplementedError

    def _llm_forward(
        self,
        vit_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ot_object_embeds: torch.FloatTensor = None,
        vprompt_flags: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict
        
        B, N = input_ids.shape
        temp_input_ids = input_ids.clone().flatten()
        temp_input_ids[temp_input_ids == self.vpt_content_token_idx] = self.model.img_context_token_id
        input_embeds = self.model.language_model.get_input_embeddings()(temp_input_ids.reshape(B, N)).clone()

        # input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = vit_embeds.shape[0]
        
        B, N, C  = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        
        skip_this_case=False
        if ot_object_embeds is not None:    
            try:
                ot_object_embeds = ot_object_embeds[vprompt_flags > 0]
                selected = (input_ids == self.vpt_content_token_idx)
                input_embeds[selected] = input_embeds[selected] * 0.0 + ot_object_embeds
                skip_this_case=False
            except:
                print(f"The number of the provided object embeds is not match with vprompt_flags or VPT_CONTENT_TOKEN.")
                selected = (input_ids == self.vpt_content_token_idx)
                input_embeds[selected] = input_embeds[selected] * 0.0 + ot_object_embeds.mean(dim=0, keepdim=True).to(input_embeds.dtype)
                skip_this_case=True
    
        selected = (input_ids == self.model.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f"warning: {e}, input_embeds[selected].shape="
                  f"{input_embeds[selected].shape}, "
                  f"vit_embeds.shape={vit_embeds.shape}")
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
        input_embeds = input_embeds.reshape(B, N, C)

        if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
            print(f"dynamic ViT batch size: {vit_batch_size}, "
                  f"images per sample: {vit_batch_size}/B, "
                  f"dynamic token length: {N}")
        self._count += 1

        outputs = self.model.language_model(
            inputs_embeds = input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shit so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), skip_this_case
    
    def batch_chat(self, pixel_values, questions, merged_visual_prompts, 
                   generation_config, num_patches_list=None, history=None, return_history=False,
                   ):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError
        
        # tokenize and embed the text prompts
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and DEFAULT_IMAGE_TOKEN not in question:
                question = DEFAULT_IMAGE_TOKEN + '\n' + question
            template = get_conv_template(self.config.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.patch_token * num_patches + IMG_END_TOKEN
            query = query.replace(DEFAULT_IMAGE_TOKEN, image_tokens, 1)
            queries.append(query)

        self.tokenizer.padding_side = 'left'  # Important, When training the model, this attribute is setted to 'right'
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        
        vit_embeds = self.model.extract_feature(merged_visual_prompts)
        
        B, N, C  = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        
        selected = (input_ids == self.model.img_context_token_id)
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        input_embeds = input_embeds.reshape(B, N, C)

        # 4. generate
        generate_output = self.generate(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            **generation_config
        )

        responses = self.tokenizer.batch_decode(generate_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]

        return responses
    
    @torch.no_grad()
    def generate(
        self,
        input_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        outputs = self.model.language_model.generate(
            inputs_embeds = input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

        

        
        
        

        
        
