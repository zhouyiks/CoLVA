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
from accelerate import init_empty_weights

import torch.utils
import torch.utils.checkpoint
from xtuner.registry import BUILDER
from xtuner.model.modules import dispatch_modules
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from transformers import (AutoModel, AutoConfig, AutoTokenizer, BitsAndBytesConfig, 
                          GenerationConfig, AutoImageProcessor, AutoProcessor)
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_outputs import BaseModelOutputWithPast
from .utils import (LoadWoInit, traverse_dict, make_inputs_require_grad, find_all_linear_names,
                    guess_load_checkpoint, get_peft_model_state_dict)
from ..dataset.utils import (get_conv_template, IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                             VPT_CONTEXT_TOKEN, VPT_START_TOKEN, VPT_END_TOKEN)

class WrapQwen2VL(BaseModel):
    def __init__(self, 
                 mllm,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 freeze_connector=False,
                 freeze_ot_mlp=False,
                 unfreeze_vocab=False,
                 unfreeze_lm_head=False,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 pretrained_pth=None,
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

        traverse_dict(mllm)
        model_clazz = mllm.pop('type')
        self.model = model_clazz(**mllm)
        self.model.model.config.use_cache = False
        dispatch_modules(self.model.model)

        self.model.model.forward = MethodType(Qwen2VLModel_forward, self.model.model)

        if use_object_tokens:
            # 1280
            ot_config = AutoConfig.from_pretrained(object_tokenizer["pretrained_model_name_or_path"], trust_remote_code=True)
            self.ot_config = ot_config
            traverse_dict(object_tokenizer)
            ot_clazz = object_tokenizer.pop('type')
            self.object_tokenizer = ot_clazz(**object_tokenizer)  # !!! accelerate >= 1.0.0
            ot_hidden_size = self.object_tokenizer.model.num_features
            llm_hidden_size = self.model.model.config.hidden_size
            self.ot_mlp1 = nn.Sequential(
                nn.LayerNorm(ot_hidden_size,),
                nn.Linear(ot_hidden_size, llm_hidden_size,),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        else:
            self.object_tokenizer = None
            self.ot_mlp1 = None
            self.ot_config = None
        
        self.processor = AutoProcessor.from_pretrained(mllm["pretrained_model_name_or_path"])
        
        self._add_special_tokens()

        if self.freeze_llm:
            self.model.model.requires_grad_(False)
        if self.freeze_visual_encoder:
            assert self.freeze_connector
            self.model.visual.requires_grad_(False)
        if self.object_tokenizer is not None:
            self.object_tokenizer.requires_grad_(False)
        if self.freeze_ot_mlp and self.ot_mlp1 is not None:
            self.ot_mlp1.requires_grad_(False)
        
        if use_activation_checkpointing:
            # it is necessary when using gradient checkpointing
            if hasattr(self.model.model, 'enable_input_require_grads'):
                    self.model.model.enable_input_require_grads()
            else:
                self.model.model.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)
        # put this after llm_lora
        if self.unfreeze_vocab:
            self.model.get_input_embeddings().requires_grad_(True)
        else:
            self.model.get_input_embeddings().requires_grad_(False)
        if self.unfreeze_lm_head:
            self.model.get_output_embeddings().requires_grad_(True)
        else:
            self.model.get_output_embeddings().requires_grad_(False)
        
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
           
            mllm_state_dict = {}
            for k, v in pretrained_state_dict.items():
                if k.startswith('model.'):
                    mllm_state_dict[k[len('model.'):]] = v
            if len(mllm_state_dict) != 0:
                self.model.load_state_dict(mllm_state_dict, strict=False)
            
            if use_object_tokens:
                ot_adapter_state_dict = {}
                for k, v in pretrained_state_dict.items():
                    if k.startswith('ot_mlp1.'):
                        ot_adapter_state_dict[k[len('ot_mlp1.'):]] = v
                if len(ot_adapter_state_dict) != 0:
                    self.ot_mlp1.load_state_dict(ot_adapter_state_dict, strict=False)
         
                for k, v in self.ot_mlp1.named_parameters():
                    assert v.equal(ot_adapter_state_dict[k])
            
            print(f"Load pretrained weight from {pretrained_pth}")
        
        self._count = 0
        print_log(self, logger="current")
        print_log('Qwen2-VL construction is complete', logger='current')

    def _add_special_tokens(self):
        assert hasattr(self, "processor")

        special_tokens = [VPT_CONTEXT_TOKEN, ]
        num_new_tokens = self.processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
        print_log(f"Added {num_new_tokens} special tokens.")
        
        self.vpt_content_token_idx = self.processor.tokenizer(VPT_CONTEXT_TOKEN, add_special_tokens=False).input_ids[0]
        image_token = "<|image_pad|>" if not hasattr(self.processor.tokenizer, "image_token") else self.processor.tokenizer.image_token
        self.img_context_token_idx = self.processor.tokenizer(image_token, add_special_tokens=False).input_ids[0]

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
            lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model.model = prepare_model_for_kbit_training(self.model.model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.model)
            lora_config.target_modules = modules
        
        self.model.model = get_peft_model(self.model.model, lora_config)
    
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
                    self.model.visual, state_dict=state_dict))
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
                for k, v in state_dict.items() if 'model.model.'
            })
        
        # Custom part
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
        pixel_values = data['pixel_values'].to(self.model.visual.dtype)
        merged_visual_prompts = data['merged_visual_prompts'].to(self.model.visual.dtype)
        has_ot_input = data['ot_pixel_values'] is not None
        image_grid_thw = data['image_grid_thw']

        vit_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        
        if has_ot_input and self.object_tokenizer:
            ot_pixel_values = data['ot_pixel_values'].to(self.object_tokenizer.dtype)
            ot_h, ot_w = ot_pixel_values.shape[-2:]
            ot_num_tokens_h, ot_num_tokens_w = ot_h // self.ot_config.patch_size, ot_w // self.ot_config.patch_size
            summary, ot_embeds = self.object_tokenizer(ot_pixel_values)
            with torch.amp.autocast(device_type='cuda', dtype=self.model.visual.dtype):
                ot_embeds = self.ot_mlp1(ot_embeds)
        
        if self.object_tokenizer_pretrain:
            region_ids = data['region_ids']

            num_images = data['num_images']
            batch_size = len(num_images)
            num_vprompts = data['num_vprompts']
            visual_prompts = data['visual_prompts']
            image_grid_thw = data['image_grid_thw']
            merge_length = self.processor.image_processor.merge_size ** 2
            image_num_tokens = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2] // merge_length
            split_vit_embeds = torch.split(vit_embeds, [num_tokens for num_tokens in image_num_tokens])
            split_num_vprompts = torch.split(num_vprompts, [num_img for num_img in num_images])

            object_embeds_in_batch = []
            valid_flag_in_batch = []
            start_idx = 0
            for bidx in range(batch_size):
                num_vprompts = split_num_vprompts[bidx]
                visual_prompts_bi = torch.split(visual_prompts[bidx], [nvp for nvp in num_vprompts])
                split_vit_embeds_bi = split_vit_embeds[start_idx:start_idx+num_images[bidx]]
                start_idx = start_idx + num_images[bidx]

                object_embed_list, valid_flag_list = [], []
                for fidx, visual_prompts_fi in enumerate(visual_prompts_bi):
                    h, w = visual_prompts_fi.shape[-2:]
                    visual_prompts_fi = visual_prompts_fi.reshape(num_vprompts[fidx], h, w)
                    visual_prompts_fi = (visual_prompts_fi > 0.55).to(vit_embeds.dtype)
                    visual_prompts_fi = visual_prompts_fi.reshape(num_vprompts[fidx], -1)

                    num_vp_tokens = torch.sum(visual_prompts_fi, dim=-1, keepdim=False)
                    valid_flag = num_vp_tokens > 0

                    vit_embeds_fi = split_vit_embeds_bi[fidx]
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
        temp_input_ids[temp_input_ids == self.vpt_content_token_idx] = self.img_context_token_idx
        input_embeds = self.model.get_input_embeddings()(temp_input_ids.reshape(B, N)).clone()

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
        
        selected = (input_ids == self.img_context_token_idx)
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

        
        outputs = self.model(
            inputs_embeds = input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )
        
        return outputs, skip_this_case




# def Qwen2VLModel_forward

def Qwen2VLModel_forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )