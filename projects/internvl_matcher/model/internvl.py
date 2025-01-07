import copy
from collections import OrderedDict
from typing import List, Optional, Tuple, Union
from types import MethodType
import torch
import torch.distributed
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training

from xtuner.registry import BUILDER
from xtuner.model.modules import dispatch_modules
from transformers import AutoModel, AutoConfig, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutput, BaseModelOutputWithPooling
from .modules import VisualPromptEncodeModel
from .utils import (LoadWoInit, traverse_dict, make_inputs_require_grad, find_all_linear_names,
                    guess_load_checkpoint, get_peft_model_state_dict)


def vision_model_forward_cache(self,
                               pixel_values: Optional[torch.FloatTensor] = None,
                               visual_prompt_embeds: Optional[torch.FloatTensor] = None,
                               output_hidden_states: Optional[bool] = None,
                               return_dict: Optional[bool] = None,
                               pixel_embeds: Optional[torch.FloatTensor] = None,
                               )->Union[Tuple, BaseModelOutputWithPooling]:
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None and pixel_embeds is None:
        raise ValueError('You have to specify pixel_values or pixel_embeds')
    
    if pixel_embeds is not None:
        hidden_states = torch.cat([
            pixel_embeds[:, :1, :], pixel_embeds[:, 1:, :] + visual_prompt_embeds.flatten(2).transpose(1, 2)], dim=1)
    else:
        if len(pixel_values.shape) == 4:
            _pixel_embeds = self.embeddings(pixel_values)
            hidden_states = torch.cat([
                _pixel_embeds[:, :1, :], _pixel_embeds[:, 1:, :] + visual_prompt_embeds.flatten(2).transpose(1, 2)], dim=1)
        else:
            raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    last_hidden_state = encoder_outputs.last_hidden_state
    pooled_output = last_hidden_state[:, 0, :]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]
    
    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def extract_feature_cache(self, 
                          pixel_values, 
                          visual_prompt_embeds):
    if self.select_layer == -1:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            visual_prompt_embeds=visual_prompt_embeds,
            output_hidden_states=False,
            return_dict=True).last_hidden_state
    else:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            visual_prompt_embeds=visual_prompt_embeds,
            output_hidden_states=True,
            return_dict=True).hidden_states[self.select_layer]
    vit_embeds = vit_embeds[:, 1:, :]

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    vit_embeds = self.mlp1(vit_embeds)
    return vit_embeds


class WrapInternVL(BaseModel):
    def __init__(self, 
                 mllm,
                 tokenizer=None,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 freeze_connector=False,
                 unfreeze_lm_head=False,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 quantization_vit=False,
                 quantization_llm=False,
                 pretrained_pth=None,
                 use_activation_checkpointing=True,
                 ):
        super().__init__()

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_connector = freeze_connector
        self.unfreeze_lm_head = unfreeze_lm_head
        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None
        self.quantization_vit = quantization_vit
        self.quantization_llm = quantization_llm
        self.use_activation_checkpointing=use_activation_checkpointing
        if quantization_vit:
            assert visual_encoder_lora is not None
        if quantization_llm:
            assert quantization_llm and llm_lora is not None

        config = AutoConfig.from_pretrained(mllm["pretrained_model_name_or_path"], trust_remote_code=True)
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'

        if quantization_vit is False and quantization_llm is False:
            quantization = None
        else:
            llm_int8_skip_modules = ['mlp1']
            if quantization_llm and not quantization_vit:
                llm_int8_skip_modules.append('vision_model')
            
            if quantization_vit and not quantization_llm:
                llm_int8_skip_modules.append('language_model')
            
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

        with LoadWoInit():
            traverse_dict(mllm)
            model_clazz = mllm.pop('type')
            mllm.update(dict(quantization_config=quantization, config=config))
            # The weights in internvl2 modules have been loaded inside the calling of AutoModel.from_pretrained()
            self.model = model_clazz(**mllm)
        # self.model.language_model.config.use_cache = False
        dispatch_modules(self.model.language_model)
        
        self.model.vision_model.forward = MethodType(vision_model_forward_cache, self.model.vision_model)
        self.model.extract_feature = MethodType(extract_feature_cache, self.model)
        self.visual_prompt_encoder = VisualPromptEncodeModel(
            in_channels=3, vision_hidden_size=config.vision_config.hidden_size, 
            language_hidden_size=config.llm_config.hidden_size, force_image_size=config.force_image_size,
            patch_size=config.vision_config.patch_size, downsample_ratio=config.downsample_ratio).to(
            self.model.vision_model.dtype)
        
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
        if self.unfreeze_lm_head:
            # self.model.language_model.get_output_embeddings().require_grad = True
            self.model.language_model.get_output_embeddings().requires_grad_(True)
            # for name, param in self.named_parameters():
            #     if 'tok_' in name or 'lm_head' in name:
            #         print("Unfrozen {} !!!".format(name))
            #         param.requires_grad_(True)
            #     if 'output.' in name and 'llm' in name and 'lora' not in name:
            #         print("Unfrozen {} !!!".format(name))
            #         param.requires_grad_(True)
        
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
        
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(visual_encoder_lora)
        
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)  # TODO, check whether the internvl2 weights are loaded correctly.
            print(f"Load pretrained weight from {pretrained_pth}")

        self._count = 0
        print_log(self, logger="current")
        print_log('InternVL_V1_5 construction is complete', logger='current')
    
    def _add_special_tokens(self):
        assert hasattr(self, "tokenizer")
        
        mark_tokens = [f'<mark{str(ii).zfill(3)}>' for ii in range(100)]
        added_tokens_num = self.tokenizer.add_tokens(mark_tokens)
        print_log(f'{added_tokens_num} special mark tokens were added successfully.', logger='current')
        
        self.model.language_model.resize_token_embeddings(len(self.tokenizer))
        
        self.mark_token_ids = {mark_token: self.tokenizer(
            mark_token, add_special_tokens=False).input_ids[0] for mark_token in mark_tokens}

        if self.use_activation_checkpointing or self.use_llm_lora or not self.freeze_llm:
            self.model.language_model.enable_input_require_grads()
        self.added_special_token = True
        
        return

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
        self.model.language_model = prepare_model_for_kbit_training(
            self.model.language_model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.language_model)
            lora_config.target_modules = modules
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)

    def _prepare_visual_encoder_for_lora(self, lora_config):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.vision_model)
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
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.mlp1.' in k})

        # prompt related models
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'visual_prompt_encoder.' in k})

        # embeds and so on
        # vocabulary embedding
        to_return.update(
            {k: v for k, v in state_dict.items() if 'tok_' in k or 'embed_tokens' in k}
        )
        # logit head
        to_return.update(
            {k: v for k, v in state_dict.items() if
             ('output.' in k or 'lm_head' in k) and 'llm' in k and 'lora' not in k}
        )

        return to_return
    
    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values'].to(self.model.vision_model.dtype)
        visual_prompts = data['visual_prompts'].to(self.model.vision_model.dtype)
        merged_visual_prompts = data['merged_visual_prompts'].to(self.model.vision_model.dtype)
        num_patches = data['num_patches']
        num_vprompts = data['num_vprompts']
        sampled_mark_token_ids = data['sampled_mark_token_ids']

        # print('pixel values: ', pixel_values.shape)
        # print('visual prompts: ', visual_prompts.shape)
        # print('merged visual prompt: ', merged_visual_prompts.shape)
        # print('num patches: ', num_patches)
        # print('num_vprompts: ', num_vprompts)
        # exit(0)
        
        sampled_mark_tokens = [f'<mark{str(ii.item()).zfill(3)}>' for ii in sampled_mark_token_ids]
        sampled_mark_token_ids = torch.tensor(
            [self.mark_token_ids[mark_token] for mark_token in sampled_mark_tokens], 
            dtype=torch.long).to("cuda")
        # print("sampled mark tokens: ", sampled_mark_tokens)
        # print("sampled mark token ids: ", sampled_mark_token_ids)
        mark_embeddings = self.model.language_model.get_input_embeddings()(sampled_mark_token_ids)
     
        visual_prompts_patch_embeds = self.visual_prompt_encoder(
            merged_visual_prompts, visual_prompts, mark_embeddings, num_patches, num_vprompts)

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        image_flags = data['image_flags']

        labels = data['labels']
        use_cache = False

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=pixel_values,
            labels=labels,
            use_cache=use_cache,
            visual_prompt_embeds=visual_prompts_patch_embeds,
        )
        loss_dict = {'loss': outputs.loss}
        if mode == 'loss':
            return loss_dict
        else:
            raise NotImplementedError

    def _llm_forward(
        self,
        pixel_values: torch.FloatTensor,
        visual_prompt_embeds: torch.FloatTensor,
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None \
            else self.model.config.use_return_dict
        
        image_flags = image_flags.squeeze(-1)
        # We only added the clone code here to avoid the error. Error will be thrown in the below try...except... codes
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        # input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        vit_embeds = self.model.extract_feature(pixel_values, visual_prompt_embeds)
        # vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B*N, C)

        if torch.distributed.get_rank() == 0 and self._count % 100 == 0:
            print(f"dynamic ViT batch size: {vit_batch_size}, "
                  f"images per sample: {vit_batch_size}/B, "
                  f"dynamic token length: {N}")
        self._count += 1

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)
   
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C).to(input_embeds.dtype)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f"warning: {e}, input_embeds[selected].shape="
                  f"{input_embeds[selected].shape}, "
                  f"vit_embeds.shape={vit_embeds.shape}")
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token].to(input_embeds.dtype)
        
        input_embeds = input_embeds.reshape(B, N, C)

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
        )
 