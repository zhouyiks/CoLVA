from collections import OrderedDict
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoTokenizer, BitsAndBytesConfig, LlavaForConditionalGeneration)
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from xtuner.registry import BUILDER
from xtuner.model.utils import find_all_linear_names, get_peft_model_state_dict, guess_load_checkpoint, make_inputs_require_grad



class LLaVAModel(BaseModel):

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

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization,
            trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.tokenizer = tokenizer
        if special_tokens is not None:
            self._add_special_tokens(special_tokens)

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_tower.requires_grad_(False)

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
        self.model.vision_model = get_peft_model(self.model.vision_model,
                                                 lora_config)

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
                    self.model.vision_tower, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'model.vision_tower.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.language_model, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'model.language_model.' in k
            })
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'model.multi_modal_projector.' in k})
        return to_return

    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']
        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']

        labels = data['labels']
        use_cache = False

        # for lora
        outputs = self._llm_forward(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    pixel_values=pixel_values,
                                    labels=labels,
                                    use_cache=use_cache,
                                    output_hidden_states=True,
                                    )
        return outputs


    def _llm_forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.model.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.model.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                if type(pixel_values) is list:
                    pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                    ]
                    pixel_values = torch.cat(
                        [image.to(self.model.vision_tower.dtype) for image in pixel_values], dim=0)
                else:
                    _bs, _n_img, _, _h, _w = pixel_values.shape
                    pixel_values = pixel_values.flatten(0, 1).to(self.model.vision_tower.dtype)
                image_outputs = self.model.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer].to(pixel_values.dtype)

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.model.config.vision_feature_select_strategy}"
                    )

                image_features = self.model.multi_modal_projector(selected_image_feature)
                num_images, num_image_patches, embed_dim = image_features.shape
                image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
                image_flags = image_flags.long()
                image_features = image_features[image_flags == 1]
                real_num_images = image_features.shape[0]

                inputs_embeds = inputs_embeds.to(image_features.dtype)
                batch_size, sequence_length = input_ids.shape
                _input_ids = input_ids.reshape(batch_size * sequence_length)
                _inputs_embeds = inputs_embeds.reshape(batch_size * sequence_length, embed_dim)
                selected = (_input_ids == self.model.config.image_token_index)
                assert selected.sum() == real_num_images * num_image_patches

                _inputs_embeds[selected] = image_features.reshape(real_num_images * num_image_patches, embed_dim)
                inputs_embeds = _inputs_embeds.reshape(batch_size, sequence_length, embed_dim)

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.model.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
