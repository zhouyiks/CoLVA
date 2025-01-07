from PIL import Image
import torch
from xtuner.model import InternVL_V1_5
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from transformers import GenerationConfig

from projects.llava_sam2.models.preprocess.image_resize import DirectResize

from projects.lisa.datasets.sem_seg_dataset import dynamic_preprocess

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class InternVL_vlm(InternVL_V1_5):

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values']

        if type(pixel_values) is list or pixel_values.ndim == 5:
            if type(pixel_values) is list:
                pixel_values = [
                    x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                ]
            # b*n, c, h, w
            concat_images = torch.cat(
                [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)
        else:
            raise NotImplementedError()

        input_ids = data['input_ids']
        position_ids = data['position_ids']
        attention_mask = data['attention_mask']
        # sum is 0 are text
        image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()

        labels = data['labels']
        use_cache = False

        outputs = self._llm_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            pixel_values=concat_images,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True)
        if mode == 'loss':
            return {'llm_loss': outputs.loss,}
        else:
            return outputs

    def _llm_forward(
            self,
            pixel_values: torch.FloatTensor,
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
        # We only added the clone code here to avoid the error.
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids).clone()

        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        self._count += 1

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)
        try:
            input_embeds[selected] = vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape='
                  f'{input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
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
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.model.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.model.device
        assert self.model.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                if type(pixel_values) is list or pixel_values.ndim == 5:
                    if type(pixel_values) is list:
                        pixel_values = [
                            x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                        ]
                    # b*n, c, h, w
                    pixel_values = torch.cat(
                        [image.to(self.model.vision_model.dtype) for image in pixel_values], dim=0)
                vit_embeds = self.model.extract_feature(pixel_values.to(device))
            image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
            image_flags = image_flags.long()
            vit_embeds = vit_embeds[image_flags == 1]

            input_embeds = self.model.language_model.get_input_embeddings()(input_ids.to(device))
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def preparing_for_generation(self, metainfo, **kwargs):
        # set stop criteria and generation configs for model
        self.torch_dtype = torch.bfloat16
        assert 'tokenizer' in metainfo
        tokenizer = metainfo['tokenizer']
        tokenizer_type = tokenizer['type']
        del tokenizer['type']
        self.tokenizer = tokenizer_type(**tokenizer)

        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
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

        self.to(self.torch_dtype)

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
        # self.mllm.model.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, self.mllm.model.language_model)
        return

    def predict_forward(self, question=None, image_path=None, **kwargs):

        assert self.init_prediction_config, "Please set prediction configs using self.preparing_for_generation()"

        input_dict = {}
        # prepare images
        assert image_path is not None, "InternVL2 only support process the image from scratch !!!"

        image = Image.open(image_path).convert('RGB')
        # for pixel segmentation tasks

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

        if isinstance(question, str):
            text_prompts = [question]
        for text_prompt in text_prompts:
            # add template for text
            text_prompt = text_prompt.replace('<image>', image_token_str)
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

            generate_output = self.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                streamer=None,
                bos_token_id=self.tokenizer.bos_token_id,
                stopping_criteria=self.stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            predict = self.tokenizer.decode(
                generate_output.sequences[0], skip_special_tokens=True).strip()
            # print(predict)
            ret_predictions.append(predict)

        if len(ret_predictions) == 1:
            ret_predictions = ret_predictions[0]
        print(ret_predictions)
        ret_dict = {'prediction': ret_predictions}
        ret_dict.update(kwargs)
        return ret_dict
