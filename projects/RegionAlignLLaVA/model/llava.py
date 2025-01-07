from xtuner.model import LLaVAModel as XtunerLLaVAModel
import torch
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from transformers import GenerationConfig
from .utils import prepare_inputs_labels_for_multimodal
from xtuner.model.utils import prepare_inputs_labels_for_multimodal as xtuner_prepare_inputs_labels_for_multimodal

class LLaVAModel(XtunerLLaVAModel):
    def __init__(self,
                 llm,
                 visual_encoder,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 max_position_embeddings=None,
                 tokenizer=None,
                 inference_dtype=torch.bfloat16,
    ):
        super(LLaVAModel, self).__init__(
                 llm,
                 visual_encoder,
                 freeze_llm=freeze_llm,
                 freeze_visual_encoder=freeze_visual_encoder,
                 visual_select_layer=visual_select_layer,
                 pretrained_pth=pretrained_pth,
                 projector_depth=projector_depth,
                 llm_lora=llm_lora,
                 visual_encoder_lora=visual_encoder_lora,
                 use_activation_checkpointing=use_activation_checkpointing,
                 max_position_embeddings=max_position_embeddings)
        if tokenizer is not None:
            self.tokenizer = tokenizer
            tokenizer_type = self.tokenizer['type']
            del self.tokenizer['type']
            self.tokenizer = tokenizer_type(**self.tokenizer)
        self.visual_select_layer = visual_select_layer

        self.inference_dtype = inference_dtype

    def forward(self, data, data_samples=None, mode='loss'):
        if self.is_first_iter:
            # hardcode for qlora DeepSpeed ZeRO3, put buffers and QuantState to
            # device
            # Only required in `LLaVAModel` .
            # We do not need this in `SupervisedFinetune` .
            self.to(data['input_ids'].device)
            self.is_first_iter = False

        if 'pixel_values' in data:
            visual_outputs = self.visual_encoder(
                data['pixel_values'].to(self.visual_encoder.dtype),
                output_hidden_states=True)
            pixel_values = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
            data['pixel_values'] = pixel_values
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def preparing_for_generation(self, metainfo):
        # set stop criteria and generation configs for model
        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
        self.bot_name = 'BOT'
        if 'template' in metainfo.keys():
            template = metainfo['template']
        else:
            template = PROMPT_TEMPLATE['internlm2_chat']
        self.template = template
        stop_words = []
        stop_words += template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
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

        self.visual_encoder.to(self.inference_dtype)
        self.projector.to(self.inference_dtype)
        return

    def predict_forward(
            self, pixel_values, text_prompts, **kwargs):
        # pixel_values: image tensor
        # text_prompts: question without template
        assert self.init_prediction_config, "Please set prediction configs using self.preparing_for_generation()"
        # add template for text
        input_text = ''
        input_text += self.template['INSTRUCTION'].format(
            input=text_prompts, round=1, bot_name=self.bot_name)

        chunk_encode = []
        for idx, chunk in enumerate(input_text.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        image = pixel_values.cuda().unsqueeze(0)

        visual_outputs = self.visual_encoder(image, output_hidden_states=True)
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:])

        mm_inputs = xtuner_prepare_inputs_labels_for_multimodal(
            llm=self.llm, input_ids=ids, pixel_values=pixel_values)

        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=False,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=True).strip()
        return {'prediction': predict}

