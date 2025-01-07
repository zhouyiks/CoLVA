import torch
from PIL import Image

from transformers import Qwen2VLProcessor, AutoProcessor
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessorKwargs


class QwenVLProcessor:
    ROLE = ('user', 'assistant')

    def __init__(self, max_length=512, pretrained_model_name_or_path=None):
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path)
        self.max_length = max_length

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.processor, name)

    def build_prompt(self, query, answer, round=0, system=None):
        messages = [{"role": self.ROLE[0], "content": query}]
        if round == 0 and system:
            messages.insert(0, {"role": "system", "content": system})

        if answer is None:
            query = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            messages.append({"role": self.ROLE[1], "content": answer})
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            query, answer = prompt.split("<|im_start|>assistant\n")
            query += "<|im_start|>assistant\n"

        return query, answer

    def __call__(self, data_dict, **kwargs):
        conversations = data_dict["conversations"]
        images = data_dict.get("images", None)
        videos = data_dict.get("videos", None)

        images = data_dict.get("image", None)  # HACK: support multi images
        if images is not None:
            images = [Image.open(images).convert('RGB')]

        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(
                images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        new_conversation = []
        index = 0
        for msg in conversations:
            if msg['from'] == 'human':
                if image_grid_thw is not None:
                    merge_length = self.image_processor.merge_size**2
                    text = msg['value']
                    while "<image>" in text:
                        text = text.replace(
                            "<image>", "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1)
                        index += 1
                    text = text.replace("<|placeholder|>", "<|image_pad|>")
                    msg['value'] = text
            new_conversation.append(msg)

        input_ids, labels = [], []
        for i in range(0, len(new_conversation), 2):
            query = new_conversation[i]['value']
            answer = new_conversation[i+1]['value'] if i + \
                1 < len(new_conversation) else None
            query, answer = self.build_prompt(query, answer, round=i // 2)

            input_ids_ = self.tokenizer(
                query, add_special_tokens=True, return_attention_mask=False)['input_ids']
            labels_ = [-100] * len(input_ids_)
            if answer is not None:
                output_ids_ = self.tokenizer(answer, add_special_tokens=True,
                                             return_attention_mask=False)['input_ids']
                labels_ += output_ids_
                input_ids_ += output_ids_
            input_ids += input_ids_
            labels += labels_

        return {
            "input_ids": input_ids,
            "labels": labels,
            'pixel_values': image_inputs.get('pixel_values', None),
        }
