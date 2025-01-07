from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

import warnings
from mmengine.utils.misc import get_object_from_string
from transformers import GenerationConfig, StoppingCriteriaList
from xtuner.dataset.utils import load_image
from xtuner.registry import BUILDER
from xtuner.utils import StopWordStoppingCriteria
from xtuner.engine.hooks import EvaluateChatHook

class EvaluateChatHook_solo(EvaluateChatHook):
    priority = 'LOW'
    def __init__(self,
                 tokenizer,
                 evaluation_inputs,
                 evaluation_images=None,
                 image_processor=None,
                 system='',
                 prompt_template=None,
                 every_n_iters=None,
                 max_new_tokens=50,
                 stop_word=None,
                 stop_words=[],
                 ):
        self.evaluation_inputs = evaluation_inputs
        if isinstance(self.evaluation_inputs, str):
            self.evaluation_inputs = [self.evaluation_inputs]
        self.evaluation_images = evaluation_images
        if isinstance(self.evaluation_images, str):
            self.evaluation_images = [self.evaluation_images]
        if self.evaluation_images is not None:
            assert len(
                self.evaluation_images) in [1, len(self.evaluation_inputs)]
            if len(self.evaluation_images) == 1:
                self.evaluation_images = [self.evaluation_images[0]] * len(
                    self.evaluation_inputs)
            self.evaluation_images = [
                load_image(img) for img in self.evaluation_images
            ]
        if prompt_template is None:
            instruction = '{input}'
        else:
            if isinstance(prompt_template, str):  # for resume
                prompt_template = get_object_from_string(prompt_template)
            instruction = prompt_template.get('INSTRUCTION', '{input}')
            if system != '':
                system = prompt_template.get(
                    'SYSTEM', '{system}\n').format(system=system)
            stop_words += prompt_template.get('STOP_WORDS', [])
        if stop_word is not None:
            # TODO: deprecation, v0.3.0
            warnings.warn(
                ('The `stop_word` argument is deprecated and will be removed '
                 'in v0.3.0, use `stop_words` instead.'), DeprecationWarning)
            stop_words.append(stop_word)
        self.instruction = instruction
        self.system = system
        self.every_n_iters = every_n_iters
        self.max_new_tokens = max_new_tokens
        self.tokenizer = BUILDER.build(tokenizer)
        if image_processor is not None:
            self.image_processor = BUILDER.build(image_processor)
        self.stop_criteria = StoppingCriteriaList()
        # default generation config
        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )
        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))

        self.is_first_run = True
        self.metainfo = {
            'template': prompt_template,
        }

    def _eval_images(self,
                     runner,
                     model,
                     device,
                     max_new_tokens=None,
                     save_eval_output=False):
        if save_eval_output:
            eval_outputs = []

        for sample_image, sample_input in zip(self.evaluation_images,
                                              self.evaluation_inputs):
            image = self.image_processor.preprocess(
                sample_image, return_tensors='pt')['pixel_values'][0]
            image = image.to(device)

            sample_input = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
            model.preparing_for_generation(self.metainfo)
            generation_output = model.predict_forward(
                image, sample_input
            )['prediction']

            inputs = (self.system + self.instruction).format(
                input=sample_input, round=1, **runner.cfg)

            runner.logger.info(f'Sample output:\n'
                               f'{inputs + generation_output}\n')
            if save_eval_output:
                eval_outputs.append(f'{inputs + generation_output}\n')

        if save_eval_output:
            self._save_eval_output(runner, eval_outputs)
