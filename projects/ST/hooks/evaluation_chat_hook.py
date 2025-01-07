# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings

import torch
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.utils import mkdir_or_exist
from mmengine.utils.misc import get_object_from_string
from transformers import GenerationConfig, StoppingCriteriaList

from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria)


class EvaluateChatHook_ST(Hook):
    priority = 'LOW'

    def __init__(self,
                 tokenizer,
                 evaluation_inputs,
                 evaluation_images=None,
                 system='',
                 every_n_iters=None,
                 max_new_tokens=600,
                 stop_word=None,
                 stop_words=[],
                 generation_kwargs={}):
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
        instruction = '{input}'
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
        self.stop_criteria = StoppingCriteriaList()

        # default generation config
        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id)
        default_generation_kwargs.update(generation_kwargs)
        self.gen_config = GenerationConfig(**default_generation_kwargs)

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))

        self.is_first_run = True

    @master_only
    def _save_eval_output(self, runner, eval_outputs):
        save_path = os.path.join(runner.log_dir, 'vis_data',
                                 f'eval_outputs_iter_{runner.iter}.txt')
        mkdir_or_exist(os.path.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            for i, output in enumerate(eval_outputs):
                f.write(f'Eval output {i + 1}:\n{output}\n\n')

    def _eval_images(self,
                     runner,
                     model,
                     device,
                     max_new_tokens=None,
                     save_eval_output=False):
        if save_eval_output:
            eval_outputs = []
        model.preparing_for_generation(metainfo={})
        for sample_image, sample_input in zip(self.evaluation_images,
                                              self.evaluation_inputs):
            image = sample_image
            sample_input = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
            inputs = sample_input
            generation_output = model.predict_forward(image=image, text=inputs)
            inputs = generation_output['input_text']
            generation_output = generation_output['prediction']
            runner.logger.info(f'Sample output:\n'
                               f'{inputs + generation_output}\n')
            if save_eval_output:
                eval_outputs.append(f'{inputs + generation_output}\n')

        if save_eval_output:
            self._save_eval_output(runner, eval_outputs)

    def _eval_language(self,
                       runner,
                       model,
                       device,
                       max_new_tokens=None,
                       save_eval_output=False):
        if save_eval_output:
            eval_outputs = []

        for sample_input in self.evaluation_inputs:
            inputs = (self.system + self.instruction).format(
                input=sample_input, round=1, **runner.cfg)
            input_ids = self.tokenizer.encode(inputs, return_tensors='pt')
            input_ids = input_ids.to(device)
            generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                generation_config=self.gen_config,
                stopping_criteria=self.stop_criteria)
            generation_output = self.tokenizer.decode(generation_output[0])
            runner.logger.info(f'Sample output:\n{generation_output}\n')
            if save_eval_output:
                eval_outputs.append(f'{generation_output}\n')

        if save_eval_output:
            self._save_eval_output(runner, eval_outputs)

    def _generate_samples(self,
                          runner,
                          max_new_tokens=None,
                          save_eval_output=False):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(iter(model.parameters())).device

        if self.is_first_run:
            # hardcode for qlora DeepSpeed ZeRO3, put buffers and QuantState to
            # device
            model.to(device)
            self.is_first_run = False

        is_checkpointing = model.llm.is_gradient_checkpointing
        use_cache = model.llm.config.use_cache

        # Cast to inference mode
        model.activation_checkpointing_disable()
        model.llm.config.use_cache = True
        model.eval()
        if self.evaluation_images is not None:
            self._eval_images(runner, model, device, max_new_tokens,
                              save_eval_output)
        else:
            self._eval_language(runner, model, device, max_new_tokens,
                                save_eval_output)

        # Cast to training mode
        if is_checkpointing:
            model.activation_checkpointing_enable()
        model.llm.config.use_cache = use_cache
        model.train()

    def before_train(self, runner):
        runner.logger.info('before_train in EvaluateChatHook.')
        self._generate_samples(runner, max_new_tokens=50)

    def _is_save_checkpoint(self, runner):
        hooks = runner.hooks
        checkpoint_hook = None
        for hook in hooks:
            if type(hook).__name__ == 'CheckpointHook':
                checkpoint_hook = hook
                break
        if checkpoint_hook is None or checkpoint_hook.by_epoch:
            return False

        if checkpoint_hook.every_n_train_iters(
            runner, checkpoint_hook.interval, checkpoint_hook.save_begin) or \
                (checkpoint_hook.save_last and
                 checkpoint_hook.is_last_train_iter(runner)):
            return True

        return False

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        if self.every_n_iters is None:
            return

        save_eval_output = self._is_save_checkpoint(runner)

        do_chat = (
            save_eval_output
            or self.every_n_train_iters(runner, self.every_n_iters))
        if not do_chat:
            return

        runner.logger.info('after_train_iter in EvaluateChatHook.')
        self._generate_samples(runner, save_eval_output=save_eval_output)

    def after_train(self, runner):
        runner.logger.info('after_train in EvaluateChatHook.')
        self._generate_samples(runner)

    def after_val(self, runner) -> None:
        if self.every_n_iters is not None:
            return
        runner.logger.info('after_val in EvaluateChatHook.')
        self._generate_samples(runner)