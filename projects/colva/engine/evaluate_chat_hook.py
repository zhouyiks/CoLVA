from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, StopWordStoppingCriteria)
from xtuner.dataset.utils import load_image
from xtuner.engine.hooks import EvaluateChatHook

import warnings
import json
import copy
from distinctipy import distinctipy
from pycocotools import mask
from PIL import Image
import cv2
import numpy as np
from mmengine.utils.misc import get_object_from_string
from mmengine.model import is_model_wrapper
from transformers import GenerationConfig, StoppingCriteriaList
from transformers import AutoConfig, AutoTokenizer
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from ..dataset.process_functions import dynamic_preprocess
from ..dataset.utils import VPT_CONTEXT_TOKEN, VPT_START_TOKEN, VPT_END_TOKEN
from ..dataset.process_functions import contour_rendering


class EvaluateChatHook_withSpecialTokens(EvaluateChatHook):

    priority = 'LOW'
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 tokenizer,
                 evaluation_inputs,
                 evaluation_images=None,
                 evaluation_vprompts=None,
                 image_tokenize_config=None,
                 image_processor=None,
                 system='',
                 prompt_template=None,
                 every_n_iters=None,
                 max_new_tokens=600,
                 stop_word=None,
                 stop_words=[],
                 generation_kwargs={}):
        super().__init__(tokenizer, evaluation_inputs, evaluation_images, 
                         image_processor, system, prompt_template, every_n_iters, 
                         max_new_tokens, stop_word, stop_words, generation_kwargs)
        
        self.evaluation_inputs = evaluation_inputs
        if isinstance(self.evaluation_inputs, str):
            self.evaluation_inputs = [self.evaluation_inputs]
        self.evaluation_images = evaluation_images
        self.evaluation_merged_visual_prompts = evaluation_images
        if isinstance(self.evaluation_images, str):
            self.evaluation_images = [self.evaluation_images]
            self.evaluation_merged_visual_prompts = [self.evaluation_merged_visual_prompts]
        if self.evaluation_images is not None:
            assert len(self.evaluation_images) in [1, len(self.evaluation_inputs)]
            if len(self.evaluation_images) == 1:
                self.evaluation_images = [self.evaluation_images[0]] * len(
                    self.evaluation_inputs)
            self.evaluation_images = [
                load_image(img) for img in self.evaluation_images
            ]
            self.evaluation_merged_visual_prompts = [
                cv2.imread(img) for img in self.evaluation_merged_visual_prompts
            ]
        self.evaluation_vprompts = evaluation_vprompts
        if isinstance(self.evaluation_vprompts, str):
            self.evaluation_vprompts = [self.evaluation_vprompts]
        if self.evaluation_vprompts is not None:
            assert len(self.evaluation_vprompts) in [1, len(self.evaluation_inputs)]
            if len(self.evaluation_vprompts) == 1:
                self.evaluation_vprompts = [self.evaluation_vprompts[0]] * len(self.evaluation_inputs)

        self.min_dynamic_patch = image_tokenize_config.min_dynamic_patch
        self.max_dynamic_patch = image_tokenize_config.max_dynamic_patch
        self.image_size = image_tokenize_config.force_image_size
        self.use_thumbnail = image_tokenize_config.use_thumbnail

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        self.vprompt_transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST_EXACT),
        ])

        generation_config = dict(
            max_new_tokens=1024, do_sample=True,
        )
        self.generation_config = generation_config

        self.is_first_run = True

        self._add_special_tokens()
    
    def _add_special_tokens(self):
        special_tokens = [VPT_CONTEXT_TOKEN,]
        num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            if isinstance(object_mask, dict):
                if isinstance(object_mask["counts"], list):
                    # convert to compressed RLE
                    object_mask = mask.frPyObjects(object_mask, ori_height, ori_width)
                m = mask.decode(object_mask)
                m = m.astype(np.uint8).squeeze()
            elif object_mask:
                rles = mask.frPyObjects(object_mask, ori_height, ori_width)
                rle = mask.merge(rles)
                m = mask.decode(rle).astype(np.uint8).squeeze()
            else:
                m = np.zeros((ori_height, ori_width), dtype=np.uint8)
            binary_masks.append(m)
        masks = np.stack(binary_masks, axis=0)
        return masks

    def _eval_images(self, runner, model, device, max_new_tokens=None, save_eval_output=False):
        if save_eval_output:
            eval_outputs = []
        
        for idx, (sample_image, sample_vprompt, sample_input) in enumerate(
            zip(self.evaluation_images, self.evaluation_vprompts, self.evaluation_inputs)
            ):
            if isinstance(sample_input, str):
                sample_input = [sample_input]

            with open(sample_vprompt, 'r') as f:
                vprompt_data = json.load(f)

            ori_width, ori_height = sample_image.size

            annotations = []
            for anno in vprompt_data['objects']:
                annotation = dict()
                annotation['bbox'] = anno['bbox']
                annotation['segmentation'] = [np.array(anno['segmentation']).flatten().tolist()]
                annotations.append(annotation)
            segmentations = [anno['segmentation'] for anno in annotations]
            regions = self.decode_mask(segmentations, ori_height, ori_width)

            merged_visual_prompts = self.evaluation_merged_visual_prompts[idx]
            contour_rendering(merged_visual_prompts, regions)
            merged_visual_prompts = Image.fromarray(cv2.cvtColor(merged_visual_prompts, cv2.COLOR_BGR2RGB))
            # merged_visual_prompts.save(f'/mnt/bn/xiangtai-training-data/project/xiangtai-windows/internvl/internvl_debug_out/merged_vprompts_test.jpg')
            # exit(0)

            images, regions, merged_regions = dynamic_preprocess(
                sample_image, regions, merged_visual_prompts,
                min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                image_size=self.image_size, use_thumbnail=self.use_thumbnail)
            
            # Apply the transformation to each image and stack the results into a tensor
            pixel_values = [self.transform(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(model.model.vision_model.dtype).to("cuda")

            merged_visual_prompts = [self.transform(merged_region) for merged_region in merged_regions]
            merged_visual_prompts = torch.stack(merged_visual_prompts).to(model.model.vision_model.dtype).to("cuda")
            
            num_patches_list = [pixel_values.shape[0],]

            responses = model.batch_chat(
                pixel_values, sample_input, merged_visual_prompts,
                copy.deepcopy(self.generation_config), num_patches_list=num_patches_list,
            )

            runner.logger.info(f'Sample output:\n'
                               f'{sample_input[0] + responses[0]}\n')
            
            if save_eval_output:
                eval_outputs.append(f'{sample_input[0] + responses[0]}\n')
        
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
        
        is_checkpointing = model.model.language_model.is_gradient_checkpointing
        use_cache = model.model.language_model.config.use_cache

        # Cast to inference mode
        model.activation_checkpointing_disable()
        model.model.language_model.config.use_cache = True
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
        model.model.language_model.config.use_cache = use_cache
        model.train()