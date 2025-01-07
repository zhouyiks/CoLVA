# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel)

from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
# from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.engine.runner import TrainLoop
from projects.llava.model.llava import LLaVAModel
from xtuner.utils import PROMPT_TEMPLATE

from vlm.datasets.evaluation import MMEDataset, MultipleChoiceDataset, POPEDataset,\
    HallusionDataset, TextVQADataset, GQADataset,\
    VQAv2Dataset, ChartQADataset, GeneralVQADataset
from xtuner.dataset import ConcatDataset
from vlm.engine.runner.loops import TestLoop
from mmengine.dataset import DefaultSampler

from vlm.datasets.llava_lazy import LLaVALazyDataset

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = './pretrained/llava/vicuna-7b-v1.5'
visual_encoder_name_or_path = 'openai/clip-vit-large-patch14-336'
# Specify the pretrained pth
pretrained_pth = None  # noqa: E501

# Data
data_root = './data/llava_data/'
data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
image_folder = data_root + 'llava_images'
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = int(2048 - (336 / 14)**2)

# Scheduler & Optimizer
batch_size = 16  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''
evaluation_images = 'https://llava-vl.github.io/static/images/view.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture']

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

model = dict(
    type=LLaVAModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    pretrained_pth=pretrained_pth,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    llm_lora=dict(
        type=LoraConfig,
        r=512,
        lora_alpha=256,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path),
    visual_encoder_lora=dict(
        type=LoraConfig, r=64, lora_alpha=16, lora_dropout=0.05, bias='none'),
    tokenizer=tokenizer,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=LLaVALazyDataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=llava_dataset,
    # sampler=dict(
    #     type=LengthGroupedSampler,
    #     length_property='modality_length',
    #     per_device_batch_size=batch_size * accumulative_counts),
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

test_dataset = [
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/mmbench/MMBench_DEV_EN.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/mmbench/MMBench_TEST_EN.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    # dict(
    #     type=MMEDataset,
    #     data_file='./data/eval/mme/MME.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/seed_bench/SEEDBench_IMG.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/sqa/ScienceQA_VAL.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/sqa/ScienceQA_TEST.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/ai2d/AI2D_TEST.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/mmstar/MMStar.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    # dict(
    #     type=HallusionDataset,
    #     data_file='./data/eval/HallusionBench/HallusionBench.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    dict(
        type=POPEDataset,
        data_file=[
            './data/eval/pope/coco_pope_adversarial.json',
            './data/eval/pope/coco_pope_popular.json',
            './data/eval/pope/coco_pope_random.json',
        ],
        coco_val_path='./data/eval/val2014/',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=test_dataset),
)
test_evaluator = dict()
test_cfg = dict(type=TestLoop, select_metric='first')

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = []

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
