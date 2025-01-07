import torch
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.dataset import DefaultSampler

from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPImageProcessor, CLIPVisionModel

from xtuner.engine.runner import TrainLoop
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.utils.templates import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory, template_map_fn
from xtuner.dataset import ConcatDataset
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from mmdet.models import DiceLoss, CrossEntropyLoss

from projects.glamm.models.glamm import GLaMM
from projects.glamm.datasets import ReferSegmDataset
from projects.glamm.datasets import glamm_collate_fn

from third_parts.segment_anything import build_sam_vit_h
from third_parts.segment_anything.utils.transforms import ResizeLongestSide

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

# Scheduler & Optimizer
batch_size = 16  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)
max_length = int(2048 - (336 / 14)**2)
lazy = True

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''
evaluation_images = 'https://llava-vl.github.io/static/images/view.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture']

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
prompt_template = PROMPT_TEMPLATE.vicuna
pretrained_pth = 'MBZUAI/GLaMM-GranD-Pretrained'
pretrained_pth = None
llm_name_or_path = 'lmsys/vicuna-7b-v1.5'
visual_encoder_name_or_path = 'openai/clip-vit-large-patch14-336'

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)
extra_image_processor = dict(
    type=ResizeLongestSide,
    target_length=1024,
)

model = dict(
    type=GLaMM,
    tokenizer=tokenizer,
    freeze_llm=True,
    freeze_visual_encoder=True,
    pretrained_pth=pretrained_pth,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path),
    grounding_encoder=dict(
        type=build_sam_vit_h,
        checkpoint='checkpoints/sam_vit_h_4b8939.pth'),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=1.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=1.0),
)


#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

refcoco_segm_dataset = dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    image_processor=image_processor,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    data_root='data/coco/',
    data_prefix=dict(img_path='train2014/'),
    ann_file='refcoco/instances.json',
    split_file='refcoco/refs(unc).p',
)

refcocog_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    image_processor=image_processor,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    data_root='data/coco/',
    data_prefix=dict(img_path='train2014/'),
    ann_file='refcocog/instances.json',
    split_file='refcocog/refs(umd).p',
)
refcoco_plus_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    image_processor=image_processor,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    data_root='data/coco/',
    data_prefix=dict(img_path='train2014/'),
    ann_file='refcoco+/instances.json',
    split_file='refcoco+/refs(unc).p',
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        refcoco_segm_dataset, refcocog_segm_dataset,
        refcoco_plus_segm_dataset
    ])

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=glamm_collate_fn)
)

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
# custom_hooks = [
#     dict(type=DatasetInfoHook, tokenizer=tokenizer),
#     dict(
#         type=EvaluateChatHook,
#         tokenizer=tokenizer,
#         image_processor=image_processor,
#         every_n_iters=evaluation_freq,
#         evaluation_inputs=evaluation_inputs,
#         evaluation_images=evaluation_images,
#         system=SYSTEM,
#         prompt_template=prompt_template)
# ]

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
