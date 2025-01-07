from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from mmdet.models import DiceLoss, CrossEntropyLoss

from projects.ST.models.sa2va_ST import Sa2VASTModel
from projects.ST.models.models_modeling_qwen2mm_mmrope import Qwen2MMmropeForCausalLM
import torch
from projects.ST.dataset.vqa_dataset import LLaVADataset
from projects.ST.dataset.RefCOCO_Dataset import ReferSegmDataset
from projects.ST.dataset.collect_fns import st_collate_fn
from projects.ST.hooks.evaluation_chat_hook import EvaluateChatHook_ST

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
tokenizer_path = './pretrained/single_transformer/capcls1.0_1024M_imgfull_withpt_lr5e-4-0_rp0.1_iter62500_hf/'
path = './pretrained/single_transformer/SFT-Qwen2.5-0.5B-capcls1.0_1024M_iter_62500_lr5e-4_0_rp0.1_hf_llava/'
pretrained_pth = None

# Data
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

vision_patch_size = 16

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 32
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 5000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ['[SEG]',]

evaluation_freq = 500
evaluation_images = './projects/omg_llava/test.jpg'
evaluation_inputs = ['Please describe this picture']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=tokenizer_path,
    trust_remote_code=True,
    padding_side='right')

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=Sa2VASTModel,
    single_transformer=dict(
        type=Qwen2MMmropeForCausalLM.from_pretrained,
        pretrained_model_name_or_path=path,
        torch_dtype=torch.bfloat16,
        use_cache=False, attn_implementation="sdpa"
    ),
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    seg_hidden_states=256,
    patch_size=vision_patch_size,
    seg_pred_down_ratio=4,
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    torch_dtype=torch.bfloat16,
    pretrained_pth=None,
    loss_sample_points=True,
    num_points=12544,
    # for inference
    template=prompt_template,
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

################## image chat
llava_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='data/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    image_folder='data/llava_data/llava_images/',
    max_length=max_length,
    patch_size=vision_patch_size,
    add_cls=False,
)

refcoco_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    data_root='data/ref_seg/refcoco',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    patch_size=vision_patch_size,
    add_cls=False,
)
refcoco_plus_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    data_root='data/ref_seg/refcoco+',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    patch_size=vision_patch_size,
    add_cls=False,
)
refcocog_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    data_root='data/ref_seg/refcocog',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(umd).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    patch_size=vision_patch_size,
    add_cls=False,
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        # ref seg
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # image qa
        llava_vqa_dataset,
    ]
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=st_collate_fn)
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
    dtype='bfloat16'
)

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
custom_hooks = [
    dict(
        type=EvaluateChatHook_ST,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system='',)
]

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
        save_optimizer=False,
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
