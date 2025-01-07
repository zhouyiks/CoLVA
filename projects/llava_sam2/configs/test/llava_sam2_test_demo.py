from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory

from mmdet.models import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.lisa.models.internvl import InternVL

from projects.llava_sam2.models import VideoLLaVASAMModel, SAM2
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

from xtuner.dataset import ConcatDataset
from vlm.engine.runner.loops import TestLoop
from mmengine.dataset import DefaultSampler
from transformers import CLIPImageProcessor
image_processor = dict(
    type=CLIPImageProcessor,
    do_resize=True,
    size=1024,
    resample=3,
    do_center_crop=True,
    crop_size=1024,
    do_rescale=True,
    do_normalize=True,
    image_mean=[0.4814, 0.4578, 0.4082],
    image_std=[0.2686, 0.2613, 0.2757],
    do_convert_rgb=True
)

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = './pretrained/video_lisa/internvl2_4b/'

# Data
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 8
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
lr = 1e-6
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=VideoLLaVASAMModel,
    special_tokens=['[SEG]', '<p>', '</p>'],
    mllm=dict(
        type=InternVL,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'),
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2,
    ),
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


############### video res
data_root_revos = './data/video_datas/revos/'
video_revos_image_folder = data_root_revos
video_revos_expression_file = data_root_revos + 'meta_expressions_train_.json'
video_revos_mask_file = data_root_revos + 'mask_dict.json'

data_root_mevis = './data/video_datas/mevis/train/'
video_mevis_image_folder = data_root_mevis + 'JPEGImages'
video_mevis_expression_file = data_root_mevis + 'meta_expressions.json'
video_mevis_mask_file = data_root_mevis + 'mask_dict.json'

data_root_refytvos = './data/video_datas/rvos/'
video_refytvos_image_folder = data_root_refytvos + 'train/JPEGImages/'
video_refytvos_expression_file = data_root_refytvos + 'meta_expressions/train/meta_expressions.json'
video_refytvos_mask_file = data_root_refytvos + 'mask_dict.pkl'


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
