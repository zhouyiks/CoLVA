# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)

from vlm.engine.runner.loops import AnnoLoop
from xtuner.dataset import ConcatDataset

from projects.mllm_labeling.models import MLLM_Annotor
from projects.mllm_labeling.datasets.demo_dataset import DemoImageCap

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

llm_name_or_path = './pretrained/internvl/InternVL2-Llama3-76B-AWQ/'  # Please change to your own path
save_folder = './1215_demos/mllm_object_cap/'

model = dict(
    type=MLLM_Annotor,
    model=llm_name_or_path,
    save_folder=save_folder,
)

test_dataset = [dict(
    type=DemoImageCap,
    image_folder='./1215_demos/mask_outs/out/',
    bs=8,
)]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=test_dataset),
)

test_evaluator = dict()
test_cfg = dict(type=AnnoLoop, select_metric='first')

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
