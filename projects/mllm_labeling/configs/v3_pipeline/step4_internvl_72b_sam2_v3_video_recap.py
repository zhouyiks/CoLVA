# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)

from vlm.engine.runner.loops import AnnoLoop
from xtuner.dataset import ConcatDataset

from projects.mllm_labeling.models import MLLM_Annotor
from projects.mllm_labeling.datasets.sam2_dataset_v3_whole import SAM2DatasetV3_Videocap_whole

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

llm_name_or_path = './pretrained/internvl/InternVL2-Llama3-76B-AWQ/'  # Please change to your own path
save_folder = './whole_pesudo_cap_v3/stpe4_video_recap/'

# Data paths
video_folder = [
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_000/sav_train/sav_000/',
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_001/sav_train/sav_001/',
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_002/sav_train/sav_002/',
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_003/sav_train/sav_003/',
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_004/sav_train/sav_004/',
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_005/sav_train/sav_005/',
]
json_folder = [
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_000/sav_train/sav_000/',
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_001/sav_train/sav_001/',
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_002/sav_train/sav_002/',
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_003/sav_train/sav_003/',
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_004/sav_train/sav_004/',
    '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_005/sav_train/sav_005/',
]
image_cap_folder = './whole_pesudo_cap_v3/stpe3_image_recap/'

model = dict(
    type=MLLM_Annotor,
    model=llm_name_or_path,
    save_folder=save_folder,
)

test_dataset = [dict(
    image_cap_folder=image_cap_folder,
    type=SAM2DatasetV3_Videocap_whole,
    video_folder=video_folder,
    json_folder=json_folder,
    bs=16,
    select_frames=8,
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
