# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from vlm.engine.runner.loops import TestLoop
from xtuner.dataset import ConcatDataset

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel)


from projects.tap.datasets.llava_dataset import LLaVAPretrainDataset
from projects.tap.model.tap import TAP

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

llm_name_or_path = './pretrained/omg_llava/internlm2-chat-7b'  # Please change to your own path
_rank = 1

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

# Data paths
data_root = './data/llava_data/'
image_folder = data_root + 'LLaVA-Pretrain/images'

model = dict(
    type=TAP,
    model_type="tap_vit_l",
    checkpoint="./pretrained/tokenize-anything/models/tap_vit_l_v1_1.pkl",
    concept_weights="./pretrained/tokenize-anything/concepts/merged_2560.pkl",
    tokenizer=tokenizer,
    save_folder='./work_dirs/tap_caption_results/rank_{}'.format(_rank),
)

test_dataset = [dict(
    type=LLaVAPretrainDataset,
    image_folder=image_folder,
    split=8,
    rank=_rank,
)]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=test_dataset),
)
test_evaluator = dict()
test_cfg = dict(type=TestLoop, select_metric='first')

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
