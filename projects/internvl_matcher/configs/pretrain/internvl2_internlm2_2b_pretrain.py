import torch
from torch.optim import AdamW
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)

from peft import LoraConfig
from transformers import AutoModel, AutoTokenizer

from xtuner.dataset import InternVL_V1_5_Dataset
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.engine.runner import TrainLoop
from xtuner.engine.hooks import DatasetInfoHook

from projects.internvl_matcher.model import WrapInternVL
from projects.internvl_matcher.dataset import InternVLDataset
from projects.internvl_matcher.dataset.process_functions import (
    osprey_region_caption_map_fn, osprey_region_conversation_map_fn,
    RegionCaptionDataset_load_fn, RegionConversationDataset_load_fn)
from projects.internvl_matcher.dataset.collect_fns import internvl_collate_fn


#########################################################################
#                             PART 1  Settings                          #
#########################################################################

# Model
mllm_name_or_path = "./pretrained/internvl_matcher/InternVL2-2B"

# Data
data_root = './data/llava_data/'
data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
image_folder = data_root + 'llava_images'

glamm_data_root = "./data/glamm_data/"

region_cap_osprey_image_path = glamm_data_root + 'images/coco2014/train2014/'
region_cap_osprey_data_path = "./data/region_caption/osprey/osprey_detail_description.json"


# prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 8192
lazy_load = True

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
lr = 1e-6
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1 # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=mllm_name_or_path,
    trust_remote_code=True,
    padding_side='right',
)

model = dict(
    type=WrapInternVL,
    freeze_llm=True,
    freeze_visual_encoder=True,
    freeze_connector=True,
    unfreeze_lm_head=True,
    quantization_vit=False,
    quantization_llm=False,
    use_activation_checkpointing=True,
    mllm=dict(
        type=AutoModel.from_pretrained,
        pretrained_model_name_or_path=mllm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ),
    llm_lora=dict(
        type=LoraConfig,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=None,
        task_type='CAUSAL_LM'
    ),
    # visual_encoder_lora=dict(
    #     type=LoraConfig,
    #     r=64,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']
    # )
    tokenizer=tokenizer,
)



#########################################################################
#                    PART 3  Dataset & DataLoader                       #
#########################################################################
region_cap_osprey_dataset = dict(
    type=InternVLDataset,
    model_path=mllm_name_or_path,
    data_path=region_cap_osprey_data_path,
    image_folder=region_cap_osprey_image_path,
    dataset_map_fn=osprey_region_caption_map_fn,
    annotation_load_fn=RegionCaptionDataset_load_fn,
    dynamic_image_size=True,
    pad_image_to_square=False,
    repeat_time=1,
    max_length=max_length,
    lazy_load=True,
    group_by_length=True,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=region_cap_osprey_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=internvl_collate_fn)
)

# llava_dataset = dict(
#     type=InternVL_V1_5_Dataset,
#     model_path=mllm_name_or_path,
#     data_path=data_path,
#     image_folder=image_folder,
#     template=prompt_template,
#     max_length=max_length,
# )

# train_dataloader = dict(
#     batch_size=batch_size,
#     num_workers=dataloader_num_workers,
#     dataset=llava_dataset,
#     sampler=dict(
#         type=LengthGroupedSampler,
#         length_property='modality_length',
#         per_device_batch_size=batch_size * accumulative_counts
#     ),
#     collate_fn=dict(type=default_collate_fn)
# )


#########################################################################
#                    PART 4  Scheduler & Optimizer                      #
#########################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay
    ),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16'
)

# learning policy
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True
    )
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)


#########################################################################
#                             PART 5  Runtime                           #
#########################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer)
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
        max_keep_ckpts=save_total_limit
    ),
    # set sampler seed in distributed environment,
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl')
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from  =None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
