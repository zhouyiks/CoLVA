import torch
from torch.optim import AdamW
from mmengine.optim import CosineAnnealingLR, LinearLR #, AmpOptimWrapper
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)

from peft import LoraConfig
from transformers import (AutoModel, AutoTokenizer, AutoImageProcessor, CLIPImageProcessor,
                          Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM,
                          AutoModelForVision2Seq)

from xtuner.dataset import InternVL_V1_5_Dataset
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.engine.runner import TrainLoop
from xtuner.engine.hooks import DatasetInfoHook

from projects.colva.model import WrapQwen2VL
from projects.colva.dataset import Qwen2VLSA1BPseudoVideoDataset, Qwen2VLCombineDataset, SA1BPseudoVideoDataset
from projects.colva.dataset.collect_fns import qwen2vl_collate_fn

from projects.colva.engine import (
    DatasetInfoHook_withSpecialTokens, EvaluateChatHook_withSpecialTokens)
from projects.colva.optim import AmpOptimWrapper


#########################################################################
#                             PART 1  Settings                          #
#########################################################################

# Model
mllm_name_or_path = "./Qwen/Qwen2-VL-7B-Instruct"
radio_name_or_path = "./nvidia/RADIO"
# radio_adapter_weight = "./radio_adapter_pretrain_weight/internvl2_4b_iter_11792.pth"


# Data
data_path = './data/masa_sam_500k/sa1b_coco_fmt_iminfo_500k_redirected.json'
image_path = './data/masa_sam_500k'

lazy_load = True


# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 8
dataloader_num_workers = 1
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1 # grad clip
warmup_ratio = 0.03

# Save
save_steps = 1000
save_total_limit = 5  # Maximum checkpoints to keep (-1 means unlimited)


model = dict(
    type=WrapQwen2VL,
    freeze_llm=True,
    freeze_visual_encoder=True,
    freeze_connector=True,
    freeze_ot_mlp=False,
    unfreeze_vocab=False,
    unfreeze_lm_head=False,
    use_activation_checkpointing=True,
    vocab_embeds_name="embed_tokens",
    lm_head_name="lm_head",
    contras_loss=False,
    use_object_tokens=True,
    object_tokenizer_pretrain=True,
    mllm=dict(
        type=Qwen2VLForConditionalGeneration.from_pretrained,
        pretrained_model_name_or_path=mllm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ),
    object_tokenizer=dict(
        type=AutoModel.from_pretrained,
        pretrained_model_name_or_path=radio_name_or_path,
        trust_remote_code=True,
    ),
    pretrained_pth=None,
)


#########################################################################
#                    PART 3  Dataset & DataLoader                       #
#########################################################################
sa1b_dataset = dict(
    type=Qwen2VLSA1BPseudoVideoDataset,
    model_path=mllm_name_or_path,
    data_path=data_path,
    image_folder=image_path,
    dynamic_image_size=True,
    pad_image_to_square=False,
    repeat_time=1,
)


train_dataset = dict(
    type=Qwen2VLCombineDataset,
    datasets_cfgs=[
        sa1b_dataset,
    ],
    repeat_time=1,
    ot_image_processor=dict(
        type=CLIPImageProcessor.from_pretrained,
        pretrained_model_name_or_path=radio_name_or_path,
        trust_remote_code=True,
    ),
    qwen2_processor=dict(
        type=AutoProcessor.from_pretrained,
        pretrained_model_name_or_path=mllm_name_or_path,
        trust_remote_code=True,
    ),
    exhibit_special_tokens=False,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=qwen2vl_collate_fn)
)


#########################################################################
#                    PART 4  Scheduler & Optimizer                      #
#########################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay,
    ),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype=torch.bfloat16,
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
    # dict(type=DatasetInfoHook_withSpecialTokens, tokenizer=tokenizer),
    # dict(type=EvaluateChatHook_withSpecialTokens, 
    #      tokenizer=tokenizer,
    #      evaluation_inputs=evaluation_inputs,
    #      evaluation_images=evaluation_images,
    #      evaluation_vprompts=evaluation_vprompts,
    #      every_n_iters=1000,
    #      image_tokenize_config=dict(
    #          min_dynamic_patch=1,
    #          max_dynamic_patch=12,
    #          force_image_size=448,
    #          use_thumbnail=True,
    #      ),
    # ),
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
