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
from projects.colva.dataset import Qwen2VLDataset, Qwen2VLCombineDataset, SA1BPseudoVideoDataset
from projects.colva.dataset.collect_fns import qwen2vl_collate_fn

from projects.colva.engine import (
    DatasetInfoHook_withSpecialTokens, EvaluateChatHook_withSpecialTokens)
from projects.colva.optim import AmpOptimWrapper

from projects.colva.dataset.map_fns import (
    llava_map_fn, match_reasoning_map_fn, match_choice_only_map_fn, match_reasoning_map_fn_roi
)

from projects.colva.dataset.process_functions import (
    LLaVAInstructDataset_load_fn, MatchDataset_load_fn,
)

#########################################################################
#                             PART 1  Settings                          #
#########################################################################

# Model
mllm_name_or_path = "./Qwen/Qwen2-VL-7B-Instruct"
radio_name_or_path = "./nvidia/RADIO"
radio_adapter_weight = "./zhouyik/colva_ablation/qwen2vl_7b_radio_align_pretrain/iter_23584.pth"


# Data
data_root = './data/'
llava_data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
llava_image_path = data_root + 'images/llava_images/'

shareGPT4o_data_path = data_root + 'shareGPT4o/gpt-4o.jsonl'
shareGPT4o_image_path = data_root + 'shareGPT4o/image/'

match_choice_image_path = ""
match_choice_data_path = "./data/cross_image_reasoning/mllm_match_choices_sft.json"

match_reasoning_image_path = ""
match_reasoning_data_path = "./data/cross_image_reasoning/mllm_match_reasoning_sft.json"

lazy_load = True


# Scheduler & Optimizer
batch_size = 2  # per_device
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
    freeze_ot_mlp=True,
    unfreeze_vocab=False,
    unfreeze_lm_head=False,
    use_activation_checkpointing=True,
    vocab_embeds_name="embed_tokens",
    lm_head_name="lm_head",
    contras_loss=False,
    use_object_tokens=True,
    object_tokenizer_pretrain=False,
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
    llm_lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1, 
        bias='none',
        target_modules=None,
        task_type='CAUSAL_LM'
    ),
    pretrained_pth=radio_adapter_weight,
)


#########################################################################
#                    PART 3  Dataset & DataLoader                       #
#########################################################################
llava_dataset = dict(
    type=Qwen2VLDataset,
    data_path=llava_data_path,
    image_folder=llava_image_path,
    dataset_map_fn=llava_map_fn,
    annotation_load_fn=LLaVAInstructDataset_load_fn,
    repeat_time=1,
    lazy_load=True,
)

shareGPT4o_dataset = dict(
    type=Qwen2VLDataset,
    data_path=shareGPT4o_data_path,
    image_folder=shareGPT4o_image_path,
    dataset_map_fn=llava_map_fn,
    annotation_load_fn=LLaVAInstructDataset_load_fn,
    repeat_time=1,
    lazy_load=True,
)

match_reasoning_dataset = dict(
    type=Qwen2VLDataset,
    data_path=match_reasoning_data_path,
    image_folder=match_reasoning_image_path,
    dataset_map_fn=match_reasoning_map_fn_roi,
    annotation_load_fn=MatchDataset_load_fn,
    repeat_time=1,
    lazy_load=True,
)

match_choice_dataset = dict(
    type=Qwen2VLDataset,
    data_path=match_choice_data_path,
    image_folder=match_choice_image_path,
    dataset_map_fn=match_choice_only_map_fn,
    annotation_load_fn=MatchDataset_load_fn,
    repeat_time=1,
    lazy_load=True,
)


train_dataset = dict(
    type=Qwen2VLCombineDataset,
    datasets_cfgs=[
        llava_dataset,
        shareGPT4o_dataset,
        match_reasoning_dataset,
        match_choice_dataset,
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
