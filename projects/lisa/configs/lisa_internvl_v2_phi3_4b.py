from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.dataset import DefaultSampler
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, CLIPImageProcessor

from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory

from mmdet.models import DiceLoss, CrossEntropyLoss
from mmdet.datasets.samplers import MultiDataSampler
from peft import LoraConfig

from projects.lisa.models.internvl import InternVL
from projects.lisa.datasets.sem_seg_dataset import ADE20kSemanticSegDataset, COCOStuffSemanticSegDataset,  \
    PascalPartSemanticSegDataset, PacoSemanticSegDataset, MapillarySemanticSegDataset
from projects.lisa.datasets.vqa_dataset import LLaVADataset
from projects.lisa.datasets.refcoco_segm_dataset import ReferSegmDataset

from projects.lisa.models.lisa import LisaModel
from projects.lisa.datasets.sampler import MultiDataPseudoSampler, MultiDataSameBatchSampler
from projects.lisa.datasets.concat_dataset import ConcatDataset
from projects.glamm.datasets import glamm_collate_fn
from projects.lisa.processor.internvl_processor import InternVLProcessor
from third_parts.segment_anything import build_sam_vit_h
from third_parts.segment_anything.utils.transforms import ResizeLongestSide

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = 'OpenGVLab/InternVL2-4B'
llm_name_or_path = 'microsoft/Phi-3-mini-128k-instruct'
visual_encoder_name_or_path = 'OpenGVLab/InternViT-300M-448px'

# Data
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 10
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 3e-4
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 1000
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

processor = dict(
    type=InternVLProcessor,
    pretrained_model_name_or_path='OpenGVLab/InternVL2-4B'
)

extra_image_processor = dict(
    type=ResizeLongestSide,
    target_length=1024,
)
model = dict(
    type=LisaModel,
    mllm=dict(
        type=InternVL,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'),
    ),
    tokenizer=tokenizer,
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

semantic_seg_ade20k_dataset = dict(
    type=ADE20kSemanticSegDataset,
    data_path='projects/omg_llava/dataset/utils/ade20k_classes.json',
    image_folder='./data/ade20k/images/training/',
    processor=processor,
    extra_image_processor=extra_image_processor,
)
semantic_seg_cocostuff_dataset = dict(
    type=COCOStuffSemanticSegDataset,
    data_path='projects/omg_llava/dataset/utils/cocostuff_classes.txt',
    image_folder='./data/coco_stuff/train2017/',
    processor=processor,
    extra_image_processor=extra_image_processor,
)

semantic_seg_pascal_part_dataset = dict(
    type=PascalPartSemanticSegDataset,
    data_path='data/pascal_part/train.json',
    image_folder='data/pascal_part/VOCdevkit/VOC2010/JPEGImages/',
    processor=processor,
    extra_image_processor=extra_image_processor,
)

semantic_seg_paco_lvis_dataset = dict(
    type=PacoSemanticSegDataset,
    data_path='data/paco/annotations/paco_lvis_v1_train.json',
    image_folder='data/coco/',
    processor=processor,
    extra_image_processor=extra_image_processor,
)

semantic_seg_mapillary_dataset = dict(
    type=MapillarySemanticSegDataset,
    image_folder='data/mapillary/training/images/',
    data_path='data/mapillary/config_v2.0.json',
    processor=processor,
    extra_image_processor=extra_image_processor,
)

refcoco_segm_dataset=dict(
    type=ReferSegmDataset,
    processor=processor,
    extra_image_processor=extra_image_processor,
    data_root='data/coco/',
    data_prefix=dict(img_path='train2014/'),
    ann_file='refcoco/instances.json',
    split_file='refcoco/refs(unc).p',
)
refcoco_plus_segm_dataset=dict(
    type=ReferSegmDataset,
    processor=processor,
    extra_image_processor=extra_image_processor,
    data_root='data/coco/',
    data_prefix=dict(img_path='train2014/'),
    ann_file='refcoco+/instances.json',
    split_file='refcoco+/refs(unc).p',
)
refcocog_segm_dataset=dict(
    type=ReferSegmDataset,
    processor=processor,
    extra_image_processor=extra_image_processor,
    data_root='data/coco/',
    data_prefix=dict(img_path='train2014/'),
    ann_file='refcocog/instances.json',
    split_file='refcocog/refs(umd).p',
)

vqa_dataset = dict(
    type=LLaVADataset,
    processor=processor,
    data_path='data/llava_data/LLaVA-Instruct-150K/llava_instruct_150k.json',
    image_folder='data/coco/train2017/',
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        semantic_seg_ade20k_dataset, semantic_seg_cocostuff_dataset, semantic_seg_pascal_part_dataset,
        semantic_seg_paco_lvis_dataset, semantic_seg_mapillary_dataset, refcoco_segm_dataset,
        refcoco_plus_segm_dataset, refcocog_segm_dataset, vqa_dataset
    ]
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=train_dataset,
    sampler=dict(
        type=MultiDataPseudoSampler,
    ),
    batch_sampler=dict(
        type=MultiDataSameBatchSampler,
    ),
    collate_fn=dict(type=glamm_collate_fn))

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
# configure default hooks
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
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

log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

log_processor = dict(by_epoch=False)
