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
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils.templates import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory, template_map_fn
from xtuner.dataset import ConcatDataset
from mmdet.models import DiceLoss, CrossEntropyLoss

from projects.glamm.models.glamm import GLaMM

from projects.glamm.datasets import ADE20kSemanticSegDataset, COCOStuffSemanticSegDataset, \
    PascalPartSemanticSegDataset, PacoSemanticSegDataset, GranDfGCGDataset, RefCOCOgGCGDataset, \
    OpenPsgGCGDataset, Flickr30kGCGDataset
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
evaluation_images = 'assets/view.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture']

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
prompt_template = PROMPT_TEMPLATE.vicuna
pretrained_model = 'MBZUAI/GLaMM-GranD-Pretrained'
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
    pretrained_pth=None,
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

semantic_seg_ade20k_dataset = dict(
    type=ADE20kSemanticSegDataset,
    data_path='projects/omg_llava/dataset/utils/ade20k_classes.json',
    image_folder='./data/ade20k/images/training/',
    tokenizer=tokenizer,
    image_processor=image_processor,
    extra_image_processor=extra_image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    lazy=lazy,
    repeats=1,
    gcg_format=False,
)

semantic_seg_cocostuff_dataset = dict(
    type=COCOStuffSemanticSegDataset,
    data_path='projects/omg_llava/dataset/utils/cocostuff_classes.txt',
    image_folder='./data/coco/train2017/',
    label_path='./data/coco_stuff/stuffthingmaps_trainval2017/train2017/',
    tokenizer=tokenizer,
    image_processor=image_processor,
    extra_image_processor=extra_image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    lazy=lazy,
    repeats=1,
    gcg_format=False,
)

semantic_seg_pascal_part_dataset = dict(
    type=PascalPartSemanticSegDataset,
    data_path='data/pascal_part/train.json',
    image_folder='data/pascal_part/VOCdevkit/VOC2010/JPEGImages/',
    image_processor=image_processor,
    tokenizer=tokenizer,
    max_length=max_length,
    pad_image_to_square=True,
    extra_image_processor=extra_image_processor,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    lazy=lazy,
    repeats=1,
    gcg_format=False,
    num_classes_per_sample=3,
)

semantic_seg_paco_lvis_dataset = dict(
    type=PacoSemanticSegDataset,
    data_path='data/paco/annotations/paco_lvis_v1_train.json',
    image_folder='data/coco/',
    image_processor=image_processor,
    tokenizer=tokenizer,
    max_length=max_length,
    pad_image_to_square=True,
    extra_image_processor=extra_image_processor,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    lazy=lazy,
    repeats=1,
    gcg_format=False,
    num_classes_per_sample=3,
)

grandf_gcg_dataset=dict(
    type=GranDfGCGDataset,
    image_folder='data/GranDf/GranDf_HA_images/train',
    image_processor=image_processor,
    data_path='./data/GranDf/annotations/train/GranDf_HA_GCG_train.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=2048,
    pad_image_to_square=True,
    repeats=1,
    num_classes_per_sample=3,
    extra_image_processor=extra_image_processor)

refcocog_gcg_dataset = dict(
    type=RefCOCOgGCGDataset,
    image_folder='data/coco/train2014/',
    image_processor=image_processor,
    data_path='./data/GranDf/annotations/train/RefCOCOg_GCG_train.json',
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=2048,
    pad_image_to_square=True,
    repeats=1,
    num_classes_per_sample=3,
    extra_image_processor=extra_image_processor)

openpsg_pcg_dataset=dict(
    type=OpenPsgGCGDataset,
    image_folder='data/coco/',
    image_processor=image_processor,
    data_path='./data/GranDf/annotations/train/OpenPsgGCG_train.json',
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=2048,
    pad_image_to_square=True,
    repeats=1,
    num_classes_per_sample=3,
    extra_image_processor=extra_image_processor)

flickr30k_gcg_dataset=dict(
    type=Flickr30kGCGDataset,
        image_folder='data/flickr30k/flickr30k-images/',
        image_processor=image_processor,
        data_path='./data/GranDf/annotations/train/flickr_mergedGT_GCG_train.json',
        tokenizer=tokenizer,
        template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template),
        max_length=2048,
        pad_image_to_square=True,
        repeats=1,
        num_classes_per_sample=3,
        extra_image_processor=extra_image_processor)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        semantic_seg_ade20k_dataset, semantic_seg_cocostuff_dataset,
        semantic_seg_pascal_part_dataset, semantic_seg_paco_lvis_dataset,
        grandf_gcg_dataset,refcocog_gcg_dataset, openpsg_pcg_dataset,flickr30k_gcg_dataset
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
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        image_processor=image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template)
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
