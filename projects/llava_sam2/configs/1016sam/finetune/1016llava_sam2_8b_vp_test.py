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

from projects.llava_sam2.models.internvl import InternVL_Slowfast

from projects.llava_sam2.models import VideoLLaVASAMModel, SAM2, VideoLLaVASAMModel_zero3
from projects.llava_sam2.datasets import VideoReVOSDataset, VideoMeVISDataset, VideoRefYoutubeVOSDataset, video_lisa_collate_fn, VideoSAM2Dataset
from projects.video_lisa.datasets import VideoChatUniViDataset
from projects.llava_sam2.datasets import RefCOCOgGCGDataset, OpenPsgGCGDataset, FlickrGCGDataset, GranDfGCGDataset, OspreyDataset, OspreyDescriptionDataset, OspreyShortDescriptionDataset
from projects.llava_sam2.datasets import LLaVADataset
from projects.llava_sam2.datasets import ReferSegmDataset
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = './pretrained/internvl/InternVL2-8B/'
pretrained_pth = None

# Data
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 4
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
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '/vp']

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
    type=VideoLLaVASAMModel_zero3,
    special_tokens=special_tokens,
    frozen_sam2_decoder=False,
    template=prompt_template,
    mllm=dict(
        type=InternVL_Slowfast,
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
        special_tokens=special_tokens,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2,
    ),
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
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    # loss_sample_points=False,
    bs=batch_size,
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

video_revos_dataset = dict(
    type=VideoReVOSDataset,
    image_folder=video_revos_image_folder,
    expression_file=video_revos_expression_file,
    mask_file=video_revos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=10,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

video_mevis_dataset = dict(
    type=VideoMeVISDataset,
    image_folder=video_mevis_image_folder,
    expression_file=video_mevis_expression_file,
    mask_file=video_mevis_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

video_refytvos_dataset = dict(
    type=VideoRefYoutubeVOSDataset,
    image_folder=video_refytvos_image_folder,
    expression_file=video_refytvos_expression_file,
    mask_file=video_refytvos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

################### Video chat
data_root_video_chatunivi = '/mnt/bn/xiangtai-training-data-video/dataset/video_vlm/video_chat/'
video_chatunivi_image_folder = data_root_video_chatunivi + 'Activity_Videos/'
video_chatunivi_json_file = data_root_video_chatunivi+ 'video_chat.json'

video_qa_dataset = dict(
    type=VideoChatUniViDataset,
    image_folder=video_chatunivi_image_folder,
    json_file=video_chatunivi_json_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

################## image chat
llava_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='data/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    image_folder='data/llava_data/llava_images/',
)

################## image res
# semantic_seg_ade20k_dataset = dict(
#     type=ADE20kSemanticSegDataset,
#     tokenizer=tokenizer,
#     prompt_template=prompt_template,
#     special_tokens=['[SEG]'],
#     data_path='projects/omg_llava/dataset/utils/ade20k_classes.json',
#     image_folder='./data/semantic_seg/ADEChallengeData2016/images/training/',
#     extra_image_processor=extra_image_processor,
#     max_length=max_length,
# )

refcoco_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='data/ref_seg/refcoco',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
refcoco_plus_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='data/ref_seg/refcoco+',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)
refcocog_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='data/ref_seg/refcocog',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(umd).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

# image gcg datas
glamm_data_root = './data/glamm_data/'

refcocog_image_path = glamm_data_root + 'images/coco2014/train2014/'
refcocog_ann_file = glamm_data_root + 'annotations/RefCOCOg_GCG_train.json'

grandf_image_path = glamm_data_root + 'images/grandf/train/'
grandf_ann_file = glamm_data_root + 'annotations/GranDf_HA_GCG_train.json'

flickr_image_path = glamm_data_root + 'images/flickr30k/Flickr30K/'
flickr_ann_file = glamm_data_root + 'annotations/flickr_mergedGT_GCG_train.json'

psg_image_path = glamm_data_root + 'images/coco2017/'
psg_ann_file = glamm_data_root + 'annotations/OpenPsgGCG_train.json'

glamm_refcocog_dataset = dict(
    type=RefCOCOgGCGDataset,
    image_folder=refcocog_image_path,
    data_path=refcocog_ann_file,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

glamm_grandf_dataset = dict(
    type=GranDfGCGDataset,
    data_path=grandf_ann_file,
    image_folder=grandf_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=10,
)

glamm_psg_dataset = dict(
    type=OpenPsgGCGDataset,
    data_path=psg_ann_file,
    image_folder=psg_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

glamm_flickr_dataset = dict(
    type=FlickrGCGDataset,
    data_path=flickr_ann_file,
    image_folder=flickr_image_path,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
)

# sam2 data
data_sam2_folder = '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/'
data_sam2_expression_file = './whole_pesudo_cap/sam_v_final_v2.json'
video_sam2_dataset = dict(
    type=VideoSAM2Dataset,
    sam2_folder=data_sam2_folder,
    expression_file=data_sam2_expression_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=2,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    select_number=5,
)

# osprey
data_osprey_file = '/mnt/bn/xiangtai-training-data-video/dataset/osprey-724k/Osprey-724K/osprey_conversation.json'
data_osprey_image_folders = [
    '/mnt/bn/xiangtai-training-data/project/xiangtai-windows/tt_vlm/data/glamm_data/images/coco2014/train2014/',
    '/mnt/bn/xiangtai-training-data/datasets/coco/val2014/',
    '/mnt/bn/xiangtai-training-data/datasets/coco/train2017/',
    '/mnt/bn/xiangtai-training-data/datasets/coco/val2017/',
]

image_osprey_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_detail_description_file = '/mnt/bn/xiangtai-training-data-video/dataset/osprey-724k/Osprey-724K/osprey_detail_description.json'
image_osprey_description_dataset = dict(
    type=OspreyDescriptionDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_detail_description_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_short_file = '/mnt/bn/xiangtai-training-data-video/dataset/osprey-724k/Osprey-724K/osprey_short_form.json'
image_osprey_short_dataset = dict(
    type=OspreyShortDescriptionDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_short_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_part_file = '/mnt/bn/xiangtai-training-data-video/dataset/osprey-724k/Osprey-724K/osprey_part_level.json'
image_osprey_part_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_part_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

data_osprey_positive_neg_file = '/mnt/bn/xiangtai-training-data-video/dataset/osprey-724k/Osprey-724K/osprey_lvis_positive_negative.json'
image_osprey_positive_neg_dataset = dict(
    type=OspreyDataset,
    image_folder=data_osprey_image_folders,
    data_path=data_osprey_positive_neg_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        # sem seg
        # semantic_seg_ade20k_dataset,
        # ref seg
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # # image qa
        # llava_vqa_dataset,
        # # video res
        # video_mevis_dataset, video_revos_dataset, video_refytvos_dataset,
        # # video chat
        # video_qa_dataset,
        # # sam2 pesudo
        # # video_sam2_dataset,
        # # gcg data
        # glamm_psg_dataset,
        # glamm_grandf_dataset,
        # glamm_flickr_dataset,
        # glamm_refcocog_dataset,
        # visual prompt
        image_osprey_dataset, image_osprey_description_dataset,
        image_osprey_part_dataset, image_osprey_short_dataset,
        image_osprey_positive_neg_dataset,
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
    collate_fn=dict(type=video_lisa_collate_fn)
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
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
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
