from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.dataset import DefaultSampler
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, CLIPImageProcessor


from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory

from mmdet.models import DiceLoss, CrossEntropyLoss
from mmdet.datasets.samplers import MultiDataSampler
from peft import LoraConfig

from projects.lisa.models.internvl import InternVL
from projects.lisa.datasets.sem_seg_dataset import ADE20kSemanticSegDataset, COCOStuffSemanticSegDataset,  \
    PascalPartSemanticSegDataset, PacoSemanticSegDataset, MapillarySemanticSegDataset


from projects.lisa.models.lisa import LisaModel
from projects.lisa.datasets.sampler import MultiDataPseudoSampler, MultiDataSameBatchSampler
from projects.glamm.datasets import glamm_collate_fn

from third_parts.segment_anything import build_sam_vit_h
from third_parts.segment_anything.utils.transforms import ResizeLongestSide


from projects.lisa.datasets.vqa_dataset import LLaVADataset
from projects.llava_sam2.datasets import ReferSegmDataset

from vlm.datasets.evaluation import MMEDataset, MultipleChoiceDataset, POPEDataset,\
    HallusionDataset, TextVQADataset, GQADataset,\
    VQAv2Dataset, ChartQADataset, GeneralVQADataset, RESDataset
from xtuner.dataset import ConcatDataset
from vlm.engine.runner.loops import TestLoop
from mmengine.dataset import DefaultSampler

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
accumulative_counts = 10
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 3e-4
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.03

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

# Save
save_steps = 1000
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

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
        checkpoint='./pretrained/video_lisa/sam_vit_h_4b8939.pth'),
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
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=['[SEG]'],
    data_path='projects/omg_llava/dataset/utils/ade20k_classes.json',
    image_folder='./data/ade20k/images/training/',
    extra_image_processor=extra_image_processor,
    max_length=max_length,
)
semantic_seg_cocostuff_dataset = dict(
    type=COCOStuffSemanticSegDataset,
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=['[SEG]'],
    data_path='projects/omg_llava/dataset/utils/cocostuff_classes.txt',
    image_folder='./data/coco_stuff/train2017/',
    extra_image_processor=extra_image_processor,
    max_length=max_length,
)

semantic_seg_pascal_part_dataset = dict(
    type=PascalPartSemanticSegDataset,
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=['[SEG]'],
    data_path='data/pascal_part/train.json',
    image_folder='data/pascal_part/VOCdevkit/VOC2010/JPEGImages/',
    extra_image_processor=extra_image_processor,
    max_length=max_length,
)

semantic_seg_paco_lvis_dataset = dict(
    type=PacoSemanticSegDataset,
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=['[SEG]'],
    data_path='data/paco/annotations/paco_lvis_v1_train.json',
    image_folder='data/coco/',
    max_length=max_length,
    extra_image_processor=extra_image_processor,
)

semantic_seg_mapillary_dataset = dict(
    type=MapillarySemanticSegDataset,
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    special_tokens=['[SEG]'],
    image_folder='data/mapillary/training/images/',
    data_path='data/mapillary/config_v2.0.json',
    max_length=max_length,
    extra_image_processor=extra_image_processor,
)

refcoco_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=['[SEG]'],
    extra_image_processor=extra_image_processor,
    data_root='data/coco/',
    data_prefix=dict(img_path='train2014/'),
    ann_file='refcoco/instances.json',
    split_file='refcoco/refs(unc).p',
    prompt_template=prompt_template,
    max_length=max_length
)
refcoco_plus_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=['[SEG]'],
    extra_image_processor=extra_image_processor,
    data_root='data/coco/',
    data_prefix=dict(img_path='train2014/'),
    ann_file='refcoco+/instances.json',
    split_file='refcoco+/refs(unc).p',
    prompt_template=prompt_template,
    max_length=max_length
)
refcocog_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=['[SEG]'],
    extra_image_processor=extra_image_processor,
    data_root='data/coco/',
    data_prefix=dict(img_path='train2014/'),
    ann_file='refcocog/instances.json',
    split_file='refcocog/refs(umd).p',
    prompt_template=prompt_template,
    max_length=max_length
)

vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path='data/llava_data/LLaVA-Instruct-150K/llava_instruct_150k.json',
    prompt_template=prompt_template,
    special_tokens=['[SEG]'],
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


test_dataset = [
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/mmbench/MMBench_DEV_EN.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    #     metainfo=dict(
    #         template=prompt_template,
    #     ),
    #     ori_image=True,
    # ),
    # dict(
    #     type=MMEDataset,
    #     data_file='./data/eval/mme/MME.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    #     metainfo=dict(
    #         template=prompt_template,
    #     )
    # ),
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/seed_bench/SEEDBench_IMG.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    #     metainfo=dict(
    #         template=prompt_template,
    #     ),
    # ori_image=True,
    # ),
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/sqa/ScienceQA_TEST.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    #     metainfo=dict(
    #         template=prompt_template,
    #     ),
    #     ori_image=True,
    # ),
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/ai2d/AI2D_TEST.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    #     metainfo=dict(
    #         template=prompt_template,
    #     ),
    #     ori_image=True,
    # ),
    # dict(
    #     type=MultipleChoiceDataset,
    #     data_file='./data/eval/mmstar/MMStar.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    #     metainfo=dict(
    #         template=prompt_template,
    #     ),
    #     ori_image=True,
    # ),
    # dict(
    #     type=POPEDataset,
    #     data_file=[
    #         './data/eval/pope/coco_pope_adversarial.json',
    #         './data/eval/pope/coco_pope_popular.json',
    #         './data/eval/pope/coco_pope_random.json',
    #     ],
    #     coco_val_path='./data/eval/val2014/',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    #     metainfo=dict(
    #         template=prompt_template,
    #     )
    # ),
    dict(
        type=RESDataset,
        dataset_name='refcoco',
        image_folder='./data/glamm_data/images/coco2014/train2014/',
        image_processor=image_processor,
        data_path="./data/ref_seg/",
        pad_image_to_square=True,
        split='val',
        metainfo=dict(
            template=prompt_template,
        ),
        ori_image=True,
    ),
    # dict(
    #     type=RESDataset,
    #     dataset_name='refcoco_plus',
    #     image_folder='./data/glamm_data/images/coco2014/train2014/',
    #     image_processor=image_processor,
    #     data_path="./data/ref_seg/",
    #     pad_image_to_square=True,
    #     split='val',
    #     metainfo=dict(
    #         template=prompt_template,
    #     ),
    #     ori_image=True,
    # ),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=test_dataset),
)
test_evaluator = dict()
test_cfg = dict(type=TestLoop, select_metric='first')

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
