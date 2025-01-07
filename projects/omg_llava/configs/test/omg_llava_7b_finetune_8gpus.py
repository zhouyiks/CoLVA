# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel)

from projects.omg_llava.dataset import LLaVADataset
from projects.omg_llava.dataset.collect_fns import omg_llava_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from projects.omg_llava.dataset import GranDfGCGDataset, FlickrGCGDataset, OpenPsgGCGDataset, RefCOCOgGCGDataset,\
    CombineDataset, glamm_refcocog_map_fn, glamm_openpsg_map_fn, glamm_flickr_map_fn, glamm_granf_map_fn,\
    ADE20kSemanticSegDataset, COCOStuffSemanticSegDataset, semantic_seg_map_fn, MapillarySemanticSegDataset,\
    PascalPartSemanticSegDataset, pascal_part_map_fn, PacoSemanticSegDataset,\
    RefcocoReferringSegDataset, referring_seg_map_fn, Refcoco_plus_ReferringSegDataset,\
    Refcocog_ReferringSegDataset, Refclef_ReferringSegDataset,\
    OspreyRegionCaptionDataset, osprey_region_caption_map_fn,\
    OspreyRegionConversationDataset, osprey_region_conversation_map_fn,\
    MDPVPointDetailedCaptionDataset, mdpv_points_map_fn, MDPVPointBriefCaptionDataset,\
    semantic_seg_gcg_format_map_fn, pascal_part_gcg_format_map_fn,\
    referring_seg_gcg_format_map_fn, osprey_region_caption_gcg_format_map_fn
from xtuner.dataset.samplers import LengthGroupedSampler
from projects.omg_llava.engine import DatasetInfoHook_withSpecoalTokens, EvaluateChatHook_withSpecialTokens
from xtuner.engine.runner import TrainLoop
from projects.omg_llava.model import OMG_LLaVA
from xtuner.utils import PROMPT_TEMPLATE
from projects.omg_llava.model import OpenCLIPBackbone_omgseg
from projects.omg_llava.model import OMGSegVisualEncoder, Mask2FormerVideoSemSamHead

from torch.nn import GroupNorm, ReLU

from mmdet.models import BatchFixedSizePad, CrossEntropyLoss, \
    DiceLoss, MaskFormerFusionHead, FocalLoss
from projects.omg_llava.model.omg_seg.msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder
from mmdet.models.task_modules.assigners import HungarianAssigner, CrossEntropyLossCost, DiceCost
from mmdet.models.task_modules.samplers import MaskPseudoSampler

from vlm.datasets.evaluation import MMEDataset, MultipleChoiceDataset, POPEDataset,\
    HallusionDataset, TextVQADataset, GQADataset,\
    VQAv2Dataset, ChartQADataset, GeneralVQADataset, RESDataset
from xtuner.dataset import ConcatDataset
from vlm.engine.runner.loops import TestLoop
from mmengine.dataset import DefaultSampler


#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# mode
lazy = False

# Model
llm_name_or_path = './pretrained/omg_llava/internlm2-chat-7b'  # Please change to your own path
omg_ov_class_embed_path = './pretrained/omg_llava/convnext_large_d_320_CocoPanopticOVDataset.pth' # Please change to your own path
omg_head_pretrain_pth_path = './pretrained/omg_llava/omg_seg_convl.pth'  # Please change to your own path

prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = int(2048 - (1024 / 64)**2 - 100)

# Data
data_root = './data/llava_data/'
data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
image_folder = data_root + 'llava_images'

# Scheduler & Optimizer
batch_size = 8  # per_device
accumulative_counts = 2
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03


# Save
save_steps = 2000
save_total_limit = 4  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 2000
SYSTEM = ''
evaluation_images = './projects/omg_llava/test.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture',
                     'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.']

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

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

class_embed = 'convnext_large_d_320_CocoPanopticOVDataset'
num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes

omgseg_model = dict(
    type=OMGSegVisualEncoder,
    data_preprocessor=None,
    pixel_shuffle_down_ratio=2,
    backbone=dict(
        type=OpenCLIPBackbone_omgseg,
        model_name='convnext_large_d_320',
        fix=True,
        init_cfg=dict(
            type='clip_pretrain',
            checkpoint='laion2b_s29b_b131k_ft_soup'
        )
    ),
    panoptic_head=dict(
        type=Mask2FormerVideoSemSamHead,
        sphere_cls=True,
        ov_path=omg_ov_class_embed_path,
        enable_box_query=False,
        ov_classifier_name=class_embed,
        logit=None,
        in_channels=[192, 384, 768, 1536],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=300,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type=MSDeformAttnPixelDecoder,
            num_outs=3,
            norm_cfg=dict(type=GroupNorm, num_groups=32),
            act_cfg=dict(type=ReLU),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type=ReLU, inplace=True)))),
        positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * 240 + [0.1]),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        loss_iou=dict(
            type=FocalLoss,
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='mean')
    ),
    panoptic_fusion_head=dict(
        type=MaskFormerFusionHead,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                # dict(type=FlexibleClassificationCost, weight=2.0),
                dict(type=CrossEntropyLossCost, weight=5.0, use_sigmoid=True),
                dict(type=DiceCost, weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type=MaskPseudoSampler)),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=omg_head_pretrain_pth_path,
    )
)

model = dict(
    type=OMG_LLaVA,
    freeze_llm=True,
    freeze_visual_encoder=True,
    require_omg_decoder=False,
    pretrained_pth=None,
    text2vision_projector=True,
    pixel_shuffle_ratio=2,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    llm_lora=dict(
        type=LoraConfig,
        r=512,
        lora_alpha=256,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'),
    visual_encoder=omgseg_model,
    tokenizer=tokenizer,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

llava_dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    lazy=lazy,
)

train_dataset = dict(
    type=CombineDataset,
    datasets_cfgs=[llava_dataset,],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=omg_llava_collate_fn))

test_dataset = [
    dict(
        type=RESDataset,
        dataset_name='refcoco',
        image_folder='./data/glamm_data/images/coco2014/train2014/',
        image_processor=image_processor,
        data_path="./data/ref_seg/",
        pad_image_to_square=True,
        split='val',
    ),
    dict(
        type=RESDataset,
        dataset_name='refcoco_plus',
        image_folder='./data/glamm_data/images/coco2014/train2014/',
        image_processor=image_processor,
        data_path="./data/ref_seg/",
        pad_image_to_square=True,
        split='val',
    ),
    dict(
        type=RESDataset,
        dataset_name='refcocog',
        image_folder='./data/glamm_data/images/coco2014/train2014/',
        image_processor=image_processor,
        data_path="./data/ref_seg/",
        pad_image_to_square=True,
        split='val',
    ),
    dict(
        type=MultipleChoiceDataset,
        data_file='./data/eval/mmbench/MMBench_DEV_EN.tsv',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=MultipleChoiceDataset,
        data_file='./data/eval/mmbench/MMBench_TEST_EN.tsv',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=MMEDataset,
        data_file='./data/eval/mme/MME.tsv',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=MultipleChoiceDataset,
        data_file='./data/eval/seed_bench/SEEDBench_IMG.tsv',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=MultipleChoiceDataset,
        data_file='./data/eval/sqa/ScienceQA_VAL.tsv',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=MultipleChoiceDataset,
        data_file='./data/eval/sqa/ScienceQA_TEST.tsv',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=MultipleChoiceDataset,
        data_file='./data/eval/ai2d/AI2D_TEST.tsv',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=MultipleChoiceDataset,
        data_file='./data/eval/mmstar/MMStar.tsv',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=HallusionDataset,
        data_file='./data/eval/HallusionBench/HallusionBench.tsv',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
    dict(
        type=POPEDataset,
        data_file=[
            './data/eval/pope/coco_pope_adversarial.json',
            './data/eval/pope/coco_pope_popular.json',
            './data/eval/pope/coco_pope_random.json',
        ],
        coco_val_path='./data/eval/val2014/',
        image_processor=image_processor,
        pad_image_to_square=True,
    ),
    # dict(
    #     type=GeneralVQADataset,
    #     data_file='./data/eval/docvqa/DocVQA_VAL.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    # dict(
    #     type=GeneralVQADataset,
    #     data_file='./data/eval/ocrvqa/OCRVQA_TEST.tsv',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),
    # dict(
    #     type=VQAv2Dataset,
    #     question_file='./data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl',
    #     answer_file='llava_vqav2_testdev_balanced_merge.jsonl', # file name of predicted answer
    #     test_file='./data/eval/vqav2/llava_vqav2_mscoco_test2015.jsonl',
    #     prediction_file='vqav2_testdev_balanced_predictions.json', # file name of formatted predicted answer
    #     image_folder='./data/eval/vqav2/test2015',
    #     image_processor=image_processor,
    #     pad_image_to_square=True,
    # ),  # only prediction
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
# Log the dialogue periodically during the training process, optional
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
