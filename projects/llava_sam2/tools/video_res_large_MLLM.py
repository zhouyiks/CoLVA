# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp
from distutils.command.config import config

import numpy as np
import torch
import tqdm
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from mmengine.utils.dl_utils import set_multi_processing
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict

import logging
from mmengine import print_log
from PIL import Image
from pycocotools import mask
import torch.nn.functional as F

from projects.llava_sam2.configs.test.llava_sam2_test_gcg_26b import test_dataset
from projects.omg_llava.dataset.utils import expand2square
from projects.omg_llava.dataset.utils.refcoco_refer import REFER
from projects.omg_llava.tools.utils_refcoco import AverageMeter, Summary, intersectionAndUnionGPU


def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth_model', help='pth model file')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
        default='refcoco',
        help='Specify a ref dataset')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default='internlm2_chat',
        help='Specify a prompt template')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=100,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args

DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco+'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
}

@master_only
def master_print(msg):
    print(msg)

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.launcher != 'none':
        set_multi_processing(distributed=True)
        init_dist(args.launcher)

        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    # if args.cfg_options is not None:
        # cfg.merge_from_dict(args.cfg_options)

    model_name = cfg.model.type if isinstance(cfg.model.type,
                                              str) else cfg.model.type.__name__

    model = BUILDER.build(cfg.model)
    backend = get_file_backend(args.pth_model)

    # if os.path.exists(cfg.pretrained_pth):
    #     if isinstance(backend, PetrelBackend):
    #         from xtuner.utils.fileio import patch_fileio
    #         with patch_fileio():
    #             state_dict = guess_load_checkpoint(cfg.pretrained_pth)
    #     else:
    #         state_dict = guess_load_checkpoint(cfg.pretrained_pth)
    #
    #     # del state_dict['llm.base_model.model.model.tok_embeddings.weight']
    #     model.load_state_dict(state_dict, strict=False)
    #     print(f'Load pre PTH model from {cfg.pretrained_pth}')

    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    datasets = []
    datasets_configs = cfg.test_dataset
    for dataset_config in datasets_configs:
        _type = dataset_config['type']
        del dataset_config['type']
        datasets.append(_type(**dataset_config))

    # model.cuda()
    model.grounding_encoder.cuda()
    model.text_hidden_fcs.cuda()
    model.eval()


    for i_dataset, dataset in enumerate(datasets):
        model.preparing_for_generation(dataset.metainfo)
        results = []
        n_samples = len(dataset)
        per_rank_samples = math.ceil(n_samples / world_size)
        per_rank_ids = range(per_rank_samples * rank,
                             min(n_samples, per_rank_samples * (rank + 1)))
        for idx in tqdm.tqdm(per_rank_ids):
            data_batch = dataset[idx]
            prediction = {'video_id': data_batch['video_id']}
            outputs = model.predict_forward(**data_batch)
            prediction.update(outputs)
            results.append(prediction)

        results = collect_results(results, len(dataset))
        if get_rank() == 0:
            metric = dataset.evaluate(results, './work_dirs')
            objects = [metric]
        else:
            objects = [None]
        print(f"Done eval of dataset {i_dataset}.")

if __name__ == '__main__':

    main()
