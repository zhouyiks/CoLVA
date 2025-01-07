# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os.path as osp

import torch

from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)

from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict

from transformers import AutoConfig

def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--mllm-model-path', type=str, default='./OpenGVLab/InternVL2-4B', help='directory path to the base model.')
    parser.add_argument("--radio-path", type=str, default='./nvidia/RADIO', help='directory path to the radio model.')
    parser.add_argument(
        '--save-path', type=str, default='./work_dirs/hf_model', help='save folder name')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

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
    model = BUILDER.build(cfg.model)

    model._merge_lora()
    model.model.transfer_to_hf = True

    all_state_dict = model.all_state_dict()
    
    all_state_dict_new = {}
    for key in all_state_dict.keys():
        new_key = copy.deepcopy(key)
        if new_key.startswith('model.'):
            new_key = new_key[len('model.'):]
        all_state_dict_new[new_key] = all_state_dict[key]

    from projects.colva.colva_hf.internvl2_4b.configuration_internvl_chat import InternVLChatConfig
    from projects.colva.colva_hf.internvl2_4b.modeling_internvl_chat import InternVLChatModel

    mllm_config = AutoConfig.from_pretrained(args.mllm_model_path, trust_remote_code=True)
    mllm_config_dict = mllm_config.to_dict()
    radio_config = AutoConfig.from_pretrained(args.radio_path, trust_remote_code=True)
    radio_config_dict = radio_config.to_dict()
    radio_config_dict['auto_map'] = {
        'AutoConfig': "configuraion_radio.RADIOConfig",
        "AutoModel": "modeling_radio.RADIOModel"
    }
    mllm_config_dict.update({"radio_config": radio_config_dict})
    colva_hf_config = InternVLChatConfig(**mllm_config_dict)
    colva_hf_model = InternVLChatModel(colva_hf_config, vision_model=model.model.vision_model, language_model=model.model.language_model)
    
    colva_hf_model.load_state_dict(all_state_dict_new)
    colva_hf_model.save_pretrained(args.save_path)
    print(f"Save the hf model into {args.save_path}")

if __name__ == '__main__':

    main()

    
