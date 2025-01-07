# Copyright (c) OpenMMLab. All rights reserved.
import os
from huggingface_hub import snapshot_download
from peft import PeftModel
import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)

import argparse
import os.path as osp

from mmengine.config import Config, DictAction

from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument(
        'model_name_or_path', help='Hugging Face model name or path')
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument(
        '--adapter', default=None, help='adapter name or path')
    parser.add_argument(
        '--config', default='config.py', help='config.py')
    parser.add_argument(
        '--pth-path', default='./converted.pth', help='./converted.pth')
    adapter_group.add_argument(
        '--llava', default=None, help='llava name or path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument('--image', default=None, help='image')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template')
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template',
        choices=SYSTEM_TEMPLATE.keys(),
        default=None,
        help='Specify a system template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    args = parser.parse_args()
    return args


def get_input():
    """Helper function for getting input from users."""
    sentinel = ''  # ends when this string is seen
    result = None
    while result is None:
        print(('\ndouble enter to end input (EXIT: exit chat, '
               'RESET: reset history) >>> '),
              end='')
        try:
            result = '\n'.join(iter(input, sentinel))
        except UnicodeDecodeError:
            print('Invalid characters detected. Please enter again.')
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # build llm
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }
    if args.with_plugins is None:
        inner_thoughts_open = False
        calculate_open = False
        solve_open = False
        search_open = False
    else:
        assert args.prompt_template == args.system_template == 'moss_sft'
        calculate_open = 'calculate' in args.with_plugins
        solve_open = 'solve' in args.with_plugins
        search_open = 'search' in args.with_plugins
        # pre-import for api and model preparation
        if calculate_open:
            from plugins import calculate  # noqa: F401
        if solve_open:
            from plugins import solve  # noqa: F401
        if search_open:
            from plugins import search  # noqa: F401
    # build llm
    llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                               **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        encode_special_tokens=True)
    print(f'Load LLM from {args.model_name_or_path}')
    if args.adapter is not None:
        llm = PeftModel.from_pretrained(
            llm,
            args.adapter,
            offload_folder=args.offload_folder,
            trust_remote_code=True)
        print(f'Load adapter from {args.adapter}')
    if args.llava is not None:
        llava_path = snapshot_download(
            repo_id=args.llava) if not osp.isdir(
            args.llava) else args.llava

        # build visual_encoder
        if 'visual_encoder' in os.listdir(llava_path):
            assert args.visual_encoder is None, (
                "Please don't specify the `--visual-encoder` since passed "
                '`--llava` contains a visual encoder!')
            visual_encoder_path = osp.join(llava_path, 'visual_encoder')
        else:
            assert args.visual_encoder is not None, (
                'Please specify the `--visual-encoder`!')
            visual_encoder_path = args.visual_encoder
        visual_encoder = CLIPVisionModel.from_pretrained(
            visual_encoder_path,
            torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
        print(f'Load visual_encoder from {visual_encoder_path}')

        # load adapter
        if 'llm_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'llm_adapter')
            llm = PeftModel.from_pretrained(
                llm,
                adapter_path,
                offload_folder=args.offload_folder,
                trust_remote_code=True)
            print(f'Load LLM adapter from {args.llava}')
        if 'visual_encoder_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
            visual_encoder = PeftModel.from_pretrained(
                visual_encoder,
                adapter_path,
                offload_folder=args.offload_folder)
            print(f'Load visual_encoder adapter from {args.llava}')

        # build projector
        projector_path = osp.join(llava_path, 'projector')
        projector = AutoModel.from_pretrained(
            projector_path,
            torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
            trust_remote_code=True)
        print(f'Load projector from {args.llava}')

        projector.cuda()
        projector.eval()
        visual_encoder.cuda()
        visual_encoder.eval()
    llm.eval()

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
    if 'LLaVAModel' or 'OMG' in model_name:
        cfg.model.pretrained_pth = None

    model = BUILDER.build(cfg.model)
    model.llm = llm
    model.visual_encoder = visual_encoder
    model.projector = projector

    state_dict = model.state_dict()
    torch.save(state_dict, args.pth_path)
    print('Save the converted pth to {}'.format(args.pth_path))

if __name__ == '__main__':
    main()