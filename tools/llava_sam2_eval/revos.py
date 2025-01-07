import argparse
import copy
import os.path as osp

import torch
from torch.utils.data import DataLoader

from mmengine import Config
from mmengine.dist import init_dist, get_dist_info
from mmengine.utils.dl_utils import set_multi_processing
from transformers import GenerationConfig
from xtuner.configs import cfgs_name_path
from xtuner.registry import BUILDER
from xtuner.tools.chat import TORCH_DTYPE_MAP
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import PROMPT_TEMPLATE


from projects.llava_sam2.datasets import video_lisa_collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth_model', default=None, help='pth model file')
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

def main():
    args = parse_args()
    if args.launcher != 'none':
        set_multi_processing(distributed=True)
        init_dist(args.launcher)

        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
    print(f'Rank: {rank} / World size: {world_size}')

    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    cfg = Config.fromfile(args.config)
    model_name = cfg.model.type if isinstance(cfg.model.type, str) else cfg.model.type.__name__
    assert model_name in ('VideoLLaVASAMModel',)

    model = BUILDER.build(cfg.model)

    if args.pth_model is not None:
        state_dict_pth = args.pth_model
        state_dict = torch.load(state_dict_pth, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    model.to('cuda:0')
    model.eval()

    # define some pointers
    tokenizer = model.tokenizer

    # gen_configs
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    stop_words = args.stop_words
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)

    data_cfg = copy.deepcopy(cfg.video_revos_dataset)
    data_cfg.update(expression_file=data_cfg.expression_file.replace('train', 'val'))
    dataset = BUILDER.build(cfg.video_revos_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        collate_fn=video_lisa_collate_fn,
    )

    for data_item in dataloader:
        data_item = model.data_preprocessor(data_item)
        inputs, data_samples = data_item['data'], data_item['data_samples']
        g_pixel_values = inputs.pop('g_pixel_values', None)
        gt_masks = inputs.pop('masks', None)

        output = model.mllm.generate(
            pixel_values=inputs['pixel_values'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            visual_features=None,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=tokenizer.bos_token_id,
            stopping_criteria=stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        print(1)
    print(1)


if __name__ == '__main__':
    main()
