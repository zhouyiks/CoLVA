import argparse
import copy
import math
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
import random

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from datasets import RESDataset

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco_plus'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
}

IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'
DATA_PATH = './data/ref_seg/'


def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]

    dataset = RESDataset(
        image_folder=IMAGE_FOLDER,
        dataset_name=dataset_info['dataset_name'],
        data_path=DATA_PATH,
        split=args.split,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        prediction = {'img_id': data_batch['img_id'], 'gt_masks': data_batch['gt_masks']}
        prediction['gt_masks'] = mask_to_rle(prediction['gt_masks'].cpu().numpy())
        del data_batch['img_id'], data_batch['gt_masks']

        texts = data_batch['text']
        del data_batch['text']
        pred_masks = []
        for text in texts:
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text
            pred_mask = model.predict_forward(**_data_batch, tokenizer=tokenizer)['prediction_masks']
            if len(pred_mask) == 0:
                # give a zero mask
                print("No seg pred !!!")
                pred_masks.append(None)
            else:
                _ret_mask = pred_mask[0].cpu().numpy()
                _ret_mask = mask_to_rle(_ret_mask)
                pred_masks.append(_ret_mask)

        prediction.update({'prediction_masks': pred_masks})
        results.append(prediction)

    tmpdir = './dist_test_temp_res_' + args.dataset + args.split + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        metric = dataset.evaluate(results, './work_dirs')
        print(metric)

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

if __name__ == '__main__':
    main()
