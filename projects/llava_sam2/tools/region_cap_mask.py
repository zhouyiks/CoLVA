# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import math
import os
import os.path as osp
import re
import torch
import tqdm

from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from mmengine.utils.dl_utils import set_multi_processing
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from projects.omg_llava.model.utils import prepare_inputs_labels_for_multimodal_with_visual_prompts
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)

from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict

from PIL import Image
import torch.nn.functional as F
from projects.omg_llava.dataset.utils import expand2square, expand2square_mask
from pycocotools import mask

from pycocotools.coco import COCO
import numpy as np

def bbox_to_x1y1x2y2(bbox):
    x1, y1, w, h = bbox
    bbox = [x1, y1, x1 + w, y1 + h]

    return bbox

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
        '--output-path', type=str, default='./work_dirs/region_cap_pred.json', help='Name for Bot')
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
        default=300,
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


@master_only
def master_print(msg):
    print(msg)

class RegionCap_Inference_Dataset(Dataset):
    def __init__(self,
                 image_folder,
                 annotation_file=None,
                 metainfo=None,
                 ):
        self.metainfo = metainfo
        self.image_folder = image_folder
        self.image_h, self.image_w = 1024, 1024

        self.down_ratio = 1

        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())

    def __len__(self):
        return len(self.image_dict_keys)

    def decode_mask(self, annotation, image_info):
        flag = False
        masks = []

        for ann_id in range(1):

            ann = {"segmentation": annotation}

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
        masks = np.stack(masks, axis=0)

        return masks

    def get_questions(self):
        # question = "<image>Can you provide me with a detailed description of the region in the picture marked by region1."
        question = "<image>Please give me a short description of the region in the picture marked by region1."
        return question

    def __getitem__(self, index):

        data_dict = {}

        image_id = self.image_dict_keys[index]
        image_file = self.image_dict[image_id]['file_name']

        questions = self.get_questions()

        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        masks = self.ann_dict[image_id]['segmentation']
        image_info = self.image_dict[image_id]
        masks = self.decode_mask(masks, image_info)

        data_dict['pixel_values'] = image
        data_dict['ori_image'] = image
        data_dict['text_prompts'] = questions
        ori_width, ori_height = image.size
        data_dict['ori_image_size'] = (ori_width, ori_height)
        data_dict['img_id'] = image_id
        data_dict['vp'] = True
        data_dict['mask_prompts'] = [masks]

        return data_dict

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

    datasets_configs = cfg.test_dataset

    model.cuda()
    # model.grounding_encoder.cuda()
    # model.text_hidden_fcs.cuda()
    model.eval()

    dataset = RegionCap_Inference_Dataset(
        annotation_file='./data/region_caption/refcocog/finetune_refcocog_val_with_mask.json',
        image_folder='./data/glamm_data/images/coco2014/train2014/',
        metainfo=datasets_configs[0]['metainfo'],
        # debug=True,
    )
    datasets = [dataset]


    for i_dataset, dataset in enumerate(datasets):
        model.preparing_for_generation(dataset.metainfo)
        results = []
        n_samples = len(dataset)
        per_rank_samples = math.ceil(n_samples / world_size)
        per_rank_ids = range(per_rank_samples * rank,
                             min(n_samples, per_rank_samples * (rank + 1)))
        for idx in tqdm.tqdm(per_rank_ids):
            data_batch = dataset[idx]
            prediction = {'img_id': data_batch['img_id']}
            outputs = model.predict_forward(**data_batch)
            prediction.update(outputs)
            # results.append(prediction)

            text_output = outputs['prediction'].replace("<s>", "").replace("\n", "") \
                .replace("region1", '').replace("Region1", '') \
                .replace(':', '').replace("   ", " ").replace("  ", " ")
            text_output = text_output.split("ASSISTANT: ")[-1]
            cleaned_str = re.sub(r'<.*?>', '', text_output)
            cleaned_str = cleaned_str.replace('[SEG]', '')
            cleaned_str = ' '.join(cleaned_str.split()).strip("'")
            cleaned_str = cleaned_str.strip()

            result_dict = {}
            result_dict["image_id"] = data_batch['img_id']
            result_dict["caption"] = cleaned_str
            result_dict["image_file"] = data_batch['image_file']
            result_dict["prediction"] = cleaned_str
            results.append(result_dict)
            print(cleaned_str)

        results = collect_results(results, n_samples)

        if get_rank() == 0:
            with open(args.output_path, 'w') as json_file:
                json.dump(results, json_file, indent=2)

if __name__ == '__main__':

    main()
