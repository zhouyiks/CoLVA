# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
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
        '--output-path', type=str, default='./1215_demos/object_cap_sa2va.json', help='Name for Bot')
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
                 metainfo=None,
                 ):
        self.metainfo = metainfo
        self.image_folder = image_folder

        image_files = []
        for file_name in os.listdir(self.image_folder):
            if 'out' not in file_name and '.jpg' in file_name:
                image_files.append(file_name)

        json_files = []
        for file_name in image_files:
            json_files.append(file_name.replace('.jpg', '_out.json'))

        self.image_files = image_files
        self.json_files = json_files

        self.data_dicts = []
        for image_file, json_file in zip(image_files, json_files):
            with open(os.path.join(image_folder, json_file), 'r') as f:
                _datas = json.load(f)
            for _data in _datas:
                self.data_dicts.append({'image_file': image_file, 'object_anno': _data})

    def __len__(self):
        return len(self.data_dicts)

    def decode_mask(self, rle):
        m = mask.decode(rle)[None]
        print(m.shape)
        return m

    def get_questions(self):
        # question = "<image>Can you provide me with a detailed description of the region in the picture marked by region1."
        question = "<image>Please give me a short description of the region in the picture marked by region1."
        return question

    def __getitem__(self, index):

        _json_info = self.data_dicts[index]

        data_dict = {}

        image_id = index
        image_file = _json_info['image_file']

        questions = self.get_questions()

        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        masks = _json_info['object_anno']['segmentation']

        masks = self.decode_mask(masks)

        data_dict['pixel_values'] = image
        data_dict['ori_image'] = image
        data_dict['text_prompts'] = questions
        ori_width, ori_height = image.size
        data_dict['ori_image_size'] = (ori_width, ori_height)
        data_dict['img_id'] = image_id
        data_dict['vp'] = True
        data_dict['mask_prompts'] = [masks]

        mask_image = self.get_mask_image(image, masks[0])
        mask_image.save(os.path.join('./1215_demos/object_demos/', f"{image_id}.png"))
        return data_dict

    def get_mask_image(self, image, mask):

        image_shape = image.size
        mask = torch.Tensor(mask).unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(
            mask,
            size=(image_shape[1], image_shape[0]),
            mode='nearest').squeeze(0).squeeze(0)
        mask = mask.numpy()

        image = copy.deepcopy(image)
        image = np.array(image)

        image = image * 0.5
        image[:, :, 0] = image[:, :, 0] + mask * 255 * 0.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)

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
        image_folder='./1215_demos/mask_outs/out/',
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

            with open(os.path.join('./1215_demos/object_demos/', f"{data_batch['img_id']}.txt"), 'w') as f:
                f.write(cleaned_str)
            print(cleaned_str)

        results = collect_results(results, n_samples)

        if get_rank() == 0:
            with open(args.output_path, 'w') as json_file:
                json.dump(results, json_file, indent=2)

if __name__ == '__main__':

    main()
