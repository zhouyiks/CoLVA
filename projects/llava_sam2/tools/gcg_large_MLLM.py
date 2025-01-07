# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import math
import os
import os.path as osp
import re
from importlib.metadata import files

import torch
import tqdm

from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from mmengine.utils.dl_utils import set_multi_processing
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
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
from xtuner.dataset.utils import expand2square
from pycocotools import mask as mask_utils


def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


GCG_QUESTIONS = [
    'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Please describe in detail the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a comprehensive explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me an elaborate explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth_model', help='pth model file')
    parser.add_argument(
        '--output-name', type=str, default='gcg', help='save folder name')
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


@master_only
def master_print(msg):
    print(msg)

class GCD_Inference_Dataset(Dataset):
    def __init__(self,
                 image_folder,
                 debug=False,
                 metainfo=None,
                 save_dir=None,
                 ):
        self.debug = debug
        self.image_folder = image_folder
        self.metainfo = metainfo

        self.images = os.listdir(image_folder)

        if save_dir is not None:
            # filter evaluated
            self.save_dir = save_dir
            exsits_files = os.listdir(self.save_dir)
            exsits_files = [_file[:-5] for _file in exsits_files]
            _images = []
            for item in self.images:
                if item[:-4] not in exsits_files:
                    _images.append(item)
            self.images = _images


        if debug:
            self.images = self.images[:20]

    def __len__(self):
        return len(self.images)

    def get_questions(self):
        question = "Could you please give me a detailed description of the image? Please respond with interleaved \
    segmentation masks for the corresponding parts of the answer."
        return question

    def __getitem__(self, index):

        data_dict = {}

        questions = self.get_questions()
        image_file = self.images[index]
        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        # print(image_file)
        image = Image.open(image_file).convert('RGB')
        data_dict['pixel_values'] = image
        data_dict['ori_image'] = image
        data_dict['text_prompts'] = "<image>\n" + questions
        ori_width, ori_height = image.size
        data_dict['ori_image_size'] = (ori_width, ori_height)
        data_dict['img_id'] = image_file
        data_dict['mode'] = 'demo'
        data_dict['masks'] = 'none'
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
    dataset = GCD_Inference_Dataset(
        image_folder='./data/glamm_data/images/grandf/val_test/',
        debug=False,
        metainfo=datasets_configs[0]['metainfo'],
        save_dir="./work_dirs/{}/".format(args.output_name),
        # debug=True,
    )
    datasets = [dataset]

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
            prediction = {'img_id': data_batch['img_id']}
            outputs = model.predict_forward(**data_batch)
            prediction.update(outputs)
            results.append(prediction)
            if 'prediction_masks' not in prediction.keys():
                print("No SEG !!!")
                print(prediction['prediction'])
                w, h = data_batch['ori_image_size']
                prediction['prediction_masks'] = torch.zeros((0, h, w), dtype=torch.bool)
            else:
                prediction['prediction_masks'] = torch.stack(prediction['prediction_masks'], dim=0)[:, 0]
            # print(prediction['prediction'], "----", len(prediction['prediction_masks']))
            process_and_save_output(
                "./work_dirs/{}/".format(args.output_name),
                data_batch['image_file'],
                prediction['prediction'],
                prediction['prediction_masks']
            )

        # results = collect_results(results, len(dataset))
        # if get_rank() == 0:
        #     metric = dataset.evaluate(results, './work_dirs')
        #     objects = [metric]
        # else:
        #     objects = [None]
        print(f"Done eval of dataset {i_dataset}.")

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def process_and_save_output(output_dir, image_name, text_output, pred_masks):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # Convert the predicted masks into RLE format
    pred_masks_tensor = pred_masks.cpu()
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))

    # Create results dictionary
    # print(f"clean_str: {cleaned_str}")
    result_dict = {
        "image_id": image_name[:-4],
        "caption": cleaned_str,
        "phrases": phrases,
        "pred_masks": rle_masks
    }

    # print(cleaned_str)
    # print(phrases)

    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)

    return

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out

def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle

if __name__ == '__main__':

    main()
