import argparse
import math
import os
import torch
import tqdm
from pycocotools import mask as mask_utils

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from utils import _init_dist_pytorch, get_dist_info, collect_results_cpu
from PIL import Image
import re
import json

def parse_args():
    parser = argparse.ArgumentParser(description='GCG')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--save_dir',
        default='./gcg_pred/',
        help='save path')
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

IMAGE_FOLDER = './data/glamm_data/images/grandf/val_test/'


class GCGInferenceDataset:
    def __init__(self,
                 image_folder,
                 save_dir=None,
                 ):
        self.image_folder = image_folder

        self.images = os.listdir(image_folder)

        if save_dir is not None:
            # filter evaluated
            self.save_dir = save_dir
            exsits_files = os.listdir(self.save_dir)
            exsits_files = [_file[:-5] for _file in exsits_files]
            _images = []
            for i, item in enumerate(self.images):
                if item[:-4] not in exsits_files:
                    _images.append(item)
            self.images = _images

    def __len__(self):
        return len(self.images)

    def get_questions(self):
        question = "Could you please give me a brief description of the image? Please respond with interleaved \
    segmentation masks for the corresponding parts of the answer."
        return question

    def __getitem__(self, index):
        data_dict = {}
        questions = self.get_questions()
        image_file = self.images[index]
        data_dict['image_file'] = image_file

        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        data_dict['image'] = image
        data_dict['text'] = "<image>\n" + questions

        data_dict['img_id'] = image_file
        return data_dict

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

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    dataset = GCGInferenceDataset(
        image_folder=IMAGE_FOLDER,
        save_dir=args.save_dir,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        prediction = {'img_id': data_batch['img_id'], 'image_file': data_batch['image_file']}
        del data_batch['img_id'], data_batch['image_file']

        w, h = data_batch['image'].size

        pred_dict = model.predict_forward(**data_batch, tokenizer=tokenizer)
        if 'prediction_masks' not in pred_dict.keys() or pred_dict['prediction_masks'] is None or len(pred_dict['prediction_masks']) == 0:
            print("No SEG !!!")
            prediction['prediction_masks'] = torch.zeros((0, h, w), dtype=torch.bool)
        else:
            prediction['prediction_masks'] = torch.stack(pred_dict['prediction_masks'], dim=0)[:, 0]
        process_and_save_output(
            args.save_dir,
            prediction['image_file'],
            pred_dict['prediction'],
            prediction['prediction_masks']
        )
        results.append(pred_dict['prediction'])

    results = collect_results_cpu(results, len(dataset), tmpdir='./gcg_eval_tmp')


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
