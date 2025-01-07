from mmdet.datasets import RefCocoDataset

from mmdet.datasets.transforms import LoadAnnotations
from mmdet.evaluation import RefSegMetric
import argparse
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from xtuner.utils.constants import DEFAULT_IMAGE_TOKEN
from accelerate import Accelerator
from accelerate.utils import gather_object
from mmdet.structures.mask import BitmapMasks
from mmcv.transforms import LoadImageFromFile
from tqdm import tqdm
import torch
import torch.nn.functional as F
from time import time

from projects.f_llm.datasets.transforms import PILLoadImageFromFile, RefCOCO2PNG
from projects.lisa.datasets.refcoco_segm_dataset import ReferSegmDataset
from projects.glamm.datasets.collate_fns.glamm_collate_fn import glamm_collate_fn
from third_parts.segment_anything.utils.transforms import ResizeLongestSide
from pycocotools import mask as mask_utils
from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST
extra_image_processor = ResizeLongestSide(
    target_length=1024,
)
import copy
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from xtuner.utils import PROMPT_TEMPLATE
template = PROMPT_TEMPLATE.phi3_chat
_system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'
# _system = 'You are an AI assistant whose name is Phi-3.'
_system = ''
begin_str = f'{DEFAULT_IMAGE_TOKEN}\n'
template['INSTRUCTION'] = '<|user|>\n{input}<|end|><|assistant|>\n'

transformer = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def get_inputid_labels(conversations, image_token_str):
    input = ''
    out_conversation = []
    while conversations and conversations[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        conversations = conversations[1:]
    for msg in conversations:
        if msg['from'] == 'human':
            if image_token_str is None and '<image>' in msg['value']:
                msg['value'] = msg['value'].replace('<image>', '')
            if '<image>' in msg['value']:
                msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
            input += msg['value'].strip()
        elif msg['from'] == 'gpt':
            out_conversation.append({
                'input': input,
                'output': msg['value'].strip()
            })
            input = ''
        else:
            raise NotImplementedError
    input_ids, labels = [], []
    for i, single_turn_conversation in enumerate(out_conversation):
        input = single_turn_conversation.get('input', '')
        if input is None:
            input = ''
        input_text = template.INSTRUCTION.format(
            input=input, round=i + 1)
        if i == 0:
            if _system != '' and _system is not None:
                system = template.SYSTEM.format(system=_system)
                input_text = system + input_text
            input_encode = tokenizer.encode(input_text, add_special_tokens=True)
        else:
            input_encode = tokenizer.encode(input_text, add_special_tokens=False)
        input_ids += input_encode
        labels += [-100] * len(input_encode)
        output_text = single_turn_conversation.get('output', '')
        if template.get('SUFFIX', None):
            output_text += template.SUFFIX
        output_encode = tokenizer.encode(
            output_text, add_special_tokens=False)
        input_ids += output_encode
        labels += copy.deepcopy(output_encode)
    max_length = 8192
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return input_ids, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(messages)

    cfg = Config.fromfile(args.config)
    tokenizer = cfg.tokenizer
    tokenizer = BUILDER.build(tokenizer)
    tokenizer.add_tokens(['[SEG]'], special_tokens=True)

    model = BUILDER.build(cfg.model)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")

    model = model.to(device=accelerator.device)
    model.eval()
    model.to(torch.bfloat16)

    dataset = RefCocoDataset(
            data_root='data/coco/',
            data_prefix=dict(img_path='train2014/'),
            text_mode='select_first',
            ann_file='refcoco/instances.json',
            split_file='refcoco/refs(unc).p',
            split='val'
        )
    accelerator.wait_for_everyone()

    data_ids = list(range(len(dataset)))

    results = []
    from PIL import Image
    import numpy as np
    from projects.lisa.datasets.sem_seg_dataset import dynamic_preprocess
    with accelerator.split_between_processes(data_ids) as sub_ids:
        for idx in tqdm(sub_ids, disable=not accelerator.is_main_process):
            ann_info = dataset[idx]
            image = Image.open(ann_info['img_path']).convert('RGB')
            width, height = image.size
            g_image = np.array(image)  # for grounding
            g_image = extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            
            images = dynamic_preprocess(image, 1, 12, 448, True)
            pixel_values = [transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            patch_token = int((448 // 14)**2 * (0.5**2))
            num_image_tokens = pixel_values.shape[0] * patch_token
            image_token_str = f'<img>' + '<IMG_CONTEXT>' * num_image_tokens+ '</img>' 

            instances, phrases = ann_info['instances'], ann_info['text']
            for inst, phrase in zip(instances, phrases):
                if '.' == phrase[-1]:
                    phrase = phrase[:-1]
                binary_mask = np.zeros((height, width), dtype=np.uint8)
                for seg in inst["mask"]:
                    rles = mask_utils.frPyObjects([seg], height, width)
                    m = mask_utils.decode(rles)
                    m = m.astype(np.uint8)
                    binary_mask += m.squeeze()
                
                import random
                conversation = []
                question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
                question = begin_str + question
                conversation.append({'from':'human', 'value': question})
                conversation.append({'from':'gpt', 'value': ''})

                input_ids, labels = get_inputid_labels(conversation, image_token_str)
                input_ids = input_ids[:-1] # remove <|end|>
                out_data_dict = {
                    'input_ids': torch.tensor(input_ids),
                    'labels': torch.tensor(labels),
                    'g_pixel_values': g_pixel_values,
                    'pixel_values': pixel_values,
                    'masks': binary_mask[None],
                }
            
                data_sample = glamm_collate_fn([out_data_dict])
                with torch.no_grad():
                    outputs = model(**data_sample, mode='predict')
                
                gt_masks = binary_mask[None] > 0
                pred_mask_logits = outputs['pred_mask_logits']
                if pred_mask_logits is None:
                    pred_masks = torch.zeros_like(gt_masks)
                else:
                    pred_masks = pred_mask_logits.sigmoid().cpu() > 0.5

                assert len(pred_masks) == len(gt_masks)
                mask_cnt = pred_masks.shape[0]
                results.append(
                    dict(
                        pred_instances=dict(masks=pred_masks),
                        gt_masks=BitmapMasks(
                            masks=gt_masks,
                            height=gt_masks.shape[1],
                            width=gt_masks.shape[2]))
                    )
        results = gather_object(results)

    if accelerator.is_main_process:
        accelerator.print(
            f"Collected {len(results)} result samples from all gpus")
        evaluator = RefSegMetric(metric=['cIoU', 'mIoU'])
        evaluator.process(data_batch=dict(), data_samples=results)
        metrics = evaluator.compute_metrics(evaluator.results)
        accelerator.print(f"Evaluation results on : {metrics}")
