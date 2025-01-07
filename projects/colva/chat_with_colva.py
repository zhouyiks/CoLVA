import os
import json
import cv2
import random
from typing import List
import pycocotools.mask as mask_util
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from transformers import CLIPImageProcessor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VPT_CONTEXT_TOKEN = '<VPT_CONTEXT>'

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6, upscale=False):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file.convert('RGB')

    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    masks = mask_util.decode(rles)
    reduced = np.add.reduce(masks, axis=2)
    m = np.where(reduced>=2, 0, reduced)
    # rle = mask_util.merge(rles)
    return m.astype(bool)

from distinctipy import distinctipy
def contour_rendering(image, masks, mask_ids=None):
    colors = distinctipy.get_colors(len(masks)+1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 2
    font_scale_list = []
    label_list = []
    color_list = []
    label_loc_list = []
    for anno_i in range(len(masks)):
        mask = masks[anno_i]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if colors[anno_i][0] > 0.9 and colors[anno_i][1] > 0.9 and colors[anno_i][2] > 0.9:
            color_anno_i = (colors[-1][2] * 255, colors[-1][1] * 255, colors[-1][0] * 255)
        else:
            color_anno_i = (colors[anno_i][2] * 255, colors[anno_i][1] * 255, colors[anno_i][0] * 255)
        
        cv2.drawContours(image, contours, -1, color=color_anno_i, thickness=2)

        cnt_area = []
        cnt_centroid = []
        cnt_bbox = []
        for cnt in contours:
            cnt_area.append(cv2.contourArea(cnt))
            M = cv2.moments(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w/2, y + h/2
            cnt_centroid.append((cx, cy))
            cnt_bbox.append((w, h))
        select_cnt = 0
        if len(cnt_area) > 1:
            select_cnt = np.argmax(np.array(cnt_area))
        select_centroid = cnt_centroid[select_cnt]
        visual_prompt_id = anno_i+1 if mask_ids is None else mask_ids[anno_i]
        boxW, boxH = cnt_bbox[select_cnt]
        if max(boxH, boxW) < 25:
            thickness=1
        else:
            thickness=text_thickness

        # find the optimal font scale: text width/height close to 1/5 of the bbox width/height
        ok = False
        for scale in reversed(range(5, 60, 1)):
            textSize = cv2.getTextSize(f"{visual_prompt_id}", font, scale/10, thickness)
            textW, textH = textSize[0][0], textSize[0][1]
            if textH / boxH > 0.15 or textW / boxW > 0.15:
                continue
            font_scale_list.append(scale/10)
            ok = True
            break
        if not ok:
            font_scale_list.append(0.5)
        label_list.append(visual_prompt_id)
        color_list.append(color_anno_i)

        (base_w, base_h), bottom = cv2.getTextSize(f"{visual_prompt_id}", font, font_scale_list[-1], thickness)
        label_loc_list.append((
            int(select_centroid[0] - base_w/2),
            int(select_centroid[1] + (base_h+bottom)/2)
        ))
    font_scale = min(font_scale_list)
    for anno_i in range(len(label_list)):
        (base_w, base_h), bottom = cv2.getTextSize(f"{label_list[anno_i]}", font, font_scale, thickness)
        cv2.rectangle(image, (label_loc_list[anno_i][0], int(label_loc_list[anno_i][1]-base_h-bottom/2)),
                      (label_loc_list[anno_i][0]+base_w, int(label_loc_list[anno_i][1]+bottom/2)),
                      color_list[anno_i], -1, 8)
        cv2.putText(image, f"{label_list[anno_i]}", label_loc_list[anno_i], font, font_scale,
                    (255, 255, 255), thickness)
    
    return None


def main():
    path = "./work_dirs/colva_internvl2_4b"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    generation_config = dict(max_new_tokens=1024, do_sample=True)

    # # pure-text conversation
    # question = 'Hello, who are you?'
    # response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    # print(f'User: {question}\nAssistant: {response}')

    # question = 'Can you tell me a story?'
    # response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
    # print(f'User: {question}\nAssistant: {response}')
    
    # pixel_values = load_image(os.path.join(path, "examples/image1.jpg"), max_num=12).to(torch.bfloat16).cuda()
    # question = '<image>\nPlease describe the image shortly.'
    # response = model.chat(tokenizer, pixel_values, question, generation_config)
    # print(f'User: {question}\nAssistant: {response}')
    
    image_path_list = [os.path.join(path, "examples/match_case/FRAME00_ORI.jpg"), os.path.join(path, "examples/match_case/FRAME01_ORI.jpg")]
    anno_file_list = [os.path.join(path, "examples/match_case/FRAME00.json"), os.path.join(path, "examples/match_case/FRAME01_CAND.json")]
    
    # load annotations
    region_list = []
    for query_json_file in anno_file_list[:-1]:
        with open(query_json_file, 'r') as f:
            query_anno = json.load(f)
        ori_height, ori_width = query_anno[0]['height'], query_anno[0]['width']
        segm = query_anno[0]['segmentation']
        segm = [np.array(poly) for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
        mask = polygons_to_bitmask(segm, ori_height, ori_width)
        region_list.append(mask[np.newaxis, :, :].astype(np.uint8))
    with open(anno_file_list[-1], 'r') as f:
        query_anno = json.load(f)
    all_masks = []
    for idx in range(len(query_anno)):
        ori_height, ori_width = query_anno[idx]['height'], query_anno[idx]['width']
        segm = query_anno[idx]['segmentation']
        segm = [np.array(poly) for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
        mask = polygons_to_bitmask(segm, ori_height, ori_width)
        all_masks.append(mask)
    all_masks = np.stack(all_masks, axis=0)
    region_list.append(all_masks.astype(np.uint8))
    
    # draw the visual prompts on the image
    overlied_images = [cv2.imread(img_file) for img_file in image_path_list]
    for fidx, (image, regions) in enumerate(zip(overlied_images[:-1], region_list[:-1])):
        for region in regions:
            contours, hierarchy = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlied_images[fidx], contours, -1, color=(255, 255, 0), thickness=2)
    random_id = list(range(1, len(region_list[-1])+1))
    random.shuffle(random_id)
    all_region_ids = random_id
    contour_rendering(overlied_images[-1], region_list[-1], random_id)

    for fidx, overlied_image in enumerate(overlied_images):
        cv2.imwrite(f"./overlied_image_{fidx+1}.jpg", overlied_image)

    overlied_images = [Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB)) for item in overlied_images]

    # prepare radio inputs
    ot_image_processor = CLIPImageProcessor.from_pretrained("./nvidia/RADIO", trust_remote_code=True)
    ot_images = [Image.open(image_name).convert('RGB') for image_name in image_path_list]
    ot_pixel_values, ot_visual_prompts = [], []
    for fi, image in enumerate(ot_images):
        w, h = image.size
        if w > h:
            target_size = (1024, int(h/w*1024))
        else:
            target_size = (int(w/h*1024), 1024)
        resized_image = image.resize(target_size)
        cur_w, cur_h = resized_image.size
        padded_image = np.ones(shape=(1024, 1024, 3), dtype=np.uint8) * 255
        padded_image[:cur_h, :cur_w, :] = np.array(resized_image)

        ot_pixel_values.append(ot_image_processor(images=Image.fromarray(padded_image), return_tensors='pt').pixel_values)
    ot_pixel_values = torch.cat(ot_pixel_values).to(torch.bfloat16).cuda()

    for regions in region_list:
        h, w = regions.shape[-2:]
        regions = torch.from_numpy(regions).to(ot_pixel_values.dtype).to(ot_pixel_values.device)
        if h > w:
            padded_regions = regions.new_zeros((regions.shape[0], h, h))
        else:
            padded_regions = regions.new_zeros((regions.shape[0], w, w))
        padded_regions[:, :h, :w] = regions
        resized_padded_regions = F.interpolate(padded_regions.unsqueeze(0), size=(1024, 1024), mode='bilinear').squeeze(0)
        ot_visual_prompts.append(resized_padded_regions)

    # prepare choice items
    choice_names = [f"{chr(i)}" for i in range(65,91)]
    if len(regions) > len(choice_names) - 1:
        valid_num = len(choice_names) - 1
    else:
        valid_num = len(regions)
    region_ids = random_id[:valid_num]
    choice_names = choice_names[:valid_num+1]

    region_ids.sort()
    multi_choices_str = ""
    for choice_name, region_id in zip(choice_names[:-1], region_ids):
        multi_choices_str = multi_choices_str + f"{choice_name}. {region_id}\n"
    multi_choices_str = multi_choices_str + f"{choice_names[-1]}. None of the above choices are correct\n"

    question = "Here are two images. In the second image, I have marked several "\
        "visual objects with their contours in different colors, and each "\
        "is identified by a white numeric ID against a background that "\
        "matches the contour's color. Could you please tell me which of "\
        "these marked objects is the same as the object marked with a cyan "\
        "contour in the first image? Please make a choice from the following options: \n"
    
    object_token_str = ""
    for fidx in range(len(overlied_images)-1):
        object_token_str = object_token_str + f"Objects in Image-{fidx+1}: <query object>{VPT_CONTEXT_TOKEN}\n"
    object_token_str = object_token_str + f"Objects in Image-{len(overlied_images)}: "
    sorted_indices = sorted(range(len(all_region_ids)), key=lambda k: all_region_ids[k])
    for sorted_idx in sorted_indices:
        object_token_str = object_token_str + f"<object-{all_region_ids[sorted_idx]}>{VPT_CONTEXT_TOKEN}, "
    object_token_str = object_token_str[:-2] + '.\n'
    prefix_str = f"Image-1: <image>\nImage-2: <image>\n" + object_token_str
    question = prefix_str + question + multi_choices_str

    num_patches_list = []
    pixel_values_list = []
    for overlied_image in overlied_images:
        pixel_values = load_image(overlied_image, max_num=12).to(torch.bfloat16).cuda()
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.size(0))
    pixel_values = torch.cat(pixel_values_list, dim=0)

    response, history = model.chat(tokenizer, pixel_values, question, generation_config, return_history=True, 
                                   num_patches_list=num_patches_list, ot_pixel_values=ot_pixel_values, ot_visual_prompts=ot_visual_prompts)
    print(f'User: {question}\nAssistant: {response}')

    question = "Why are they the same one?"
    response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True, 
                                   num_patches_list=num_patches_list, ot_pixel_values=ot_pixel_values, ot_visual_prompts=ot_visual_prompts)
    print(f'User: {question}\nAssistant: {response}')
    


if __name__ == '__main__':
    main()
