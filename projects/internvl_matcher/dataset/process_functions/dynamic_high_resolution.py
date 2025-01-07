from PIL import Image
import numpy as np
import cv2

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
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, regions, merged_regions, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    assert image.size == merged_regions.size
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
    resized_merged_regions = merged_regions.resize((target_width, target_height))

    # resize the regions
    resized_regions = cv2.resize(np.transpose(regions, (1, 2, 0)), dsize=(target_width, target_height), interpolation=cv2.INTER_NEAREST_EXACT)
    if resized_regions.ndim < 3:
        resized_regions = resized_regions[:, :, np.newaxis]
    # for r in range(resized_regions.shape[-1]):
    #     mask = resized_regions[:, :, r]
    #     new_img = np.zeros((resized_regions.shape[0], resized_regions.shape[1], 3), dtype=np.uint8)
    #     new_img[:, :, 0] = mask * 255
    #     cv2.imwrite(f"./{r}.png", new_img)

    processed_images = []
    processed_merged_regions = []
    processed_regions = [[] for _ in range(resized_regions.shape[-1])]
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

        # split the visual prompt canvas
        split_mrgn = resized_merged_regions.crop(box)
        processed_merged_regions.append(split_mrgn)
        
        split_rgn = resized_regions[box[1]:box[3], box[0]:box[2], :]
        for r in range(resized_regions.shape[-1]):
            processed_regions[r].append(split_rgn[:, :, r])
            
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

        thumbnail_mrng = merged_regions.resize((image_size, image_size))
        processed_merged_regions.append(thumbnail_mrng)
        
        thumbnail_rng = cv2.resize(np.transpose(regions, (1, 2, 0)), dsize=(image_size, image_size), interpolation=cv2.INTER_NEAREST_EXACT)
        if thumbnail_rng.ndim < 3:
            thumbnail_rng = thumbnail_rng[:, :, np.newaxis]
        for r in range(regions.shape[0]):
            processed_regions[r].append(thumbnail_rng[:, :, r])

    return processed_images, processed_regions, processed_merged_regions