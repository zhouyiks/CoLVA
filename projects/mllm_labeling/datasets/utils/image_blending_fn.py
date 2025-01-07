from PIL import Image, ImageDraw
import random, math
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon
from scipy.stats import multivariate_normal
from pycocotools import mask
import cv2
import copy
from typing import Tuple


color_pool = {
    'red': (255, 0, 0),
    'lime': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'fuchsia': (255, 0, 255),
    'aqua': (0, 255, 255),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'gold': (255, 215, 0),

}


def get_random_point_within_polygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    trial_num = 0
    while True:
        if trial_num < 50:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            if polygon.contains(point):
                return x, y
            trial_num += 1
        else:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            return x, y

def get_random_point_within_bbox(bbox):
    left, top, right, bottom = bbox
    x = np.random.uniform(left, right)
    y = np.random.uniform(top, bottom)
    return x, y
        
def is_max_angle_less_than_150(points):
    for i in range(3):
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % 3])
        p3 = np.array(points[(i + 2) % 3])

        a = np.linalg.norm(p3 - p2)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        # Calculate angle at p2 using cosine rule
        angle_at_p2 = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2*a*c)))

        if angle_at_p2 > 150:
            return False
    return True

def draw_rectangle(canvas, bbox_coord, outline_color, width):
    left, top, right, bottom = bbox_coord
    canvas.rectangle([(left, top), (right, bottom)], outline=outline_color, width=width)

def draw_ellipse(canvas, bbox_coord, mask_polygon, outline_color, width, size_ratio=1, aspect_ratio=1.0):
    if mask_polygon != None:
        minx, miny, maxx, maxy = mask_polygon.bounds
    else:
        minx, miny, maxx, maxy = bbox_coord
    
    # Calculate the center of the bounding box
    center_x = (maxx + minx) / 2
    center_y = (maxy + miny) / 2

    # Calculate the dimensions of the new bounding box
    new_width = (maxx - minx) * size_ratio * aspect_ratio
    new_height = (maxy - miny) * size_ratio / aspect_ratio

    # Calculate the new minx, miny, maxx, maxy based on the new dimensions
    minx = center_x - new_width / 2
    miny = center_y - new_height / 2
    maxx = center_x + new_width / 2
    maxy = center_y + new_height / 2

    # Draw the ellipse
    bbox = [minx, miny, maxx, maxy]
    canvas.ellipse(bbox, outline=outline_color, width=width)

def draw_arrow(canvas, bbox_coord, outline_color, line_width, max_arrow_length=100):
    left, top, right, bottom = bbox_coord
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2

    # Arrow length related to the bounding box size
    bounding_box_size_length = min(right - left,  bottom - top)
    if 0.8 * bounding_box_size_length > max_arrow_length:
        min_arrow_length = 0.8 * bounding_box_size_length
    else:
        min_arrow_length = max_arrow_length
        max_arrow_length = 0.8 * bounding_box_size_length
    arrow_length = random.uniform(min_arrow_length, max_arrow_length)

    # Randomize the arrow angle
    angle = random.uniform(0, 2 * math.pi)
    center_x += random.uniform(-0.25, 0.25) * (right - left)
    center_y += random.uniform(-0.25, 0.25) * (bottom - top)

    # Arrowhead size related to arrow length
    arrow_head_size = max(random.uniform(0.2, 0.5) * arrow_length, 6)
    
    # Recalculate the arrow end to ensure it connects properly with the arrowhead
    arrow_end_x = center_x + (arrow_length - arrow_head_size) * math.cos(angle)
    arrow_end_y = center_y + (arrow_length - arrow_head_size) * math.sin(angle)

    if random.random() < 0.5:
        # Draw with a "wobble" to mimic human drawing
        mid_x = (center_x + arrow_end_x) / 2 + random.uniform(-5, 5)
        mid_y = (center_y + arrow_end_y) / 2 + random.uniform(-5, 5)
        canvas.line([(center_x, center_y), (mid_x, mid_y), (arrow_end_x, arrow_end_y)], 
                    fill=outline_color, width=line_width)
    else:
        # Draw the arrow line
        canvas.line([(center_x, center_y), (arrow_end_x, arrow_end_y)], fill=outline_color, width=line_width)
    arrow_end_x = center_x
    arrow_end_y = center_y
    # Draw the arrow head
    if random.random() < 0.5:
        canvas.polygon([
            (arrow_end_x + arrow_head_size * math.cos(angle + math.pi / 3),
             arrow_end_y + arrow_head_size * math.sin(angle + math.pi / 3)),
            (arrow_end_x, arrow_end_y),
            (arrow_end_x + arrow_head_size * math.cos(angle - math.pi / 3),
             arrow_end_y + arrow_head_size * math.sin(angle - math.pi / 3))
        ], fill=outline_color)
    else:
        canvas.line([
            (arrow_end_x + arrow_head_size * math.cos(angle + math.pi / 3),
             arrow_end_y + arrow_head_size * math.sin(angle + math.pi / 3)),
            (arrow_end_x, arrow_end_y),
            (arrow_end_x + arrow_head_size * math.cos(angle - math.pi / 3),
             arrow_end_y + arrow_head_size * math.sin(angle - math.pi / 3))
        ], fill=outline_color, width=line_width)
    
def draw_rounded_triangle(canvas, bbox_coord, mask_polygon, outline_color, width):
    while True:
        points = []
        for _ in range(3):
            if mask_polygon != None:
                point = get_random_point_within_polygon(mask_polygon)
            else:
                point = get_random_point_within_polygon(bbox_coord)
            points.append(point)
        if is_max_angle_less_than_150(points):
            break
    canvas.line([points[0], points[1], points[2], points[0]], fill=outline_color, width=width, joint='curve')

def draw_point(canvas, bbox_coord, mask_polygon, outline_color=(255, 0, 0), radius=3, aspect_ratio=1.0):
    # Calculate the center and covariance matrix for multivariate normal distribution
    if mask_polygon != None:
        minx, miny, maxx, maxy = mask_polygon.bounds
    else:
        minx, miny, maxx, maxy = bbox_coord
    mean = [(maxx + minx) / 2, (maxy + miny) / 2]
    cov = [[(maxx - minx) / 8, 0], [0, (maxy - miny) / 8]]

    # Initialize counter for fail-safe mechanism
    counter = 0

    # Generate a random central point within the mask using a normal distribution
    max_tries = 10
    while True:
        cx, cy = multivariate_normal.rvs(mean=mean, cov=cov)
        center_point = Point(cx, cy)
        if mask_polygon.contains(center_point):
            break
        counter += 1
        if counter >= max_tries:
            cx, cy = multivariate_normal.rvs(mean=mean, cov=cov)
            center_point = Point(cx, cy)
            break
    
    x_radius = radius * aspect_ratio
    y_radius = radius / aspect_ratio
    bbox = [cx - x_radius, cy - y_radius, cx + x_radius, cy + y_radius]

    # Draw the ellipse and fill it with color
    canvas.ellipse(bbox, outline=outline_color, fill=outline_color)

def draw_scribble(canvas, bbox_coord, mask_polygon, outline_color=(255, 0, 0), width=3):
    prev_point = None # Initailize prev_point outside the loop
    if mask_polygon != None:
        p0 = get_random_point_within_polygon(mask_polygon)
        p1 = get_random_point_within_polygon(mask_polygon)
        p2 = get_random_point_within_polygon(mask_polygon)
        p3 = get_random_point_within_polygon(mask_polygon)
    else:
        p0 = get_random_point_within_bbox(bbox_coord)
        p1 = get_random_point_within_bbox(bbox_coord)
        p2 = get_random_point_within_bbox(bbox_coord)
        p3 = get_random_point_within_bbox(bbox_coord)
    
    for t in np.linspace(0, 1, 1000):
        x = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
        
        current_point = (x, y)
        if prev_point:
            canvas.line([prev_point, current_point], fill=outline_color, width=width)
        
        prev_point = current_point  # Update prev_point to the current ending point
 
def draw_mask_contour(canvas, bbox_coord, segmentation_coords, color="red", width=1):
    if segmentation_coords == None:
        segmentation_coords = [[bbox_coord[0], bbox_coord[1], bbox_coord[0], bbox_coord[3],
                                bbox_coord[2], bbox_coord[3], bbox_coord[2], bbox_coord[1]]]
    for segment in segmentation_coords:
        coords = [(segment[i], segment[i+1]) for i in range(0, len(segment), 2)]
        for dx in range(-width, width+1):
            for dy in range(-width, width+1):
                shifted_coords = [(x + dx, y + dy) for x, y in coords]
                canvas.polygon(shifted_coords, outline=color)

def draw_mask(canvas, bbox_coord, segmentation_coords, color="red", width=1):
    for segment in segmentation_coords:
        coords = [(segment[i], segment[i+1]) for i in range(0, len(segment), 2)]
        canvas.polygon(coords, outline=None, fill=color, width=width)
    


def image_blending(image, shape='rectangle', bbox_coord=None, segmentation=None, 
                   ori_height=None, ori_width=None, alpha=None, rgb_value=None):
    visual_prompt_img = Image.new('RGBA', (ori_width, ori_height), (0, 0, 0, 0))
    visual_prompt_img_canvas = ImageDraw.Draw(visual_prompt_img)
    if alpha == None:
        alpha = random.randint(96, 255) if shape != 'mask' else random.randint(48, 128)
    color_alpha = rgb_value + (alpha, )
    if isinstance(segmentation, dict):
        if isinstance(segmentation['counts'], list):
            # convert to compressed RLE
            segmentation = mask.frPyObjects(segmentation, ori_height, ori_width)
        m = mask.decode(segmentation)
        m = m.astype(np.uint8).squeeze()
        contours, hierarchy = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour.flatten() for contour in contours]
        try:
            polygons = []
            for contour in contours:
                mask_polygon = Polygon([(contour[i], contour[i+1]) for i in range(0, len(contour), 2)])
                polygons.append(mask_polygon)
            mask_polygon = random.choice(polygons)
            try:
                all_polygons_union = unary_union(polygons)
            except:
                all_polygons_union = None
        except:
            mask_polygon = None
    elif segmentation:
        contours = segmentation
        try:
            polygons = []
            for segmentation_coord in segmentation:
                mask_polygon = Polygon([(segmentation_coord[i], segmentation_coord[i+1])
                                         for i in range(0, len(segmentation_coord), 2)])
                polygons.append(mask_polygon)
            mask_polygon = polygons[0]
            try:
                all_polygons_union = unary_union(polygons)
            except:
                all_polygons_union = None
        except:
            mask_polygon = None
    else:
        contours = None
        all_polygons_union = None
        mask_polygon = None

    if shape == 'rectangle':
        line_width = random.choice([2, 3, 4, 5, 6, 7, 8])
        draw_rectangle(visual_prompt_img_canvas, bbox_coord, color_alpha, line_width)
    elif shape == 'ellipse':
        line_width = random.choice([2, 3, 4, 5, 6, 7, 8])
        size_ratio = random.uniform(1, 1.5)
        draw_ellipse(visual_prompt_img_canvas, bbox_coord, all_polygons_union, 
                     color_alpha, line_width, size_ratio=size_ratio)
    elif shape == 'arrow':
        line_width = random.choice([1, 2, 3, 4, 5, 6])
        max_arrow_length = 50
        draw_arrow(visual_prompt_img_canvas, bbox_coord, color_alpha, line_width, max_arrow_length)
    elif shape == 'triangle':
        line_width = random.choice([2, 3, 4, 5, 6, 7, 8])
        draw_rounded_triangle(visual_prompt_img_canvas, bbox_coord, all_polygons_union, color_alpha, line_width)
    elif shape == 'point':
        radius = random.choice(list(range(3, 10)))
        aspect_ratio = 1 if random.random() < 0.5 else random.uniform(0.5, 2.0)
        draw_point(visual_prompt_img_canvas, bbox_coord, mask_polygon, color_alpha, radius, aspect_ratio)
    elif shape == 'scribble':
        line_width = random.choice(list(range(2, 13)))
        draw_scribble(visual_prompt_img_canvas, bbox_coord, mask_polygon, color_alpha, line_width)
    elif shape == 'mask_contour':
        line_width = random.choice([1, 2, 3, 4])
        draw_mask_contour(visual_prompt_img_canvas, bbox_coord, contours, color_alpha, line_width)
    else:
        raise NotImplementedError
    
    image = image.convert('RGBA')
    image = Image.alpha_composite(image, visual_prompt_img)
    image = image.convert('RGB')

    visual_prompt_img = np.array(visual_prompt_img.convert('RGB'))
    visual_prompt_img = np.uint8(np.sum(visual_prompt_img, axis=-1) > 10)
    
    return image, visual_prompt_img

        
def point_rendering(points, colors, ori_height, ori_width):
    merged_visual_prompts = Image.new('RGB', (ori_width, ori_height), (0, 0, 0))
    radius = random.choice(list(range(3, 11)))
    aspect_ratio = 1 if random.random() < 0.5 else random.uniform(0.5, 2.0)
    alpha = random.randint(96, 255)

    _regions = []
    for i, point in enumerate(points):
        vprompt_img = Image.new('RGBA', (ori_width, ori_height), (0, 0, 0, 0))
        canvas = ImageDraw.Draw(vprompt_img)
        color = (int(colors[i][0] * 255), int(colors[i][1] * 255), int(colors[i][2] * 255))
        if color[0] == 0 and color[1] == 0 and color[2] == 0:
            color = (int(colors[-1][0] * 255), int(colors[-1][1] * 255), int(colors[-1][2] * 255))
        color_alpha = color + (alpha, )
        for _point in point:
            cx, cy = _point[0], _point[1]
            x_radius = radius * aspect_ratio
            y_radius = radius * aspect_ratio
            bbox = [cx - x_radius, cy - y_radius, cx + x_radius, cy + y_radius]
            canvas.ellipse(bbox, outline=color_alpha, fill=color_alpha)
        merged_visual_prompts = merged_visual_prompts.convert('RGBA')
        merged_visual_prompts = Image.alpha_composite(merged_visual_prompts, vprompt_img)
        merged_visual_prompts = merged_visual_prompts.convert('RGB')

        vprompt_img = np.array(vprompt_img.convert('RGB'))
        vprompt_img = np.uint8(np.sum(vprompt_img, axis=-1) > 10)
        _regions.append(vprompt_img)
    _regions = np.stack(_regions, axis=0)  # n, h, w

    return _regions, merged_visual_prompts

def box_rendering(boxes, colors, ori_height, ori_width):
    merged_visual_prompts = Image.new('RGB', (ori_width, ori_height), (0, 0, 0))
    # merged_visual_prompts = image
    alpha = random.randint(96, 255)
    line_width = random.choice([2, 3, 4, 5, 6, 7,])

    _regions = []
    for i, box in enumerate(boxes):
        vprompt_img = Image.new('RGBA', (ori_width, ori_height), (0, 0, 0, 0))
        canvas = ImageDraw.Draw(vprompt_img)
        color = (int(colors[i][0] * 255), int(colors[i][1] * 255), int(colors[i][2] * 255))
        if color[0] == 0 and color[1] == 0 and color[2] == 0:
            color = (int(colors[-1][0] * 255), int(colors[-1][1] * 255), int(colors[-1][2] * 255))
        color_alpha = color + (alpha, )

        left, top, right, bottom = box
        canvas.rectangle([(left, top), (right, bottom)], outline=color_alpha, width=line_width)
        
        merged_visual_prompts = merged_visual_prompts.convert('RGBA')
        merged_visual_prompts = Image.alpha_composite(merged_visual_prompts, vprompt_img)
        merged_visual_prompts = merged_visual_prompts.convert('RGB')

        vprompt_img = np.array(vprompt_img.convert('RGB'))
        vprompt_img = np.uint8(np.sum(vprompt_img, axis=-1) > 10)
        _regions.append(vprompt_img)
    _regions = np.stack(_regions, axis=0)  # n, h, w

    return _regions, merged_visual_prompts

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