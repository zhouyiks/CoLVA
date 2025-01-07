import numpy as np
from matplotlib import path


def _spaced_points(low, high,n):
    """ We want n points between low and high, but we don't want them to touch either side"""
    padding = (high-low)/(n*2)
    return np.linspace(low + padding, high-padding, num=n)

def make_mask(height, width, box, polygons_list):
    """
    Mask size: int about how big mask will be
    box: [x1, y1, x2, y2,*]
    polygons_list: List of polygons that go inside the box
    """
    mask = np.zeros((height, width), dtype=np.bool_)

    xy = np.meshgrid(_spaced_points(box[0], box[2], n=width),
                     _spaced_points(box[1], box[3], n=height)) 
    xy_flat = np.stack(xy, 2).reshape((-1, 2))

    for polygon in polygons_list:
        polygon_path = path.Path(polygon)
        mask |= polygon_path.contains_points(xy_flat).reshape((height, width))
    return mask.astype(np.float32)

def vcr_decode_mask_fn(bboxes, segms, ori_height, ori_width):
    pred_masks = np.zeros((len(segms), ori_height, ori_width))
    for i, segm in enumerate(segms):
        int_box = [round(box) for box in bboxes[i][:4]]
        
        height_ = int(int_box[3] - int_box[1])
        width_ = int(int_box[2] - int_box[0])
        box_mask = make_mask(height_, width_, bboxes[i], segm)

        pred_masks[i, int_box[1]:int_box[3], int_box[0]:int_box[2]] = box_mask
    
    return pred_masks