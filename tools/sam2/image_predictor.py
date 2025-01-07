import torch
import numpy as np
import mmcv
from mmengine.visualization import Visualizer

from third_parts.sam2.build_sam import build_sam2
from third_parts.sam2.sam2_image_predictor import SAM2ImagePredictor
from mmdet.structures.mask import bitmap_to_polygon

IMG_PATH = 'assets/view.jpg'
MODEL_CKPT = "work_dirs/ckpt/sam2_hiera_large.pt"
MODEL_CFG = "sam2_hiera_l.yaml"


def prepare():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


if __name__ == '__main__':
    prepare()
    sam2_model = build_sam2(MODEL_CFG, MODEL_CKPT, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    image = mmcv.imread(IMG_PATH)
    predictor.set_image(image)
    input_point = np.array([[500, 475]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]


    visualizer = Visualizer(image=image)
    masks = masks.astype(bool)
    masks = masks[0:1]
    polygons = []
    for i, mask in enumerate(masks):
        contours, _ = bitmap_to_polygon(mask)
        polygons.extend(contours)
    visualizer.draw_polygons(polygons, edge_colors='w', alpha=0.8)
    visualizer.draw_binary_masks(masks, alphas=0.8)

    visualizer.draw_points(input_point, 'r', marker='*')

    result = visualizer.get_image()
    