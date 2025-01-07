import os

import torch
import numpy as np
import mmcv
import mmengine
from mmengine.visualization import Visualizer

from third_parts.sam2.build_sam import build_sam2_video_predictor
from mmdet.structures.mask import bitmap_to_polygon

VID_PATH = 'assets/vid_view'
MODEL_CKPT = "work_dirs/ckpt/sam2_hiera_large.pt"
MODEL_CFG = "sam2_hiera_l.yaml"


def prepare():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


if __name__ == '__main__':
    prepare()
    predictor = build_sam2_video_predictor(MODEL_CFG, MODEL_CKPT)
    inference_state = predictor.init_state(video_path=VID_PATH)

    input_point = np.array([[255, 475]])
    input_label = np.array([1])

    ann_frame_idx = 0
    ann_obj_id = 1

    _frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=input_point,
        labels=input_label,
    )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }



    # Visualization
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(VID_PATH)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    mmengine.mkdir_or_exist("./result")
    for idx in range(len(frame_names)):
        image = mmcv.imread(os.path.join(VID_PATH, frame_names[idx]))
        visualizer = Visualizer(image=image)
        masks = video_segments[idx]
        polygons = []
        vis_masks = []
        for i, mask in masks.items():
            contours, _ = bitmap_to_polygon(mask[0])
            polygons.extend(contours)

            vis_masks.append(mask[0])
        visualizer.draw_polygons(polygons, edge_colors='w', alpha=0.8)
        visualizer.draw_binary_masks(np.concatenate(vis_masks, axis=0), alphas=0.8)

        # visualizer.draw_points(input_point, 'r', marker='*')

        result = visualizer.get_image()
        mmcv.imwrite(result, os.path.join('./result', frame_names[idx]))
