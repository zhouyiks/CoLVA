import copy
from third_parts.tokenize_anything import model_registry
from third_parts.tokenize_anything.utils.image import im_rescale
from third_parts.tokenize_anything.utils.image import im_vstack
from mmengine.model import BaseModel
import cv2
import numpy as np
import torch
import os
import json
import pycocotools.mask as maskUtils


class TAP(BaseModel):
    def __init__(self,
        model_type="tap_vit_l",
        checkpoint="../models/tap_vit_l_v1_1.pkl",
        concept_weights="../concepts/merged_2560.pkl",
        tokenizer=None,
        save_folder='./work_dirs/tap_caption_results/rank_0',
        ):
        super().__init__()
        if isinstance(model_type, list) or isinstance(model_type, tuple):
            model_type = model_type[0]
        self.tap_model = model_registry[model_type](checkpoint=checkpoint)
        self.tap_model.concept_projector.reset_weights(concept_weights)
        self.tap_model.text_decoder.reset_cache(max_batch_size=256)
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        self.item_index = 0

        if tokenizer is not None:
            self.tokenizer = tokenizer
            tokenizer_type = self.tokenizer['type']
            del self.tokenizer['type']
            self.tokenizer = tokenizer_type(**self.tokenizer)

    def _mesh_grids_coords(self, image_size, grid_size=16):
        h, w = image_size
        x_stride = w * 1.0 / grid_size
        y_stride = h * 1.0 / grid_size
        x_start = x_stride / 2.0
        y_start = y_stride / 2.0
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = x_start + i * x_stride
                y = y_start + j * y_stride
                grid_points.append(
                    [[x, y, 1], [0, 0, 4]]
                )
        grid_points = np.array(grid_points, "float32")
        return grid_points

    def forward(self, image_path):
        image = cv2.imread(image_path)

        img_list, img_scales = im_rescale(image, scales=[1024], max_size=1024)
        input_size, original_size = img_list[0].shape, image.shape[:2]

        img_batch = im_vstack(img_list, fill_value=self.tap_model.pixel_mean_value, size=(1024, 1024))
        inputs = self.tap_model.get_inputs({"img": img_batch})
        inputs.update(self.tap_model.get_features(inputs))

        # original_coordinates
        inputs["points"] = self._mesh_grids_coords(original_size, grid_size=16)
        inputs["points"][:, :, :2] *= np.array(img_scales, "float32")

        # Decode outputs for the point prompt.
        outputs = self.tap_model.get_outputs(inputs)

        # Select final mask.
        iou_score, mask_pred = outputs["iou_pred"], outputs["mask_pred"]
        iou_score[:, 0] -= 1000.0  # Penalize the score of boundary boxes.
        mask_index = torch.arange(iou_score.shape[0]), iou_score.argmax(1)

        # Upscale masks to the original image resolution.
        iou_scores, masks = iou_score[mask_index], mask_pred[mask_index]
        masks = self.tap_model.upscale_masks(masks[:, None], img_batch.shape[1:-1])
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = self.tap_model.upscale_masks(masks, original_size).gt(0).cpu().numpy()

        # Predict concepts and generate captions.
        sem_tokens, sem_embeds = outputs["sem_tokens"], outputs["sem_embeds"]
        concepts, scores = self.tap_model.predict_concept(sem_embeds[mask_index])
        captions = self.tap_model.generate_text(sem_tokens[mask_index])

        masks = masks[:, 0]

        # self.visualize(masks, iou_scores, scores, concepts, captions, image)
        return masks, iou_scores, scores, concepts, captions, image

    def predict_forward(self, image_path, **kwargs):
        masks, iou_scores, scores, concepts, captions, image = self.forward(image_path)
        iou_scores = iou_scores.cpu().numpy()
        scores = scores[:, 0]
        masks, captions = self.filter(masks, iou_scores, scores, concepts, captions)
        # self.visualize_filtered(masks, captions, image)
        image_name = image_path.replace('./data/llava_data/'+'LLaVA-Pretrain/images/', '')

        self.save_results(masks, captions, image_name)
        return {}

    def save_results(self, masks, captions, image_name):
        json_file_path = os.path.join(self.save_folder, '{}.json'.format(self.item_index))
        self.item_index += 1

        data = {'image_name': image_name}
        objects = []

        for i in range(len(masks)):
            _mask = masks[i]
            area = int(np.sum(_mask))
            _caption = captions[i]
            _caption = [str(item) for item in _caption]
            rle = maskUtils.encode(np.asfortranarray(_mask).astype(np.uint8))
            rle['counts'] = str(rle['counts'], encoding='utf-8')
            _object = {'segm': rle, 'captions': _caption, 'area': area}
            objects.append(_object)
        data['objects'] = objects
        with open(json_file_path, 'w') as f:
            json.dump(data, f)
        return

    def filter(self, masks, iou_scores, scores, concepts, captions):

        # filter according scores
        keep_indexes = []
        for i in range(len(masks)):
            if iou_scores[i] > 0.8 and np.sum(masks[i]) > 80:
                keep_indexes.append(i)

        masks = masks[keep_indexes]
        scores = scores[keep_indexes]
        iou_scores = iou_scores[keep_indexes]
        captions = captions[keep_indexes]


        overall_scores = scores * iou_scores

        keep, keep_remove_dict = non_maximum_suppression(masks, overall_scores, iou_threshold=0.5)
        masks = masks[keep]
        ret_captions = []
        for idx in keep:
            caption_idxs = keep_remove_dict[idx]
            ret_captions.append(captions[caption_idxs])

        return masks, ret_captions

    def visualize(self, masks, iou_scores, scores, concepts, captions, image):
        save_folder = os.path.join('./work_dirs/', 'tap_sample')
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        # masks
        save_folder_mask = os.path.join(save_folder, 'mask')
        if not os.path.exists(save_folder_mask):
            os.mkdir(save_folder_mask)
        n_objects = len(masks)
        _, h, w = masks.shape
        for i_obj, obj_mask in enumerate(masks):
            obj_mask = obj_mask * 255
            obj_mask = np.stack([obj_mask * 1, obj_mask * 0, obj_mask * 0], axis=2)
            obj_mask = obj_mask * 0.5 + copy.deepcopy(image) * 0.5
            obj_mask = obj_mask.astype(np.uint8)
            cv2.imwrite(os.path.join(save_folder_mask, '{}.png'.format(i_obj)), obj_mask)
            _str = "iou_score: {},\n score: {},\n concept: {},\n caption: {}\n".format(
                iou_scores[i_obj], scores[i_obj], concepts[i_obj], captions[i_obj]
            )
            with open(os.path.join(save_folder_mask, '{}.json'.format(i_obj)), 'w') as f:
                json.dump([_str], f)
        return

    def visualize_filtered(self, masks, captions, image):
        save_folder = os.path.join('./work_dirs/', 'tap_sample_filtered')
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        # masks
        save_folder_mask = os.path.join(save_folder, 'mask')
        if not os.path.exists(save_folder_mask):
            os.mkdir(save_folder_mask)
        n_objects = len(masks)
        _, h, w = masks.shape
        for i_obj, obj_mask in enumerate(masks):
            obj_mask = obj_mask * 255
            obj_mask = np.stack([obj_mask * 1, obj_mask * 0, obj_mask * 0], axis=2)
            obj_mask = obj_mask * 0.5 + copy.deepcopy(image) * 0.5
            obj_mask = obj_mask.astype(np.uint8)
            cv2.imwrite(os.path.join(save_folder_mask, '{}.png'.format(i_obj)), obj_mask)
            _obj_captions = captions[i_obj]
            _str = ""
            for _caption in _obj_captions:
                _str += _caption
                _str += '\n'
            with open(os.path.join(save_folder_mask, '{}.json'.format(i_obj)), 'w') as f:
                json.dump([_str], f)
        return

    def gradient_checkpointing_disable(self):
        return

    def gradient_checkpointing_enable(self):
        return

    def preparing_for_generation(self, *args, **kwargs):
        return

    def forward_points(self, image):
        return


def mask_iou(mask1, other_masks):
    """
    mask1 (h, w)
    other_masks (n, h, w)
    """
    mask1 = mask1.astype(np.float32)
    other_masks = other_masks.astype(np.float32)
    area1 = np.sum(mask1)  # int
    area_other = np.sum(np.sum(other_masks, axis=2), axis=1)  # (n, )

    mask1 = np.expand_dims(mask1, axis=0)
    intersection = np.sum(np.sum(mask1 * other_masks, axis=2), axis=1)  # (n, )

    ious = intersection / (area1 + area_other - intersection + 1e-4)  # (n, )
    return ious


def non_maximum_suppression(masks, scores, iou_threshold=0.7):
    # masks (n, h, w)
    keep = []
    keep_remove_dict = {}

    order = np.argsort(scores)[::-1]

    while order.size > 0:
        i = order[0]
        keep.append(i)

        ious = mask_iou(masks[i], masks[order[1:]])
        remove_idx = np.where(ious > iou_threshold)[0] + 1
        keep_remove_dict[i] = [i] + order[remove_idx].tolist()

        order = order[1:][~np.isin(np.arange(len(order[1:])), remove_idx)]

    return keep, keep_remove_dict
