import torch
import torch.nn as nn
import torch.nn.functional as F

from xtuner.registry import BUILDER

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from xtuner.model.utils import guess_load_checkpoint

from mmcv.ops import point_sample
from mmdet.models.utils import get_uncertain_point_coords_with_randomness

from mmengine.model import BaseModel
from projects.ST.dataset.utils import convert_image_to_patches
from projects.ST.dataset.collect_fns import create_single_prefix_mask
from einops import rearrange
from transformers import DynamicCache, GenerationConfig
import copy
from mmengine.config import Config, ConfigDict
from peft import get_peft_model, prepare_model_for_kbit_training

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    if 'output_layer' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('output_layer')
    return list(lora_module_names)

NON_VISION_TOKEN = -1
PROMPT_TMPL = '<|im_start|>user\n{input}<|im_end|>\n'

class Sa2VASTModel(BaseModel):
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"
    def __init__(self,
                 single_transformer,
                 tokenizer,
                 single_transformer_lora=None,
                 seg_hidden_states=256,
                 patch_size=32,
                 seg_pred_down_ratio=4,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 # for inference
                 template=None,
                 add_cls=False,
                 bs=1,
                 ):
        super().__init__()
        self.add_cls = add_cls
        self.bs = bs
        self.patch_size = patch_size
        self.seg_pred_down_ratio = seg_pred_down_ratio
        self.seg_hidden_states = seg_hidden_states
        if special_tokens is None:
            special_tokens = ['[SEG]']
        self.special_tokens = special_tokens
        self.single_transformer = BUILDER.build(single_transformer)
        self.llm = self.single_transformer

        self.tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens()

        in_dim = self.single_transformer.config.hidden_size # the hidden states of llm
        out_dim = seg_hidden_states
        self.seg_token_projector = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        out_dim = seg_hidden_states * (patch_size // seg_pred_down_ratio) ** 2
        self.image_feature_projector = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        if single_transformer_lora is not None:
            self.single_transformer.requires_grad_(False)
            self.activation_checkpointing_enable()
            self.single_transformer.enable_input_require_grads()
            self._prepare_llm_for_lora(single_transformer_lora)
            self.single_transformer.model.base_model.get_input_embeddings().requires_grad_(True)
            self.single_transformer.lm_head.requires_grad_(True)

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.template = template
        self.template['INSTRUCTION'] = PROMPT_TMPL

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.single_transformer.model = prepare_model_for_kbit_training(
            self.single_transformer.model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.single_transformer.model)
            lora_config.target_modules = modules
        self.single_transformer.model = get_peft_model(self.single_transformer.model,
                                                   lora_config)

    def activation_checkpointing_disable(self):
        self.single_transformer.gradient_checkpointing_disable()

    def activation_checkpointing_enable(self):
        self.single_transformer.gradient_checkpointing_enable()

    def _add_special_tokens(self):

        self.tokenizer.vis_beg_tok = "<vision>"
        self.tokenizer.vis_patch_tok = "<vpatch>"
        self.tokenizer.vis_rsep_tok = "<vrow_sep>"
        self.tokenizer.vis_frm_tok = "<vframe_sep>"
        self.tokenizer.vis_end_tok = "</vision>"
        self.tokenizer.vis_cls_tok = "<|vis_cls|>"

        special_tokens = self.special_tokens
        _num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        if _num_new_tokens > 0:
            self.single_transformer.resize_token_embeddings(len(self.tokenizer))
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.vision_patch_idx = self.tokenizer("<vpatch>", add_special_tokens=False).input_ids[0]

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

    def _get_pesudo_data(self, device):
        gt_masks = torch.zeros((1, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return gt_masks

    def get_mask_prediction(self, seg_embeddings_list, image_seg_features):
        # seg_embedding (N, C)
        # image_feature (H, W, C)
        ret = []
        for seg_embeddings, image_seg_feature in zip(seg_embeddings_list, image_seg_features):
            pred_masks = torch.einsum("qc,hwc->qhw", seg_embeddings, image_seg_feature)
            ret.append(pred_masks)
        return ret

    def forward(self, data, data_samples=None, mode='loss'):
        gt_masks = data.pop('masks', None)
        patch_nums_per_images = data.pop('patch_nums_per_images', None)
        input_ids = data['input_ids']

        if 'vision_patches' in data.keys() and data['vision_patches'] is not None:
            data['vision_patches'] = data['vision_patches'].flatten(1).to(self.torch_dtype)

        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            gt_masks = self._get_pesudo_data(
                device=input_ids.device,
            )
        else:
            seg_valid = True

        output = self.single_transformer(**data, return_dict=True, output_hidden_states=True)
        hidden_states = output.hidden_states
        # using last layer hidden states
        hidden_states = hidden_states[-1]

        # obtain image features
        image_token_mask = input_ids == self.vision_patch_idx
        vision_features = self.image_feature_projector(hidden_states[image_token_mask])  # (N, 256 * sub_pixels * sub_pixels)
        patch_split_nums = [item[0] * item[1] for item in patch_nums_per_images]
        vision_features = torch.split(vision_features, patch_split_nums, dim=0)
        all_image_features = []
        for patch_num, image_features in zip(patch_nums_per_images, vision_features):
            sub_pixels = self.patch_size // self.seg_pred_down_ratio
            h_patches, w_patches = patch_num
            if h_patches * w_patches == 0:
                # no image
                all_image_features.append(None)
            else:
                image_features = image_features.reshape(h_patches, w_patches, self.seg_hidden_states, sub_pixels, sub_pixels)
                image_features = image_features.permute(0, 3, 1, 4, 2)  # (h_patches, sub_pixels, w_patches, sub_pixels, seg_hidden_states)
                image_features = image_features.flatten(0, 1).flatten(1, 2)  # (h // down_ratio, w // down_ratio, c)
                all_image_features.append(image_features)

        # obtain seg tokens
        seg_token_mask = input_ids == self.seg_token_idx
        if seg_valid:
            seg_token_features = self.seg_token_projector(hidden_states[seg_token_mask])
        else:
            seg_token_features = self.seg_token_projector(hidden_states[:, :1].flatten(0, 1))
        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 1

        seg_embeddings_list_ = torch.split(seg_token_features, seg_token_counts.tolist(), dim=0)
        seg_embeddings_list = []
        image_seg_features = []
        gt_masks_ = []
        for idx, item in enumerate(seg_embeddings_list_):
            if len(item) != 0 and all_image_features[idx] is not None:
                seg_embeddings_list.append(item)
                image_seg_features.append(all_image_features[idx])
                gt_masks_.append(gt_masks[idx])
        gt_masks = gt_masks_

        pred_masks = self.get_mask_prediction(seg_embeddings_list, image_seg_features)
        if not self.loss_sample_points:
            gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_mask.shape[-2:], mode='nearest').squeeze(0) for
                        gt_mask, pred_mask in zip(gt_masks, pred_masks)]

        loss_mask, loss_dice = 0, 0
        n_masks = 0
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            # pred and gt mask, (n, h, w)
            if len(pred_mask) != len(gt_mask):
                # drop this data
                print(f"Pred mask shape {pred_mask.shape} is not equal to gt_mask shape {gt_mask.shape} !!!")
                min_num = min(len(pred_mask), len(gt_mask))
                pred_mask = pred_mask[:min_num]
                gt_mask = gt_mask[:min_num]
                _seg_valid = False
            else:
                _seg_valid = True

            if self.loss_sample_points:
                sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_mask, gt_mask)
                sam_loss_dice = self.loss_dice(
                    sampled_pred_mask,
                    sampled_gt_mask, avg_factor=(1 + 1e-4))
                sam_loss_mask = self.loss_mask(
                    sampled_pred_mask.reshape(-1),
                    sampled_gt_mask.reshape(-1),
                    avg_factor=(sampled_pred_mask.shape[1] + 1e-4))
            else:
                sam_loss_mask = self.loss_mask(pred_mask, gt_mask) * len(pred_mask)
                sam_loss_dice = self.loss_dice(pred_mask, gt_mask) * len(pred_mask)

            if _seg_valid and seg_valid:
                _scale = 1.0
                n_masks += len(pred_mask)
            else:
                _scale = 0.0

            loss_mask += sam_loss_mask * _scale
            loss_dice += sam_loss_dice * _scale

        if loss_mask == 0.0:
            _llm_loss_scale = 1.0
        else:
            _llm_loss_scale = 0.1

        loss_dict = {
            'loss_mask': loss_mask / (n_masks + 1e-4) + output.loss * 0.0,
            'loss_dice': loss_dice / (n_masks + 1e-4) + output.loss * 0.0,
            'llm_loss': output.loss * _llm_loss_scale,
        }
        return loss_dict

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def preparing_for_generation(self, metainfo, **kwargs):
        # set stop criteria and generation configs for model
        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
        self.bot_name = 'BOT'
        if 'template' in metainfo.keys():
            template = metainfo['template']
        else:
            template = PROMPT_TEMPLATE['phi3_chat']
        if self.template is None:
            self.template = template
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            temperature=0,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True

        self.single_transformer.to(self.torch_dtype)
        self.seg_token_projector.to(self.torch_dtype)
        self.image_feature_projector.to(self.torch_dtype)
        return

    def prepare_image_textual_seq_norowsep(self, h, w):
        image_token_patch_indices = []
        seq = ""
        tok_len = 0

        seq += self.IMG_START_TOKEN
        tok_len += 1
        image_token_patch_indices.append(NON_VISION_TOKEN)

        seq += self.IMG_CONTEXT_TOKEN * (w * h)
        tok_len += (w * h)
        image_token_patch_indices += [idx for idx in range(w * h)]

        seq += self.IMG_END_TOKEN
        tok_len += 1
        image_token_patch_indices.append(NON_VISION_TOKEN)

        if self.add_cls:
            seq += self.CLS_TOKEN
            tok_len += 1
            image_token_patch_indices.append(NON_VISION_TOKEN)
        return seq, tok_len, image_token_patch_indices

    def predict_forward(
            self,
            image=None,
            text=None,
            past_text='',
    ):
        assert self.tokenizer

        input_dict = {}
        ori_image_size = image.size

        if image is None:
            input_dict['vision_patches'] = None
            input_dict['patch_nums_per_images'] = (0, 0)
            image_token_str = ''
            image_token_patch_indices = []
        else:
            image_patches = convert_image_to_patches(image, self.patch_size)
            # tensor, (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
            h_patches, w_patches = image_patches.shape[:2]
            n_patches = h_patches * w_patches
            # input_dict['vision_patches'] = image_patches.view(n_patches, -1)  # (n_patches, 3*patch_size*patch_size)
            input_dict['vision_patches'] = image_patches.flatten(0, 1).flatten(1)  # (n_patches, 3*patch_size*patch_size)
            input_dict['patch_nums_per_images'] = (h_patches, w_patches)
            image_token_str, image_token_len, image_token_patch_indices = \
                self.prepare_image_textual_seq_norowsep(
                    image_patches.shape[0], image_patches.shape[1]
                )

        ret_masks = []
        if '<image>' in text:
            assert past_text is None or len(past_text) == 0
            first_conv = True
        else:
            first_conv = False
        text = text.replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
        input_text = ''
        input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
        if first_conv:
            input_text = image_token_str + input_text
        else:
            input_text = past_text + input_text

        ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        vision_start_end = self.search_vision_tokens(ids)

        attention_mask = create_single_prefix_mask(vision_start_end, len(ids)).unsqueeze(0).unsqueeze(0).cuda()
        # attention_mask = create_single_prefix_mask(vision_start_end, len(ids)).unsqueeze(0).cuda()

        ids = torch.tensor(ids).cuda().unsqueeze(0)
        position_ids = generate_mm_pos_ids_singleit(
            ids[0].cpu().numpy().tolist(), self.vision_patch_idx,
            input_dict['patch_nums_per_images'][0], input_dict['patch_nums_per_images'][1]).unsqueeze(1).cuda()

        vision_patch_indices = []
        vision_patch_indices += image_token_patch_indices
        vision_patch_indices += [NON_VISION_TOKEN] * (ids.shape[-1] - len(vision_patch_indices))

        vision_patch_indices = torch.tensor(vision_patch_indices).cuda().unsqueeze(0)

        padding_attention_mask = torch.ones_like(ids).cuda()

        mm_inputs = {
            'vision_patches': input_dict['vision_patches'].flatten(1).cuda().to(self.torch_dtype),
            # 'vision_patches': None,
            'input_ids': ids,
            'attention_mask': padding_attention_mask,
            'position_ids': position_ids,
            'labels': None,
            'vision_patch_indices': vision_patch_indices,
        }

        # first forward for none casual image tokens
        image_tokens_len = vision_start_end[-1] + 1
        cached_inputs = dict(
            input_ids=ids[:, :image_tokens_len],
            position_ids=position_ids[:, :, :image_tokens_len],
            attention_mask=attention_mask[:, :, :image_tokens_len, :image_tokens_len],
            vision_patches=mm_inputs['vision_patches'],
            vision_patch_indices=vision_patch_indices[:, :image_tokens_len],
            use_cache=True
        )
        prefix_cache = DynamicCache()
        with torch.no_grad():
            prefix_cache = self.single_transformer.forward(**cached_inputs, past_key_values=prefix_cache,
                                                           return_dict=True, output_hidden_states=True)
            past_hidden_states = prefix_cache.hidden_states
            prefix_cache = prefix_cache.past_key_values
        past_key_values = copy.deepcopy(prefix_cache)

        generate_output = self.single_transformer.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=False).strip()

        # past key tokens
        last_past_hidden_states = past_hidden_states[-1][0]

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)

        last_hidden_states = torch.cat([last_past_hidden_states, last_hidden_states], dim=0)

        # obtain image features
        image_token_mask = ids[0] == self.vision_patch_idx
        vision_features = self.image_feature_projector(
            last_hidden_states[:len(ids[0])][image_token_mask])  # (N, 256 * sub_pixels * sub_pixels)
        patch_split_nums = [item[0] * item[1] for item in [input_dict['patch_nums_per_images']]]
        vision_features = torch.split(vision_features, patch_split_nums, dim=0)
        all_image_features = []
        for patch_num, image_features in zip([input_dict['patch_nums_per_images']], vision_features):
            sub_pixels = self.patch_size // self.seg_pred_down_ratio
            h_patches, w_patches = patch_num
            if h_patches * w_patches == 0:
                # no image
                all_image_features.append(None)
            else:
                image_features = image_features.reshape(h_patches, w_patches, self.seg_hidden_states, sub_pixels,
                                                        sub_pixels)
                image_features = image_features.permute(0, 3, 1, 4,
                                                        2)  # (h_patches, sub_pixels, w_patches, sub_pixels, seg_hidden_states)
                image_features = image_features.flatten(0, 1).flatten(1, 2)  # (h // down_ratio, w // down_ratio, c)
                all_image_features.append(image_features)
        image_features = all_image_features[0]

        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.seg_token_projector(seg_hidden_states)
        if all_seg_hidden_states.shape[0] == 0:
            ret_masks = None
        else:
            pred_masks = torch.einsum("qc,hwc->qhw", all_seg_hidden_states, image_features)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)[0]
            masks = masks.sigmoid() > 0.5
            # masks = masks.cpu().numpy()
            masks = masks.cpu()
            ret_masks.append(masks)

        return {'prediction': predict, 'prediction_masks': ret_masks, 'input_text': ''}

    def search_vision_tokens(self, input_ids):
        image_start_idx = self.tokenizer(self.IMG_START_TOKEN, add_special_tokens=False).input_ids[0]
        image_end_idx = self.tokenizer(self.IMG_END_TOKEN, add_special_tokens=False).input_ids[0]
        if image_start_idx not in input_ids:
            return None
        else:
            start_idx = input_ids.index(image_start_idx)
            end_idx = input_ids.index(image_end_idx)
            return [start_idx+1, end_idx]

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]


def generate_mm_pos_ids_singleit(input_ids, vpatch_id, h, w):
    input_ids_pt = torch.Tensor(input_ids).int()
    vpatch_pos = torch.argwhere(input_ids_pt == vpatch_id)
    vpatch_start_pos = vpatch_pos[0].item()
    nt = len(input_ids) - (h * w) + 1

    # v_pos
    t_indices = torch.arange(1)
    h_indices = torch.arange(h)
    w_indices = torch.arange(w)
    v_pos_id = torch.stack(torch.meshgrid(t_indices, h_indices, w_indices, indexing='ij'), dim=0)
    v_pos_id = rearrange(v_pos_id, "d t h w -> (t h w) d")  # [h*w, 3]
    v_pos_id += vpatch_start_pos
    position_id = torch.cat(
        [
            torch.arange(vpatch_start_pos).unsqueeze(-1).repeat(1, 3),
            v_pos_id,
            torch.arange(nt - vpatch_start_pos - 1).unsqueeze(-1).repeat(1, 3) + v_pos_id.max() + 1,
        ],
        dim=0
    )
    assert len(input_ids) == position_id.size(0)
    position_id = rearrange(position_id, "slen d -> d slen").long()

    return position_id
