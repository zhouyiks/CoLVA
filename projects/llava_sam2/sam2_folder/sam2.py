import torch
import torch.nn as nn

from .sam2_predictor import SAM2VideoPredictor
from .sam2_implementation.modeling.backbones.hieradet import Hiera
from .sam2_implementation.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from .sam2_implementation.modeling.position_encoding import PositionEmbeddingSine
from .sam2_implementation.modeling.memory_encoder import MemoryEncoder
from .sam2_implementation.modeling.memory_attention import MemoryAttentionLayer, MemoryAttention
from .sam2_implementation.modeling.sam.transformer import RoPEAttention
from .sam2_implementation.modeling.memory_encoder import MaskDownSampler
from .sam2_implementation.modeling.memory_encoder import Fuser
from .sam2_implementation.modeling.memory_encoder import CXBlock

def load_checkpoint_with_prefix(filename, prefix=None, map_location='cpu', logger='current'):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.
        logger: logger

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if not prefix:
        return state_dict
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict

def load_state_dict_to_model(model, state_dict,  logger='current'):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    if missing_keys:
        print(missing_keys)
        raise RuntimeError()
    if unexpected_keys:
        print(unexpected_keys)
        raise RuntimeError()
    print("Loaded checkpoint successfully")

class SAM2(nn.Module):
    def __init__(
            self,
            ckpt_path: str = None,
    ):
        super().__init__()

        image_encoder = self.build_image_encoder()
        memory_attention = self.build_memory_attention()
        memory_encoder = self.build_memory_encoder()
        sam2_model = SAM2VideoPredictor(
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            num_maskmem = 7,
            image_size = 1024,
            # apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
            sigmoid_scale_for_mem_enc = 20.0,
            sigmoid_bias_for_mem_enc = -10.0,
            use_mask_input_as_output_without_sam = True,
            # Memory
            directly_add_no_mem_embed = True,
            # use high-resolution feature map in the SAM mask decoder
            use_high_res_features_in_sam = True,
            # output 3 masks on the first click on initial conditioning frames
            multimask_output_in_sam = True,
            # SAM heads
            iou_prediction_use_sigmoid = True,
            # cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
            use_obj_ptrs_in_encoder = True,
            add_tpos_enc_to_obj_ptrs = False,
            only_obj_ptrs_in_the_past_for_eval = True,
            # object occlusion prediction
            pred_obj_scores = True,
            pred_obj_scores_mlp = True,
            fixed_no_obj_ptr = True,
            # multimask tracking settings
            multimask_output_for_tracking = True,
            use_multimask_token_for_obj_ptr = True,
            multimask_min_pt_num = 0,
            multimask_max_pt_num = 1,
            use_mlp_for_obj_ptr_proj = True,
            # Compilation flag
            compile_image_encoder = False,
            sam_mask_decoder_extra_args={
                'dynamic_multimask_via_stability':True,
                'dynamic_multimask_stability_delta': 0.05,
                'dynamic_multimask_stability_thresh': 0.98,
            }
        )
        if ckpt_path is not None:
            state_dict = load_checkpoint_with_prefix(ckpt_path)
            load_state_dict_to_model(sam2_model, state_dict)

        self.sam2_model = sam2_model

        self.hidden_dim = self.sam2_model.hidden_dim

        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

    def build_image_encoder(self):
        def build_trunk():
            embed_dim = 144
            num_heads = 2
            stages = [2, 6, 36, 4]
            global_att_blocks = [23, 33, 43]
            window_pos_embed_bkg_spatial_size = [7, 7]
            window_spec = [8, 4, 16, 8]
            ret = Hiera(
                embed_dim=embed_dim,
                num_heads=num_heads,
                stages=stages,
                global_att_blocks=global_att_blocks,
                window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size,
                window_spec=window_spec,
            )
            return ret
        def build_neck():
            def build_position_encoding():
                num_pos_feats = 256
                normalize = True
                scale = None
                temperature = 10000
                ret = PositionEmbeddingSine(
                    num_pos_feats=num_pos_feats,
                    normalize=normalize,
                    scale=scale,
                    temperature=temperature,
                )
                return ret
            d_model = 256
            backbone_channel_list = [1152, 576, 288, 144]
            fpn_top_down_levels = [2, 3]  # output level 0 and 1 directly use the backbone features
            fpn_interp_model = 'nearest'
            position_encoding = build_position_encoding()
            ret = FpnNeck(
                d_model=d_model,
                position_encoding=position_encoding,
                backbone_channel_list=backbone_channel_list,
                fpn_top_down_levels=fpn_top_down_levels,
                fpn_interp_model=fpn_interp_model,
            )
            return ret
        scalp = 1
        trunk = build_trunk()
        neck = build_neck()
        ret = ImageEncoder(scalp=scalp, trunk=trunk, neck=neck)
        return ret

    def build_memory_attention(self):
        def build_layer():
            def build_self_attention():
                rope_theta = 10000.0
                feat_sizes = [32, 32]
                embedding_dim = 256
                num_heads = 1
                downsample_rate = 1
                dropout = 0.1
                ret = RoPEAttention(
                    rope_theta=rope_theta,
                    feat_sizes=feat_sizes,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    downsample_rate=downsample_rate,
                    dropout=dropout
                )
                return ret
            def build_cross_attention():
                rope_theta = 10000.0
                feat_sizes = [32, 32]
                rope_k_repeat = True
                embedding_dim = 256
                num_heads = 1
                downsample_rate = 1
                dropout = 0.1
                kv_in_dim = 64
                ret = RoPEAttention(
                    rope_theta=rope_theta,
                    feat_sizes=feat_sizes,
                    rope_k_repeat=rope_k_repeat,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    downsample_rate=downsample_rate,
                    dropout=dropout,
                    kv_in_dim=kv_in_dim
                )
                return ret
            activation = 'relu'
            dim_feedforward = 2048
            dropout = 0.1
            pos_enc_at_attn = False
            d_model = 256
            pos_enc_at_cross_attn_keys = True
            pos_enc_at_cross_attn_queries = False
            self_attention = build_self_attention()
            cross_attention = build_cross_attention()
            ret = MemoryAttentionLayer(
                activation=activation,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pos_enc_at_attn=pos_enc_at_attn,
                d_model=d_model,
                pos_enc_at_cross_attn_queries=pos_enc_at_cross_attn_queries,
                pos_enc_at_cross_attn_keys=pos_enc_at_cross_attn_keys,
                self_attention=self_attention,
                cross_attention=cross_attention,
            )
            return ret
        d_model = 256
        pos_enc_at_input = True
        num_layers = 4
        layer = build_layer()
        ret = MemoryAttention(
            d_model=d_model,
            pos_enc_at_input=pos_enc_at_input,
            num_layers=num_layers,
            layer=layer,
        )
        return ret

    def build_memory_encoder(self):
        def build_position_encoding():
            num_pos_feats = 64
            normalize = True
            scale = None
            temperature = 10000
            ret = PositionEmbeddingSine(
                num_pos_feats=num_pos_feats,
                normalize=normalize,
                scale=scale,
                temperature=temperature,
            )
            return ret

        def build_mask_downsampler():
            kernel_size = 3
            stride = 2
            padding = 1
            ret = MaskDownSampler(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            return ret

        def build_fuser():
            def build_layer():
                dim = 256
                kernel_size = 7
                padding = 3
                layer_scale_init_value = 1e-6
                use_dwconv = True  # depth-wise convs
                ret = CXBlock(
                    dim=dim, kernel_size=kernel_size,
                    padding=padding, layer_scale_init_value=layer_scale_init_value,
                    use_dwconv=use_dwconv,
                )
                return ret

            num_layers = 2
            layer = build_layer()
            ret = Fuser(
                layer=layer,
                num_layers=num_layers
            )
            return ret

        out_dim = 64
        position_encoding = build_position_encoding()
        mask_downsampler = build_mask_downsampler()
        fuser = build_fuser()
        ret = MemoryEncoder(
            out_dim=out_dim,
            position_encoding=position_encoding,
            mask_downsampler=mask_downsampler,
            fuser=fuser,
        )
        return ret

    def inject_language_embd(self, inference_state, language_embd):
        num_frame = len(language_embd)
        num_obj = len(language_embd[0])
        mask_out = []
        for frame_idx in range(num_frame):
            frame_mask_out = []
            for obj_idx in range(num_obj):
                _language_embd = language_embd[frame_idx][obj_idx][None][None]
                _, _, out_mask_logits = self.sam2_model.add_language_embd(inference_state, frame_idx, obj_idx + 100, _language_embd)
                frame_mask_out.append(out_mask_logits)
            frame_mask_out = torch.cat(frame_mask_out, dim=1)
            mask_out.append(frame_mask_out)
        mask_out = torch.cat(mask_out, dim=0)
        return mask_out


    def language_embd_inference(self, inference_state, language_embd):
        num_frame = len(language_embd)
        num_obj = len(language_embd[0])
        mask_out = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for frame_idx in range(num_frame):
                frame_mask_out = []

                for obj_idx in range(num_obj):
                    _language_embd = language_embd[frame_idx][obj_idx][None][None]
                    _, _, out_mask_logits = self.sam2_model.add_language_embd(
                        inference_state,
                        frame_idx,
                        obj_idx + 100,
                        _language_embd,
                        inference=True,
                    )
                    frame_mask_out.append(out_mask_logits)
                frame_mask_out = torch.cat(frame_mask_out, dim=1)
                mask_out.append(frame_mask_out)


            mask_out = []
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(inference_state):
                mask_out.append(out_mask_logits)
            mask_out = torch.cat(mask_out, dim=0)
        return mask_out

    def get_sam2_embeddings(self, images):
        return self.sam2_model.init_state(images)

    def forward(self, batch):
        raise NotImplementedError

    def preprocess_image(self, image: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        image = image / 255.

        img_mean = torch.tensor(self.img_mean, dtype=dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std

        return image
