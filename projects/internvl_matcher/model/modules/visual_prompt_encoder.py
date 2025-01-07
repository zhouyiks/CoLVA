import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig


class VisualPromptEncodeConfig(PretrainedConfig):
    model_type = 'vision_encoder'
    _auto_class = 'AutoConfig'
    main_input_name = "visual_prompts"

    def __init__(self, 
                 vision_hidden_size: int,
                 language_hidden_size: int,
                 patch_size: int,
                 downsample_ratio,
                 **kwargs):
        super().__init__(**kwargs)


class VisualPromptEncodeModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 vision_hidden_size: int,
                 language_hidden_size: int,
                 force_image_size: int,
                 patch_size: int,
                 downsample_ratio: int,
                 ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.vision_hidden_size = vision_hidden_size
        self.language_hidden_size = language_hidden_size
        self.force_image_size = force_image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels, out_channels=vision_hidden_size, 
            kernel_size=patch_size, stride=patch_size
        )
        
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(language_hidden_size),
            nn.Linear(language_hidden_size, vision_hidden_size),
            nn.GELU(),
            nn.Linear(vision_hidden_size, vision_hidden_size)
        )

        self.patch_edge_token = force_image_size// patch_size

    def forward(self, merged_visual_prompts, visual_prompts, mark_embeddings, num_patches, num_vprompts):
        patch_embeds = self.patch_embedding(merged_visual_prompts)  # shape = [*, channel, height, width]
        split_size = [npatch * nvprompt for (npatch, nvprompt) in zip(num_patches, num_vprompts)]
        resized_visual_prompts = F.interpolate(visual_prompts.unsqueeze(1), 
                                               size=self.patch_edge_token,
                                               mode="nearest")
        resized_visual_prompts_per_batch = torch.split(resized_visual_prompts, split_size, dim=0)
        split_size = [nvp for nvp in num_vprompts]
        mark_embeddings_per_batch = torch.split(mark_embeddings, split_size, dim=0)
        batch_vprompts_input = []
        for i, (per_visual_prompts, per_mark_embeddings) in enumerate(zip(
            resized_visual_prompts_per_batch, mark_embeddings_per_batch)):
            per_visual_prompts = per_visual_prompts.view(
                num_vprompts[i], num_patches[i], 1, self.patch_edge_token, self.patch_edge_token)
            per_background = torch.ones_like(per_visual_prompts) - per_visual_prompts
            per_vprompts_input = torch.zeros(
                (num_vprompts[i], num_patches[i], self.language_hidden_size,
                 self.patch_edge_token, self.patch_edge_token), 
                 dtype=mark_embeddings.dtype).to(mark_embeddings.device)
            per_vprompts_input = per_vprompts_input * per_background + \
                per_mark_embeddings[:, None, :, None, None] * per_visual_prompts
            #TODO numeric stability for multi-granularity prompts (one pixel covered by multiple visual prompts)
            per_vprompts_input = torch.sum(per_vprompts_input, dim=0)
            batch_vprompts_input.append(per_vprompts_input)
        batch_vprompts_input = torch.cat(batch_vprompts_input, dim=0)
        batch_vprompts_input = batch_vprompts_input.permute(0, 2, 3, 1).flatten(1, 2)
        batch_vprompts_input = self.mlp1(batch_vprompts_input).view(
            -1, self.patch_edge_token, self.patch_edge_token, self.vision_hidden_size).permute(0, 3, 1, 2)
        # this version not consider color prompt
        patch_embeds = patch_embeds * 0.0 + batch_vprompts_input
        return patch_embeds

        

    
