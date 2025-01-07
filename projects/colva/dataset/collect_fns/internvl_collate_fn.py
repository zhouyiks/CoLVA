from typing import Dict, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def internvl_collate_fn(instances: Sequence[Dict],
                        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                        return_hf_format: bool = False,
                        use_varlen_attn: bool = False):
    seq_parallel_world_size = get_sequence_parallel_world_size()

    input_ids, labels = [], []
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    has_vprompt = any(inst.get('merged_visual_prompts') is not None for inst in instances)
    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'
    
    if has_image:
        pixel_values = []
    if has_vprompt:
        merged_visual_prompts = []

    first = instances[0]
    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['position_ids']))
        
        if has_image:
            pixel_values.append(example['pixel_values'])
        
        if has_vprompt:
            merged_visual_prompts.append(example['merged_visual_prompts'])
        
    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
    
    if use_varlen_attn:
        assert input_ids.size(1) % seq_parallel_world_size == 0
        attention_mask = None
        position_ids = torch.stack(position_ids, dim=0)
    else:
        # Some tokenizers have the same eos token and pad token, so input_ids
        # cannot be masked directly based on the pad token id.
        attention_mask = torch.zeros_like(input_ids).bool()
        for i, length in enumerate(ori_length):
            attention_mask[i, :length] = True

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)
    
    if seq_parallel_world_size > 1:
        input_ids = pad_for_sequence_parallel(input_ids, pad_index)
        labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
        position_ids = pad_for_sequence_parallel(position_ids, 0)
        if attention_mask is not None:
            attention_mask = pad_for_sequence_parallel(attention_mask, 0)
    
    if use_varlen_attn:
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }
    
    if has_image:
        pixel_values = torch.cat(pixel_values)
        data_dict["pixel_values"] = pixel_values
    if has_vprompt:
        merged_visual_prompts = torch.cat(merged_visual_prompts)
        data_dict['merged_visual_prompts'] = merged_visual_prompts
    for k, v in first.items():
        if k not in ('image_flags', 'pixel_values', 'merged_visual_prompts', 'visual_prompts',
                     'num_patches', 'num_vprompts', 'has_visual_prompt', 'region_ids', 'vprompt_flags') \
        and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                if all([example[k].size()==v.size() for example in instances]):
                    data_dict[k] = torch.stack([example[k] for example in instances])
            elif isinstance(v, np.ndarray):
                if all([example[k].shape==first.shape] for example in instances):
                    data_dict[k] = torch.tensor(np.stack([example[k] for example in instances]))
            else:
                data_dict[k] = torch.tensor([example[k] for example in instances])
        if k in ('image_flags', 'visual_prompts', 'num_images', 'ot_pixel_values', 'ot_visual_prompts'):
            if isinstance(v, torch.Tensor):
                data_dict[k] = torch.cat([example[k] for example in instances])
            elif isinstance(v, np.ndarray):
                data_dict[k] = torch.tensor(np.stack([example[k] for example in instances]))
            else:
                data_dict[k] = torch.tensor([example[k] for example in instances])
        if k in ('num_patches', 'num_vprompts', 'has_visual_prompt'):
            extend_list = []
            for example in instances:
                extend_list.extend(example[k])
            data_dict[k] = torch.tensor(extend_list, dtype=torch.long)
        if k in ('region_ids', ):
            data_dict[k] = [example[k] for example in instances]
        if k in ('vprompt_flags',):
            extend_list = []
            for example in instances:
                for ele in example[k]:
                    extend_list.extend(ele)
            data_dict[k] = torch.tensor(extend_list)

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
    