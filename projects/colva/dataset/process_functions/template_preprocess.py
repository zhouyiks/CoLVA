import io
import os
import random
import re
from typing import Dict
import copy

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode
from xtuner.utils import IGNORE_INDEX
IGNORE_TOKEN_ID = IGNORE_INDEX
from mmengine.config import ConfigDict

from ..utils import (get_conv_template, IMG_CONTEXT_TOKEN, IMG_START_TOKEN, 
                     IMG_END_TOKEN, DEFAULT_VISION_PROMPT_TOKEN, VPT_START_TOKEN, 
                     VPT_END_TOKEN, VPT_CONTEXT_TOKEN)

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
except ImportError as E:
    print('petrel_client is not installed. If you read data locally instead of from ceph, ignore it.')
import sys


def preprocess(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        object_tokens_str: str = "",
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
                conversation = conversation.replace('<OBJECT_TOKENS>', object_tokens_str, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            logger.info(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )




def preprocess_mpt(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        object_tokens_str: str = ""
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
                conversation = conversation.replace('<OBJECT_TOKENS>', object_tokens_str, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|im_end|><|im_start|>assistant\n
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids) + 1

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids)

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )



def preprocess_phi3_debug(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        object_tokens_str: str = ""
) -> Dict:
    conversations = sources[0]
    input = ''
    out_conversation = []
    while conversations and conversations[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        conversations = conversations[1:]
    
    for msg in conversations:
        if msg['from'] == 'human':
            msg_value = msg['value']
            if not text_only:
                for i in range(num_image):
                    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                    msg_value = msg_value.replace('<image>', image_tokens, 1)
                msg_value = msg_value.replace('<OBJECT_TOKENS>', object_tokens_str, 1).strip()
            input += msg_value
        elif msg['from'] == 'gpt':
            out_conversation.append({
                'input': input,
                'output': msg['value'].strip(),
            })
            input = ''
        else:
            raise NotImplementedError

    _system = 'You are an AI assistant whose name is Phi-3.'
    PROMPT_TEMPLATE = ConfigDict(
        phi3_chat=dict(
            SYSTEM='<|system|>\n{system}<|end|>\n',
            INSTRUCTION='<|user|>\n{input}<|end|>\n<|assistant|>\n',
            SUFFIX='<|end|>',
            SUFFIX_AS_EOS=True,
            SEP='\n',
            STOP_WORDS=['<|end|>'],
        )
    )
    template = PROMPT_TEMPLATE.phi3_chat
    template['INSTRUCTION'] = '<|user|>\n{input}<|end|><|assistant|>\n'
    
    input_ids, labels = [], []
    for i, single_turn_conversation in enumerate(out_conversation):
        input = single_turn_conversation.get('input', '')
        if input is None:
            input = ''
        input_text = template.INSTRUCTION.format(input=input, round=i+1)

        if i == 0:
            system = template.SYSTEM.format(system=_system)
            input_text = system + input_text
            input_encode = tokenizer.encode(input_text, add_special_tokens=True)
        else:
            input_encode = tokenizer.encode(input_text, add_special_tokens=False)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)

        output_text = single_turn_conversation.get('output', '')
        if template.get('SUFFIX', None):
            output_text += template.SUFFIX
        output_encode = tokenizer.encode(output_text, add_special_tokens=False)
        input_ids += output_encode
        labels += copy.deepcopy(output_encode)
    
    if len(input_ids) > tokenizer.model_max_length:
        input_ids = input_ids[:tokenizer.model_max_length]
        labels = labels[:tokenizer.model_max_length]
        print(
            f"Warning: input_ids length({len(input_ids)})"
            f"is longer than max_length, cut to {tokenizer.model_max_length}"
        )
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )



def preprocess_phi3(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        object_tokens_str: str = ""
) -> Dict:
   
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            # conversation = conversation.replace('<OBJECT_TOKENS>', object_tokens_str, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    tokenizer.padding_side = 'right'
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|end|>\n<|assistant|>
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(int(tokenizer.pad_token_id)).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        target[target == endoftext_id] = IGNORE_TOKEN_ID

        # print("turns: ", turns[3:])
        # print("re_turns: ", re_turns[1:])
        # exit(0)

        for i, turn in enumerate(re_turns):
            if turn == '':
                # print("turn == ''")
                break
            if i == 0:
                turn_len = len(tokenizer(turn).input_ids)
            else:
                turn_len = len(tokenizer(turn).input_ids) - 1
            parts = turn.split(sep)
            if len(parts) != 2:
                print("len(parts) != 2")
                break
            parts[0] += sep

            if i == 0:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            else:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()
        exit(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )



def preprocess_internlm(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        object_tokens_str: str = "",
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            sentence['value'] = sentence['value'].strip()
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
                conversation = conversation.replace('<OBJECT_TOKENS>', object_tokens_str, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.')
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_qwen2vl(conversations, object_tokens_str, num_images=0):
    out_conversation_list = [{
        "role": "system", 
        "content": [{
            "type": "text", 
            "text": "You are a helpful assistant."}]
        }]
    
    if conversations[0]['from'] != 'human':
        conversations = conversations[1:]
    
    total_images = 0
    for msg in conversations:
        if msg['from'] == 'human':
            msg_value = msg['value']
            cur_image_count = msg_value.count('<image>\n')
            total_images += cur_image_count
            msg_value = msg_value.replace('<OBJECT_TOKENS>', object_tokens_str, 1)
            if cur_image_count == 0:
                # pure text
                out_conversation_list.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": msg_value,
                        },
                    ],
                })
            else:
                out_contents = []
                text_str_list = msg_value.split('<image>\n')
                for idx, text_str in enumerate(text_str_list):
                    if idx > 0 and cur_image_count > 0:
                        out_contents.append({
                            "type": "image",
                        })
                        cur_image_count = cur_image_count - 1
                    
                    if text_str.strip() == '':
                        continue
                    else:
                        out_contents.append({
                            "type": "text",
                            "text": text_str,
                        })
                out_conversation_list.append({
                    "role": "user",
                    "content": out_contents,
                })
        elif msg['from'] == 'gpt':
            msg_value = msg['value']
            out_conversation_list.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": msg_value,
                    },
                ],
            })
    if total_images != num_images:
        return None
    else:
        return out_conversation_list


def preprocess_llava(conversations, object_tokens_str, num_images=0):
    out_conversation_list = []
    
    if conversations[0]['from'] != 'human':
        conversations = conversations[1:]
    
    total_images = 0
    for msg in conversations:
        if msg['from'] == 'human':
            msg_value = msg['value']
            cur_image_count = msg_value.count('<image>\n')
            total_images += cur_image_count
            msg_value = msg_value.replace('<OBJECT_TOKENS>', object_tokens_str, 1)
            if cur_image_count == 0:
                # pure text
                out_conversation_list.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": msg_value,
                        },
                    ],
                })
            else:
                out_contents = []
                text_str_list = msg_value.split('<image>\n')
                for idx, text_str in enumerate(text_str_list):
                    if idx > 0 and cur_image_count > 0:
                        out_contents.append({
                            "type": "image",
                        })
                        cur_image_count = cur_image_count - 1
                    
                    if text_str.strip() == '':
                        continue
                    else:
                        out_contents.append({
                            "type": "text",
                            "text": text_str,
                        })
                out_conversation_list.append({
                    "role": "user",
                    "content": out_contents,
                })
        elif msg['from'] == 'gpt':
            msg_value = msg['value']
            out_conversation_list.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": msg_value,
                    },
                ],
            })
    if total_images != num_images:
        return None
    else:
        return out_conversation_list
