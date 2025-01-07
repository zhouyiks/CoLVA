import numpy as np
import random
from xtuner.utils import DEFAULT_IMAGE_TOKEN
import re

PREFIX_REASONING_STR = [
    'This conclusion is based on several observations: ',
    'Because: ',
    "This conclusion stems from several key factors: ",
    "The reasoning behind this conclusion includes: ",
    "Several observations lead to this conclusion: ",
    "The underlying reasons are: ",
    "The evidence supporting this conclusion includes: ",
    "This is justified by: ",
]

CONTOUR_QUESTIONS = [
    "Here are two images. In the second image, I have marked several "\
    "visual objects with their contours in different colors, and each "\
    "is identified by a white numeric ID against a background that "\
    "matches the contour's color. Could you please tell me which of "\
    "these marked objects is the same as the object marked with a {color} "\
    "contour in the first image?",
    "Observe the two images provided. In the second image, several objects "\
    "are outlined in various colors, each accompanied by a white numeric ID "\
    "on a matching color background. Can you identify which object corresponds "\
    "to the one outlined in {color} in the first image?",
    "You have two images in front of you. The second image contains multiple "\
    "objects, each highlighted with a distinct color contour and labeled with "\
    "a numeric ID. Please determine which object matches the one outlined in "\
    "{color} in the first image?",
    "Examine the pair of images. In the second image, objects are marked with "\
    "different colored contours, each paired with a white numeric ID on a "\
    "corresponding colored background. Which object is identical to the one "\
    "marked with a {color} contour in the first image?",
    "Here are two images for comparison. The second image features several "\
    "objects, each enclosed in a uniquely colored contour and identified by "\
    "a numeric ID. Can you select the object that matches the one outlined "\
    "in {color} in the first image?",
    "Look at the two images provided. In the second image, objects are "\
    "highlighted with various colored contours, each with a white numeric "\
    "ID on a matching background. Which of these objects is the same as the "\
    "one outlined in {color} in the first image?"
]

CHOICE_STR = " Please make a choice from the following options: \n{choices}"

def match_reasoning_preprocess(example):
    conversations = []
    conversations.append({"from": "human", "value": random.choice(CONTOUR_QUESTIONS) + CHOICE_STR})
    conversations.append({"from": 'gpt', "value": '{answer}'})
    conversations.append({"from": 'human', "value": "Why?"})
    conversations.append({"from": 'gpt', "value": random.choice(PREFIX_REASONING_STR) + example['description']})
    
    for i, conversation in enumerate(conversations):
        if i == 0:
            role = conversation['from']
            assert role == 'human'
            question = f"Image-1: {DEFAULT_IMAGE_TOKEN}\nImage-2: {DEFAULT_IMAGE_TOKEN}\n<OBJECT_TOKENS>\n"
            question = question + conversation['value']
            conversation['value'] = question
    
    example['conversations'] = conversations
    return example

def match_reasoning_map_fn(example):
    example = match_reasoning_preprocess(example)

    return example


BBOX_QUESTIONS = [
    "Here are two images. In the second image, I have marked several "\
    "visual objects with their bounding boxes in different colors, and each "\
    "is identified by a white numeric ID against a background that "\
    "matches the bounding box color. Could you please tell me which of "\
    "these marked objects is the same as the object marked with a {color} "\
    "bounding box in the first image?",
    "Observe the two images provided. In the second image, several objects "\
    "are outlined in various colors, each accompanied by a white numeric ID "\
    "on a matching color background. Can you identify which object corresponds "\
    "to the one outlined in {color} in the first image?",
    "You have two images in front of you. The second image contains multiple "\
    "objects, each highlighted with a distinct color bounding box and labeled with "\
    "a numeric ID. Please determine which object matches the one outlined in "\
    "{color} in the first image?",
    "Examine the pair of images. In the second image, objects are marked with "\
    "different colored bounding boxes, each paired with a white numeric ID on a "\
    "corresponding colored background. Which object is identical to the one "\
    "marked with a {color} bounding box in the first image?",
    "Here are two images for comparison. The second image features several "\
    "objects, each enclosed in a uniquely colored bounding box and identified by "\
    "a numeric ID. Can you select the object that matches the one outlined "\
    "in {color} in the first image?",
    "Look at the two images provided. In the second image, objects are "\
    "highlighted with various colored bounding boxes, each with a white numeric "\
    "ID on a matching background. Which of these objects is the same as the "\
    "one outlined in {color} in the first image?"
]


def match_choice_only_preprocess(example):
    conversations = []
    if example['vprompt_type'] == "mask":
        conversations.append({"from": "human", "value": random.choice(CONTOUR_QUESTIONS) + CHOICE_STR})
    elif example["vprompt_type"] == "bbox":
        conversations.append({"from": "human", "value": random.choice(BBOX_QUESTIONS) + CHOICE_STR})
    else:
        raise NotImplementedError
    conversations.append({"from": 'gpt', "value": '{answer}'})
    
    for i, conversation in enumerate(conversations):
        if i == 0:
            role = conversation['from']
            assert role == 'human'
            question = f"Image-1: {DEFAULT_IMAGE_TOKEN}\nImage-2: {DEFAULT_IMAGE_TOKEN}\n<OBJECT_TOKENS>\n"
            question = question + conversation['value']
            conversation['value'] = question
    
    example['conversations'] = conversations
    return example

def match_choice_only_map_fn(example):
    example = match_choice_only_preprocess(example)

    return example


ROI_QUESTIONS = [
    "Here are two images. In the first image, I have specified a query object, "\
    "and in the second image, there are multiple candidate objects. Could you "\
    "identify which candidate object is the same as the query object?",
]

def match_reasoning_preprocess_roi(example):
    conversations = []
    conversations.append({"from": "human", "value": random.choice(ROI_QUESTIONS) + CHOICE_STR})
    conversations.append({"from": 'gpt', "value": '{answer}'})
    conversations.append({"from": 'human', "value": "Why?"})
    conversations.append({"from": 'gpt', "value": random.choice(PREFIX_REASONING_STR) + example['description']})
    
    for i, conversation in enumerate(conversations):
        if i == 0:
            role = conversation['from']
            assert role == 'human'
            question = f"Image-1: {DEFAULT_IMAGE_TOKEN}\nImage-2: {DEFAULT_IMAGE_TOKEN}\n<OBJECT_TOKENS>\n"
            question = question + conversation['value']
            conversation['value'] = question
    
    example['conversations'] = conversations
    return example

def match_reasoning_map_fn_roi(example):
    example = match_reasoning_preprocess_roi(example)

    return example