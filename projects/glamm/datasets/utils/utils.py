from PIL import Image



def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
CAPTION_QUESTIONS = [
    'Could you please give me a detailed description of the image?',
    'Can you provide a thorough description of the this image?',
    'Please provide a thorough description of the this image',
    'Please provide a thorough description of the this image.',
    'Please describe in detail the contents of the image.',
    'Please describe in detail the contents of the image',
    'Could you give a comprehensive explanation of what can be found within this picture?',
    'Could you give me an elaborate explanation of this picture?',
    'Could you provide me with a detailed analysis of this photo?',
    'Could you please give me a detailed description of the image?',
    'Can you provide a thorough description of the this image?',
    'Please describe in detail the contents of the image',
    'Please describe in detail the contents of the image.',
    'Can you give a comprehensive explanation of this photo',
    'Please provide an elaborate explanation of this picture.',
    'Please provide an elaborate explanation of this picture',
    'Could you provide me with a detailed analysis of this photo',
]

REGION_QUESTIONS = [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
]

REGION_GROUP_QUESTIONS = [
    'Could you please give me a detailed description of these areas <region>?',
    'Can you provide a thorough description of the regions <region> in this image?',
    'Please describe in detail the contents of the boxed areas <region>.',
    'Could you give a comprehensive explanation of what can be found within <region> in the picture?',
    'Could you give me an elaborate explanation of the <region> regions in this picture?',
    'Can you provide a comprehensive description of the areas identified by <region> in this photo?',
    'Help me understand the specific locations labeled <region> in this picture in detail, please.',
    'What is the detailed information about the areas marked by <region> in this image?',
    'Could you provide me with a detailed analysis of the regions designated <region> in this photo?',
    'What are the specific features of the areas marked <region> in this picture that you can describe in detail?',
    'Could you elaborate on the regions identified by <region> in this image?',
    'What can you tell me about the areas labeled <region> in this picture?',
    'Can you provide a thorough analysis of the specific locations designated <region> in this photo?',
    'I am interested in learning more about the regions marked <region> in this image. Can you provide me with more information?',
    'Could you please provide a detailed description of the areas identified by <region> in this photo?',
    'What is the significance of the regions labeled <region> in this picture?',
    'I would like to know more about the specific locations designated <region> in this image. Can you provide me with more information?',
    'Can you provide a detailed breakdown of the regions marked <region> in this photo?',
    'What specific features can you tell me about the areas identified by <region> in this picture?',
    'Could you please provide a comprehensive explanation of the locations labeled <region> in this image?',
    'Can you provide a detailed account of the regions designated <region> in this photo?',
    'I am curious about the areas marked <region> in this picture. Can you provide me with a detailed analysis?',
    'What important details can you tell me about the specific locations identified by <region> in this image?',
    'Could you please provide a detailed description of the regions labeled <region> in this photo?',
    'What can you tell me about the features of the areas designated <region> in this picture?',
    'Can you provide a comprehensive overview of the regions marked <region> in this image?',
    'I would like to know more about the specific locations identified by <region> in this photo. Can you provide me with more information?',
    'What is the detailed information you have on the areas labeled <region> in this picture?',
    'Could you provide me with a thorough analysis of the regions designated <region> in this image?',
    'Can you provide a detailed explanation of the specific locations marked by <region> in this photo?'
]

GCG_QUESTIONS = [
    'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Please describe in detail the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a comprehensive explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me an elaborate explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]