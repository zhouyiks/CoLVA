from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, is_vision_available, logging

from typing import Dict, List, Optional, Union
from math import ceil
from torchvision.transforms import Resize

def get_resize_output_image_size_long(
    image_size, PATCH_SIZE=32, MAX_RESOLUTION = 1024, MIN_RESOLUTION = 448,
) -> tuple:
    l1, l2 = image_size  # 540, 32
    short, long = (l2, l1) if l2 <= l1 else (l1, l2)

    # set the nearest multiple of PATCH_SIZE for `long`
    requested_new_long = min(
        [
            ceil(long / PATCH_SIZE) * PATCH_SIZE,
            MAX_RESOLUTION,
        ]
    )

    requested_new_long = max(requested_new_long, MIN_RESOLUTION)

    new_long, new_short = requested_new_long, int(requested_new_long * short / long)
    # Find the nearest multiple of 64 for new_short
    new_short = ceil(new_short / PATCH_SIZE) * PATCH_SIZE
    return (new_long, new_short) if l2 <= l1 else (new_short, new_long)


class SoloCLIPImageProcessor(CLIPImageProcessor):

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        PATCH_SIZE=32,
        MAX_RESOLUTION=1024,
        MIN_RESOLUTION=448,
        **kwargs,
    ) -> None:
        super(SoloCLIPImageProcessor, self).__init__(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )
        self.PATCH_SIZE = PATCH_SIZE
        self.MAX_RESOLUTION = MAX_RESOLUTION
        self.MIN_RESOLUTION = MIN_RESOLUTION

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        return_dict = super(SoloCLIPImageProcessor, self).preprocess(
            images=images,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            return_tensors=return_tensors,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        pixel_values = return_dict['pixel_values'][0]
        _, height, width = pixel_values.size()
        height, width = get_resize_output_image_size_long(
            (height, width),
            PATCH_SIZE=self.PATCH_SIZE,
            MAX_RESOLUTION=self.MAX_RESOLUTION,
            MIN_RESOLUTION=self.MIN_RESOLUTION,
        )
        pixel_values = Resize(size=(height, width))(pixel_values)
        return_dict['pixel_values'] = pixel_values.unsqueeze(0)
        return return_dict