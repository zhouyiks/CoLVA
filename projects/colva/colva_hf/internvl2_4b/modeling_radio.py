# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import namedtuple
from typing import Optional, List, Union

from timm.models import VisionTransformer
import torch
from transformers import PreTrainedModel

# Import all required modules.
from .radio_adaptor_base import AdaptorBase, RadioOutput, AdaptorInput
from .radio_adaptor_generic import GenericAdaptor, AdaptorBase
from .radio_adaptor_mlp import create_mlp_from_state
from .radio_adaptor_registry import adaptor_registry
from .radio_cls_token import ClsToken
from .radio_enable_cpe_support import enable_cpe
from .radio_enable_spectral_reparam import configure_spectral_reparam_from_args
from .radio_eradio_model import eradio
from .radio_model import create_model_from_args
from .radio_model import RADIOModel as RADIOModelBase, Resolution
from .radio_input_conditioner import get_default_conditioner, InputConditioner
from .radio_open_clip_adaptor import OpenCLIP_RADIO
from .radio_vit_patch_generator import ViTPatchGenerator
from .radio_vitdet import apply_vitdet_arch, VitDetArgs

# Register extra models
from .radio_extra_timm_models import *

from .configuration_radio import RADIOConfig

class RADIOModel(PreTrainedModel):
    """Pretrained Hugging Face model for RADIO.

    This class inherits from PreTrainedModel, which provides
    HuggingFace's functionality for loading and saving models.
    """

    config_class = RADIOConfig

    def __init__(self, config):
        super().__init__(config)

        RADIOArgs = namedtuple("RADIOArgs", config.args.keys())
        args = RADIOArgs(**config.args)
        self.config = config

        model = create_model_from_args(args)
        input_conditioner: InputConditioner = get_default_conditioner()

        dtype = getattr(args, "dtype", torch.float32)
        if isinstance(dtype, str):
            # Convert the dtype's string representation back to a dtype.
            dtype = getattr(torch, dtype)
        model.to(dtype=dtype)
        input_conditioner.dtype = dtype

        summary_idxs = torch.tensor(
            [i for i, t in enumerate(args.teachers) if t.get("use_summary", True)],
            dtype=torch.int64,
        )

        adaptor_names = config.adaptor_names
        if adaptor_names is not None:
            raise NotImplementedError(
                f"Adaptors are not yet supported in Hugging Face models. Adaptor names: {adaptor_names}"
            )

        adaptors = dict()

        self.radio_model = RADIOModelBase(
            model,
            input_conditioner,
            summary_idxs=summary_idxs,
            patch_size=config.patch_size,
            max_resolution=config.max_resolution,
            window_size=config.vitdet_window_size,
            preferred_resolution=config.preferred_resolution,
            adaptors=adaptors,
        )

    @property
    def model(self) -> VisionTransformer:
        return self.radio_model.model

    @property
    def input_conditioner(self) -> InputConditioner:
        return self.radio_model.input_conditioner

    def forward(self, x: torch.Tensor):
        return self.radio_model.forward(x)