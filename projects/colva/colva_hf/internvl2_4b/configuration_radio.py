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
from typing import Optional, List, Union, NamedTuple
import torch
from transformers import PretrainedConfig

from .radio_common import RESOURCE_MAP, DEFAULT_VERSION

from .radio_model import Resolution

class RADIOConfig(PretrainedConfig):
    """Pretrained Hugging Face configuration for RADIO models."""

    def __init__(
        self,
        args: Optional[dict] = None,
        version: Optional[str] = DEFAULT_VERSION,
        patch_size: Optional[int] = None,
        max_resolution: Optional[int] = None,
        preferred_resolution: Optional[Resolution] = None,
        adaptor_names: Union[str, List[str]] = None,
        vitdet_window_size: Optional[int] = None,
        **kwargs,
    ):
        self.args = args
        for field in ["dtype", "amp_dtype"]:
            if self.args is not None and field in self.args:
                # Convert to a string in order to make it serializable.
                # For example for torch.float32 we will store "float32",
                # for "bfloat16" we will store "bfloat16".
                self.args[field] = str(args[field]).split(".")[-1]
        self.version = version
        resource = RESOURCE_MAP[version]
        self.patch_size = patch_size or resource.patch_size
        self.max_resolution = max_resolution or resource.max_resolution
        self.preferred_resolution = (
            preferred_resolution or resource.preferred_resolution
        )
        self.adaptor_names = adaptor_names
        self.vitdet_window_size = vitdet_window_size
        super().__init__(**kwargs)