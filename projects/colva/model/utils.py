# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.model.utils import *
from typing import List, Optional
import torch
from transformers import PreTrainedModel
from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX