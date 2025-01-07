from .mme_dataset import MMEDataset
from .multiple_choice_dataset import MultipleChoiceDataset
from .pope_dataset import POPEDataset
from .hallusion_dataset import HallusionDataset
from .textvqa_dataset import TextVQADataset
# from .gqa_dataset import GQADataset
from .gqa_llava_eval_dataset import GQADataset
# from .vqav2_dataset import VQAv2Dataset
from .vqav2_llava_eval_dataset import VQAv2Dataset
from .chartqa_dataset import ChartQADataset
from .general_vqa_dataset import GeneralVQADataset
from .referring_expression_seg_dataset import RESDataset

__all__ = ['MMEDataset', 'MultipleChoiceDataset', 'POPEDataset', 'HallusionDataset', 'TextVQADataset', 'GQADataset',
           'VQAv2Dataset', 'ChartQADataset', 'GeneralVQADataset', 'RESDataset']
