from torch.utils.data import Dataset
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoImageProcessor
from .utils import DEFAULT_VISION_PROMPT_TOKEN, VPT_CONTEXT_TOKEN, VPT_START_TOKEN, VPT_END_TOKEN

class LLaVACombineDataset(Dataset):
    def __init__(self,
                 datasets_cfgs,
                 exhibit_special_tokens=False,
                 llava_processor=None,
                 ot_image_processor=None,
                 repeat_time=1,
                 ):
        super().__init__()

        self.datasets = []
        self.datasets_length = []
        
        if ot_image_processor:
            process_clazz = ot_image_processor.pop('type')
            ot_image_processor = process_clazz(**ot_image_processor)
        else:
            ot_image_processor = None
        if llava_processor:
            llava_processor_clazz = llava_processor.pop('type')
            self.llava_processor = llava_processor_clazz(**llava_processor)
        else:
            self.llava_processor = None
        if not exhibit_special_tokens:
            self._add_special_tokens()

        for dataset_cfg in datasets_cfgs:
            dataset = dataset_cfg['type']
            ori_repeat_time = dataset_cfg['repeat_time']
            del dataset_cfg['type']
            dataset_cfg.update(dict(ot_image_processor=ot_image_processor,
                                    llava_processor=self.llava_processor,
                                    repeat_time=ori_repeat_time*repeat_time))
            dataset = dataset(**dataset_cfg)
            self.datasets.append(dataset)
            self.datasets_length.append(len(dataset))
        
        self.dataset_threshold = []
        for i, length in enumerate(self.datasets_length):
            if i == 0:
                self.dataset_threshold.append(length)
            else:
                self.dataset_threshold.append(length + self.dataset_threshold[i - 1])
        
        np.random.seed(42)
        self.shuffled_index = np.arange(self.dataset_threshold[-1])
        np.random.shuffle(self.shuffled_index)

    def _add_special_tokens(self):
        special_tokens = [VPT_CONTEXT_TOKEN,]
        num_new_tokens = self.llava_processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    @property
    def modality_length(self):
        length_list = []
        for dataset in self.datasets:
            length_list += dataset.modality_length
        return length_list
    
    def __len__(self):
        return self.dataset_threshold[-1]
    
    def __getitem__(self, index):
        index = int(self.shuffled_index[index])
        for i, thred in enumerate(self.dataset_threshold):
            if index < thred:
                break
        if i == 0:
            _index = index
        else:
            _index = index - self.dataset_threshold[i - 1]
        
        return self.datasets[i][_index]