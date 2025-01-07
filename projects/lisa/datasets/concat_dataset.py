from torch.utils.data import ConcatDataset as TorchConcatDataset
import bisect
from xtuner.registry import BUILDER

class ConcatDataset(TorchConcatDataset):

    def __init__(self, datasets):
        datasets_instance = []
        for cfg in datasets:
            datasets_instance.append(BUILDER.build(cfg))
        super().__init__(datasets=datasets_instance)

    def __repr__(self):
        main_str = 'Dataset as a concatenation of multiple datasets. \n'
        main_str += ',\n'.join(
            [f'{repr(dataset)}' for dataset in self.datasets])
        return main_str

    def get_dataset_source(self, idx: int) -> int:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        return dataset_idx