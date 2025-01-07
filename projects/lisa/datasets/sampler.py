import torch
from torch.utils.data import Sampler, BatchSampler

from typing import Iterator, Optional, Sized

from mmengine.dist import get_dist_info

class MultiDataPseudoSampler(Sampler):
    def __init__(self, dataset, seed=None, round_up=True):
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        pass

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class MultiDataSameBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size: int, drop_last: bool = True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        rank, world_size = get_dist_info()
        self.world_size = world_size
        self.rank = rank

        self.dataset = sampler.dataset

        total_batches = 0
        for start, end in zip([0] + self.dataset.cumulative_sizes[:-1], self.dataset.cumulative_sizes):
            total_batches_ = (end - start) // self.batch_size
            total_batches += total_batches_
        
        self.num_samples = total_batches // self.world_size * self.batch_size
        self.total_size = self.num_samples * self.world_size

        self.epoch = 0

    def __iter__(self):
        indices = self._shuffle()
        indices = indices[self.rank:self.total_size // self.batch_size:self.world_size]
        assert len(indices) * self.batch_size == self.num_samples
        return iter(indices)

    def _shuffle(self):
        g = torch.Generator()
        g.manual_seed(42 + self.epoch)

        indices = []
        for start, end in zip([0] + self.dataset.cumulative_sizes[:-1], self.dataset.cumulative_sizes):
            indices_ = torch.randperm(end-start, generator=g) + start
            if len(indices_) % self.batch_size:
                indices_ = indices_[:-(len(indices_) % self.batch_size)]
            indices_ = indices_.view(-1, self.batch_size)
            indices += indices_
        indices = torch.stack(indices)
        indices = indices[torch.randperm(len(indices), generator=g)]

        return indices.tolist()

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
