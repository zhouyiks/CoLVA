import os
from mmengine.dist import master_only
from vlm.datasets.evaluation.base_eval_dataset import BaseEvalDataset

class ImageDataset(BaseEvalDataset):
    METAINFO: dict = dict(name='image dataset')
    def __init__(
            self,
            image_folder,
            bs=8,
    ):
        super().__init__()
        self.image_files = os.listdir(image_folder)
        # self.image_paths = [os.path.join(image_folder, file_name) for file_name in self.image_files]
        self.image_paths = self._get_image_paths(image_folder)
        self.bs = bs

    def _get_image_paths(self, folder_path):
        ret = []
        sub_files = os.listdir(folder_path)
        for file_name in sub_files:
            if '.png' not in file_name and '.jpg' not in file_name:
                # a folder
                sub_folder_files = self._get_image_paths(os.path.join(folder_path, file_name))
                ret.extend(sub_folder_files)
            else:
                ret.append(os.path.join(folder_path, file_name))
        return ret

    def __len__(self):
        return len(self.image_paths) // self.bs

    def __getitem__(self, idx):
        start = idx * self.bs
        end = start + self.bs
        data_dict = {'image_paths': self.image_paths[start:end]}
        return data_dict

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}
