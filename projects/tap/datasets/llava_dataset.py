import os
from mmengine.dist import master_only
from vlm.datasets.evaluation.base_eval_dataset import BaseEvalDataset

class LLaVAPretrainDataset(BaseEvalDataset):
    METAINFO: dict = dict(name='llava_pretrain')
    def __init__(
            self,
            image_folder,
            split=8,
            rank=0,
    ):
        super().__init__()
        self.image_files = os.listdir(image_folder)
        # self.image_paths = [os.path.join(image_folder, file_name) for file_name in self.image_files]
        self.image_paths = self._get_image_paths(image_folder)

        assert rank < split
        size = len(self.image_paths) // split + 1
        i_s, i_e = size * rank, min(size * (rank + 1), len(self.image_paths))
        self.image_paths = self.image_paths[i_s: i_e]

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
        return len(self.image_paths)

    def __getitem__(self, idx):
        data_dict = {'image_path': self.image_paths[idx]}
        data_dict.update({'text_prompts': self.image_paths[idx], 'pixel_values': None, 'img_id': str(idx)})
        return data_dict

    @master_only
    def evaluate(self, **kwargs):
        return {'Acc': 0}
