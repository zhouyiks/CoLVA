import json
import os

from mmengine.dist import (master_only)

from vlm.datasets.evaluation.base_eval_dataset import BaseEvalDataset

from mmengine.logging import print_log


class FigureReasoningDataset(BaseEvalDataset):
    METAINFO: dict = dict(name='FigureReasoning dataset')

    def __init__(self,
        data_file,
        image_folder,
        metainfo={},
    ):
        super().__init__(metainfo)
        self.data_file = data_file
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder

        self.type_map = {
            'yangshiguilv': 'Style',
            'shuliangguilv': 'Quantity',
            'weizhiguilv': 'Positional',
            'shuxingguilv': 'Attribute',
            'kongjianguilv': 'Spatial',
            'other': 'Others',
        }
        self.difficulty_nums = {'hard': 0, 'medium': 0, 'easy': 0}
        self.difficulty_tp = {'hard': 0, 'medium': 0, 'easy': 0}
        self.type_nums = {'Style': 0, 'Quantity': 0, 'Positional': 0, 'Attribute': 0, 'Spatial': 0, 'Others': 0}
        self.type_tp = {'Style': 0, 'Quantity': 0, 'Positional': 0, 'Attribute': 0, 'Spatial': 0, 'Others': 0}
        self.all_nums = 0
        self.all_tp = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_type = data['type']
        difficulty = data['difficulty']

        data_type = data_type.split('/')[0]
        data_type = self.type_map[data_type]

        image = data['image']
        image_path = os.path.join(self.image_folder, image)

        # question = data['conversations'][0]['value'] + '\nPlease directly give the answer.'
        question = data['conversations'][0]['value']

        answer = data['answer']
        ret = {'difficulty': difficulty, 'type': data_type, 'image_path': image_path,
               'question': question, 'answer': answer,
               'text_prompts': None, 'pixel_values': None,
               'img_id': data['id'],
               }
        return ret

    def _get_answer(self, prediction):
        answer_list = ['A', 'B', 'C', 'D']
        prediction = prediction.replace('.', ' ').split(' ')
        for _aws in answer_list:
            if _aws in prediction:
                return _aws
        return 'None'

    @master_only
    def evaluate(self, results, work_dir):
        for result in results:
            prediction = result['prediction']
            prediction = self._get_answer(prediction)

            answer = result['answer']
            difficulty = result['difficulty']
            data_type = result['type']

            self.difficulty_nums[difficulty] += 1
            self.type_nums[data_type] += 1
            self.all_nums += 1
            # print(prediction, '---', answer)
            if prediction == answer:
                # correct
                self.difficulty_tp[difficulty] += 1
                self.type_tp[data_type] += 1
                self.all_tp += 1
        print_log('============================================', 'current')
        print_log(f'Figure reasoning dataset successfully finished evaluating, {len(results)} items' 'current')
        print_log(f'Average: {self.all_tp / (self.all_nums + 1e-4)}', 'current')
        for _type in self.type_nums.keys():
            print_log(f'{_type}: {self.type_tp[_type] / (self.type_nums[_type] + 1e-4)}', 'current')
        for _type in self.difficulty_nums.keys():
            print_log(f'{_type}: {self.difficulty_tp[_type] / (self.difficulty_nums[_type] + 1e-4)}', 'current')
        print_log('============================================', 'current')
        return None
