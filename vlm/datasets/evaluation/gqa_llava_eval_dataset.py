import os
import os.path as osp
import json
from mmengine.dist import master_only
from .base_eval_dataset import BaseEvalDataset
from xtuner.registry import BUILDER
from .eval_gqa import eval_gqa
from .utils import custom_data_process

class GQADataset(BaseEvalDataset):

    METAINFO: dict = dict(name='gqa')

    def __init__(
        self,
        question_file,
        answer_file,
        prediction_file,
        image_folder,
        image_processor,
        tier='val',
        test_question_file=None,
        pad_image_to_square=True,
        metainfo=None,
    ):
        super().__init__(metainfo)
        self.image_folder = image_folder
        self.data_file = question_file
        self.answer_file = answer_file
        self.prediction_file = prediction_file
        self.test_question_file = test_question_file
        self.tier = tier

        self.image_processor = BUILDER.build(image_processor)
        self.pad_image_to_square = pad_image_to_square
        self.data = self.load_data_list()

    def load_data_list(self):
        question_data = [json.loads(q) for q in open(os.path.expanduser(self.data_file), "r")]
        data_list = []
        for idx in range(len(question_data)):
            sample = question_data[idx]
            index = sample['question_id']
            image_path = sample['image']
            question = sample['text']
            category = sample['category']

            data = {
                'img_id': idx,
                'index': index,
                'image_path': image_path,
                'question': question,
                'category': category,
            }
            data_list.append(data)
            idx += 1

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = custom_data_process(self, data)
        return data_dict

    @master_only
    def evaluate(self, results, work_dir):
        answers_file = osp.join(work_dir, self.answer_file)
        ans_file = open(answers_file, "w")

        for pred_dict in results:
            idx = pred_dict["img_id"]
            gt_data = self.data[idx]

            ans_file.write(
                json.dumps(
                    {
                        "question_id": gt_data['index'],
                        "prompt": gt_data['question'],
                        "text": pred_dict['prediction'],
                        "metadata": {},
                    }
                )
                + "\n"
            )
        ans_file.close()

        all_answers = []
        for line_idx, line in enumerate(open(answers_file)):
            res = json.loads(line)
            question_id = res['question_id']
            text = res['text'].rstrip('.').lower()
            all_answers.append({"questionId": question_id, "prediction": text})

        prediction_file = osp.join(work_dir, self.prediction_file)
        with open(prediction_file, 'w') as f:
            json.dump(all_answers, f)

        evaluator = eval_gqa(questions=self.test_question_file, predictions=prediction_file)
        evaluator.forward()
