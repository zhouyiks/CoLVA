import json
import os.path

from mmengine.model import BaseModel

from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import torch
import torch.nn as nn

class LLM_Annotor(BaseModel):
    def __init__(self,
        model=None,
        save_folder='./work_dirs/qwen2_72b_obj_referring/'
        ):
        super().__init__()
        print(torch.cuda.device_count())
        print(f"\n\n Using {model} !!! \n\n")
        pipe = pipeline(model,
                        backend_config=TurbomindEngineConfig(
                            # session_len=8192,
                            session_len=4096,
                            tp=torch.cuda.device_count()),
                        )
        self.pipe = [pipe]
        self._zero = nn.Linear(10, 10)
        self.results_list = []
        self.item_idx = 0

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self.save_folder = save_folder

    def forward(self, **kwargs):
        return None

    def predict_forward_text_vertify(self, data_dicts):
        prompts = []

        print('vertify forward !!!')

        # remove ignore items
        if 'ignore' in data_dicts[0].keys():
            data_dicts_ = []
            for _item in data_dicts:
                if _item['ignore']:
                    continue
                data_dicts_.append(_item)
            data_dicts = data_dicts_

        for data_dict in data_dicts:
            texts = data_dict['text_prompt']
            prompts.append(texts)

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            print('\n\n', text, '\n\n')
            if 'No conflict' in text:
                results_list.append({
                    'video_id': data_dicts[i]['video_id'],
                    'obj_id': data_dicts[i]['obj_id'],
                    'ori_captions': data_dicts[i]['ori_captions'],
                })
                print('\n\n', data_dicts[i]['ori_captions'], '\n\n')

        self.results_list += results_list

        if len(self.results_list) > 100:
            self.save_step()

        return {}

    def predict_forward_text_summarize(self, data_dicts):
        prompts = []

        print('summarize forward !!!')

        for data_dict in data_dicts:
            texts = data_dict['text_prompt']
            prompts.append(texts)

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):

            results_list.append({
                'video_id': data_dicts[i]['video_id'],
                'obj_id': data_dicts[i]['obj_id'],
                'ori_captions': data_dicts[i]['ori_captions'],
                'summarized': text
            })
            print('\n\n', data_dicts[i]['ori_captions'], '\n', text, '\n\n')

        self.results_list += results_list

        if len(self.results_list) > 100:
            self.save_step()

        return {}

    def predict_forward_text_formatting(self, data_dicts):
        prompts = []

        print('formatting forward !!!')

        for data_dict in data_dicts:
            texts = data_dict['text_prompt']
            prompts.append(texts)

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            text = text.split(':')[-1]
            text = text.strip()
            text = text.replace('\"', '')
            results_list.append({
                'video_id': data_dicts[i]['video_id'],
                'obj_id': data_dicts[i]['obj_id'],
                'ori_captions': data_dicts[i]['ori_captions'],
                'summarized': data_dicts[i]['summarized'],
                'formated': text
            })
            print('\n\n', text, '\n\n')

        self.results_list += results_list

        if len(self.results_list) > 100:
            self.save_step()

        return {}

    def predict_forward_short_cap(self, data_dicts):
        prompts = []

        print('short caption forward !!!')

        for data_dict in data_dicts:
            texts = data_dict['text_prompt']
            prompts.append(texts)

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            print('\n\n', data_dicts[i]['formated'], '\n', text, '\n\n')
            results_list.append({
                'video_id': data_dicts[i]['video_id'],
                'obj_id': data_dicts[i]['obj_id'],
                'formated': data_dicts[i]['formated'],
                'short_cap': text
            })

        self.results_list += results_list

        if len(self.results_list) > 100:
            self.save_step()

        return {}

    def predict_forward_filter_unindentified(self, data_dicts):
        prompts = []

        print('filter_unindentified forward !!!')

        for data_dict in data_dicts:
            texts = data_dict['text_prompt']
            prompts.append(texts)

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            # print('\n\n', data_dicts[i]['caption'], '\n', text, '\n\n')
            if "Unidentified" in text or "unidentified" in text:
                pass
            else:
                results_list.append({
                    'video_id': data_dicts[i]['video_id'],
                    'obj_id': data_dicts[i]['obj_id'],
                    'caption': data_dicts[i]['caption'],
                    'type': data_dicts[i]['type'],
                    'category': text,
                })

        self.results_list += results_list
        if len(self.results_list) > 100:
            self.save_step()
        return {}

    def predict_forward_consistency(self, data_dicts):
        prompts = []

        print('Consistency forward !!!')

        for data_dict in data_dicts:
            texts = data_dict['text_prompt']
            prompts.append(texts)

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        out_num = 0
        for i, text in enumerate(text_lsit):
            if "Yes" in text or "yes" in text:
                print('\n\n', data_dicts[i]['text_prompt'], '\n', text, '\n\n')
                out_num += 1
                results_list.append({
                    'video_id': data_dicts[i]['video_id'],
                    'obj_id': data_dicts[i]['obj_id'],
                    'caption': data_dicts[i]['caption'],
                    'category': data_dicts[i]['category'],
                })

        print(f"***************Input {len(text_lsit)} items and keep {out_num} items !!!\n")
        self.results_list += results_list
        if len(self.results_list) > 100:
            self.save_step()
        return {}

    def predict_forward_re_consistency(self, data_dicts):
        prompts = []

        print('Re consistency forward !!!')

        for data_dict in data_dicts:
            texts = data_dict['text_prompt']
            prompts.append(texts)

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        out_num = 0
        for i, text in enumerate(text_lsit):
            if "Yes" in text or "yes" in text:
                # print('\n\n', data_dicts[i]['text_prompt'], '\n', text, '\n\n')
                out_num += 1
                results_list.append({
                    'video_id': data_dicts[i]['video_id'],
                    'obj_id': data_dicts[i]['obj_id'],
                    'crop_caption': data_dicts[i]['crop_caption'],
                    'crop_category': data_dicts[i]['crop_category'],
                    'image_caption': data_dicts[i]['image_caption'],
                    'video_caption': data_dicts[i]['video_caption'],
                })
            # else:
            #     print('\n\n', data_dicts[i]['text_prompt'], '\n', text, '\n\n')
        print(f"***************Input {len(text_lsit)} items and keep {out_num} items !!!\n")
        self.results_list += results_list
        if len(self.results_list) > 100:
            self.save_step()
        return {}

    def predict_forward_change_style(self, data_dicts):
        prompts = []

        print('Change Style forward !!!')

        for data_dict in data_dicts:
            texts = data_dict['text_prompt']
            prompts.append(texts)

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            results_list.append({
                'video_id': data_dicts[i]['video_id'],
                'obj_id': data_dicts[i]['obj_id'],
                'crop_caption': data_dicts[i]['crop_caption'],
                'crop_category': data_dicts[i]['crop_category'],
                'image_caption': data_dicts[i]['image_caption'],
                'video_caption': data_dicts[i]['video_caption'],
                'final_caption': text,
            })
            # print('\n\n', data_dicts[i]['text_prompt'], '\n', text, '\n\n')
        self.results_list += results_list
        if len(self.results_list) > 100:
            self.save_step()
        return {}

    def predict_forward_translation(self, data_dicts):
        prompts = []

        print('translation forward !!!')

        for data_dict in data_dicts:
            texts = data_dict['text_prompt']
            prompts.append(texts)

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            results_list.append({
                'video_id': data_dicts[i]['video_id'],
                'obj_id': data_dicts[i]['obj_id'],
                'crop_caption': data_dicts[i]['crop_caption'],
                'crop_category': data_dicts[i]['crop_category'],
                'image_caption': data_dicts[i]['image_caption'],
                'video_caption': data_dicts[i]['video_caption'],
                'final_caption': data_dicts[i]['final_caption'],
                'translation': text
            })
            print(text, '\n')
            # print('\n\n', data_dicts[i]['text_prompt'], '\n', text, '\n\n')
        self.results_list += results_list
        if len(self.results_list) > 100:
            self.save_step()
        return {}

    def predict_forward_text(self, data_dicts):
        if 'task' in data_dicts[0].keys() and data_dicts[0]['task'] == 'vertify':
            return self.predict_forward_text_vertify(data_dicts)

        prompts = []

        # remove ignore items
        if 'ignore' in data_dicts[0].keys():
            data_dicts_ = []
            for _item in data_dicts:
                if _item['ignore']:
                    continue
                data_dicts_.append(_item)
            data_dicts = data_dicts_

        for data_dict in data_dicts:
            texts = data_dict['text_prompt']
            prompts.append(texts)

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            results_list.append({
                'video_id': data_dicts[i]['video_id'],
                'obj_id': data_dicts[i]['obj_id'],
                'caption': text,
            })
            print('\n\n', text, '\n\n')

        self.results_list += results_list

        if len(self.results_list) > 100:
            self.save_step()

        return {}

    def save_step(self, last=False):
        if last:
            save_list = self.results_list
        else:
            save_list = self.results_list[:100]
            self.results_list = self.results_list[100:]

        json_path = os.path.join(self.save_folder, f'{self.item_idx}.json')
        self.item_idx += 1
        with open(json_path, 'w') as f:
            json.dump(save_list, fp=f)
        return

    def predict_forward(self, image_paths, **kwargs):

        if 'type' in kwargs.keys() and kwargs['type'] == 'text':
            if 'task' in kwargs.keys() and kwargs['task'] == 'vertify':
                return self.predict_forward_text_vertify(kwargs['data_dicts'])
            elif 'task' in kwargs.keys() and kwargs['task'] == 'summarize':
                return self.predict_forward_text_summarize(kwargs['data_dicts'])
            elif 'task' in kwargs.keys() and kwargs['task'] == 'formatting':
                return self.predict_forward_text_formatting(kwargs['data_dicts'])
            elif 'task' in kwargs.keys() and kwargs['task'] == 'short_cap':
                return self.predict_forward_short_cap(kwargs['data_dicts'])
            elif 'task' in kwargs.keys() and kwargs['task'] == 'filter_unindentified':
                return self.predict_forward_filter_unindentified(kwargs['data_dicts'])
            elif 'task' in kwargs.keys() and kwargs['task'] == 'consistency':
                return self.predict_forward_consistency(kwargs['data_dicts'])
            elif 'task' in kwargs.keys() and kwargs['task'] == 're_consistency':
                return self.predict_forward_re_consistency(kwargs['data_dicts'])
            elif 'task' in kwargs.keys() and kwargs['task'] == 'change_style':
                return self.predict_forward_change_style(kwargs['data_dicts'])
            elif 'task' in kwargs.keys() and kwargs['task'] == 'translation':
                return self.predict_forward_translation(kwargs['data_dicts'])
            return self.predict_forward_text(kwargs['data_dicts'])

        images = [load_image(image_path) for image_path in image_paths]
        prompts = [('Please briefly describe this image in a sentence.', image) for image in images]
        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]
        print(text_lsit)
        return {}
