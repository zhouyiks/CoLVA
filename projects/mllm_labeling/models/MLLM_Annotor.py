import json
import os.path

from mmengine.model import BaseModel

from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import torch
import torch.nn as nn

class MLLM_Annotor(BaseModel):
    def __init__(self,
        model=None,
        save_folder='./work_dirs/internvl72b_sam2_obj_cap/'
        ):
        super().__init__()
        print(torch.cuda.device_count())
        pipe = pipeline(model,
                        backend_config=TurbomindEngineConfig(
                            # session_len=8192,
                            session_len=16000,
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

    def predict_forward_sam2(self, data_dicts):
        prompts = []
        for data_dict in data_dicts:
            images = data_dict['images']
            texts = data_dict['text_prompt']
            prompts.append((texts, images))

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            if '图' in text or len(text) < 15:
                # wrong response
                continue
            results_list.append({
                'video_id': data_dicts[i]['video_id'],
                'obj_id': data_dicts[i]['obj_id'],
                'caption': text,
                'type': data_dicts[i]['type'],
            })
            # print('\n\n', results_list[-1], '\n\n')

        self.results_list += results_list

        if len(self.results_list) > 100:
            self.save_step()
            print("Saved !!!")

        return {}

    def predict_forward_sam2_recap(self, data_dicts):
        prompts = []
        for data_dict in data_dicts:
            images = data_dict['images']
            texts = data_dict['text_prompt']
            prompts.append((texts, images))

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            if '图' in text or len(text) < 15:
                # wrong response
                continue
            results_list.append({
                'video_id': data_dicts[i]['video_id'],
                'obj_id': data_dicts[i]['obj_id'],
                'crop_caption': data_dicts[i]['crop_caption'],
                'crop_category': data_dicts[i]['crop_category'],
                'caption': text,
            })
            # print('\n\n', prompts[i], '\n', results_list[-1], '\n\n')

        self.results_list += results_list

        if len(self.results_list) > 100:
            self.save_step()
            print("Saved !!!")

        return {}

    def predict_forward_sam2_video_recap(self, data_dicts):
        prompts = []
        for data_dict in data_dicts:
            images = data_dict['images']
            texts = data_dict['text_prompt']
            prompts.append((texts, images))

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            if '图' in text or len(text) < 15:
                # wrong response
                continue
            results_list.append({
                'video_id': data_dicts[i]['video_id'],
                'obj_id': data_dicts[i]['obj_id'],
                'crop_caption': data_dicts[i]['crop_caption'],
                'crop_category': data_dicts[i]['crop_category'],
                'image_caption': data_dicts[i]['image_caption'],
                'video_caption': text,
            })
            # print('\n\n', prompts[i], '\n', results_list[-1], '\n\n')

        self.results_list += results_list

        if len(self.results_list) > 100:
            self.save_step()
            print("Saved !!!")

        return {}

    def predict_forward_image_dense_cap_objcap(self, data_dicts):
        prompts = []
        for data_dict in data_dicts:
            images = data_dict['images']
            texts = data_dict['text_prompt']
            prompts.append((texts, images))

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            results_list.append({
                'image_id': data_dicts[i]['image_id'],
                'caption': text,
                'object_anno': data_dicts[i]['object_anno']
            })
            print('\n\n', prompts[i], '\n', results_list[-1], '\n\n')

        self.results_list += results_list

        if len(self.results_list) > 100:
            self.save_step()
            print("Saved !!!")

        return {}

    def predict_forward_image_dense_cap_overallcap(self, data_dicts):
        prompts = []
        for data_dict in data_dicts:
            images = data_dict['images']
            texts = data_dict['text_prompt']
            prompts.append((texts, images))

        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]

        results_list = []

        for i, text in enumerate(text_lsit):
            results_list.append({
                'image_id': data_dicts[i]['image_id'],
                'caption': text,
            })
            print('\n\n', prompts[i], '\n', results_list[-1], '\n\n')

        self.results_list += results_list

        if len(self.results_list) > 100:
            self.save_step()
            print("Saved !!!")

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

        if 'type' in kwargs.keys() and kwargs['type'] == 'sam2':
            return self.predict_forward_sam2(kwargs['data_dicts'])
        elif 'type' in kwargs.keys() and kwargs['type'] == 'sam2_recap':
            return self.predict_forward_sam2_recap(kwargs['data_dicts'])
        elif 'type' in kwargs.keys() and kwargs['type'] == 'sam2_video_recap':
            return self.predict_forward_sam2_video_recap(kwargs['data_dicts'])
        elif 'type' in kwargs.keys() and kwargs['type'] == 'demo_imgcap':
            return self.predict_forward_image_dense_cap_objcap(kwargs['data_dicts'])
        elif 'type' in kwargs.keys() and kwargs['type'] == 'demo_imgcap_overall':
            return self.predict_forward_image_dense_cap_overallcap(kwargs['data_dicts'])

        images = [load_image(image_path) for image_path in image_paths]
        prompts = [('Please briefly describe this image in a sentence.', image) for image in images]
        response_list = self.pipe[0](prompts)
        text_lsit = [item.text for item in response_list]
        print(text_lsit)
        return {}
