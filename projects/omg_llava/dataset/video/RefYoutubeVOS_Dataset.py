from .ReVOS_Dataset import VideoReVOSDataset
import json
import pickle

class VideoRefYoutubeVOSDataset(VideoReVOSDataset):
    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file,
                 image_processor,
                 tokenizer=None,
                 select_number=5,
                 sampled_frames=10,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 lazy=True,
                 repeats=1,):
        super().__init__(
            image_folder=image_folder,
            expression_file=expression_file,
            mask_file=mask_file,
            image_processor=image_processor,
            tokenizer=tokenizer,
            select_number=select_number,
            sampled_frames=sampled_frames,
            offline_processed_text_folder=offline_processed_text_folder,
            template_map_fn=template_map_fn,
            max_length=max_length,
            pad_image_to_square=pad_image_to_square,
            lazy=lazy,
            repeats=repeats,
        )

    def json_file_preprocess(self, expression_file, mask_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['videos']

        metas = []
        anno_count = 0  # serve as anno_id
        vid2metaid = {}
        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]

            vid_frames = sorted(vid_express_data['frames'])
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['video'] = vid_name
                meta['exp'] = exp_dict['exp']  # str
                meta['mask_anno_id'] = [str(anno_count), ]

                if 'obj_id' in exp_dict.keys():
                    meta['obj_id'] = exp_dict['obj_id']
                else:
                    meta['obj_id'] = [0, ]  # Ref-Youtube-VOS only has one object per expression
                meta['anno_id'] = [str(anno_count), ]
                anno_count += 1
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id

                meta['length'] = vid_len
                metas.append(meta)
                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name] = []
                vid2metaid[vid_name].append(len(metas) - 1)

        # process mask annotation files
        with open(mask_file, 'rb') as f:
            mask_dict = pickle.load(f)
        return vid2metaid, metas, mask_dict
