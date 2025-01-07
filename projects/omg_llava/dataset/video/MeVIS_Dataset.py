from .ReVOS_Dataset import VideoReVOSDataset


class VideoMeVISDataset(VideoReVOSDataset):
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
