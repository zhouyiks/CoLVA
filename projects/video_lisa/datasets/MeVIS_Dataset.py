from .ReVOS_Dataset import VideoReVOSDataset


class VideoMeVISDataset(VideoReVOSDataset):
    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file,
                 extra_image_processor=None,
                 tokenizer=None,
                 select_number=5,
                 sampled_frames=10,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=2048,
                 lazy=True,
                 repeats=1,
                 special_tokens=None,
    ):
        super().__init__(
            image_folder=image_folder,
            expression_file=expression_file,
            mask_file=mask_file,
            tokenizer=tokenizer,
            extra_image_processor=extra_image_processor,
            select_number=select_number,
            sampled_frames=sampled_frames,
            offline_processed_text_folder=offline_processed_text_folder,
            template_map_fn=template_map_fn,
            max_length=max_length,
            lazy=lazy,
            repeats=repeats,
            special_tokens=special_tokens,
        )
