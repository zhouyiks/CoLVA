## Single Transformer SFT code


#### Train 

For CMD (single node):

```commandline
./tools/dist.sh train projects/single_transformer/configs/solo_sft_our_mmistral_llava_resize448.py 8 
```

the logs and ckpts will be saved in "work_dirs/solo_sft_our_mmistral_llava_resize448""

For multi-node training: 

see this trial [link](https://ml.bytedance.net/development/instance/jobs/3662970356615cd3?trialId=32717073), fork this trial and modified number gpus and nodes.

#### Evaluation

For CMD (single node):

```commandline
./tools/dist.sh test projects/single_transformer/configs/solo_sft_our_mmistral_llava_resize448_test.py 8 --checkpoint work_dirs/solo_sft_our_mmistral_llava_resize448/xxxx.pth/
```

Note that please set the test datasets in the configs.


#### Datasets

put the datasets in the ./data folder.

% training data
./data
---./Cambrian-10M
---./llava_data
---------./llava_images
---------./LLaVA-Instruct-150K
---------./LLaVA-Pretrain

% evaluation data 
./data
---./eval

Download pre-processed data from huggingface. https://huggingface.co/datasets/OMG-Research/VLM_EVAL

See the folder: /mnt/bn/xiangtai-training-data/project/VLM/data/eval


#### Configs

