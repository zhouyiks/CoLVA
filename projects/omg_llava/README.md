# OMG LLaVA

```commandline
# Pretrain
bash tools/dist.sh train projects/omg_llava/configs/pretrain/omg_llava_7b_pretrain_8gpus.py 8 --deepspeed deepspeed_zero2

# Finetune
bash tools/dist.sh train projects/omg_llava/configs/finetune/omg_llava_7b_finetune_8gpus.py 8 --deepspeed deepspeed_zero2
```
