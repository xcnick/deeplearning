# 文本分类算法

- 使用`THUCNews`数据集
- 使用`transformers`进行文本分类


## 训练脚本

```
python -m torch.distributed.launch --nproc_per_node=2 run_classifier.py \
  --model_type bert \
  --model_name_or_path /home/orient/chi/model/chinese_wwm_ext/ \
  --do_train \
  --do_eval \
  --data_dir data \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --output_dir output \
  --overwrite_output_dir \
  --do_lower_case \
  --save_steps 10000
```