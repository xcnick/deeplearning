# 文本分类算法

- 使用`THUCNews`数据集
- 使用`transformers`进行文本分类


## 训练脚本

```
python -m torch.distributed.launch --nproc_per_node=2 run_classifier.py \
  --model_type bert \
  --model_path /home/orient/chi/model/chinese_wwm_ext/bert_model_pt.bin \
  --config_name /home/orient/chi/model/chinese_wwm_ext/config.json \
  --tokenizer_name /home/orient/chi/model/chinese_wwm_ext/vocab.txt \
  --do_train \
  --do_eval \
  --data_dir /home/orient/chi/data/THUCNews/ \
  --per_gpu_train_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --output_dir out/ \
  --overwrite_output_dir \
  --do_lower_case \
  --save_steps 10000
```