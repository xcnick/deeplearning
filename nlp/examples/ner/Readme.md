# 命名实体识别算法

- 使用`MSRA`数据集
- 使用`transformers`进行Token分类


## 训练脚本

```
python -m torch.distributed.launch --nproc_per_node=2 run_ner_msra.py \
  --model_type bert \
  --model_path /home/orient/chi/model/chinese_wwm_ext/bert_model_pt.bin \
  --config_name /home/orient/chi/model/chinese_wwm_ext/config.json \
  --tokenizer_name /home/orient/chi/model/chinese_wwm_ext/vocab.txt \
  --do_train \
  --do_eval \
  --data_dir /home/orient/chi/models/ner/data/ \
  --labels /home/orient/chi/models/ner/data/tags.txt \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --output_dir output/ \
  --overwrite_output_dir \
  --do_lower_case \
  --save_steps 10000
```