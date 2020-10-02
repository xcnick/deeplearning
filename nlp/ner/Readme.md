# 命名实体识别（Named-entity recognition）

本项目使用MSRA NER数据集。

## PyTorch Transformers

### 原始版本

```
python -m torch.distributed.launch --nproc_per_node=8 run_ner_msra.py  --model_type bert  --model_name_or_path /code/chinese_wwm_ext/  --do_train  --do_eval --data_dir /mnt/cephfs/dataset/ner/ --labels /mnt/cephfs/dataset/ner/tags.txt --per_gpu_train_batch_size 64  --learning_rate 3e-5 --num_train_epochs 4  --max_seq_length 256  --output_dir /code/logs/pytorch-ner/ --overwrite_output_dir --do_lower_case --save_steps=10000
```

### 修改为动态padding

`DataLoader` 中指定 `collate_fn` 参数，可在每个batch动态padding

性能测试指标如下：

测试硬件：RTX2080Ti * 2：
参数：
--per_gpu_train_batch_size 32  --learning_rate 3e-5 --num_train_epochs 4 --max_seq_length 128  --output_dir output/ --overwrite_output_dir --do_lower_case --save_steps=10000 --fp16

- 动态Padding，单个epoch时长，02:30
- 静态Padding，单个epoch时长，02:30

可能是静态Padding长度为128，未带来太大的计算量

将 `max_seq_length` 改为256时:

- 动态Padding，02:50
- 静态Padding，03:50

若改为512，应该会有更大的差别

### 使用Trainer

```
python -m torch.distributed.launch --nproc_per_node=2 run_ner.py  --model_name_or_path /home/orient/chi/model/chinese_wwm_ext/ --do_train  --do_eval --data_dir data/ --labels data/tags.txt  --per_device_train_batch_size 32   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 128  --output_dir output/ --overwrite_output_dir --save_steps=10000
```


## TensorFlow2 Transformers

### 训练脚本

```
CUDA_VISIBLE_DEVICES=0,1 python run_tf_ner.py --model_name_or_path /home/orient/chi/model/chinese_wwm_ext/ --mode token-classification --do_train  --do_eval --data_dir data/ --labels data/tags.txt  --per_device_train_batch_size 32   --learning_rate 3e-5   --num_train_epochs 2  --max_seq_length 128 --logging_dir runs --output_dir output/ --overwrite_output_dir --save_steps=10000 --fp16
```