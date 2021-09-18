# 对联生成算法

## 使用NER方式（不推荐）

* 使用BERT提取特征
* 使用命名实体识别方式进行训练，即对每个token进行分类，模拟生成对联的方式

```
python -m torch.distributed.launch --nproc_per_node=2 run_couplet.py  --model_name_or_path /home/orient/chi/model/chinese_wwm_ext/ --do_train  --do_predict --data_dir data/ --per_device_train_batch_size 32   --learning_rate 3e-5   --num_train_epochs 2  --max_seq_length 128  --output_dir output/ --overwrite_output_dir --save_steps 10000
```

## 使用Seq2Seq模型（推荐）

* 使用BERT提取特征
* 使用UniLM方式进行训练

## 原始版本

```
python -m torch.distributed.launch --nproc_per_node=2 run_couplet_seq.py  --model_name_or_path /home/orient/chi/model/chinese_wwm_ext/ --do_train  --do_predict --data_dir data/ --per_device_train_batch_size 32   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 128  --output_dir output/ --overwrite_output_dir --save_steps 10000
```

## 使用Trainer

```
python -m torch.distributed.launch --nproc_per_node=2 run_couplet_trainer.py  --model_name_or_path /home/orient/chi/model/chinese_wwm_ext/ --do_train  --do_predict --data_dir data/ --per_device_train_batch_size 32   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 128  --output_dir output/ --overwrite_output_dir --save_steps 10000 --per_device_eval_batch_size 1
```

## 推理

- MLflow启动脚本

```
mlflow models serve --model-uri couplets/ --host=0.0.0.0 --port=5000 --no-conda
```

- 客户端访问脚本

```
curl -H "Content-Type: application/json" -X POST -d '{"columns":["input"],"data": [["有朋自远方来"]]}' http://192.168.4.11:5000/invocations
```