# Transformer

- 基于Transformer实现的各种预训练模型，包含：
  - Bert
  - Albert
  - Electra
  - GPT2
  - TTA
- 使用PyTorch和TensorFlow2.0实现
- 仅包含基本的模型实现，下游任务需自行定义网络模型

## 转换脚本

支持从TensorFlow版本模型和Huggingface PyTorch版本模型转换为TensorFlow ckpt模型或PyTorch模型。

```
python convert_weights.py --model_type bert \
  --from_model tf \
  --from_path $MODEL_PATH \
  --config_path $CONFIG_FILE_PATH \
  --to_model tf \
  --dump_path $DUMP_PATH
```

## 示例代码

在 `examples` 文件夹中。

- 文本分类
- NER