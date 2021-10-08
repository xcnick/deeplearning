# Transformer

- 基于Transformer实现的各种预训练模型，包含：
  - Bert
  - Albert
  - Electra
  - GPT2
  - TTA
- 使用 PyTorch / TensorFlow2.0 / MindSpore 实现
- 仅包含基本的模型实现，下游任务需自行定义网络模型

## 转换脚本

支持从 TensorFlow 版本模型和 Huggingface PyTorch 版本模型转换为 TensorFlow ckpt 模型、PyTorch 模型、MindSpore 。

```
python convert_weights.py --model_type bert \
  --from_model tf \
  --from_path $MODEL_PATH \
  --config_path $CONFIG_FILE_PATH \
  --to_model tf \
  --dump_path $DUMP_PATH
```

## 示例代码

### 文本分类算法

以 TensorFlow 2 版本实现为例

```bash
# 将本项目加入 `PYTHONPATH`
export PYTHONPATH=$PWD:${PYTHONPATH}

# 使用 tools/train_tf.py 进行训练
cd transformer/tools
python train_tf.py --config=../configs/models/bert/seqcls.py --train_url=/workspace/outputs/thucnews/ --fp16
```