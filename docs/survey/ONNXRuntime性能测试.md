# ONNXRuntime性能测试

## TensorFlow saved_model

### Resnet50

- 使用 `tensorflow.keras.applications.resnet50` 的 `ResNet50` 类导出 `saved_model` 权重
- 使用 `tf2onnx` 将 `saved_model` 转换为 `onnx` 格式

```bash
python -m tf2onnx.convert --saved-model /mnt/data/models/resnet50/saved_model --output /mnt/data/models/resnet50/resnet.onnx
```

- 测试环境
  - 硬件
    - CPU：Intel i9 9900x
    - GPU：RTX 2080Ti
  - 软件
    - TensorFlow 2.1.0
    - cudatoolkit 10.1.243
    - cudnn 7.6.5
    - onnxruntime 1.4.0

- 测试代码

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

model_path = "/mnt/data/models/resnet50/saved_model"
model = tf.saved_model.load(model_path)
# keras model 模式
# model = tf.keras.models.load_model(model_path)

dummy_input = tf.random.normal([8, 224, 224, 3], 0, 1, tf.float32)
# keras model 模式
# tf_output = model.predict(dummy_input, batch_size=8)

import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("/mnt/data/models/resnet50/resnet.onnx")

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and TensorFlow results
np.testing.assert_allclose(tf_output, ort_outs[0], rtol=1e-03, atol=1e-05)

# benchmark
import timeit

batch_size = [1, 2, 4, 8, 16]
for bs in batch_size:
    dummy_input = tf.random.normal([bs, 224, 224, 3], 0, 1, tf.float32)
    print('tf: bs {}'.format(bs), timeit.timeit('model.predict_on_batch(dummy_input)',
                                                number=1000, globals=globals()))
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    print('ort: bs {}'.format(bs), timeit.timeit("ort_session.run(None, ort_inputs)",
                                                number=1000, globals=globals()))
```

- 测试结果

推理代码如上所示，推理1000次记录总耗时

| batch_size | keras predict(s) | keras call(s) | keras predict_on_batch(s) | tf.saved_model(s) | onnxruntime(s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 28.10 | 52.17 | 9.08 | 7.61 | 3.83 |
| 2 | 27.14 | 53.65 | 9.35 | 7.38 | 5.66 |
| 4 | 30.35 | 52.85 | 8.08 | 7.34 | 9.35 |
| 8 | 36.03 | 53.96 | 11.66 | 11.42 | 17.35 |
| 16 | 44.77 | 53.61 | 20.70 | 20.46 | 32.53 |

在 `tf.keras` 中， `predict` 和  `__call__` 方法都是报warning

```bash
WARNING:tensorflow:5 out of the last 5 calls to <function recreate_function.<locals>.restored_function_body at 0x7f2b5411dcb0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details
```

### Bert

- 使用 `tensorflow.hub` 下载 Bert 的 `saved_model` 权重
- 使用 `tf2onnx` 将 `saved_model` 转换为 `onnx` 格式，其中 `opset` 设置为12，因为该版本后才能支持 `einsum` op

```bash
python -m tf2onnx.convert --saved-model /mnt/data/models/nlp/bert_model/ --opset 12 --output /mnt/data/models/nlp/bert_model/bert_model.onnx
```

- 测试环境
  - 硬件
    - CPU：Intel i9 9900x
    - GPU：RTX 2080Ti
  - 软件
    - TensorFlow 2.1.0
    - cudatoolkit 10.1.243
    - cudnn 7.6.5
    - onnxruntime 1.4.0

- 测试代码

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


# 加载模型
model_path = "/mnt/data/models/nlp/bert_model/"
model = tf.saved_model.load(model_path)
#model = tf.keras.models.load_model(model_path)

dummy_input = tf.random.uniform([8, 128], 1, 8000, tf.int32)
tf_output = model({"input_word_ids":dummy_input, "input_mask": dummy_input, "input_type_ids": dummy_input})


# onnxruntime
import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("/mnt/data/models/nlp/bert_model/bert_model.onnx")

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy(),
              ort_session.get_inputs()[1].name: dummy_input.numpy(),
              ort_session.get_inputs()[2].name: dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and TensorFlow results
np.testing.assert_allclose(tf_output["pooled_output"].numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)


# benchmark
import timeit

batch_size = [1, 2, 4, 8, 16]
seq_len = [10, 50, 100, 200, 300, 400]
for bs in batch_size:
    for sl in seq_len:
        dummy_input = tf.random.uniform([bs, sl], 1, 8000, tf.int32)
        print('tf: batch_size {} seq_len {} '.format(bs, sl), timeit.timeit('model({"input_word_ids":dummy_input, "input_mask": dummy_input, "input_type_ids": dummy_input})',
                                                    number=1000, globals=globals()))
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy(),
                    ort_session.get_inputs()[1].name: dummy_input.numpy(),
                    ort_session.get_inputs()[2].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        print('ort: batch_size {} seq_len {} '.format(bs, sl), timeit.timeit("ort_session.run(None, ort_inputs)",
                                                    number=1000, globals=globals()))
```

- 测试结果

推理代码如上所示，推理1000次记录总耗时

| batch_size | seq_len | tf.saved_model(s) | onnxruntime(s) |
| --- | --- | --- | --- |
| 1 | 10 | 10.20 | 6.27 |
| 1 | 50 | 7.86 | 7.65 |
| 1 | 100 | 8.81 | 9.81 |
| 1 | 200 | 9.36 | 12.96 |
| 1 | 300 | 10.40 | 16.45 |
| 1 | 400 | 14.41 | 21.22 |
| 1 | 500 | 19.08 | 25.47 |
| 4 | 10 | 7.41 | 7.03 |
| 4 | 50 | 10.13 | 11.98 |
| 4 | 100 | 12.46 | 17.60 |
| 4 | 200 | 23.61 | 30.82 |
| 4 | 300 | 36.08 | 46.21 |
| 4 | 400 | 49.42 | 62.79 |
| 4 | 500 | 63.80 | 80.64 |
| 8 | 10 | 8.16 | 8.06 |
| 8 | 50 | 12.19 | 17.36 |
| 8 | 100 | 22.32 | 29.41 |
| 8 | 200 | 44.72 | 59.19 |
| 8 | 300 | 68.87 | 88.73 |
| 8 | 400 | 94.78 | 119.24 |
| 8 | 500 | 124.46 | 156.63 |
| 16 | 10 | 8.93 | 10.13 |
| 16 | 50 | 21.68 | 28.83 |
| 16 | 100 | 42.45 | 56.39 |
| 16 | 200 | 85.39 | 115.94 |
| 16 | 300 | 134.33 | 176.11 |
| 16 | 400 | 186.09 | 235.80 |
| 16 | 500 | 243.73 | 322.51 |

## PyTorch

### Resnet50

- 使用 `torchvision.models.resnet50` 类导出权重文件
- 使用 `torch.onnx.export` 将 `pth` 转换为 `onnx` 格式，设置 `batch_size` 维度为动态

```python
input_names = [ "input" ]
output_names = [ "output" ]

torch.onnx.export(model, dummy_input, "/mnt/data/models/resnet50/resnet50.onnx", verbose=True, input_names=input_names, output_names=output_names, dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
```

- 测试环境
  - 硬件
    - CPU：Intel i9 9900x
    - GPU：RTX 2080Ti
  - 软件
    - PyTorch 1.6.0
    - cudatoolkit 10.1.243
    - cudnn 7.6.5
    - onnxruntime 1.4.0

- 测试代码

```python
import torch
import torchvision

dummy_input = torch.randn((10, 3, 224, 224), device='cuda')
model = torchvision.models.resnet50(pretrained=False).cuda()
pth_file = "/mnt/data/models/nlp/resnet50-19c8e357.pth"
model.load_state_dict(torch.load(pth_file))
model.eval()

torch_out = model(dummy_input)


import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("/mnt/data/models/resnet50/resnet50.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


# benchmark
import timeit

batch_size = [1, 2, 4, 8, 16]
for bs in batch_size:
    dummy_input = torch.randn((bs, 3, 224, 224), device='cuda')
    print('pytorch: bs {}'.format(bs), timeit.timeit('model(dummy_input)',
                                                number=1000, globals=globals()))
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    print('ort: bs {}'.format(bs), timeit.timeit("ort_session.run(None, ort_inputs)",
                                                number=1000, globals=globals()))
```

- 测试结果

推理代码如上所示，推理1000次记录总耗时

| batch_size | PyTorch(s) | onnxruntime(s) |
| --- | --- | --- |
| 1 | 4.28 | 3.01 |
| 2 | 4.32 | 3.96 |
| 4 | 5.79 | 6.03 |
| 8 | 10.24 | 10.68 |
| 16 | 19.88 | 19.08 |

### Bert

- 使用 `transformers` 框架下载预训练 Bert 权重
- 使用 `torch.onnx.export` 将 `bin` 转换为 `onnx` 格式，设置输入输出 `batch_size` 、 `seq_len` 维度为动态

```python
torch.onnx.export(model=model,
                args=(input_ids, ),
                verbose=True,
                f="/mnt/data/models/nlp/chinese_wwm_ext/bert_model.onnx",
                opset_version=10,
                input_names=['input'],
                output_names=['output_1', 'output_2'],
                dynamic_axes={
                    'input': [0, 1],
                    'output_1': [0, 1],
                    'output_2': [0],
                })
```

- 测试环境
  - 硬件
    - CPU：Intel i9 9900x
    - GPU：RTX 2080Ti
  - 软件
    - PyTorch 1.6.0
    - cudatoolkit 10.1.243
    - cudnn 7.6.5
    - onnxruntime 1.4.0

- 测试代码

```python
import torch
import transformers

test_device = torch.device('cuda')
model = transformers.BertModel.from_pretrained("/mnt/data/models/nlp/chinese_wwm_ext/")
model.to(test_device)
model.eval()

cfg = model.config
input_ids = torch.randint(low=0,
                          high=cfg.vocab_size - 1,
                          size=(4, 128),
                          dtype=torch.long,
                          device=test_device)

torch_out = model(input_ids)


# onnxruntime
import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("/mnt/data/models/nlp/chinese_wwm_ext/bert_model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)


# benchmark
import timeit

batch_size = [1, 4, 8, 16]
seq_len = [10, 50, 100, 200, 300, 400, 500]
for bs in batch_size:
    for sl in seq_len:
        input_ids = torch.randint(low=0,
                                high=cfg.vocab_size - 1,
                                size=(bs, sl),
                                dtype=torch.long,
                                device=test_device)
        print('pytorch: batch_size {} seq_len {} '.format(bs, sl), timeit.timeit('model(input_ids)',
                                                    number=1000, globals=globals()))
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids)}
        ort_outs = ort_session.run(None, ort_inputs)
        print('ort: batch_size {} seq_len {} '.format(bs, sl), timeit.timeit("ort_session.run(None, ort_inputs)",
                                                    number=1000, globals=globals()))
```

- 测试结果

推理代码如上所示，推理1000次记录总耗时

| batch_size | seq_len | PyTorch(s) | onnxruntime(s) |
| --- | --- | --- | --- |
| 1 | 10 | 7.68 | 1.83 |
| 1 | 50 | 7.69 | 2.42 |
| 1 | 100 | 7.69 | 3.94 |
| 1 | 200 | 7.73 | 5.74 |
| 1 | 300 | 8.59 | 7.61 |
| 1 | 400 | 13.22 | 10.26 |
| 1 | 500 | 14.89 | 12.61 |
| 4 | 10 | 7.65 | 2.29 |
| 4 | 50 | 7.74 | 5.43 |
| 4 | 100 | 11.51 | 8.82 |
| 4 | 200 | 18.45 | 16.82 |
| 4 | 300 | 28.16 | 26.67 |
| 4 | 400 | 41.98 | 35.15 |
| 4 | 500 | 50.94 | 45.03 |
| 8 | 10 | 7.68 | 3.01 |
| 8 | 50 | 11.32 | 8.65 |
| 8 | 100 | 17.32 | 15.95 |
| 8 | 200 | 38.48 | 32.00 |
| 8 | 300 | 53.21 | 51.20 |
| 8 | 400 | 73.36 | 69.44 |
| 8 | 500 | 95.24 | 87.56 |
| 16 | 10 | 7.70 | 4.39 |
| 16 | 50 | 16.96 | 15.65 |
| 16 | 100 | 36.43 | 30.25 |
| 16 | 200 | 66.20 | 62.18 |
| 16 | 300 | 103.63 | 98.97 |
| 16 | 400 | 142.59 | 135.01 |
| 16 | 500 |  |  |

batch_size=16, seq_len=500，CUDA OOM
