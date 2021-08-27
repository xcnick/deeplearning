# Ray serve 调研

Ray serve 是一个基于 ray 的模型服务框架，其特点是：

1. 与深度学习引擎无关，可支持 TensorFlow, PyTorch, Keras 等框架；
2. 使用 Python 代码实现模型推理服务，而不是 YAML 或 JSON 配置文件。

主要支持两种使用方式：

1. 使用内置的 HTTP 服务，并使用 Python 函数和类部署模型服务；
2. 使用其他的 Web 服务，并使用内置的 Python [ServeHandle API](https://docs.ray.io/en/master/serve/package-ref.html#servehandle-api)

> 从 Ray 1.2.0 版本开始，Ray Serve 使用 Starlette 请求对象替代 Flask 请求对象，具体参考[迁移说明](https://docs.google.com/document/d/1CG4y5WTTc4G_MRQGyjnb_eZ7GK3G9dUX6TNLKLnKRAc/edit)。迁移原因是 Starlette 请求对象被 Web 框架广泛使用，简化 Ray serve 框架集成其他 Web 框架的过程。下文以 Ray 1.2.0 版本为例进行说明。

## 快速开始

将 Python 函数封装成服务：

```python
import ray
from ray import serve
import requests

ray.init(num_cpus=4)
client = serve.start()

def say_hello(request):
    return "hello " + request.query_params["name"] + "!"

# Form a backend from our function and connect it to an endpoint.
client.create_backend("my_backend", say_hello)
client.create_endpoint("my_endpoint", backend="my_backend", route="/hello")

# Query our endpoint in two different ways: from HTTP and from Python.
print(requests.get("http://127.0.0.1:8000/hello?name=serve").text)
# > hello serve!
print(ray.get(client.get_handle("my_endpoint").remote(name="serve")))
# > hello serve
```

将有状态类封装成服务：

```python
import ray
from ray import serve
import requests

ray.init(num_cpus=4)
client = serve.start()

class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, request):
        self.count += 1
        return {"count": self.count}

# Form a backend from our class and connect it to an endpoint.
client.create_backend("my_backend", Counter)
client.create_endpoint("my_endpoint", backend="my_backend", route="/counter")

# Query our endpoint in two different ways: from HTTP and from Python.
print(requests.get("http://127.0.0.1:8000/counter").json())
# > {"count": 1}
print(ray.get(client.get_handle("my_endpoint").remote()))
# > {"count": 2}
```

Ray serve 中的核心概念：`backends`, `endpoints`。

## Ray Serve 优势

深度学习模型推理服务有两种：

1. 使用传统 web server，例如 Flask 框架。缺点是难以水平扩展服务规模。
2. 使用公有云服务，如 SageMaker 等，或者使用特定引擎的框架，如 TFServing。缺点是灵活性一般。

Ray serve 可以解决上述问题，它提供了一个简单的 web server（也可以用外部web server），同时也提供了路由处理、水平扩展等功能，包括：

- 水平扩展多副本 backends；
- 通过去耦合路由逻辑与响应处理逻辑，实现网络流量分离；
- 批量请求，提升推理性能；
- CPU、GPU 资源管理，可实现多个模型复用 GPU。

Ray Serve 适用场景：

- Python-based 机器学习模型部署；
- 模型部署规模的需要具有水平扩展能力；
- 有大规模批量推理需求；
- 后端不同模型需要进行 A/B 测试，或控制不同模型之间的流量。


## Backend

### 部署 Backend

client 可以部署多个 backend。每个 backend 可以有多个副本，每个副本是 Ray 集群中单独的进程。每个 backend 需要定义一个 handler，用于处理 Starlette 格式的请求，并返回 JSON 格式的输出。handler 可以是函数或者类。

定义 backend

```python
def handle_request(starlette_request):
  return "hello world"

class RequestHandler:
  # Take the message to return as an argument to the constructor.
  def __init__(self, msg):
      self.msg = msg

  def __call__(self, starlette_request):
      return self.msg

client.create_backend("simple_backend", handle_request)
# Pass in the message that the backend will return as an argument.
# If we call this backend, it will respond with "hello, world!".
client.create_backend("simple_backend_class", RequestHandler, "hello, world!")
```

查看和删除 backend

```bash
>> client.list_backends()
{
    'simple_backend': {'accepts_batches': False, 'num_replicas': 1, 'max_batch_size': None},
    'simple_backend_class': {'accepts_batches': False, 'num_replicas': 1, 'max_batch_size': None},
}
>> client.delete_backend("simple_backend")
>> client.list_backends()
{
    'simple_backend_class': {'accepts_batches': False, 'num_replicas': 1, 'max_batch_size': None},
}
```

### 暴露 backend

创建 endpoint 即可将 backend 以 HTTP 服务的形式暴露。

创建 endpoint

```python
client.create_endpoint("simple_endpoint", backend="simple_backend", route="/simple", methods=["GET"])
```

检查服务是否正常启动

```python
import requests
print(requests.get("http://127.0.0.1:8000/simple").text)
```

使用 ServerHandle 接口查询 endpoint

```python
handle = client.get_handle("simple_endpoint")
print(ray.get(handle.remote()))
```

查看所有 endpoint

```bash
>>> client.list_endpoints()
{'simple_endpoint': {'route': '/simple', 'methods': ['GET'], 'traffic': {}}}
```

删除 endpoint

```python
client.delete_endpoint("simple_endpoint")
```

### 配置 backend

将一个 backend 水平扩展成多个副本，即多个进程

```python
config = {"num_replicas": 10}
client.create_backend("my_scaled_endpoint_backend", handle_request, config=config)

# scale it back down...
config = {"num_replicas": 2}
client.update_backend_config("my_scaled_endpoint_backend", config)
```

资源管理，指定每个副本所需的硬件资源

```python
config = {"num_gpus": 1}
client.create_backend("my_gpu_backend", handle_request, ray_actor_options=config)
```

多副本复用GPU

```python
half_gpu_config = {"num_gpus": 0.5}
client.create_backend("my_gpu_backend_1", handle_request, ray_actor_options=half_gpu_config)
client.create_backend("my_gpu_backend_2", handle_request, ray_actor_options=half_gpu_config)
```

Ray serve 默认设置环境变量 `OMP_NUM_THREADS=1`，可根据需要进行重新设置

```bash
OMP_NUM_THREADS=12 ray start --head
OMP_NUM_THREADS=12 ray start --address=$HEAD_NODE_ADDRESS
```

```python
class MyBackend:
    def __init__(self, parallelism):
        os.environ["OMP_NUM_THREADS"] = parallelism
        # Download model weights, initialize model, etc.

client.create_backend("parallel_backend", MyBackend, 12)
```

批量推理设置：

- 在 config 中设置 `max_batch_size`
- 在 backend 实现中设置接收批量输入

```python
class BatchingExample:
    def __init__(self):
        self.count = 0

    @serve.accept_batch
    def __call__(self, requests):
        responses = []
            for request in requests:
                responses.append(request.json())
        return responses

config = {"max_batch_size": 5}
client.create_backend("counter1", BatchingExample, config=config)
client.create_endpoint("counter1", backend="counter1", route="/increment")
```

可指定 conda 环境，在下面的示例中，两个 backend 分别使用 tf1 和 tf2 环境

```python
import requests
from ray import serve
from ray.serve import CondaEnv
import tensorflow as tf

client = serve.start()

def tf_version(request):
    return ("Tensorflow " + tf.__version__)

client.create_backend("tf1", tf_version, env=CondaEnv("ray-tf1"))
client.create_endpoint("tf1", backend="tf1", route="/tf1")
client.create_backend("tf2", tf_version, env=CondaEnv("ray-tf2"))
client.create_endpoint("tf2", backend="tf2", route="/tf2")

print(requests.get("http://127.0.0.1:8000/tf1").text)  # Tensorflow 1.15.0
print(requests.get("http://127.0.0.1:8000/tf2").text)  # Tensorflow 2.3.0
```

## 部署 Ray Serve

Ray Serve 实例生命周期

- `serve.start` 将会创建本地 ray 集群，并启动 serve 服务，但代码结束时，serve 服务将停止，ray 集群也被销毁
- 为了实现 ray 集群的长期服务，仍需要使用传统 ray 集群启动的方式，即 `ray start --head` 命令，启动 ray 集群，代码中用 `ray.init(address="auto")` 的方式连接集群，部署 ray serve 服务有两种方式：
  - 代码中使用 `serve.start(detached=True)`
  - 使用 `serve start` 命令，并在代码种使用 `serve.connect` 方法


### 在单节点上部署

方式1，不推荐：

```python
import ray
from ray import serve

# This will start Ray locally and start Serve on top of it.
client = serve.start()

def my_backend_func(request):
  return "hello"

client.create_backend("my_backend", my_backend_func)

# Serve will be shut down once the script exits, so keep it alive manually.
while True:
    time.sleep(5)
    print(serve.list_backends())
```

方式2，推荐

```bash
ray start --head # Start local Ray cluster.
serve start # Start Serve on the local Ray cluster.
```

```python
import ray
from ray import serve

# This will connect to the running Ray cluster.
ray.init(address="auto")
client = serve.connect()

def my_backend_func(request):
  return "hello"

client.create_backend("my_backend", my_backend_func)
```

### 在 K8s 上部署

部署的方式与其他 ray 程序基本类似

## 部署深度学习模型

启动 Ray 集群

```bash
ray start --head
```

启动 Ray Serve 服务

```bash
serve start
```

使用代码注册 backend, endpoint 等，进行模型部署

- 服务端关键代码

```python
class TFModel(object):

    def __init__(self):
        root_path = os.path.dirname(os.path.realpath(__file__))

        infer_config_url = os.path.join(root_path, 'infer_config.json')
        if not os.path.exists(infer_config_url):
            raise ValueError("Can not find 'infer_config.json'")
        with open(infer_config_url, 'r') as f:
            infer_config = json.load(f)

        model_config, infer_model_config, self.data_config = \
            infer_config['model_config'], infer_config['infer_model_config'], infer_config['data_config']

        model_weights_url = os.path.join(root_path, 'infer_model.h5')
        if not os.path.exists(model_weights_url):
            raise ValueError("Can not find 'infer_model.h5'")

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # Build Faster R-CNN
        model = faster_rcnn.build_faster_rcnn_model(**model_config)
        model.load_weights(model_weights_url)
        # Build inference model
        self.infer_model = FasterRCNNInferModel(model, **infer_model_config)

        logging.info("Model serving start ...")

    async def __call__(self, input):

        image_bytes = await input.body()
        image_bytes = base64.b64decode(image_bytes)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)

        boxes, scores, classes = self.infer_model.predict(image)
        assert len(boxes) == len(scores) == len(classes)

        outputs = []
        for i in range(len(boxes)):
            bbox = np.array(boxes[i]).astype('int32').tolist()
            bbox = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]  # y1x1y2x2 -> x1y1wh
            outputs.append({'bbox': bbox, 'score': float(scores[i]), 'class': int(classes[i])})

        res = {
            "results": outputs,
            "label_names": self.data_config['label_names'],
            "label_colors": self.data_config['label_colors']
        }

        return res

......


ray.init(address="auto")
client = serve.connect()

# 指定使用gpu数量，不设置则不使用gpu
config = {"num_gpus": 1}
client.create_backend("faster_rcnn", TFModel, ray_actor_options=config)
client.create_endpoint("object_detection", backend="faster_rcnn", route="/object_detection", methods=["POST"])
```

客户端关键代码

```python
with open(path, "rb") as f:
    image_bytes = f.read()

data = base64.encodebytes(image_bytes)

response = requests.post(url='{uri}/object_detection'.format(uri=uri),
                         data=data)

if response.status_code != 200:
    raise Exception("Status Code {status_code}. {text}".format(
        status_code=response.status_code,
        text=response.text
    ))
```

批量推理关键代码

```python
# 服务端
@serve.accept_batch
async def __call__(self, input):

    responses = []
    for input_image in input:
        image = await input_image.body()
        image_bytes = base64.b64decode(image)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)

        boxes, scores, classes = self.infer_model.predict(image)
        assert len(boxes) == len(scores) == len(classes)

        outputs = []
        for i in range(len(boxes)):
            bbox = np.array(boxes[i]).astype('int32').tolist()
            bbox = [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]  # y1x1y2x2 -> x1y1wh
            outputs.append({'bbox': bbox, 'score': float(scores[i]), 'class': int(classes[i])})

        res = {
            "results": outputs,
            "label_names": self.data_config['label_names'],
            "label_colors": self.data_config['label_colors']
        }
        responses.append(res)

    return responses

......

client = serve.start()

config = {"num_gpus": 1}
client.create_backend("faster_rcnn", TFModel, ray_actor_options=config, config={"max_batch_size": 8, "batch_wait_timeout": 1})
client.create_endpoint("object_detection", backend="faster_rcnn", route="/object_detection", methods=["POST"])


# 客户端
import requests
with open("000000000785.jpg", "rb") as f:
    image_bytes = f.read()
data = base64.encodebytes(image_bytes)

handle = client.get_handle("object_detection")

import timeit
print(timeit.timeit("handle.remote(data)", number=1000, globals=globals()))
```

## 性能测试

测试环境

- 硬件
  - CPU：Intel i9 9900x
  - GPU：RTX 2080Ti
- 软件
  - TensorFlow 2.1.0
  - cudatoolkit 10.1.243
  - cudnn 7.6.5
  - ray 1.2.0
- 算法模型
  - faster_rcnn

测试目的，对比 Ray serve 与 MLflow 的性能

使用一张图片，推理 1000 次总耗时

| 推理方式 | 耗时(s) |
| --- | --- |
| Ray Serve | 108.76 |
| Ray Serve (bs=8) | 98.85 |
| MLflow | 117.80 |

其中，Ray Serve 批量推理时，仅指 Web 层面批量数据处理，与模型层面的批量推理无关。

## Ray Serve 系统架构

![Ray](assets/architecture.svg)

Ray Serve 整体系统如图所示。使用命令 `ray start --head`，启动 Ray 集群。使用命令 `serve start`，启动 Ray Serve 服务实例，在集群中创建了两个进程：

1. ServeContoller 进程，对应途中的 Controller 进程。整个集群一个 serve 实例仅有一个此进程，该 Controller 进程主要负责管理 backend、endpoint 等 actor 的创建、更新、销毁等。
2. HTTPProxyActor 进程，对应图中的 HTTP Server 进程，每个节点都有一个，是基于 uvicorn 的 HTTP Server，用于处理 HTTP 请求，并将流量转发到相应的 worker 副本，并进行响应。

启动 worker 副本，将启动 `num_replicas` 个进程，代码如下：

```python
import ray
from ray import serve

def my_backend_func(request):
  return "hello"

ray.init(address="auto")
client = serve.connect()

config = {"num_replicas": 10}
client.create_backend("my_backend", my_backend_func, config=config)
client.create_endpoint("hello", backend="my_backend", route="/hello", methods=["GET"])
```

此时集群中存在上述 `num_replicas` + 2 个进程。

在多节点场景下，将另一个节点加入 Ray 集群，即在 head 节点启动后，使用类似 `ray start --address='192.168.19.130:6379' --redis-password='5241590000000000'` 命令将另一个节点加入集群，在 dashboard 中可观察到两个节点。存在以下与官网文档描述不符的现象：

1. 官网架构图中提到，Router 进程（HTTPProxyActor进程）在每个节点都有一个。但实际观察的现象是，仅在启动 serve 实例的节点上存在 Router 进程。
2. 由于上述现象，导致水平扩展测试时，最大仅能支持与当前节点 worker 数量相同的副本数量，无法使用其他节点的资源。