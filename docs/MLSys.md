# 深度学习引擎

- [AI 框架基础技术之自动求导机制 (Autograd)
](https://zhuanlan.zhihu.com/p/347385418)
- [Tensor 的简易实现](https://zhuanlan.zhihu.com/p/340228853)


# 分布式训练

- [Optimizer state sharding (ZeRO)](https://zhuanlan.zhihu.com/p/394064174)
- [即时翻译技术](https://zhuanlan.zhihu.com/p/381119145)
- [AMP 混合精度](https://zhuanlan.zhihu.com/p/375224982)
- [模型量化](https://zhuanlan.zhihu.com/p/354921065)

# 引擎编译实验

- 使用自定义 Docker 镜像
- 区分 CUDA 10.1/11.1 等版本

## MindSpore 编译

仅验证 CUDA 10.1 版本

```bash
# 进入容器
docker run -it --rm --gpus=all -v /home/orient/code/mindspore/:/workspace devenv:0.3 bash

# 进行编译
sh build.sh -e gpu -j 16

# 编译完成后，生成的 pip 安装包路径
build/package/

# 将 build 路径加入 PYTHONPATH
export PYTHONPATH=$PWD:${PYTHONPATH}
```

## OneFlow 编译

```bash
# 进入容器
docker run -it --rm --gpus=all -v /home/orient/code/oneflow/:/workspace devenv:0.3 bash

# 进行编译
mkdir build
cd build
cmake .. -C ../cmake/caches/cn/cuda.cmake

# 将 oneflow 加入 PYTHONPATH
source source.sh
```

## Paddle 编译实验

```bash
# 进入容器
docker run -it --rm --gpus=all -v /home/orient/chi/Paddle/:/workspace mindspore paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82 bash

# 进行编译
mkdir -p /workspace/build && cd /workspace/build
cmake .. -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
make -j 16

# 编译完成后，生成 pip 包安装路径
build/python/dist/

# 进入 build/python 路径，将此路径加入 PYTHONPATH
export PYTHONPATH=$PWD:${PYTHONPATH}

# 修改 GCC 版本
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100

# 查看设置结果
sudo update-alternatives --config gcc
```

## 编译性能

| 引擎          | 硬件          | 耗时   |
| ------------- | ------------- | ------ |
| MindSpore 1.3 | 6248 64 core  | 11 min |
| MindSpore 1.3 | 9900x 20 core | 20 min |
| MindSpore 1.3 | 5900x 24 core | 15 min |
| OneFlow 0.4   | 6248 64 core  | 17 min |
| OneFlow 0.4   | 9900x 20 core | 25 min |
| OneFlow 0.4   | 5900x 24 core | 17 min |
| PyTorch 1.10  | 6248 64 core  | 21 min |
| PyTorch 1.10  | 9900x 20 core | 36 min |
| PyTorch 1.10  | 5900x 24 core | 67 min |