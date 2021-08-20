# CUDA

# grid、block、thread关系

```cpp
dim3 grid(4, 2, 1), block(8, 6, 1)

/*
gridDim.x = 4
gridDim.y = 2
gridDim.z = 1
blockIdx.x = [0, 3]
blockIdx.y = [0, 1]
blockIdx.z = 0
*/

/*
blockDim.x = 8
blockDim.y = 6
blockDim.z = 1
threadIdx.x = [0, 7]
threadIdx.y = [0, 5]
threadIdx.z = 0
*/

/*
二维
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
*/
```


## rdc=True 动态并行
