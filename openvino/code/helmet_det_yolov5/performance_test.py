import cv2
import numpy as np

# input
srcimg = cv2.imread("./10.jpg")
img = srcimg[..., ::-1]
h, w, c = img.shape
target = 640

# Scale ratio (new / old)
scale = min(target / h, target / w)
# if not scaleup:  # only scale down, do not scale up (for better val mAP)
#     r = min(r, 1.0)
# Compute padding
new_unpad = int(round(w * scale)), int(round(h * scale))
dw, dh = target - new_unpad[0], target - new_unpad[1]  # wh padding
dw //= 2  # divide padding into 2 sides
dh //= 2

img = cv2.resize(img, new_unpad)

top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

img = cv2.copyMakeBorder(
    img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
)  # add border

img = np.expand_dims(img, axis=0)

img = np.float32(img)
img /= 255  # 0 - 255 to 0.0 - 1.0
img = np.transpose(img, (0, 3, 1, 2))


# PyTorch
import torch

torch_model = torch.jit.load("./helmet_head_person_s.torchscript")

# ONNXRuntime
import onnxruntime as ort

ort_session = ort.InferenceSession(
    "./onnx/helmet_head_person_s.onnx", providers=["CPUExecutionProvider"]
)

# OpenVINO
from openvino.runtime import Core, AsyncInferQueue

ie = Core()
onnx_model_path = "./onnx/helmet_head_person_s.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
input_layer = next(iter(model_onnx.inputs))
compiled_model_onnx = ie.compile_model(
    model=model_onnx, device_name="CPU", config={"PERFORMANCE_HINT": "THROUGHPUT"}
)
request = compiled_model_onnx.create_infer_request()

with torch.no_grad():
    torch_output = torch_model(torch.tensor(img, dtype=torch.float32))
ort_output = ort_session.run(None, {ort_session.get_inputs()[0].name: img})
request.infer({input_layer.any_name: img})
ov_output = request.get_output_tensor(0).data

np.testing.assert_allclose(
    torch_output[0].numpy(), ort_output[0], rtol=1e-03, atol=1e-05
)
np.testing.assert_allclose(torch_output[0].numpy(), ov_output, rtol=1e-03, atol=1e-05)


import time

warm_up_iters = 100
inference_iters = 1000

batch_size = [1, 2, 4, 8, 16]
for bs in batch_size:
    input = np.concatenate([img] * bs, axis=0)

    # TorchScript
    with torch.no_grad():
        for _ in range(warm_up_iters):
            torch_output = torch_model(torch.tensor(input))
        # inference test
        start_time = time.time()
        for _ in range(inference_iters):
            torch_model(torch.tensor(input))
        torch_time = time.time() - start_time
        print(f"torchscript: bs {bs}, {torch_time:.2f} s")

    # onnxruntime
    ort_inputs = {ort_session.get_inputs()[0].name: input}
    # warm up
    for _ in range(warm_up_iters):
        ort_session.run(None, ort_inputs)
    # inference test
    start_time = time.time()
    for _ in range(inference_iters):
        ort_session.run(None, ort_inputs)
    ort_time = time.time() - start_time
    print(f"onnxruntime: bs {bs}, {ort_time:.2f} s")

    # openvino
    infer_queue = AsyncInferQueue(compiled_model_onnx, 16)
    # warm up
    for _ in range(warm_up_iters):
        infer_queue.start_async(inputs={input_layer.any_name: input})
    infer_queue.wait_all()
    # inference test
    start_time = time.time()
    for _ in range(inference_iters):
        infer_queue.start_async(inputs={input_layer.any_name: input})
    infer_queue.wait_all()
    ov_time = time.time() - start_time
    print(f"openvino-fp32: bs {bs}, {ov_time:.2f} s")
