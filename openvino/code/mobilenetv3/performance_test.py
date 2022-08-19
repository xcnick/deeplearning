# TensorFlow Saved Model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf

saved_model_path = "saved_model"
tf_model = tf.saved_model.load(saved_model_path)

# ONNXRuntime
import onnxruntime as ort

ort_session = ort.InferenceSession(
    "./onnx/mobilenetv3.onnx", providers=["CPUExecutionProvider"]
)

# OpenVINO
from openvino.runtime import Core, AsyncInferQueue

ie = Core()
onnx_model_path = "./onnx/mobilenetv3.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
input_layer = next(iter(model_onnx.inputs))
compiled_model_onnx = ie.compile_model(
    model=model_onnx, device_name="CPU", config={"PERFORMANCE_HINT": "THROUGHPUT"}
)
request = compiled_model_onnx.create_infer_request()
# INT8
ir_model_path = "./int8/mobilenetv3.xml"
model_ir = ie.read_model(model=ir_model_path)


# input
import numpy as np

dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)

tf_output = tf_model(tf.convert_to_tensor(dummy_input))
ort_output = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input})
request.infer({input_layer.any_name: dummy_input})
ov_output = request.get_output_tensor(0).data

np.testing.assert_allclose(tf_output, ort_output[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(tf_output, ov_output, rtol=1e-03, atol=1e-05)

import time

warm_up_iters = 100
inference_iters = 1000

batch_size = [1, 2, 4, 8, 16]
for bs in batch_size:
    dummy_input = np.random.randn(bs, 224, 224, 3).astype(np.float32)

    # tensorflow saved_model
    tf_input = tf.convert_to_tensor(dummy_input)
    # warm up
    for _ in range(warm_up_iters):
        tf_model(tf_input)
    # inference test
    start_time = time.time()
    for _ in range(inference_iters):
        tf_model(tf_input)
    tf_time = time.time() - start_time
    print(f"tensorflow: bs {bs}, {tf_time:.2f} s")

    # onnxruntime
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
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
    model_onnx.reshape([bs, 224, 224, 3])
    compiled_model_onnx = ie.compile_model(
        model=model_onnx, device_name="CPU", config={"PERFORMANCE_HINT": "THROUGHPUT"}
    )
    infer_queue = AsyncInferQueue(compiled_model_onnx, 16)
    # warm up
    for _ in range(warm_up_iters):
        infer_queue.start_async(inputs={input_layer.any_name: dummy_input})
    infer_queue.wait_all()
    # inference test
    start_time = time.time()
    for _ in range(inference_iters):
        infer_queue.start_async(inputs={input_layer.any_name: dummy_input})
    infer_queue.wait_all()
    ov_time = time.time() - start_time
    print(f"openvino-fp32: bs {bs}, {ov_time:.2f} s")

    # openvino INT8
    model_ir.reshape([bs, 224, 224, 3])
    compiled_model_ir = ie.compile_model(
        model=model_ir, device_name="CPU", config={"PERFORMANCE_HINT": "THROUGHPUT"}
    )
    infer_queue = AsyncInferQueue(compiled_model_ir, 16)
    # warm up
    for _ in range(warm_up_iters):
        infer_queue.start_async(inputs={input_layer.any_name: dummy_input})
    infer_queue.wait_all()
    # inference test
    start_time = time.time()
    for _ in range(inference_iters):
        infer_queue.start_async(inputs={input_layer.any_name: dummy_input})
    infer_queue.wait_all()
    ov_time = time.time() - start_time
    print(f"openvino-int8: bs {bs}, {ov_time:.2f} s")
