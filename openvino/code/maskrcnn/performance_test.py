# preprocess
import numpy as np
from PIL import Image


def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize(
        (int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR
    )

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype("float32")

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math

    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, : image.shape[1], : image.shape[2]] = image
    image = padded_image

    return image


img = Image.open("./bus.jpg")
img_data = preprocess(img)


# ONNXRuntime
import onnxruntime as ort

ort_session = ort.InferenceSession(
    "./onnx/mask_rcnn_R_50_FPN_1x.onnx", providers=["CPUExecutionProvider"]
)

# OpenVINO
from openvino.runtime import Core, AsyncInferQueue

ie = Core()
onnx_model_path = "./onnx/mask_rcnn_R_50_FPN_1x.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
input_layer = next(iter(model_onnx.inputs))
compiled_model_onnx = ie.compile_model(
    model=model_onnx, device_name="CPU", config={"PERFORMANCE_HINT": "THROUGHPUT"}
)
request = compiled_model_onnx.create_infer_request()

# input
import numpy as np

image_input = img_data

ort_output = ort_session.run(None, {ort_session.get_inputs()[0].name: image_input})
request.infer({input_layer.any_name: image_input})
ov_output_boxes_0 = request.get_output_tensor(0).data
ov_output_labels_1 = request.get_output_tensor(1).data
ov_output_scores_2 = request.get_output_tensor(2).data
ov_output_masks_3 = request.get_output_tensor(3).data

np.testing.assert_allclose(ov_output_boxes_0, ort_output[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(ov_output_labels_1, ort_output[1], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(ov_output_scores_2, ort_output[2], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(ov_output_masks_3, ort_output[3], rtol=1e-03, atol=1e-05)


import time

warm_up_iters = 10
inference_iters = 100

# onnxruntime
ort_inputs = {ort_session.get_inputs()[0].name: image_input}
# warm up
for _ in range(warm_up_iters):
    ort_session.run(None, ort_inputs)
# inference test
start_time = time.time()
for _ in range(inference_iters):
    ort_session.run(None, ort_inputs)
ort_time = time.time() - start_time
print(f"onnxruntime: {ort_time:.2f} s")

# openvino
compiled_model_onnx = ie.compile_model(
    model=model_onnx, device_name="CPU", config={"PERFORMANCE_HINT": "THROUGHPUT"}
)
infer_queue = AsyncInferQueue(compiled_model_onnx, 16)
# warm up
for _ in range(warm_up_iters):
    infer_queue.start_async(inputs={input_layer.any_name: image_input})
infer_queue.wait_all()
# inference test
start_time = time.time()
for _ in range(inference_iters):
    infer_queue.start_async(inputs={input_layer.any_name: image_input})
infer_queue.wait_all()
ov_time = time.time() - start_time
print(f"openvino-fp32: {ov_time:.2f} s")
