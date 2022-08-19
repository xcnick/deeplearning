# PyTorch
import torch
import transformers

torch_model = transformers.BertModel.from_pretrained(
    "./models/chinese_roberta_wwm_ext_pytorch"
)
torch_model.eval()

# ONNXRuntime
import onnxruntime as ort

ort_session = ort.InferenceSession(
    "./onnx/bert_model.onnx", providers=["CPUExecutionProvider"]
)

# OpenVINO
from openvino.runtime import Core, AsyncInferQueue

ie = Core()
onnx_model_path = "./onnx/bert_model.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
input_layer = next(iter(model_onnx.inputs))
compiled_model_onnx = ie.compile_model(
    model=model_onnx, device_name="CPU", config={"PERFORMANCE_HINT": "THROUGHPUT"}
)
request = compiled_model_onnx.create_infer_request()

# input
import numpy as np

dummy_input = np.random.randint(
    low=1, high=torch_model.config.vocab_size - 1, size=(1, 128), dtype=np.int64
)

with torch.no_grad():
    torch_output = torch_model(torch.tensor(dummy_input))
ort_output = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input})
request.infer({input_layer.any_name: dummy_input})
ov_output_0 = request.get_output_tensor(0).data
ov_output_1 = request.get_output_tensor(1).data

np.testing.assert_allclose(
    torch_output[0].numpy(), ort_output[0], rtol=1e-03, atol=1e-05
)
np.testing.assert_allclose(torch_output[0].numpy(), ov_output_0, rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(
    torch_output[1].numpy(), ort_output[1], rtol=1e-03, atol=1e-05
)
np.testing.assert_allclose(torch_output[1].numpy(), ov_output_1, rtol=1e-03, atol=1e-05)

import time

warm_up_iters = 10
inference_iters = 100

batch_size = [1, 4, 8, 16]
seq_len = [10, 50, 100, 200, 300, 400, 500]
for bs in batch_size:
    for sl in seq_len:
        dummy_input = np.random.randint(
            low=1, high=torch_model.config.vocab_size - 1, size=(bs, sl), dtype=np.int64
        )

        # pytorch
        # warm up
        with torch.no_grad():
            for _ in range(warm_up_iters):
                torch_output = torch_model(torch.tensor(dummy_input))
            # inference test
            start_time = time.time()
            for _ in range(inference_iters):
                torch_model(torch.tensor(dummy_input))
            torch_time = time.time() - start_time
            print(f"pytorch: batch_size {bs}, seq_len {sl}, {torch_time:.2f} s")

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
        print(f"onnxruntime: batch_size {bs}, seq_len {sl}, {ort_time:.2f} s")

        # openvino
        model_onnx.reshape([bs, sl])
        compiled_model_onnx = ie.compile_model(
            model=model_onnx,
            device_name="CPU",
            config={"PERFORMANCE_HINT": "THROUGHPUT"},
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
        print(f"openvino-fp32: batch_size {bs}, seq_len {sl}, {ov_time:.2f} s")
