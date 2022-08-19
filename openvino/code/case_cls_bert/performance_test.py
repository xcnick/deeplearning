import numpy as np
import pandas as pd
import onnxruntime as ort
from openvino.runtime import Core, AsyncInferQueue
from tokenizer import Tokenizer


dummy_input = np.random.randint(low=1, high=10000, size=(1, 128), dtype=np.int64)

# Onnxruntime
ort_sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# OpenVINO
ie = Core()
model_ir = ie.read_model(model="model.xml")
input_layer = next(iter(model_ir.inputs))
compiled_model_ir = ie.compile_model(
    model=model_ir, device_name="CPU", config={"PERFORMANCE_HINT": "THROUGHPUT"}
)
request = compiled_model_ir.create_infer_request()

ort_output = ort_sess.run(
    None,
    {
        ort_sess.get_inputs()[0].name: dummy_input,
        ort_sess.get_inputs()[1].name: dummy_input,
    },
)
request.infer(
    {
        model_ir.inputs[0].any_name: dummy_input,
        model_ir.inputs[1].any_name: dummy_input,
    }
)
ov_output_0 = request.get_output_tensor(0).data

np.testing.assert_allclose(ov_output_0, ort_output[0], rtol=1e-03, atol=1e-05)


tokenizer = Tokenizer(vocab_file="./models/bert_base/vocab.txt", do_lower_case=True)


def convert_to_input(tokenizer, input_text):
    max_seq_length = 512
    special_tokens_count = 2
    cls_token = "[CLS]"
    sep_token = "[SEP]"

    tokens = tokenizer.tokenize(input_text)
    if len(tokens) > (max_seq_length - special_tokens_count):
        tokens = tokens[: (max_seq_length - special_tokens_count)]
    tokens = [cls_token] + tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    return input_ids, input_mask


# 读取原始数据
raw_df = pd.read_csv("./data/mhwy_cls.csv")


# ONNXRuntime
res = []
batch_count = 0

import time

start_time = time.time()
for _, line in raw_df.iterrows():
    input_text = line["案件信息"].strip().replace("\n", "")

    input_ids, input_mask = convert_to_input(tokenizer, input_text)

    outputs = ort_sess.run(
        None,
        {
            "input_1": np.array(np.expand_dims(input_ids, axis=0)),
            "input_2": np.array(np.expand_dims(input_mask, axis=0)),
        },
    )

    res.extend(np.argmax(outputs, axis=-1).tolist()[0])

ort_time = time.time() - start_time
print(f"ort: {len(raw_df)} samples in {ort_time:.2f} s ")


# OpenVINO
batch = 64
res = []
batch_count = 0
infer_queue = AsyncInferQueue(compiled_model_ir, batch)

import time

start_time = time.time()
for _, line in raw_df.iterrows():
    input_text = line["案件信息"].strip().replace("\n", "")

    input_ids, input_mask = convert_to_input(tokenizer, input_text)

    infer_queue.start_async(
        {
            "input_1": np.array(np.expand_dims(input_ids, axis=0)),
            "input_2": np.array(np.expand_dims(input_mask, axis=0)),
        }
    )

    batch_count += 1
    if batch_count % batch == 0:
        infer_queue.wait_all()
        for req in infer_queue:
            res.append(np.argmax(req.get_output_tensor().data, axis=-1).tolist()[0])

if len(infer_queue) > 0:
    infer_queue.wait_all()
    for req in infer_queue:
        res.append(np.argmax(req.get_output_tensor().data, axis=-1).tolist()[0])

ort_time = time.time() - start_time
print(f"openvino: {len(raw_df)} samples in {ort_time:.2f} s ")
