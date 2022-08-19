import torch
import transformers

model = transformers.BertModel.from_pretrained(
    "./models/chinese_roberta_wwm_ext_pytorch"
)

dummy_input = torch.randint(
    low=1,
    high=model.config.vocab_size - 1,
    size=(1, 128),
    dtype=torch.long,
    device="cpu",
)

torch.onnx.export(
    model=model,
    args=(dummy_input,),
    verbose=True,
    f="onnx/bert_model.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output_1", "output_2"],
    dynamic_axes={"input": [0, 1], "output_1": [0, 1], "output_2": [0],},
)
