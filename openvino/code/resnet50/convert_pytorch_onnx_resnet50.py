import torch
import torchvision

input_names = ["input"]
output_names = ["output"]


dummy_input = torch.randn((1, 3, 224, 224), device="cpu")
model = torchvision.models.resnet50(pretrained=False)
model.load_state_dict(torch.load("resnet50-0676ba61.pth"))

torch.onnx.export(
    model,
    dummy_input,
    "onnx/resnet50-nchw.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
