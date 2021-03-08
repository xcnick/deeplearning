import torch.nn as nn
from .darknet import DarkNet
from .neck import YoloV3Neck


class YoloV3(nn.Module):
    def __init__(self):
        super(YoloV3, self).__init__()
        self.backbone = DarkNet([1, 2, 8, 8, 4])
        self.neck = YoloV3Neck()

    def forward(self, x):
        out = self.backbone(x)
        out = self.neck(out)
        return out
