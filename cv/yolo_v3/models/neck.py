import torch.nn as nn
import torch.nn.functional as F
import torch

from .darknet import Conv2d_Bn_Leaky#, DarkNet


class YoloBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(YoloBlock, self).__init__()
        self.conv1 = Conv2d_Bn_Leaky(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = Conv2d_Bn_Leaky(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = Conv2d_Bn_Leaky(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv4 = Conv2d_Bn_Leaky(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv5 = Conv2d_Bn_Leaky(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out


class YoloV3Neck(nn.Module):
    def __init__(self, num_anchors=3, num_classes=80):
        super(YoloV3Neck, self).__init__()

        out_filters = num_anchors * (num_classes + 5)
        in_channels = [1024, 512, 256]
        out_channels = [512, 256, 128]

        self.conv79 = YoloBlock(in_channels=in_channels[0], out_channels=out_channels[0])
        self.conv80 = Conv2d_Bn_Leaky(
            in_channels=out_channels[0],
            out_channels=in_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.y1 = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_filters, kernel_size=1, stride=1, padding=0
        )

        self.conv82 = Conv2d_Bn_Leaky(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv91 = YoloBlock(
            in_channels=in_channels[1] + out_channels[1], out_channels=out_channels[1]
        )
        self.conv92 = Conv2d_Bn_Leaky(
            in_channels=out_channels[1],
            out_channels=in_channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.y2 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_filters, kernel_size=1, stride=1, padding=0
        )

        self.conv94 = Conv2d_Bn_Leaky(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv103 = YoloBlock(
            in_channels=in_channels[2] + out_channels[2], out_channels=out_channels[2]
        )
        self.conv104 = Conv2d_Bn_Leaky(
            in_channels=out_channels[2],
            out_channels=in_channels[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.y3 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_filters, kernel_size=1, stride=1, padding=0
        )

    def forward(self, feats):
        feat1, feat2, feat3 = feats

        conv79 = self.conv79(feat1)
        out = self.conv80(conv79)
        y1 = self.y1(out)

        out = self.conv82(conv79)
        out = torch.cat((feat2, F.interpolate(out, scale_factor=2)), dim=1)
        conv91 = self.conv91(out)
        out = self.conv92(conv91)
        y2 = self.y2(out)

        out = self.conv94(conv91)
        out = torch.cat((feat3, F.interpolate(out, scale_factor=2)), dim=1)
        out = self.conv103(out)
        out = self.conv104(out)
        y3 = self.y3(out)

        return (y1, y2, y3)


# if __name__ == "__main__":
#     net = DarkNet([1, 2, 8, 8, 4])
#     net.eval()
#     inputs = torch.rand(1, 3, 416, 416)
#     level_outputs = net(inputs)
#     for level_out in level_outputs:
#         print(tuple(level_out.shape))

#     neck = YoloV3Neck()
#     neck.eval()
#     neck_outputs = neck(level_outputs)
#     for neck_out in neck_outputs:
#         print(tuple(neck_out.shape))
