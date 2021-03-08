import torch.nn as nn
import torch


class Conv2d_Bn_Leaky(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ):
        super(Conv2d_Bn_Leaky, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.leaky_relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        inner_channels = channels // 2

        self.conv1 = Conv2d_Bn_Leaky(
            in_channels=channels, out_channels=inner_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = Conv2d_Bn_Leaky(
            in_channels=inner_channels, out_channels=channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        return out


class DarkNet(nn.Module):
    def __init__(self, layer_list: list):
        super(DarkNet, self).__init__()

        filter_list = [64, 128, 256, 512, 1024]

        self.conv1 = Conv2d_Bn_Leaky(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.stage1 = self._make_layers(filter_list[0], layer_list[0])
        self.stage2 = self._make_layers(filter_list[1], layer_list[1])
        self.stage3 = self._make_layers(filter_list[2], layer_list[2])
        self.stage4 = self._make_layers(filter_list[3], layer_list[3])
        self.stage5 = self._make_layers(filter_list[4], layer_list[4])

    def _make_layers(self, channels, num_blocks):
        layers = []
        layers.append(
            Conv2d_Bn_Leaky(
                in_channels=channels // 2, out_channels=channels, kernel_size=3, stride=2, padding=1
            )
        )
        for _ in range(num_blocks):
            layers.append(ResBlock(channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out1 = self.stage3(out)
        out2 = self.stage4(out1)
        out3 = self.stage5(out2)

        return (out3, out2, out1)


# if __name__ == "__main__":
#     net = DarkNet([1, 2, 8, 8, 4])
#     net.eval()
#     inputs = torch.rand(1, 3, 416, 416)
#     level_outputs = net(inputs)
#     for level_out in level_outputs:
#         print(tuple(level_out.shape))
