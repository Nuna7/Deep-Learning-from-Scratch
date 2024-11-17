import torch
from torch import nn

class DepthwiseSeparableLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(DepthwiseSeparableLayer, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Define example depthwise and pointwise kernels for illustration
        self.depthwise_kernel = torch.randn(in_channels, 1, 3, 3)  # For illustration only
        self.pointwise_kernel = torch.randn(out_channels, in_channels, 1, 1)  # For illustration only


    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return self.relu(x)

    # Illustration of the idea
    def depthwise_convolution(self, x):
        n, c, h, w = x.size()
        new_h = (h + 2 * self.padding - 3) // self.stride + 1  # kernel_size = 3
        new_w = (w + 2 * self.padding - 3) // self.stride + 1
        output = torch.zeros((n, c, new_h, new_w))

        for instance in range(n):
            for channel in range(c):
                for i in range(new_h):
                    for j in range(new_w):
                        top = i * self.stride
                        left = j * self.stride
                        bottom = top + 3
                        right = left + 3
                        region = x[instance, channel, top:bottom, left:right]
                        # Using the example depthwise kernel for this channel
                        output[instance, channel, i, j] = (region * self.depthwise_kernel[channel]).sum()

        return output

    # Illustration of the idea
    def pointwise_convolution(self, x):
        n, c, h, w = x.size()
        output = torch.zeros((n, self.out_channels, h, w))

        for instance in range(n):
            for i in range(h):
                for j in range(w):
                    for k in range(self.out_channels):
                        region = x[instance, :, i, j]
                        # Using the example pointwise kernel for this output channel
                        output[instance, k, i, j] = (region * self.pointwise_kernel[k].squeeze()).sum()

        return output

class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()

        def conv_dw(in_channels, out_channels, stride):
            return nn.Sequential(
                DepthwiseSeparableLayer(in_channels, out_channels, stride),
            )

        self.model = nn.Sequential(
            ConvolutionLayer(3, 32, kernel_size=3, stride=2, padding=1),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return F.softmax(x, dim=1)
