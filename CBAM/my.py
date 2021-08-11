import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = x

        out = self.ca(out) * out
        out = self.sa(out) * out

        return out


if __name__ == '__main__':
    pass

    a = torch.randint(1, 10, [5, 32, 224, 224], dtype=torch.float32)
    # 随机产生1-10之间的整数，其shape为(3,3)

    # print(a)

    b = ChannelAttention(in_planes=32).forward(a)

    c = SpatialAttention().forward(a)

    # print(a[0,0,0,0])

    # print(b.shape)
    # print(b[0,0,0,0])

    d=b*a
    # print(d[0,0,0,0])
    # print(d.shape)
    #
    # print(c[0,0,0,0])
    # print(c.shape)

    e=c*a
    # print(e[0,0,0,0])
    # print(e.shape)

    f=d+e
    # print(f.shape)
    # print(f[0,0,0,0])
    print(f)


# torch.Size([5, 32, 1, 1])
# torch.Size([5, 1, 224, 224])





