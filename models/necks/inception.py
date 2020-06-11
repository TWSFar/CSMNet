import torch
import torch.nn as nn
import torch.nn.functional as F
from models.necks import SELayer


class _InceptionModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, BatchNorm):
        super(_InceptionModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Inception(nn.Module):
    def __init__(self, backbone, output_stride, inplanes=None, BatchNorm=None):
        super(Inception, self).__init__()
        self.ince1 = _InceptionModule(inplanes, 64, 1, padding=0, BatchNorm=BatchNorm)
        self.ince2 = _InceptionModule(inplanes, 64, 3, padding=1, BatchNorm=BatchNorm)
        self.ince3 = _InceptionModule(inplanes, 64, 5, padding=2, BatchNorm=BatchNorm)
        self.ince4 = _InceptionModule(inplanes, 64, 7, padding=3, BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 64, 1, stride=1, bias=False),
                                             BatchNorm(64),
                                             nn.ReLU())
        self.selayer = SELayer(5*64)
        self.conv1 = nn.Conv2d(5*64, 64, 1, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.ince1(x)
        x2 = self.ince2(x)
        x3 = self.ince3(x)
        x4 = self.ince4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.selayer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
