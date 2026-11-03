"""strat_11: post-activation ResNet of Tarun's S3 recipe. build_model(cfg) -> nn.Module (cfg = the `model:` section).

Ported from the teammate notebook docs/10-31-2026 - Tarun/kaggle_train_tarun.ipynb (src/models.py, version 25fd8d1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:  # projection skip: 1x1 conv + BN to match the shape
            self.skip = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, 0, bias=False), nn.BatchNorm2d(c_out))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(self.skip(x) + out)


class ResNet(nn.Module):
    def __init__(self, widths, blocks, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        layers, c_in = [], widths[0]
        for i, (w, b) in enumerate(zip(widths, blocks)):
            for j in range(b):
                layers.append(BasicBlock(c_in, w, 2 if (i > 0 and j == 0) else 1))
                c_in = w
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.layers(self.stem(x))), 1))


def build_model(cfg):
    return ResNet(cfg["widths"], cfg["blocks"], cfg["num_classes"])
