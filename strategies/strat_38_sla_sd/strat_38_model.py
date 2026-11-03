"""strat_38: build_model(cfg) -> nn.Module (cfg = the `model:` section). arch: resnet (post-activation) or wrn (pre-activation).
strat_38 adds an optional class x rotation head to the resnet (model.sla_rotations), for SLA+SD (Lee et al. 2020).

The resnet comes from the strat_10 port of the teammate notebook. The wrn follows Zagoruyko & Komodakis 2016, as in that notebook. Its module names equal strat_6_model.py, so our WRN state_dicts load.
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
    """fc: 10-way single head. fc_joint (sla_rotations > 0): class x rotation head, index = class * R + rotation."""

    def __init__(self, widths, blocks, num_classes=10, sla_rotations=0):
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
        self.fc_joint = nn.Linear(c_in, num_classes * sla_rotations) if sla_rotations > 0 else None
        init_weights(self)

    def features(self, x):
        return torch.flatten(self.pool(self.layers(self.stem(x))), 1)

    def heads(self, x):
        """(single logits, joint logits) from one pass."""
        f = self.features(x)
        return self.fc(f), self.fc_joint(f)

    def forward(self, x):
        return self.fc(self.features(x))


class WideBlock(nn.Module):
    """Pre-activation block: BN-ReLU-conv-BN-ReLU-conv, plus the skip (1x1 conv when the shape changes)."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.shortcut = None if (stride == 1 and c_in == c_out) else nn.Conv2d(c_in, c_out, 1, stride, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        y = self.conv2(F.relu(self.bn2(self.conv1(o))))
        return y + (x if self.shortcut is None else self.shortcut(o))


class WideResNet(nn.Module):
    def __init__(self, widths, blocks, stem=16, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, stem, 3, 1, 1, bias=False)
        layers, c_in = [], stem
        for i, (w, b) in enumerate(zip(widths, blocks)):
            for j in range(b):
                layers.append(WideBlock(c_in, w, 2 if (i > 0 and j == 0) else 1))
                c_in = w
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(c_in)
        self.fc = nn.Linear(c_in, num_classes)
        init_weights(self)

    def forward(self, x):
        out = F.relu(self.bn(self.blocks(self.stem(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(out, 1), 1))


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def build_model(cfg):
    if cfg.get("arch", "resnet") == "wrn":
        return WideResNet(cfg["widths"], cfg["blocks"], cfg.get("stem", 16), cfg["num_classes"])
    return ResNet(cfg["widths"], cfg["blocks"], cfg["num_classes"], cfg.get("sla_rotations", 0))
