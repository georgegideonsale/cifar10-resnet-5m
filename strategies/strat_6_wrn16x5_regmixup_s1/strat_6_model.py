"""strat_1: pre-activation WRN-16-5. build_model(cfg) -> nn.Module (cfg = the `model:` config section)."""
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


class PreActBlock(nn.Module):
    """Stage-entry block (cin != cout or stride != 1): 1x1 shortcut on o = ReLU(BN1(x)).
    Identity block: y = x + main(x). No activation after the residual add."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = conv3x3(cout, cout, 1)
        self.shortcut = None
        if cin != cout or stride != 1:
            self.shortcut = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        y = self.conv2(F.relu(self.bn2(self.conv1(o))))
        return y + (self.shortcut(o) if self.shortcut is not None else x)


class PreActWRN(nn.Module):
    def __init__(self, in_channels, stem_channels, widths, blocks_per_stage, first_strides, num_classes):
        super().__init__()
        self.stem = conv3x3(in_channels, stem_channels, 1)  # no BN/ReLU after the stem
        blocks, cin = [], stem_channels
        for width, stride in zip(widths, first_strides):
            for b in range(blocks_per_stage):
                blocks.append(PreActBlock(cin, width, stride if b == 0 else 1))
                cin = width
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # nn.Linear keeps the PyTorch default init.

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def build_model(cfg):
    return PreActWRN(
        in_channels=cfg["in_channels"],
        stem_channels=cfg["stem_channels"],
        widths=list(cfg["widths"]),
        blocks_per_stage=cfg["blocks_per_stage"],
        first_strides=list(cfg["first_strides"]),
        num_classes=cfg["num_classes"],
    )
