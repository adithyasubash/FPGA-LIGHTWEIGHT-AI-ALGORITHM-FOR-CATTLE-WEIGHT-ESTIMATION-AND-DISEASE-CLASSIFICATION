"""
cnn_model.py -- GreenCNN, a small CNN designed so it can be turned into HLS later.

Only HLS-friendly layers:  Conv3x3 (+BatchNorm, folded away at export) -> ReLU -> MaxPool2
  input 3x64x64
  block1  3 -> 16   64x64 -> 32x32
  block2 16 -> 32   32x32 -> 16x16
  block3 32 -> 48   16x16 ->  8x8
  block4 48 -> 64    8x8  ->  4x4
  global average pool -> 64 -> Linear -> 3 classes
About 50k parameters -> fits easily in the PYNQ-Z2 BRAM as int8.
"""
import torch
import torch.nn as nn

import config as C


class GreenCNN(nn.Module):
    def __init__(self, widths=C.CNN_WIDTHS, num_classes=C.NUM_CLASSES, dropout=0.3):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers += [nn.Conv2d(cin, w, 3, padding=1, bias=False),
                       nn.BatchNorm2d(w),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):          # x: float in [0,1], shape N x 3 x 64 x 64
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = GreenCNN()
    print(m)
    print("parameters:", count_params(m))
    print("output:", m(torch.zeros(2, 3, C.IMG_SIZE, C.IMG_SIZE)).shape)
