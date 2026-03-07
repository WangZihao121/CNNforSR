import torch
from torch import nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        # 第一层：Patch extraction (9x9, 64 filters)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=0)
        # 第二层：Non-linear mapping (1x1, 32 filters)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        # 第三层：Reconstruction (5x5, 1 filter)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
