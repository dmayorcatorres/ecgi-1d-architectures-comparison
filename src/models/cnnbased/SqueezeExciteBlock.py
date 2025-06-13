import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels // ratio, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        se = F.adaptive_avg_pool1d(x, 1)  # (B, C, 1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se  # Broadcast a lo largo de T