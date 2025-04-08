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
        se = F.adaptive_avg_pool1d(x, 1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se

class CNN_BiGRU(nn.Module):
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 stage1_out: int = 128, stage2_out: int = 128, stage3_out: int = 128, gru_hidden: int = 16):
        super(CNN_BiGRU, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=stage1_out, kernel_size=8, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(stage1_out)
        self.se1 = SqueezeExciteBlock(stage1_out, ratio=16)
        
        self.conv2 = nn.Conv1d(in_channels=stage1_out, out_channels=stage2_out, kernel_size=5, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(stage2_out)
        self.se2 = SqueezeExciteBlock(stage2_out, ratio=16)
        
        self.conv3 = nn.Conv1d(in_channels=stage2_out, out_channels=stage3_out, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm1d(stage3_out)

        self.bigru = nn.GRU(input_size=input_channels, hidden_size=gru_hidden,
                            num_layers=2, dropout=0.2, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.8)

        self.fc = nn.Conv1d(in_channels=stage3_out + 2 * gru_hidden, out_channels=output_channels, kernel_size=1, bias=True)
        
    def forward(self, x):
        cnn = self.conv1(x)
        cnn = self.bn1(cnn)
        cnn = F.relu(cnn)
        cnn = self.se1(cnn)
        
        cnn = self.conv2(cnn)
        cnn = self.bn2(cnn)
        cnn = F.relu(cnn)
        cnn = self.se2(cnn)
        
        cnn = self.conv3(cnn)
        cnn = self.bn3(cnn)
        cnn = F.relu(cnn)

        gru_in = x.permute(0, 2, 1)
        gru_out, _ = self.bigru(gru_in)
        gru_out = self.dropout(gru_out)
        gru_out = gru_out.permute(0, 2, 1)

        combined = torch.cat([cnn, gru_out], dim=1)
        
        out = self.fc(combined)
        return out
