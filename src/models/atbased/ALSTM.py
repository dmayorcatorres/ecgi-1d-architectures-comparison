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

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attn_fc = nn.Linear(hidden_dim, hidden_dim)
        self.attn_vector = nn.Parameter(torch.randn(hidden_dim))
    
    def forward(self, lstm_output):
        energy = torch.tanh(self.attn_fc(lstm_output))  
        attn_scores = torch.matmul(energy, self.attn_vector)  
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights

class ALSTM(nn.Module):
    def __init__(self, time_steps: int, input_channels: int, output_channels: int,
                 stage1_out: int = 128, stage2_out: int = 128, stage3_out: int = 128, lstm_hidden: int = 16):
        super(ALSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, stage1_out, kernel_size=8, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(stage1_out)
        self.se1 = SqueezeExciteBlock(stage1_out)
        
        self.conv2 = nn.Conv1d(stage1_out, stage2_out, kernel_size=5, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(stage2_out)
        self.se2 = SqueezeExciteBlock(stage2_out)
        
        self.conv3 = nn.Conv1d(stage2_out, stage3_out, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm1d(stage3_out)
        
        self.lstm = nn.LSTM(input_channels, lstm_hidden, batch_first=True)
        self.attention = AttentionLayer(lstm_hidden)
        self.dropout = nn.Dropout(0.8)
        
        self.fc = nn.Conv1d(stage3_out + lstm_hidden, output_channels, kernel_size=1, bias=True)
        
    def forward(self, x):
        cnn = F.relu(self.bn1(self.conv1(x)))
        cnn = self.se1(cnn)
        
        cnn = F.relu(self.bn2(self.conv2(cnn)))
        cnn = self.se2(cnn)
        
        cnn = F.relu(self.bn3(self.conv3(cnn)))
        
        lstm_in = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.dropout(lstm_out)
        
        context, attn_weights = self.attention(lstm_out)
        context_expanded = context.unsqueeze(-1).expand(-1, -1, x.size(2))
        
        combined = torch.cat([cnn, context_expanded], dim=1)
        out = self.fc(combined)
        return out
