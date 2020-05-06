import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate*(kernel_size-1))/2),
                    dilation=dilation_rate
                )
            )
            self.gate_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=int((dilation_rate*(kernel_size-1))/2),
                    dilation=dilation_rate
                )
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


class Wave_Classifier(nn.Module):

    def __init__(self, in_channels=7, kernel_size=3):
        super().__init__()
        self.wave_block1 = Wave_Block(in_channels, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(16, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.fc = nn.Linear(128, 11)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)
        
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x
    
    
class WaveRNN_Classifier(nn.Module):

    def __init__(self, in_channels=7, kernel_size=3):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, dilation=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.wave_block1 = Wave_Block(64, 32, 12, kernel_size)
        self.bn1         = nn.BatchNorm1d(32)
        self.wave_block2 = Wave_Block(32, 64, 8, kernel_size)
        self.bn2         = nn.BatchNorm1d(64)
        self.wave_block3 = Wave_Block(64, 128, 4, kernel_size)
        self.bn3         = nn.BatchNorm1d(128)
        self.wave_block4 = Wave_Block(128, 192, 1, kernel_size)
        self.bn4         = nn.BatchNorm1d(192)
        self.RNN         = nn.GRU(input_size=192, hidden_size=192, num_layers=1, dropout=.1, bidirectional=True)
        self.bn5         = nn.BatchNorm1d(192*2)
        self.fc = nn.Linear(192*2, 11)

    def forward(self, x):
        # [batch, sequence, feature] => [batch, feature, sequence]
        x = x.permute(0, 2, 1)
        
        x = self.cbr(x)
        x = self.wave_block1(x)
        x = self.bn1(x)
        x = self.wave_block2(x)
        x = self.bn2(x)
        x = self.wave_block3(x)
        x = self.bn3(x)
        x = self.wave_block4(x)
        x = self.bn4(x)
        
        # [batch, feature, sequence] => [sequence, batch, feature]
        x = x.permute(2, 0, 1)
        
        # [sequence, batch, feature] => [sequence, batch, 2*feature]
        x, _ = self.RNN(x)
        
        # [sequence, batch, 2*feature] => [batch, 2*feature, sequence]
        x = x.permute(1, 2, 0)
        x = self.bn5(x)
        
        # [batch, 2*feature, sequence] => [batch, sequence, 2*feature]
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x
    
    
class WaveTRSFM_Classifier(nn.Module):

    def __init__(self, in_channels=7, kernel_size=3):
        super().__init__()
        
        self.wave_block1 = Wave_Block(in_channels, 16, 12, kernel_size)
        self.bn1         = nn.BatchNorm1d(16)
        self.wave_block2 = Wave_Block(16, 32, 8, kernel_size)
        self.bn2         = nn.BatchNorm1d(32)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.bn3         = nn.BatchNorm1d(64)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.bn4         = nn.BatchNorm1d(128)
        self.TRSFM       = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=6)
        self.fc = nn.Linear(128, 11)

    def forward(self, x):
        # [batch, sequence, feature] => [batch, feature, sequence]
        x = x.permute(0, 2, 1)
        
        x = self.wave_block1(x)
        x = self.bn1(x)
        x = self.wave_block2(x)
        x = self.bn2(x)
        x = self.wave_block3(x)
        x = self.bn3(x)
        x = self.wave_block4(x)
        x = self.bn4(x)
        
        # [batch, feature, sequence] => [sequence, batch, feature]
        x = x.permute(2, 0, 1)
        x = self.TRSFM(x)
        
        # [sequence, batch, feature] => [batch, sequence, feature]
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        
        return x