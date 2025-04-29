import torch
import torch.nn as nn

class SADModel(nn.Module):
    def __init__(self, input_size=30, hidden_size=128, num_layers=3, dropout=0.5, filter_num=64):
        super(SADModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, filter_num//2, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(filter_num//2, filter_num//1, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(filter_num//1, filter_num, kernel_size=(3, 3), padding=(1, 1))
        
        self.bn1 = nn.BatchNorm2d(filter_num//2)
        self.bn2 = nn.BatchNorm2d(filter_num//1)
        self.bn3 = nn.BatchNorm2d(filter_num)
        
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(
            input_size=(input_size // 8) * filter_num,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            #dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        #flatten
        batch_size, channels, seq_len, pooled_mfcc = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, channels * pooled_mfcc)
        
        x, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size]
        x = self.fc(x)  # fc_out: [batch_size, seq_len, 1]
        return x