import torch
import torch.nn as nn

class SADModel(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, dropout=0.5, bidirectional=True):
        super(SADModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            #dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size]
        x = self.fc(x)  # fc_out: [batch_size, seq_len, 1]
        return x