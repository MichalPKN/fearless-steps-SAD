import torch
import torch.nn as nn

class SADModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(SADModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 1 output for binary classification (voice or no voice)
        self.sigmoid = nn.Sigmoid()  # For binary output
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Apply the fully connected layer to each time step
        out = self.fc(lstm_out)  # Shape: (batch_size, seq_len, 1)
        return self.sigmoid(out)