import torch
import torch.nn as nn

class SADModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(SADModel, self).__init__()
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[2], 1) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 
    
    def forward(self, x):
        #out, _ = self.lstm(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.sigmoid(self.fc4(x))
        return out