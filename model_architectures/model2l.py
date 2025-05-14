import torch
import torch.nn as nn

class SADModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(SADModel, self).__init__()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out