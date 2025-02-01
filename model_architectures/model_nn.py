import torch
import torch.nn as nn

class SADModel(nn.Module):
    def __init__(self, input_size=30, hidden_size=[512, 1024], seq_lenght=500, num_layers=3):
        super(SADModel, self).__init__()
        self.fc1 = nn.Linear(input_size * seq_lenght, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], seq_lenght)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x.unsqueeze(-1)
        return x