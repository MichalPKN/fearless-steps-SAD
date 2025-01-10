import torch
import torch.nn as nn

class SADModel(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, num_layers, seq_length):
        super(SADModel, self).__init__()
    
        self.embedding = nn.Linear(input_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embed_size))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_size, 1)
        
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x