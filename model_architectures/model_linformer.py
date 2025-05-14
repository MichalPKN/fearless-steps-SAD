import torch
import torch.nn as nn
from model_architectures.linformer_implementation import Linformer

class SADModel(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, num_layers, seq_length):
        super(SADModel, self).__init__()
    
        self.embedding = nn.Linear(input_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embed_size))
        self.layer_norm = nn.LayerNorm(embed_size)
        self.linformer = Linformer(embed_size, seq_length, depth=num_layers, k=256, heads=num_heads, dim_head=None,
                                  one_kv_head=False, share_kv=False, 
                                  reversible=False, dropout=0.)
        self.fc_out = nn.Linear(embed_size, 1)
        
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.layer_norm(x)
        x = self.linformer(x)
        x = self.fc_out(x)
        return x