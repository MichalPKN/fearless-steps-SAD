import torch
import torch.nn as nn
import torchaudio.models as models

class SADModel(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, num_layers, seq_length, ffn_dim_mult=4, kernel_size=31):
        super(SADModel, self).__init__()
    
        self.embedding = nn.Linear(input_size, embed_size)
        self.conformer = models.Conformer(
            input_dim=embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=embed_size * ffn_dim_mult,
            depthwise_conv_kernel_size=kernel_size
        )
        self.fc_out = nn.Linear(embed_size, 1)
        self.layer_norm = nn.LayerNorm(embed_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.layer_norm(x)
        batch_size, seq_len, _ = x.shape
        lengths = torch.full((batch_size,), seq_len, dtype=torch.int64, device=x.device)
        x, _ = self.conformer(x, lengths)
        x = self.fc_out(x)
        return x