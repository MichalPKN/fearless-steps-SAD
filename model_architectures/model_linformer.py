import torch
import torch.nn as nn
from model_architectures.linformer_implementation import Linformer

# class LinformerForVAD(nn.Module):
#     def __init__(self, input_dim, dim, seq_len, depth, k=256, heads=8, dim_head=None, 
#                  one_kv_head=False, share_kv=False, reversible=False, dropout=0.):
#         super().__init__()
#         self.input_proj = nn.Linear(input_dim, dim)
#         self.pos_emb = nn.Embedding(seq_len, dim)
#         self.linformer = Linformer(dim, seq_len, depth, k=k, heads=heads, dim_head=dim_head,
#                                   one_kv_head=one_kv_head, share_kv=share_kv, 
#                                   reversible=reversible, dropout=dropout)
#         self.to_output = nn.Linear(dim, 1)

#     def forward(self, x):
#         # x shape: [batch_size, sequence_size, input_dim]
#         x = self.input_proj(x)
#         seq_length = x.size(1)
#         positions = torch.arange(seq_length, device=x.device)
#         x = x + self.pos_emb(positions)
#         x = self.linformer(x)
#         logits = self.to_output(x).squeeze(-1)  # [batch_size, sequence_size]
#         return logits

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