import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossSensorAttention(nn.Module):
    """
    Fusión de [img, tab, lidar] en cada timestep mediante self-attention.
    """
    def __init__(self, input_dim, d_model=256, nhead=4, p_dropout=0.3):
        super(CrossSensorAttention, self).__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=p_dropout,
            batch_first=False
        )
        self.dropout = nn.Dropout(p_dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        => Proyectamos a (B, T, d_model)
        => self-attn en la dimensión "sensor"
        """
        B, T, _ = x.shape
        x = self.fc_in(x)              # => (B, T, d_model)
        x = x.permute(1, 0, 2)        # => (T, B, d_model)
        attn_out, _ = self.mha(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        x = x.permute(1, 0, 2)        # => (B, T, d_model)
        return x