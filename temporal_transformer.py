import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedPositionalEncoding(nn.Module):
    """
    Sencillo embedding aprendido de posiciones [0..max_len].
    """
    def __init__(self, d_model, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_embedding(positions)  # => (1, T, d_model)
        pos_emb = pos_emb.expand(B, T, D)        # => (B, T, d_model)
        return x + pos_emb


class TemporalTransformer(nn.Module):
    """
    Un stack de TransformerEncoderLayers para procesar la secuencia en el tiempo,
    usando batch_first=True para optimizar el rendimiento.
    """
    def __init__(self, d_model=256, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(TemporalTransformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = LearnedPositionalEncoding(d_model)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        x = self.pos_encoder(x)    # => (B, T, d_model)
        out = self.encoder(x)      # => (B, T, d_model) con batch_first=True
        return out