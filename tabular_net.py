import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularNet(nn.Module):
    """
    MLP tabular más profundo, sin pesos preentrenados.
    """
    def __init__(self, input_dim, hidden_dims=[128, 128, 64], output_dim=64, p_dropout=0.3):
        super(TabularNet, self).__init__()

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))
            prev_dim = hdim

        # Última capa de proyección
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)