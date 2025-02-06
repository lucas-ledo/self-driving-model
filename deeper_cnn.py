import torch
import torch.nn as nn
import torch.nn.functional as F

class DeeperCNN(nn.Module):
    """
    CNN más profunda desde cero (sin pre-entrenar).
    Bloques de conv, con mayor número de canales y dropout.
    """
    def __init__(self, in_channels, out_features, p_dropout=0.3):
        super(DeeperCNN, self).__init__()

        # Bloque 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        # Bloque 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)


        # Pooling adaptativo para reducir a un tamaño fijo
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        # => si la salida de conv4 es [B, 512, H', W'],
        #    tras pool => [B, 512, 2, 4] => 512*2*4=4096

        self.dropout = nn.Dropout(p_dropout)  # Dropout para evitar overfitting

        self.fc = nn.Linear(128 * 2 * 4, out_features)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # [B, 64, H, W]
        x = F.relu(self.bn2(self.conv2(x)))   # [B, 128, H, W]

        x = self.pool(x)                      # [B, 128, 2, 4]
        x = x.view(x.size(0), -1)             # [B, 4096]

        x = self.dropout(x)
        x = self.fc(x)                        # [B, out_features]
        return x