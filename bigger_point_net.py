import torch
import torch.nn as nn
import torch.nn.functional as F

class BiggerPointNet(nn.Module):
    """
    PointNet algo más profundo, con más capas y dropout.
    (Ojo: no incluye el T-Net, puedes agregarlo si quieres)
    """
    def __init__(self, output_dim=128, p_dropout=0.3):
        super(BiggerPointNet, self).__init__()

        self.mlp1 = nn.Linear(3, 32)
        self.bn1   = nn.BatchNorm1d(32)
        self.mlp2 = nn.Linear(32, 64)
        self.bn2   = nn.BatchNorm1d(64)
        self.mlp3 = nn.Linear(64, 128)
        self.bn3   = nn.BatchNorm1d(128)
        self.mlp4 = nn.Linear(128, 256)
        self.bn4   = nn.BatchNorm1d(256)
        self.mlp5 = nn.Linear(256, 512)
        self.bn5   = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(p_dropout)

        self.fc_final = nn.Linear(512, output_dim)

    def forward(self, x):
        """
        x: (B, N, 3)
        """
        # La BN1d hay que aplicarla con shape (B*N, C).
        # Por tanto, reordenamos a 2D antes de BN y volvemos a 3D.

        B, N, _ = x.shape

        # mlp1
        x = self.mlp1(x)              # (B, N, 64)
        x = x.view(B*N, 32)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.view(B, N, 32)

        # mlp2
        x = self.mlp2(x)              # (B, N, 128)
        x = x.view(B*N, 64)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(B, N, 64)

        # mlp3
        x = self.mlp3(x)              # (B, N, 256)
        x = x.view(B*N, 128)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(B, N, 128)

        # mlp4
        x = self.mlp4(x)              # (B, N, 512)
        x = x.view(B*N, 256)
        x = self.bn4(x)
        x = F.relu(x)
        x = x.view(B, N, 256)

        # mlp5
        x = self.mlp5(x)              # (B, N, 1024)
        x = x.view(B*N, 512)
        x = self.bn5(x)
        x = F.relu(x)
        x = x.view(B, N, 512)

        # Con dropout en la penúltima capa
        x = self.dropout(x)

        # Global Max Pool
        x = x.max(dim=1)[0]           # (B, 1024)

        # fc_final
        x = self.fc_final(x)          # (B, output_dim)

        return x

def my_collate_fn(batch):
    """
    batch: lista de samples, donde cada sample es:
      (seq_imgs, seq_tabs, seq_lidar, seq_controls)  o  None
    """
    # 1) Descarta los None
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        # Si *todo* el batch fue None, devolvemos None
        return None

    # 2) Separamos los elementos
    batch_imgs_list   = []
    batch_tabs_list   = []
    batch_lidar_list  = []
    batch_ctrls_list  = []

    for (seq_imgs, seq_tabs, seq_lidar, seq_controls) in batch:
        batch_imgs_list.append(seq_imgs)
        batch_tabs_list.append(seq_tabs)
        batch_lidar_list.append(seq_lidar)
        batch_ctrls_list.append(seq_controls)

    # 3) Unificamos en tensores con dimensión (B, T, ...) para imágenes/tab/control.
    # Para LiDAR, mantenemos la estructura de listas, ya que cada (num_points,3) puede ser diferente.

    # a) Imágenes => dict { sensor: (B, T, C, H, W) }
    from collections import defaultdict
    final_imgs = defaultdict(list)
    # batch_imgs_list es una lista de dicts, uno por sample
    # each dict => sensor -> (T, C, H, W)
    for sample_dict in batch_imgs_list:
        for sensor, seq_tensor in sample_dict.items():
            final_imgs[sensor].append(seq_tensor)  # (T, C, H, W)
    for sensor, list_of_tensors in final_imgs.items():
        final_imgs[sensor] = torch.stack(list_of_tensors, dim=0)  # => (B, T, C, H, W)

    # b) Tabular => dict { sensor: (B, T, D) }
    final_tabs = defaultdict(list)
    for sample_dict in batch_tabs_list:
        for sensor, tab_tensor in sample_dict.items():
            final_tabs[sensor].append(tab_tensor)  # (T, D)
    for sensor, list_of_tensors in final_tabs.items():
        final_tabs[sensor] = torch.stack(list_of_tensors, dim=0)  # => (B, T, D)

    # c) LiDAR => list de longitud B, cada elemento es [ (num_points, 3) x T ]
    # Lo mantenemos igual, pues cada sample puede tener distinta cantidad de puntos
    final_lidar = batch_lidar_list  # => list of length B

    # d) Controls => (B, T, 3)
    final_ctrls = torch.stack(batch_ctrls_list, dim=0)  # => (B, T, 3)

    return final_imgs, final_tabs, final_lidar, final_ctrls