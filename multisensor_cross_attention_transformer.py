import torch
import torch.nn as nn
import torch.nn.functional as F

# Importa tus subredes
from deeper_cnn import DeeperCNN
from tabular_net import TabularNet
from bigger_point_net import BiggerPointNet
from cross_attention import CrossSensorAttention
from temporal_transformer import TemporalTransformer


class MultiSensorCrossAttentionTransformer(nn.Module):
    def __init__(self,
                 sensor_list=('depth_cam', 'rgb_front'),
                 tabular_sensors=('gnss_sensor',),
                 tabular_input_dims={'gnss_sensor': 3},
                 image_feature_dim=128,
                 tabular_feature_dim=64,
                 lidar_feature_dim=128,
                 cross_d_model=256,
                 cross_nhead=4,
                 cross_dropout=0.3,
                 temporal_d_model=256,
                 temporal_nhead=4,
                 temporal_num_layers=2,
                 temporal_dim_feedforward=512,
                 temporal_dropout=0.1,
                 out_dim=3):
        super().__init__()

        # A) Subredes sensor
        self.image_nets = nn.ModuleDict()
        for sensor in sensor_list:
            in_ch = 1 if 'depth' in sensor else 3
            self.image_nets[sensor] = DeeperCNN(in_ch, image_feature_dim, p_dropout=0.3)

        self.tab_nets = nn.ModuleDict()
        for sensor in tabular_sensors:
            inp_dim = tabular_input_dims[sensor]
            self.tab_nets[sensor] = TabularNet(inp_dim, [128, 128, 64],
                                               tabular_feature_dim, p_dropout=0.3)

        self.lidar_net = BiggerPointNet(output_dim=lidar_feature_dim, p_dropout=0.3)

        # B) Cross-attention (fusión en cada timestep)
        self.fused_dim = (image_feature_dim * len(sensor_list)
                          + tabular_feature_dim * len(tabular_sensors)
                          + lidar_feature_dim)

        self.cross_attn = CrossSensorAttention(
            input_dim=self.fused_dim,
            d_model=cross_d_model,
            nhead=cross_nhead,
            p_dropout=cross_dropout
        )

        # C) Ajustar cross_d_model -> temporal_d_model si difieren
        self.same_dim = (cross_d_model == temporal_d_model)
        if not self.same_dim:
            self.proj = nn.Linear(cross_d_model, temporal_d_model)
        else:
            self.proj = None

        # D) Transformer en el tiempo
        self.temporal_transformer = TemporalTransformer(
            d_model=temporal_d_model,
            nhead=temporal_nhead,
            num_layers=temporal_num_layers,
            dim_feedforward=temporal_dim_feedforward,
            dropout=temporal_dropout
        )

        # E) Capa final
        self.fc_out = nn.Sequential(
            nn.Linear(temporal_d_model, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def _precompute_sensor_features(self, batch_imgs, batch_tabs, batch_lidar):
        """
        Calcula las features de cada sensor (CNN, TabNet, LidarNet) en TODO el
        rango temporal (T) de una sola pasada, para no repetir cálculo.

        Retorna un tensor (B, T, fused_dim).
        """
        # 1) Imagen
        feats_img_list = []
        B, T, _, _, _ = next(iter(batch_imgs.values())).shape

        for sensor, net in self.image_nets.items():
            x = batch_imgs[sensor]  # (B, T, C, H, W)
            B_, T_, C, H, W = x.shape
            # Por si 'depth_cam' trae 3 channels
            if 'depth' in sensor and C != 1:
                x = x[:, :, 0:1, :, :]
            x = x.reshape(B_ * T_, C, H, W)
            x = net(x)  # -> (B_*T_, image_feature_dim)
            x = x.view(B_, T_, -1)
            feats_img_list.append(x)

        feats_img = torch.cat(feats_img_list, dim=2) if feats_img_list else None

        # 2) Tabular
        feats_tab_list = []
        for sensor, net in self.tab_nets.items():
            x = batch_tabs[sensor]  # (B, T, D)
            B_, T_, D_ = x.shape
            x = x.view(B_ * T_, D_)
            x = net(x)  # -> (B_*T_, tabular_feature_dim)
            x = x.view(B_, T_, -1)
            feats_tab_list.append(x)

        feats_tab = torch.cat(feats_tab_list, dim=2) if feats_tab_list else None

        # 3) LIDAR
        B_, T_, N, _ = batch_lidar.shape
        x_lidar = batch_lidar.view(B_ * T_, N, 3)
        x_lidar = self.lidar_net(x_lidar)  # -> (B_*T_, lidar_feature_dim)
        x_lidar = x_lidar.view(B_, T_, -1)

        # 4) Concat final
        cat_list = []
        if feats_img is not None:
            cat_list.append(feats_img)
        if feats_tab is not None:
            cat_list.append(feats_tab)
        cat_list.append(x_lidar)
        x_all = torch.cat(cat_list, dim=2)  # (B, T, fused_dim)

        return x_all

    def _forward_transformer_in_chunks(self, x_all, chunk_size=None):
        """
        Aplica Cross-Attention + TemporalTransformer en modo 'chunked' si
        chunk_size no es None.
        """
        B, T, _ = x_all.shape
        intermediate_features = []

        if chunk_size is None:
            # Procesar toda la secuencia de una vez (sin partir)
            x_all = self.cross_attn(x_all)  # (B, T, cross_d_model)
            if self.proj is not None:
                x_all = self.proj(x_all)     # (B, T, temporal_d_model)
            temporal_out = self.temporal_transformer(x_all)  # (B, T, temporal_d_model)
            out = self.fc_out(temporal_out)                  # (B, T, out_dim)
            intermediate_features = temporal_out  # Guardar características intermedias
            return out, intermediate_features
        else:
            # Procesar por chunks en el tiempo
            chunk_outputs = []
            for start_t in range(0, T, chunk_size):
                end_t = min(start_t + chunk_size, T)
                x_chunk = x_all[:, start_t:end_t, :]  # (B, chunk_size, fused_dim)

                # Cross-attn
                x_chunk = self.cross_attn(x_chunk)    # (B, chunk, cross_d_model)
                if self.proj is not None:
                    x_chunk = self.proj(x_chunk)      # (B, chunk, temporal_d_model)

                temporal_out_chunk = self.temporal_transformer(x_chunk)  # (B, chunk, temporal_d_model)
                out_chunk = self.fc_out(temporal_out_chunk)              # (B, chunk, out_dim)
                chunk_outputs.append(out_chunk)
                intermediate_features.append(temporal_out_chunk)  # Guardar características intermedias

            # Concatenar la salida a lo largo de la dimensión temporal
            final_out = torch.cat(chunk_outputs, dim=1)  # (B, T, out_dim)
            intermediate_features = torch.cat(intermediate_features, dim=1)  # (B, T, temporal_d_model)
            return final_out, intermediate_features

    def forward(self, batch_imgs, batch_tabs, batch_lidar, chunk_size=None, return_features=False):
        """
        - Primero precomputamos features de cada sensor para toda la secuencia.
        - Luego, si chunk_size está definido, aplicamos cross-attn+transformer
        en trozos para ahorrar memoria. Sino, procesamos todo de una vez.

        Args:
            batch_imgs (dict): Diccionario de imágenes.
            batch_tabs (dict): Diccionario de datos tabulares.
            batch_lidar (torch.Tensor): Tensor de datos LiDAR.
            chunk_size (int, optional): Tamaño de los chunks temporales.
            return_features (bool, optional): Si es True, retorna características antes de `fc_out`.

        Returns:
            torch.Tensor: Salida del modelo (acciones o características).
            torch.Tensor (opcional): Características intermedias si `return_features=True`.
        """
        x_all = self._precompute_sensor_features(batch_imgs, batch_tabs, batch_lidar)
        features = self._forward_transformer_in_chunks(x_all, chunk_size=chunk_size)
        
        if return_features:
            return features  # (intermediate_features)
        else:
            return features[0]  # Solo las acciones