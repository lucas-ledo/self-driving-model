import torch
import torch.nn as nn
import torch.nn.functional as F

# Importa tus subredes
from deeper_cnn import DeeperCNN
from tabular_net import TabularNet
from bigger_point_net import BiggerPointNet
from cross_attention import CrossSensorAttention
from temporal_transformer import TemporalTransformer

from multisensor_cross_attention_transformer import MultiSensorCrossAttentionTransformer 

class MultiSensorCrossAttentionTransformerRL(MultiSensorCrossAttentionTransformer):
    """
    Extiende la arquitectura base para generar:
      - Politica (Actor): media, std de las acciones.
      - Crítico (Critic): valor del estado.
    Se parte de la backbone preentrenada y se modifica la
    última parte de la red.
    """
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
                 # El out_dim original (3) se reemplazará en RL por actor/critic heads.
                 action_dim=3  # e.g. steer, throttle, brake
                ):
        super().__init__(
            sensor_list=sensor_list,
            tabular_sensors=tabular_sensors,
            tabular_input_dims=tabular_input_dims,
            image_feature_dim=image_feature_dim,
            tabular_feature_dim=tabular_feature_dim,
            lidar_feature_dim=lidar_feature_dim,
            cross_d_model=cross_d_model,
            cross_nhead=cross_nhead,
            cross_dropout=cross_dropout,
            temporal_d_model=temporal_d_model,
            temporal_nhead=temporal_nhead,
            temporal_num_layers=temporal_num_layers,
            temporal_dim_feedforward=temporal_dim_feedforward,
            temporal_dropout=temporal_dropout,
            out_dim=action_dim  # se usará solo para inicializar la superclase, pero lo redefinimos
        )

        # Sobrescribimos la capa final con dos cabezales:
        # 1) Actor: produce la media (mu) de la acción.
        #    La std se maneja como un parámetro aparte (log_std).
        # 2) Crítico: produce un valor escalar V(s).
        
        # Quitamos la fc_out original:
        del self.fc_out

        # Actor:
        self.actor_mean = nn.Linear(temporal_d_model, action_dim)
        # log_std global para cada acción (podrías hacer uno por dimensión)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Crítico:
        self.critic_head = nn.Sequential(
            nn.Linear(temporal_d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, batch_imgs, batch_tabs, batch_lidar,
                chunk_size=None, return_value=True):
        """
        Retorna:
          - mean: (B, action_dim)
          - std:  (B, action_dim)
          - value: (B, 1) si return_value=True
        """
        # 1) Extraemos las features del backbone (CNN + PointNet + Tab + CrossAttn + Transformer)
        #    -> en RL solemos usar la última "salida" de la secuencia o un pool
        #    -> _forward_transformer_in_chunks dev. (out, temporal_feat)
        #       donde out=(B,T,action_dim) en la superclase, pero no lo usamos directamente
        #       preferimos quedarnos con la representación 'features'.
        
        x_all = self._precompute_sensor_features(batch_imgs, batch_tabs, batch_lidar)
        # Usamos la parte de cross-attn + temporal, pero "desactivamos" la fc_out
        # Llamamos manualmente al método chunk:
        B, T, _ = x_all.shape

        if chunk_size is None:
            # Procesamos toda la secuencia
            x_all_attn = self.cross_attn(x_all)  # (B, T, cross_d_model)
            if self.proj is not None:
                x_all_attn = self.proj(x_all_attn)  # => (B, T, temporal_d_model)

            temporal_out = self.temporal_transformer(x_all_attn)  # => (B, T, d_model)
        else:
            # Procesar por chunks
            chunks_rep = []
            for start_t in range(0, T, chunk_size):
                end_t = min(start_t+chunk_size, T)
                x_chunk = x_all[:, start_t:end_t, :]

                x_chunk = self.cross_attn(x_chunk)
                if self.proj is not None:
                    x_chunk = self.proj(x_chunk)

                chunk_out = self.temporal_transformer(x_chunk)
                chunks_rep.append(chunk_out)

            temporal_out = torch.cat(chunks_rep, dim=1)  # => (B, T, d_model)

        # 2) Seleccionamos el último timestep como estado actual (o un pool si prefieres)
        #    Escogemos: temporal_out[:, -1, :]
        final_feat = temporal_out[:, -1, :]  # => (B, d_model)

        # 3) Actor
        mean = self.actor_mean(final_feat)    # => (B, action_dim)
        std = self.log_std.exp().expand_as(mean)  # => (B, action_dim)

        # 4) Crítico
        value = None
        if return_value:
            value = self.critic_head(final_feat)  # => (B, 1)

        return mean, std, value

    def act(self, batch_imgs, batch_tabs, batch_lidar, chunk_size=None,
            deterministic=False):
        """
        Método auxiliar para muestrear acciones (y valor).
        Retorna:
          action => (B, action_dim)
          value  => (B, 1)
          log_prob => (B,) log-probabilidad de la acción muestreada
        """
        mean, std, value = self.forward(batch_imgs, batch_tabs, batch_lidar,
                                        chunk_size=chunk_size, return_value=True)
        if deterministic:
            action = mean
        else:
            # Muestreamos de una Gaussiana
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
        
        # Calculamos log_prob
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)  # suma sobre cada dim de acción

        return action, value, log_prob