import os
import time
import argparse
import logging
import random
import collections
from PIL import Image
import torchvision.transforms as TT

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from multisensor_cross_attention_transformer import MultiSensorCrossAttentionTransformer
from carla_gym_env import CarlaEnv

transform_depth_cam = TT.Compose([
        TT.Grayscale(num_output_channels=1),
        TT.ToTensor(),
    ])

transform_rgb_front = TT.Compose([
        TT.ToTensor(),
    ])

class ActorCritic(nn.Module):
    def __init__(self, backbone, temporal_d_model=128, action_dim=3):
        """
        backbone: instancia de MultiSensorCrossAttentionTransformer
        temporal_d_model: dimensión de salida del transformer
        action_dim: número de acciones (3 en este caso)
        """
        super().__init__()
        self.backbone = backbone
        # Usamos la fc_out preentrenada para la parte actor.
        # Para la parte de la política, definimos un parámetro de log_std (por acción)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -2)
        # Cabeza crítica que recibe las features intermedias (del temporal transformer)
        self.value_head = nn.Linear(temporal_d_model, 1)

    def forward(self, batch_imgs, batch_tabs, batch_lidar, chunk_size=16):
        """
        Retorna:
          - actions: (B, T, action_dim) -> salida de la cabeza actor (media)
          - value: (B, T, 1) -> valor estimado para cada timestep
        """
        actions_seq, features = self.backbone(batch_imgs, batch_tabs, batch_lidar,
                                               chunk_size=chunk_size,
                                               return_features=True)
        value_seq = self.value_head(features)
        return actions_seq, value_seq

    def act(self, obs, device, chunk_size=16):
        """
        Dada una observación (formato dict, de la env), procesa la imagen, datos tabulares y LiDAR,
        y retorna la acción sampleada, valor y log_prob (todo en modo no diferenciable).
        """
        self.eval()  # Modo evaluación
        with torch.no_grad():
            batch_imgs, batch_tabs, batch_lidar = process_obs(obs, device)
            actions_seq, value_seq = self.forward(batch_imgs, batch_tabs, batch_lidar,
                                                  chunk_size=chunk_size)
            # Como se procesa una secuencia de 1 timestep, usamos el último
            action_mean = actions_seq[:, -1, :]  # (B, action_dim)
            value = value_seq[:, -1, 0]          # (B,)
            std = self.log_std.exp().expand_as(action_mean)
            dist = torch.distributions.Normal(action_mean, std)
            action_raw = dist.sample()
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            steer = torch.tanh(action_raw[:, 0:1])
            bias_throttle = 0.5  # Sesgo positivo para favorecer la aceleración
            bias_brake = -0.5    # Sesgo negativo para reducir el frenado
            throttle = torch.sigmoid(action_raw[:, 1:2] + bias_throttle)
            brake = torch.sigmoid(action_raw[:, 2:3] + bias_brake)
            action_processed = torch.cat([steer, throttle, brake], dim=-1)

        self.train()  # Vuelve al modo entrenamiento
        return action_processed.cpu().detach().numpy()[0], value, log_prob

# ---------------------------------------------------------------------------
# Función para convertir la observación (dict de la env) en tensores con la forma esperada
# Las observaciones vienen en formato numpy (p.ej., imágenes HxWxC); se convierten a tensor con
# la dimensión de batch y secuencia (B, T, ...)
# ---------------------------------------------------------------------------
def process_obs(obs, device):
    """
    Recibe una observación con las claves:
      - "rgb_front": imagen RGB (H, W, 3) (uint8)
      - "depth_cam": imagen de profundidad (H, W) o (H, W, 1) (float32)
      - "gnss": vector (3,)
      - "lidar": nube de puntos (N, 3)
    Retorna:
      - batch_imgs: dict con tensores de forma (1, 1, C, H, W)
      - batch_tabs: dict con tensores de forma (1, 1, D)
      - batch_lidar: tensor de forma (1, 1, N, 3)
    """
    # Procesa RGB: convierte a float y normaliza
    rgb = obs["rgb_front"]  # shape: (H, W, 3)
    rgb_pil = Image.fromarray(rgb.astype('uint8'), 'RGB')  # Convertir a PIL Image
    rgb_tensor = transform_rgb_front(rgb_pil).to(device)
    # rgb_tensor: (1, 1, 3, H, W)

    # Procesa Depth: se asegura que tenga canal
    depth = obs["depth_cam"]  # shape: (H, W)
    if len(depth.shape) == 2:
        depth_pil = Image.fromarray(depth.astype('uint8'), 'L')  # Modo 'L' para grayscale
    else:
        depth_pil = Image.fromarray(depth.astype('uint8'), 'RGB')
    depth_tensor = transform_depth_cam(depth_pil).to(device) 

    batch_imgs = {
        "rgb_front": rgb_tensor.unsqueeze(0).unsqueeze(0),    # (1, 3, H', W')
        "depth_cam": depth_tensor.unsqueeze(0).unsqueeze(0)     # (1, 1, H', W')
    }

    # Procesa datos tabulares: mapea "gnss" a "gnss_sensor"
    gnss = obs["gnss"]
    gnss_tensor = torch.from_numpy(gnss).float().unsqueeze(0).unsqueeze(0).to(device)
    # gnss_tensor: (1, 1, 3)
    batch_tabs = {"gnss_sensor": gnss_tensor}

    # Procesa LiDAR: (N, 3) -> (1, 1, N, 3)
    lidar = obs["lidar"]
    lidar_tensor = torch.from_numpy(lidar).float().unsqueeze(0).unsqueeze(0).to(device)

    return batch_imgs, batch_tabs, lidar_tensor

# ---------------------------------------------------------------------------
# Función para computar GAE (advantage) y los returns
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """
    rewards: lista de rewards por timestep (length=T)
    values: lista de valores estimados (length=T)
    dones: lista de flags done (0 o 1) (length=T)
    next_value: valor estimado para el siguiente timestep (float)
    gamma: factor de descuento
    lam: lambda para GAE
    Retorna:
      - advantages: lista de ventajas
      - returns: lista de returns (valor objetivo para el crítico)
    """
    advantages = []
    gae = 0
    # Agregamos next_value a la lista de valores para bootstrap
    values = values + [next_value]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

# ---------------------------------------------------------------------------
# Función principal de entrenamiento
# ---------------------------------------------------------------------------
def train(args):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")

    env = CarlaEnv(
        render_display=True,
        seed=42,
        image_width=240,     # Debe coincidir con las transformaciones
        image_height=120,    # Debe coincidir con las transformaciones
        frame_skip=5,        # Por ejemplo
        camera_fov=90,
        vehicle_model="vehicle.tesla.model3",
        enable_lane_following=True
    )


    # Instancia el modelo preentrenado (backbone)
    if args.pretrained_path is not None and os.path.isfile(args.pretrained_path):
        backbone = torch.load(args.pretrained_path, map_location=device)
        logging.info(f"Cargados pesos preentrenados desde {args.pretrained_path}")
    else:
        logging.info("No se ha cargado modelo preentrenado (se usará el inicializado aleatoriamente).")
    backbone.to(device)

    # Crea el modelo Actor-Crítico
    actor_critic = ActorCritic(backbone).to(device)
    actor_critic.train()

    optimizer = optim.Adam([
        {'params': backbone.parameters(), 'lr': args.learning_rate * 0.1},
        {'params': actor_critic.value_head.parameters()},
        {'params': [actor_critic.log_std]}
    ], lr=args.learning_rate)



    # Hiperparámetros de PPO
    rollout_length = args.rollout_length      # número de timesteps por actualización
    ppo_epochs = args.ppo_epochs
    clip_param = args.clip_param
    gamma = args.gamma
    lam = args.lam

    num_updates = args.num_updates

    # Bucle de entrenamiento principal
    update = 0
    obs, _ = env.reset()
    ep_reward = 0.0
    while update < num_updates:
        # Almacena datos del rollout
        obs_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        dones_list = []

        ep_reward = 0.0

        # Recoge rollout_length timesteps
        for step in range(rollout_length):
            # Obtiene acción, valor y log_prob usando el modelo
            action, value, log_prob = actor_critic.act(obs, device)
            next_obs, reward, done, truncated, info = env.step(action)

            # Almacena datos
            processed_obs = process_obs(obs, device)  # Devuelve (batch_imgs, batch_tabs, batch_lidar)
            obs_list.append(processed_obs)
            actions_list.append(action)  # acción ya postprocesada
            log_probs_list.append(log_prob.item())
            rewards_list.append(reward)
            values_list.append(value.item())
            dones_list.append(1.0 if done else 0.0)

            obs = next_obs
            ep_reward += reward

            if done:
                logging.info(f"Fin de episodio. Recompensa total: {ep_reward:.2f}")
                obs, _ = env.reset()
                ep_reward = 0.0

        # Valor para bootstrap (última observación)
        _, next_value, _ = actor_critic.act(obs, device)

        # Computa ventajas y returns usando GAE
        advantages, returns = compute_gae(rewards_list, values_list, dones_list, next_value.item(), gamma, lam)

        # ----- PREPARACIÓN DE LOS DATOS DEL ROLLOUT -----
        # Dado que cada observación es un dict, convertimos cada uno a tensores y apilamos a lo largo
        # de la dimensión temporal para formar secuencias de forma (1, T, ...)
        batch_imgs_list = {"depth_cam": [], "rgb_front": []}
        batch_tabs_list = {"gnss_sensor": []}
        batch_lidar_list = []

        for (b_imgs, b_tabs, b_lidar) in obs_list:
            batch_imgs_list["rgb_front"].append(b_imgs["rgb_front"].squeeze(0))   # (1, 3, H, W)
            batch_imgs_list["depth_cam"].append(b_imgs["depth_cam"].squeeze(0))   # (1, 1, H, W)
            batch_tabs_list["gnss_sensor"].append(b_tabs["gnss_sensor"].squeeze(0))  # (1, 3)
            batch_lidar_list.append(b_lidar.squeeze(0))  # (1, N, 3)

        # Apila a lo largo de la dimensión temporal (T)
        batch_imgs = {}
        for key in batch_imgs_list:
            # Cada tensor se expande en la dimensión temporal y se concatenan
            # Resultado: (1, T, C, H, W)
            batch_imgs[key] = torch.cat([x.unsqueeze(1) for x in batch_imgs_list[key]], dim=1)
        batch_tabs = {}
        for key in batch_tabs_list:
            # Resultado: (1, T, D)
            batch_tabs[key] = torch.cat([x.unsqueeze(1) for x in batch_tabs_list[key]], dim=1)
        # Para LiDAR: (1, T, N, 3)
        batch_lidar = torch.cat([x.unsqueeze(1) for x in batch_lidar_list], dim=1)

        # Convierte las demás variables a tensores y agrega dimensión batch
        actions_tensor = torch.tensor(np.array(actions_list), dtype=torch.float32, device=device).unsqueeze(0)   # (1, T, action_dim)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(0)            # (1, T)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device).unsqueeze(0)        # (1, T)
        old_log_probs_tensor = torch.tensor(log_probs_list, dtype=torch.float32, device=device).unsqueeze(0)   # (1, T)


        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # ----- ACTUALIZACIÓN PPO -----
        total_loss = 0.0
        for epoch in range(ppo_epochs):
            # Realiza un forward completo sobre la secuencia del rollout
            actions_seq, values_seq = actor_critic.forward(batch_imgs, batch_tabs, batch_lidar, chunk_size=16)
            # values_seq: (1, T, 1) → quita la última dimensión
            values_seq = values_seq.squeeze(-1)  # (1, T)
            # Calcula la nueva distribución de acciones a partir de la media producida por la red
            mean_actions = actions_seq  # (1, T, action_dim)
            std = actor_critic.log_std.exp().expand_as(mean_actions)
            dist = torch.distributions.Normal(mean_actions, std)
            new_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)  # (1, T)
            entropy = dist.entropy().sum(dim=-1).mean()

            # Ratio para PPO
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            # Cálculo de la pérdida: se usa el clip de PPO
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (returns_tensor - values_seq).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.02 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / ppo_epochs
        update += 1
        logging.info(f"Actualización {update}: Pérdida = {avg_loss:.4f}, Suma rewards rollout = {sum(rewards_list):.2f}")

        # Guarda un checkpoint cada cierto número de actualizaciones
        if update % args.save_interval == 0:
            ckpt_path = f"./checkpoint/actor_critic_update_{update}.pth"
            torch.save(actor_critic, ckpt_path)
            logging.info(f"Guardado checkpoint en {ckpt_path}")

    env.close()
    logging.info("Entrenamiento finalizado.")

# ---------------------------------------------------------------------------
# Configuración de argumentos y llamada a la función principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento RL con PPO para conducción autónoma en CARLA")
    parser.add_argument("--pretrained_path", type=str, default='checkpoint_chunks_final.pt',
                        help="Ruta al modelo preentrenado (archivo .pth)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Tasa de aprendizaje")
    parser.add_argument("--rollout_length", type=int, default=32,
                        help="Cantidad de timesteps por rollout")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="Cantidad de épocas PPO por actualización")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="Valor de clip para PPO")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Factor de descuento")
    parser.add_argument("--lam", type=float, default=0.95,
                        help="Lambda para GAE")
    parser.add_argument("--num_updates", type=int, default=10000,
                        help="Cantidad de actualizaciones de PPO")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Intervalo (en actualizaciones) para guardar checkpoints")
    parser.add_argument("--seed", type=int, default=100, help="Semilla para reproducibilidad")
    args = parser.parse_args()

    train(args)