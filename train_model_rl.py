import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torchvision.transforms as TT
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importa tu modelo preentrenado
from multisensor_cross_attention_transformer import MultiSensorCrossAttentionTransformer

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo preentrenado
tabular_input_dims = {'gnss_sensor': 3}
pretrained_model = MultiSensorCrossAttentionTransformer(
    sensor_list=('depth_cam', 'rgb_front'),
    tabular_sensors=('gnss_sensor',),
    tabular_input_dims=tabular_input_dims,
    image_feature_dim=128,
    tabular_feature_dim=64,
    lidar_feature_dim=128,
    cross_d_model=128,
    cross_nhead=4,
    cross_dropout=0.3,
    temporal_d_model=128,
    temporal_nhead=4,
    temporal_num_layers=2,
    temporal_dim_feedforward=256,
    temporal_dropout=0.1,
    out_dim=3
).to(device)

pretrained_model_path = 'checkpoint_chunks_final.pt'  # Reemplaza con la ruta real
pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
pretrained_model.eval()



# Definir el extractor de características personalizado
class MultiSensorFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space,  **kwargs):
        super().__init__(observation_space,  features_dim=128)
        
        # Inicializar el modelo preentrenado
        self.pretrained_model = pretrained_model

        self.action_dist = DiagGaussianDistribution(3)
        
        # Congelar las capas del modelo preentrenado
        #for param in self.pretrained_model.parameters():
        #    param.requires_grad = False
        
    

    def forward(self, observations):
        """
        Procesa las observaciones y extrae características utilizando el modelo preentrenado.

        Args:
            observations (dict): Observaciones del entorno con claves 'rgb_front', 'depth_cam', 'gnss', 'lidar'.

        Returns:
            torch.Tensor: Características extraídas de dimensión `features_dim`.
        """
        # Extraer las observaciones del diccionario
        rgb = observations["rgb_front"]    # shape: (C, H, W)
        depth = observations["depth_cam"]  # shape: (1, H, W) o similar

        # Normalizar las imágenes si es necesario
        rgb = rgb.float() / 255.0          # Normalizar RGB a [0, 1]
        depth = depth.float() / 255.0        # Normalizar Depth a [0, 1]

        # Asegurarse de que las imágenes tengan 5 dimensiones: (B, T, C, H, W)
        # En RL, típicamente B=1 y T=1
        rgb = rgb.unsqueeze(1)    # (1, 1, C, H, W)
        depth = depth.unsqueeze(0).unsqueeze(1)  # (1, 1, 1, H, W)

        # Procesamiento de GNSS
        gnss = observations["gnss"]  # shape: (3,)
        gnss = gnss.unsqueeze(1)  # (1, 1, 3)

        # Procesamiento de LiDAR
        lidar = observations["lidar"]  # shape: (max_lidar_points, 4)
        lidar_3d = lidar[:,:,:3]        # shape: (max_lidar_points, 3)
        lidar_3d = lidar_3d.unsqueeze(1)  # (1, 1, max_lidar_points, 3)

        # Crear un diccionario para batch_imgs con la dimensión de tiempo
        batch_imgs = {
            "rgb_front": rgb.to(device),    # (1, 1, C, H, W)
            "depth_cam": depth.to(device)   # (1, 1, 1, H, W)
        }

        # Crear un diccionario para batch_tabs
        batch_tabs = {
            "gnss_sensor": gnss.to(device)  # (1, 1, 3)
        }

        # batch_lidar ya está procesado
        batch_lidar = lidar_3d.to(device)    # (1, 1, max_lidar_points, 3)
        
        # Definir chunk_size fijo para RL
        chunk_size = 1  # Puedes ajustar este valor según tu entrenamiento original
        
        # Pasar los argumentos correctos al modelo preentrenado
        features = self.pretrained_model(batch_imgs, batch_tabs, batch_lidar, chunk_size=chunk_size,return_features=True)

        features = features[:, 0, :]
        
        return features

class CustomMultiSensorPolicy(ActorCriticPolicy):
    """
    Política Actor-Crítico para PPO que:
    - Usa el MultiSensorFeaturesExtractor para obtener embedding.
    - Pasa ese embedding por la "actor" (para mu/log_std) y "critic" (para V(s)).
    """
    def __init__(self, observation_space, action_space, lr_schedule, 
                 net_arch=None, feature_extractor_class=MultiSensorFeaturesExtractor,
                 feature_extractor_kwargs=None,
                 **kwargs):
        
        # 1) Llamamos al init de ActorCriticPolicy.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # SB3 ignora net_arch si definimos un feature_extractor
            net_arch=net_arch,
            features_extractor_class=feature_extractor_class,
            features_extractor_kwargs=feature_extractor_kwargs,
            **kwargs
        )
        
        # 2) Definimos la distribución de acción: 
        #    En tu caso (steering, throttle, brake) => 3D continuo => Gaussiana
        self.action_dist = DiagGaussianDistribution(3)
        
        # 3) Dimensión de las features (por defecto 256 del extractor).
        #    la guardamos en self.features_dim
        #    (SB3 hace lo propio, pero reafirmamos por claridad).
        self.features_dim = self.features_extractor.features_dim
        
        # 4) Actor head: a partir del embedding (B, 256) -> (B, 3) (mu)
        #    + un log_std param aprendible
        self.actor_mu = nn.Linear(self.features_dim, 3)
        # log_std como un parámetro de tamaño (3,).
        self.log_std_param = nn.Parameter(torch.zeros(3))
        
        # 5) Crítico: (B, 256) -> (B, 1)
        self.critic_value = nn.Linear(self.features_dim, 1)
        
        # 6) Inicializaciones recomendadas
        self.apply(self.init_weights)
    
    def init_weights(self, m, gain=1.0):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _get_action_dist_params(self, features: torch.Tensor):
        """
        Dado el embedding 'features', retorna (mean, log_std)
        para la distribución Gaussiana de la acción.
        """
        # Mean
        action_mean = self.actor_mu(features)
        # Log std
        action_log_std = self.log_std_param.expand_as(action_mean)
        return action_mean, action_log_std
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Retorna la media de la acción (mu).
        (SB3 internamente llamará a la distribución con (mu, log_std)).
        """
        mu, _ = self._get_action_dist_params(features)
        return mu
    
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Retorna V(s).
        """
        return self.critic_value(features)
    
    def forward(self, obs, deterministic=False):
        """
        No se usa directamente en PPO de SB3, 
        pero lo definimos por completitud.
        """
        features = self.extract_features(obs)
        mu, log_std = self._get_action_dist_params(features)
        
        dist = self.action_dist.proba_distribution(mean_actions=mu, log_std=log_std)

        
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample()
        
        # Calcula V(s):
        values = self.forward_critic(features)

        # Ahora retorna los 3 elementos
        return actions, values, dist.log_prob(actions)



from carla_gym_env import CarlaEnv

env = CarlaEnv(
        render_display=False,
        seed=42,
        image_width=240,     # Debe coincidir con las transformaciones
        image_height=120,    # Debe coincidir con las transformaciones
        frame_skip=1,        # Por ejemplo
        camera_fov=90,
        vehicle_model="vehicle.tesla.model3",
        enable_lane_following=True
    )

# Configurar callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./logs/',
    name_prefix='ppo_simulator'
)

eval_callback = EvalCallback(
    env,
    best_model_save_path='./logs/best_model/',
    log_path='./logs/results/',
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Inicializar el modelo PPO con la política personalizada
model = PPO(
    policy=CustomMultiSensorPolicy,
    env=env,
    verbose=1,
    tensorboard_log="./ppo_simulator_tensorboard/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# Entrenar el modelo
total_timesteps = 100000  # Ajusta según tus necesidades
model.learn(
    total_timesteps=total_timesteps,
    callback=[checkpoint_callback, eval_callback],
    log_interval=10
)

# Guardar el modelo entrenado
model.save("ppo_simulator_trained")
print("Modelo guardado como 'ppo_simulator_trained.zip'")

# Evaluar el modelo entrenado
num_episodes = 10
episode_rewards = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()  # Opcional: Visualizar el entorno
    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Visualizar las recompensas de los episodios
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o')
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Desempeño del Agente en Diferentes Episodios")
plt.grid(True)
plt.show()
