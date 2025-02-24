import argparse
import os
import torch
import numpy as np
from collections import deque
from PIL import Image

import torchvision.transforms as TT

from multisensor_cross_attention_transformer import (
    MultiSensorCrossAttentionTransformer
)

from carla_gym_env import CarlaEnv
from train_model_rl_2 import ActorCritic

def load_trained_model(model_path, device='cuda'):
    """
    Carga el modelo entrenado desde el checkpoint.
    """

    # Verificar la existencia del modelo
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo no existe en la ruta especificada: {model_path}")

    # Cargar el estado del modelo
    print(f"Cargando modelo desde '{model_path}' en '{device}' ...")
    model = torch.load(model_path)
    model.eval()  # Modo evaluación
    return model

def get_transforms():
    """
    Define las transformaciones para las imágenes.
    """
    transform_depth_cam = TT.Compose([
        TT.Grayscale(num_output_channels=1),
        TT.ToTensor(),
    ])

    transform_rgb_front = TT.Compose([
        TT.ToTensor(),
    ])

    return transform_depth_cam, transform_rgb_front

def process_observation(obs, transform_depth_cam, transform_rgb_front, device):
    """
    Convierte la observación del entorno en tensores adecuados para el modelo.
    """
    # Procesamiento de la imagen RGB
    rgb = obs["rgb_front"]  # shape: (H, W, 3)
    rgb_pil = Image.fromarray(rgb.astype('uint8'), 'RGB')  # Convertir a PIL Image
    rgb_tensor = transform_rgb_front(rgb_pil).to(device)  # (3, 112, 112)

    # Procesamiento de la imagen Depth
    depth = obs["depth_cam"]  # shape: (H, W)
    # Asegurarse de que la imagen Depth tenga una única canal
    if len(depth.shape) == 2:
        depth_pil = Image.fromarray(depth.astype('uint8'), 'L')  # Convertir a PIL Image en modo 'L' (grayscale)
    else:
        # Si depth tiene más de un canal, ajustar según corresponda
        depth_pil = Image.fromarray(depth.astype('uint8'), 'RGB')  # o el modo adecuado
    depth_tensor = transform_depth_cam(depth_pil).to(device)  # (1, 112, 112)

    # Crear un diccionario para batch_imgs con la dimensión de tiempo
    batch_imgs = {
        "rgb_front": rgb_tensor.unsqueeze(0),    # (1, 3, 112, 112)
        "depth_cam": depth_tensor.unsqueeze(0)   # (1, 1, 112, 112)
    }

    # Procesamiento de GNSS
    gnss = obs["gnss"]  # np.array([lat, lon, alt])
    batch_tabs = {
        "gnss_sensor": torch.from_numpy(gnss).float().unsqueeze(0).to(device)  # (1, 3)
    }

    # Procesamiento de LiDAR
    lidar = obs["lidar"]  # shape: (max_lidar_points, 4)
    lidar_3d = lidar[:, :3]  # shape: (max_lidar_points, 3)
    batch_lidar = torch.from_numpy(lidar_3d).float().to(device)  # (max_lidar_points, 3)

    return batch_imgs, batch_tabs, batch_lidar


def real_time_test(
    model,
    env,
    num_episodes=10000,
    chunk_size=16,
    sequence_length=50,
    device='cuda',
    render_display=False
):
    """
    Ejecuta pruebas en tiempo real utilizando el modelo entrenado en el entorno CARLA.
    Calcula métricas R² y MAE comparando las acciones predichas con las acciones reales.
    """
    # Definir las transformaciones
    transform_depth_cam, transform_rgb_front = get_transforms()

    # Configuración de los buffers deslizantes
    img_buffer = {sensor: deque(maxlen=sequence_length) for sensor in ('depth_cam', 'rgb_front')}
    tab_buffer = {sensor: deque(maxlen=sequence_length) for sensor in ('gnss_sensor',)}
    lidar_buffer = deque(maxlen=sequence_length)

    # Acumuladores para métricas
    all_pred_actions = []
    all_true_actions = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        # Reiniciar buffers al comienzo de cada episodio
        for buffer in img_buffer.values():
            buffer.clear()
        for buffer in tab_buffer.values():
            buffer.clear()
        lidar_buffer.clear()

        while not done:
            # Procesar la observación

                # Pasar por el modelo con chunking
            with torch.no_grad():
                action, value, log_prob = model.act(obs, device)
                # outputs: (B, T, out_dim)

            # Seleccionar la acción correspondiente al último paso de la secuencia

            print(f"Acción Predicha: {action}")

            # Aplicar acción en el entorno
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward


    env.close()
    print("Pruebas completadas.")


def main():
    parser = argparse.ArgumentParser(description="Probar un modelo PyTorch entrenado en CARLA (4 observaciones).")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Ruta al modelo PyTorch entrenado (.pt)')
    parser.add_argument('--num_episodes', type=int, default=10000,
                        help='Número de episodios para probar el modelo')
    parser.add_argument('--render_display', action='store_true',
                        help='Si se especifica, se mostrará la ventana de pygame con la cámara RGB.')

    args = parser.parse_args()
    model_path = args.model_path
    num_episodes = args.num_episodes
    render_display = args.render_display

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cargar el modelo entrenado
    model = load_trained_model(model_path, device=device)

    # Crear el entorno con render opcional
    env = CarlaEnv(
        render_display=render_display,
        seed=42,
        image_width=240,     # Debe coincidir con las transformaciones
        image_height=120,    # Debe coincidir con las transformaciones
        frame_skip=1,        # Por ejemplo
        camera_fov=90,
        vehicle_model="vehicle.tesla.model3",
        enable_lane_following=True
    )

    # Ejecutar la prueba en tiempo real
    real_time_test(
        model=model,
        env=env,
        num_episodes=num_episodes,
        chunk_size=16,         # Tamaño del chunk usado durante el entrenamiento
        sequence_length=50,    # Longitud de la secuencia
        device=device,
        render_display=render_display
    )

if __name__ == "__main__":
    main()
