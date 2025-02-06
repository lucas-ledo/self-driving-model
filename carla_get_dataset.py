import csv
import gymnasium as gym
import numpy as np
import cv2
import random
import time
import carla
import math
from carla import Location, Rotation
import collections
import json

import torch

from gymnasium import spaces, ObservationWrapper
from gymnasium.spaces import Box
import os
import logging

import pygame  # Añadido para la visualización
from pygame.locals import QUIT

# Opcional: ocultar mensajes de pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
REWARD_WAYPOINT = 10.0
REWARD_FINAL = 100.0
PENALTY_COLLISION = -40.0
PENALTY_OFF_TRACK = -40.0
PENALTY_TIME_STEP = 0.01    
PENALTY_ERRATIC_STEER = 0.0001
PENALTY_ERRATIC_ACCEL = 0.01
PROGRESS_FACTOR = 5.0
REWARD_CENTER_LINE = 0.1

# Ejemplo de velocidad segura
MAX_SAFE_SPEED = 8.0  # m/s aprox 72 km/h
REWARD_SPEED_FACTOR = 0.02

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_observation(obs, image_size=(120,240)):
    """
    Preprocesa la observación de la imagen:
    - Convierte a escala de grises.
    - Redimensiona la imagen.
    - Normaliza los píxeles.
    - Añade una dimensión de canal.
    
    Parámetros:
    -----------
    - obs (np.array): Imagen RGB original.
    - image_size (tuple): Dimensiones de la imagen redimensionada.
    
    Retorna:
    --------
    - preprocessed_obs (np.array): Imagen preprocesada.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(obs["image"], cv2.COLOR_RGB2GRAY)
    # Redimensionar
    resized = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    # Normalizar
    normalized = resized / 255.0
    # Reordenar ejes: de (H, W) a (H, W,1)
    preprocessed_obs = normalized.reshape(1,image_size[0], image_size[1]).astype(np.float32)
    return preprocessed_obs

class PreprocessFrame(ObservationWrapper):
    def __init__(self, env, image_size=(120,240 )):
        super(PreprocessFrame, self).__init__(env)
        self.image_size = image_size
        # Define el nuevo espacio de observación como un diccionario
        self.observation_space = spaces.Dict({
            "image": Box(low=0, high=1.0, shape=(1,120, 240), dtype=np.float32),
            "speed": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "orientation": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "orientation_waypoint": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })

    def observation(self, obs):
        preprocessed_image = preprocess_observation(obs, self.image_size)


        speed = np.array([self.env.current_velocity / 100.0], dtype=np.float32)
        orientation = np.array([self.env.vehicle.get_transform().rotation.yaw / 180.0], dtype=np.float32)
        orientation_waypoint = obs["angle_diff"]


        return {
            "image": preprocessed_image,
            "speed": speed,
            "orientation": orientation,
            "orientation_waypoint": orientation_waypoint
        }

class CarlaEnv(gym.Env):
    """
    Entorno Gym para controlar un vehículo en CARLA en modo síncrono, 
    con una cámara RGB y sensores adicionales.
    """

    def __init__(
        self,
        host="localhost",
        port=2000,
        world_timeout=10.0,
        render_display=False,
        image_width=240,  # Aumentado para mejor visualización en Pygame
        image_height=120,
        max_episode_time=240.0,
        reward_collision=-400.0,
        reward_survival=1,
        frame_skip=3,
        seed=None,
        spectator_display=True,
        camera_fov=110,
        vehicle_model="vehicle.tesla.model3",
        map="Town01",
        enable_lane_following=True,
    ):
        """
        Inicializa el entorno CarlaEnv.

        Parámetros:
        -----------
        - host (str): IP o nombre del host donde corre CARLA.
        - port (int): Puerto de conexión (por defecto 2000 en CARLA).
        - world_timeout (float): Timeout para la conexión al mundo.
        - render_display (bool): Si True, se mostrará la imagen en ventana de Pygame.
        - image_width (int), image_height (int): Dimensiones de la cámara.
        - max_episode_time (float): Tiempo máximo (segundos) antes de terminar episodio.
        - reward_collision (float): Penalización al colisionar.
        - reward_survival (float): Recompensa por cada step sin colisión.
        - frame_skip (int): Número de ticks del mundo que se saltan antes de recibir la siguiente observación.
        - seed (int, optional): Semilla para reproducibilidad.
        - camera_fov (int): Campo de visión de la cámara.
        - vehicle_model (str): Modelo de vehículo a spawnear.
        - enable_lane_following (bool): Si True, se agrega penalización por invadir carriles.
        """

        super(CarlaEnv, self).__init__()

        # Configuración de la semilla para reproducibilidad
        self.seed(seed)

        self.host = host
        self.port = port
        self.world_timeout = world_timeout
        self.render_display = render_display
        self.spectator_display = spectator_display
        self.map = map

        self.image_width = image_width
        self.image_height = image_height
        self.max_episode_time = max_episode_time
        self.reward_collision = reward_collision
        self.reward_survival = reward_survival
        self.frame_skip = frame_skip
        self.camera_fov = camera_fov
        self.vehicle_model = vehicle_model
        self.enable_lane_following = enable_lane_following
        self.current_step = 0
        self.route_points = None  
        self.current_route_index = 1     # Índice del waypoint al que nos dirigimos
        self.route_threshold = 0.5
        self.prev_distance_wp = 0
        self.distance_2lane_center_tol = 1

        self.prev_steer = 0
        self.prev_accel = 0
        

        # Espacio de acciones: [steer, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float32
        )

        # Espacio de observación: imagen RGB
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_height, self.image_width, 3),
            dtype=np.uint8
        )

        # Inicializar Pygame si se requiere visualización
        if self.render_display:
            pygame.init()
            self.screen = pygame.display.set_mode((self.image_width, self.image_height))
            pygame.display.set_caption("CARLA Gym Environment")
            self.clock = pygame.time.Clock()
            logger.info("Pygame inicializado y ventana creada.")

        # Conexión con CARLA
        self.client = None
        self.world = None
        self._connect_to_carla()

        # Modo síncrono
        self._enable_synchronous_mode()

        # Variables y contenedores
        self.actor_list = []
        self.vehicle = None
        self.rgb_camera = None
        self.collision_sensor = None
        self.image_data = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.collision_occurred = False
        self.episode_start_time = time.time()
        self.current_velocity = 0.0
        self.action_history = collections.deque(maxlen=3)
        self.crossed_solid_line = False
        self.crossed_solid_line_count = 0
        self.crossed_broken_line = False

        # Ventana OpenCV opcional removida, ya que usaremos Pygame

    def seed(self, seed=None):
        """Fija la semilla para la reproducibilidad."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    

    def _connect_to_carla(self, max_retries=5, delay=2):
        """Conecta con el servidor CARLA con intentos de reconexión."""
        attempts = 0
        while attempts < max_retries:
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(self.world_timeout)
                self.world = self.client.load_world(self.map)
                weather = carla.WeatherParameters.CloudySunset
                self.world.set_weather(weather)

                if self.world is not None:
                    logger.info(f"Conectado a CARLA en {self.host}:{self.port}")
                    return
            except Exception as e:
                logger.warning(f"Intento {attempts + 1}/{max_retries} de conexión fallido: {e}")
                time.sleep(delay)
                attempts += 1
        raise ConnectionError(f"No se pudo conectar a CARLA después de {max_retries} intentos.")
    
    def collect_data_autopilot(self, num_steps=300, dataset_path="dataset"):
        """
        Recolecta datos mediante el autopiloto de CARLA.
        - Spawnea vehículo y sensores.
        - Activa autopiloto en el vehículo.
        - Guarda imágenes, nubes LiDAR, etc. en 'dataset/' (a través de los callbacks).
        - Registra en disco los comandos de control en 'dataset/control/' para cada frame.

        :param num_steps: cantidad de steps (ticks) a recolectar.
        :param dataset_path: ruta base donde guardar los datos.
        """
        logger.info("[DATA COLLECTION] Iniciando recolección de datos con autopiloto.")
        
        # 1. Limpiar actores previos y re-spawnear
        self._destroy_actors()
        self.route_points, spawn_point = self._generate_predefined_route()

        # 2. Spawnear vehículo
        self._spawn_vehicle(spawn_transform=spawn_point)

        # 3. Adjuntar los sensores que quieras grabar
        self._attach_cameras()       # frontal, trasera, laterales
        self._attach_depth_camera()  # depth
        self._attach_semantic_camera()  # segmentación
        self._attach_lidar_sensor()
        self._attach_radar_sensor()
        self._attach_gnss_sensor()
        self._attach_imu_sensor()
        self._attach_collision_sensor()
        if self.enable_lane_following:
            self._attach_lane_invasion_sensor()


        traffic_manager = self.client.get_trafficmanager(4203)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager_port = traffic_manager.get_port()

        # 4. Activar autopiloto
        if self.vehicle is not None:
            self.vehicle.set_autopilot(True,traffic_manager_port)
            logger.error("Activado autopilot")
        else:
            logger.error("No se pudo activar el autopiloto: vehículo nulo.")
            return

        # 5. Esperar unos ticks iniciales para estabilizar sensores
        for _ in range(5):
            self.world.tick()

        # 6. Bucle principal para recolectar datos
        start_frame = self.world.get_snapshot().frame
        for step in range(num_steps):
            # Avanza 1 tick en modo síncrono
            for _ in range(5):
                self.world.tick()
            snapshot = self.world.get_snapshot()
            frame_id = snapshot.frame

            # Guardar el control actual que el autopiloto está aplicando (si existe el vehículo)
            try:
                if self.vehicle is not None:
                    # record_control_callback es la misma función que definiste
                    self.record_control_callback(self.vehicle, frame_id)
            except Exception as e:
                logger.error(f"Error al registrar control del autopiloto: {e}")

            # Opcional: Renderizado en pantalla
            if self.spectator_display:
                self._update_spectator()

            logger.info(f"[DATA COLLECTION] Frame={frame_id}/{step} de {num_steps}, grabando datos...")

        # 7. Finalizada la recolección
        logger.info("[DATA COLLECTION] Recolección de datos finalizada.")
        # Puedes dejar spawneado el vehículo o destruirlo
        # self._destroy_actors()


    def _distance_to_lane_center(self) -> float:
        """
        Retorna la distancia (en metros) del vehículo al centro del carril más cercano.
        """
        if not self.vehicle:
            return 0.0  # o algún valor por defecto si no tienes vehículo aún

        # 1. Transform del vehículo
        vehicle_transform = self.vehicle.get_transform()  # carla.Transform
        vehicle_location = vehicle_transform.location      # carla.Location

        # 2. Waypoint del carril en la ubicación actual
        #    lane_type=carla.LaneType.Driving te asegura buscar carriles transitables
        waypoint = self.world.get_map().get_waypoint(
            vehicle_location, 
            project_to_road=True, 
            lane_type=carla.LaneType.Driving
        )
        
        # 3. Distancia euclídea
        center_location = waypoint.transform.location
        dx = vehicle_location.x - center_location.x
        dy = vehicle_location.y - center_location.y

        distance = math.sqrt(dx**2 + dy**2)
        return distance

    def _enable_synchronous_mode(self):
        """Activa el modo síncrono en el mundo."""
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1  # ~20 FPS
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)
            logger.info("Modo síncrono activado.")
        except Exception as e:
            logger.error(f"Error al habilitar modo síncrono: {e}")
            raise

    def _disable_synchronous_mode(self):
        """Desactiva el modo síncrono en el mundo."""
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            logger.info("Modo síncrono desactivado.")
        except Exception as e:
            logger.error(f"Error al deshabilitar modo síncrono: {e}")

    def spawn_area_free(self):
        try:
            self.route_points, spawn_point = self._generate_predefined_route()
            if len(self.route_points) == 0:
                raise ValueError("No hay puntos de ruta definidos.")
            # Spawnear vehículo en la ubicación del primer waypoint
            # Ajustar un 'carla.Transform' a partir de la 'carla.Location':
            self._spawn_vehicle(spawn_transform=spawn_point)
        except Exception as e: 
            logger.error(f"Error al spawnear el vehículo:{e}")
            time.sleep(0.5)
            self.spawn_area_free()

    def reset(self, seed=None, options=None):

        logger.info("Reiniciando el entorno.")
        # Establecer la semilla si se proporciona
        if seed is not None:
            self.seed(seed)

        if self.route_points is None:
            self.route_points, spawn_point = self._generate_predefined_route()
        
        # Limpiar actores pasados
        self._destroy_actors()
        logger.info(f"Paso {self.current_step}")

        # Resetear banderas
        self.done = False
        self.collision_occurred = False
        self.crossed_solid_line = False
        self.crossed_solid_line_count = 0
        self.current_route_index = 0
        self.episode_start_time = time.time()

        if len(self.route_points) == 0:
            raise ValueError("No hay puntos de ruta definidos.")

        try:
            self._spawn_vehicle(spawn_transform=spawn_point)
        except Exception as e:
            logger.error(f"Error al spawnear el vehículo: {e}")
            self.spawn_area_free()


        self._attach_cameras()
        self._attach_collision_sensor()

        # Spawnear sensores adicionales si es necesario
        if self.enable_lane_following:
            self._attach_lane_invasion_sensor()

        # Tick inicial para que arranque la cámara y los sensores
        for _ in range(5):
            self.world.tick()

        # Resetear velocidad
        self.current_velocity = 0

        # Devolver la primera observación y un diccionario vacío de información
        observation = self._get_observation()
        info= {}
        return observation, info

    def step(self, action):
        """
        Avanza el entorno un paso dado 'action' = [steer, throttle, brake].
        Retorna (obs, reward, done, truncated, info).
        """
        self.action_history.append(action)
        self.current_step += 1

        # Suavizar las acciones promediando el historial
        avg_action = np.mean(self.action_history, axis=0)
        steer, throttle, brake = avg_action

        # Construir el control del vehículo
        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -0.5, 0.5))
        control.throttle = float(np.clip(throttle, 0.0, 1.0))
        control.brake = float(np.clip(brake, 0.0, 0.5))
        control.hand_brake = False
        control.manual_gear_shift = False

        try:
            self.vehicle.apply_control(control)
        except Exception as e:
            logger.error(f"Error al aplicar control al vehículo: {e}")

        # Hacer 'frame_skip' ticks para simular la acción por varios frames
        for _ in range(self.frame_skip):
            try:
                self.world.tick()
            except Exception as e:
                logger.error(f"Error durante el tick del mundo: {e}")
                break  # Salir del loop si hay un error

        # Obtener la observación
        obs = self._get_observation()

        # Actualizar velocidad
        velocity = self.vehicle.get_velocity()
        self.current_velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # --- Lógica de ruta predefinida ---
        # 1) Obtener la ubicación actual del vehículo
        vehicle_loc = self.vehicle.get_transform().location

        # 2) Obtener el waypoint objetivo
        if self.current_route_index < len(self.route_points):
            target_loc = self.route_points[self.current_route_index]
            dist_to_target = vehicle_loc.distance(target_loc)

            # 3) Si estamos suficientemente cerca, pasar al siguiente waypoint
            if dist_to_target < self.route_threshold:
                self.current_route_index += 1

        # Calcular la recompensa
        reward = self._compute_reward(action,obs)

        # Verificar final de episodio
        done = self._check_done()
        self.prev_steer = steer
        self.prev_accel = throttle

        # Información adicional
        info = {
            "velocity": self.current_velocity,
            "collision": self.collision_occurred,
            "crossed_solid_line_count": self.crossed_solid_line_count,
            "elapsed_time": time.time() - self.episode_start_time
        }

        if self.enable_lane_following:
            info["lane_invasion"] = getattr(self, 'lane_invasion_occurred', False)

        self.crossed_broken_line = False

        if self.spectator_display:
            self._update_spectator()

        if self.render_display:
            self.render()
        return obs, reward, done,False, info

    def render(self, mode='human'):
        """
        Muestra la imagen capturada en una ventana de Pygame.
        """
        if self.render_display:
            try:
                # Convertir la imagen a formato Pygame
                image = self.image_data
                # Asegurarse de que la imagen tenga el tamaño correcto
                image = cv2.resize(image, (self.image_width, self.image_height))
                # Convertir de RGB a Surface de Pygame
                surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
                # Blitear la superficie en la pantalla
                self.screen.blit(surface, (0, 0))
                pygame.display.flip()

                # Gestionar eventos de Pygame para evitar que la ventana se congele
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.close()

                # Controlar la tasa de refresco de Pygame
                self.clock.tick(60)

            except Exception as e:
                logger.error(f"Error al renderizar la imagen: {e}")

    def close(self):
        """Finaliza el entorno."""
        logger.info("Cerrando el entorno.")
        self._destroy_actors()
        self._disable_synchronous_mode()
        if self.render_display:
            pygame.quit()

    def _generate_predefined_route(self):
        """
        Genera una lista de waypoints o localizaciones predefinidas en el mapa.
        """
        route = []
        # Mapa de CARLA
        carla_map = self.world.get_map()
        
        # 1) Tomar un spawn point cualquiera
        spawn_points = carla_map.get_spawn_points()
        start_spawn = self.np_random.choice(spawn_points)  # O uno aleatorio, si prefieres.
        
        # 2) De ese spawn point, tomas el waypoint más cercano:
        start_waypoint = carla_map.get_waypoint(start_spawn.location)
        
        # 3) Crear una ruta de varios waypoints hacia adelante
        #    Por ejemplo, 50 metros de distancia cada punto:
        distance_between_points = 1.5
        num_points = 8
        route.append(start_waypoint.transform.location)  # Primer punto
        
        current_waypoint = start_waypoint
        for i in range(num_points - 1):
            next_wp = current_waypoint.next(distance_between_points)[0]  
            # next() retorna una lista de waypoints posibles, tomamos el primero
            route.append(next_wp.transform.location)
            current_waypoint = next_wp
        
        #for i in range(len(route) - 1):
        #    self.world.debug.draw_line(
        #        route[i],
        #        route[i+1],
        #        thickness=0.1,
        #        color=carla.Color(r=255, g=0, b=0),
        #        life_time=0
        #    )

        return route, start_spawn

    def _spawn_vehicle(self, spawn_transform):
        def is_spawn_area_free(spawn_point, radius):
            vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in vehicles:
                distance = vehicle.get_location().distance(spawn_point.location)
                if distance < radius:
                    return False
            return True

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self.vehicle_model)[0]

        self.vehicle = None
        for attempt in range(20):
            candidate = spawn_transform
            if is_spawn_area_free(candidate, 5.0):
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, candidate)
                if self.vehicle is not None:
                    logger.info(f"Vehículo spawneado en {candidate.location}")
                    self.actor_list.append(self.vehicle)
                    break
                else:
                    logger.info(f"SpawnPoint2 no válido en {candidate.location}")
            else:
                logger.info(f"SpawnPoint no válido en {candidate.location}")
                self._destroy_actors()
        if self.vehicle is None:
            raise Exception("No se pudo spawnear el vehículo después de varios intentos.")


    def _attach_cameras(self):
        # Cámara delantera
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_width))
        camera_bp.set_attribute("image_size_y", str(self.image_height))
        camera_bp.set_attribute("fov", str(self.camera_fov))

        camera_transform = carla.Transform(carla.Location(x=1.6, y=0.0, z=1.7))
        try:
            self.rgb_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle,attachment_type=carla.AttachmentType.Rigid)
            self.actor_list.append(self.rgb_camera)
            self.rgb_camera.listen(lambda image: self._on_camera_image("rgb_front",image))
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara: {e}")
            raise

        # Cámara trasera
        blueprint_library = self.world.get_blueprint_library()
        camera_rear_bp = blueprint_library.find("sensor.camera.rgb")
        camera_rear_bp.set_attribute("image_size_x", str(self.image_width))
        camera_rear_bp.set_attribute("image_size_y", str(self.image_height))
        camera_rear_bp.set_attribute("fov", str(self.camera_fov))

        cam_rear_transform = carla.Transform(
            carla.Location(x=-1.6, y=0.0, z=1.7),
            carla.Rotation(yaw=180.0)
        )
        try:
            self.rgb_rear_camera = self.world.spawn_actor(camera_rear_bp, cam_rear_transform, attach_to=self.vehicle,attachment_type=carla.AttachmentType.Rigid)
            self.actor_list.append(self.rgb_rear_camera)
            self.rgb_rear_camera.listen(lambda image: self._on_camera_image("rgb_rear",image))
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara: {e}")
            raise

        # Lateral izquierda
        cam_left_bp = blueprint_library.find('sensor.camera.rgb')
        cam_left_bp.set_attribute('image_size_x', str(self.image_width))
        cam_left_bp.set_attribute('image_size_y', str(self.image_height))
        cam_left_bp.set_attribute('fov', str(self.camera_fov))
        cam_left_transform = carla.Transform(
            carla.Location(x=0.0, y=-0.6, z=1.7),
            carla.Rotation(yaw=-90)
        )
        self.cam_left = self.world.spawn_actor(
            cam_left_bp,
            cam_left_transform,
            attach_to=self.vehicle
        )
        try:
            self.actor_list.append(self.cam_left)
            self.cam_left.listen(lambda image: self._on_camera_image("rgb_left",image))
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara: {e}")
            raise

        # Lateral derecha
        cam_right_bp = blueprint_library.find('sensor.camera.rgb')
        cam_right_bp.set_attribute('image_size_x', str(self.image_width))
        cam_right_bp.set_attribute('image_size_y', str(self.image_height))
        cam_right_bp.set_attribute('fov', str(self.camera_fov))
        cam_right_transform = carla.Transform(
            carla.Location(x=0.0, y=0.6, z=1.7),
            carla.Rotation(yaw=90)
        )
        self.cam_right = self.world.spawn_actor(
            cam_right_bp,
            cam_right_transform,
            attach_to=self.vehicle
        )
        try:
            self.actor_list.append(self.cam_right)
            self.cam_right.listen(lambda image: self._on_camera_image("rgb_left",image))
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara: {e}")
            raise

    def _attach_depth_camera(self):
        # Cámara de profundidad
        blueprint_library = self.world.get_blueprint_library()
        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(self.image_width))
        depth_bp.set_attribute('image_size_y', str(self.image_height))
        depth_bp.set_attribute('fov', str(self.camera_fov))
        depth_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.depth_cam = self.world.spawn_actor(
            depth_bp,
            depth_transform,
            attach_to=self.vehicle
        )
        try:
            self.actor_list.append(self.depth_cam)
            self.depth_cam.listen(lambda image: self.camera_depth_callback("depth_cam",image))
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara: {e}")
            raise

    def _attach_semantic_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(self.image_width))
        seg_bp.set_attribute('image_size_y', str(self.image_height))
        seg_bp.set_attribute('fov', str(self.camera_fov))
        seg_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.seg_cam = self.world.spawn_actor(
            seg_bp,
            seg_transform,
            attach_to=self.vehicle
        )
        try:
            self.actor_list.append(self.seg_cam)
            self.seg_cam.listen(lambda image: self.camera_semantic_callback("seg_cam",image))
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara: {e}")
            raise

    def _attach_lidar_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '56000') 
        lidar_bp.set_attribute('rotation_frequency', '10')  

        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        self.lidar_sensor = self.world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        try:
            self.actor_list.append(self.lidar_sensor)
            self.lidar_sensor.listen(lambda image: self.process_lidar("lidar_sensor",image))
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara: {e}")
            raise

    def _attach_radar_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '35') 
        radar_bp.set_attribute('vertical_fov', '20')  
        radar_bp.set_attribute('range', '30')         

        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar_sensor = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.vehicle)
        try:
            self.actor_list.append(self.radar_sensor)
            self.radar_sensor.listen(lambda image: self.process_radar("radar_sensor",image))
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara: {e}")
            raise

    def _attach_gnss_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('noise_alt_stddev', '0.01')
        gnss_bp.set_attribute('noise_lat_stddev', '0.0001')
        gnss_bp.set_attribute('noise_lon_stddev', '0.0001')

        gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
        self.gnss_sensor = self.world.spawn_actor(gnss_bp, gnss_transform, attach_to=self.vehicle)
        try:
            self.actor_list.append(self.gnss_sensor)
            self.gnss_sensor.listen(lambda image: self.process_gnss("gnss_sensor",image))
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara: {e}")
            raise

    def _attach_imu_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        imu_bp = blueprint_library.find('sensor.other.imu')
        # Ajustar si quieres ruidos específicos
        imu_bp.set_attribute('noise_accel_stddev_x', '0.01')
        imu_bp.set_attribute('noise_gyro_bias_x', '0.001')
        # ...

        imu_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
        self.imu_sensor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        try:
            self.actor_list.append(self.imu_sensor)
            self.imu_sensor.listen(lambda image: self.process_imu("imu_sensor",image))
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara: {e}")
            raise


    def _attach_collision_sensor(self):
        """
        Adjunta un sensor de colisión para detectar choques.
        """
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))

        try:
            self.collision_sensor = self.world.spawn_actor(
                collision_bp,
                collision_transform,
                attach_to=self.vehicle
            )
            self.actor_list.append(self.collision_sensor)
            self.collision_sensor.listen(lambda event: self._on_collision(event))
            logger.info("Sensor de colisión adjuntado al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar el sensor de colisión: {e}")
            raise

    def _attach_lane_invasion_sensor(self):
        """
        Adjunta un sensor de invasión de carril para detectar desviaciones.
        """
        blueprint_library = self.world.get_blueprint_library()
        lane_invasion_bp = blueprint_library.find("sensor.other.lane_invasion")
        lane_invasion_transform = carla.Transform()

        try:
            self.lane_invasion_sensor = self.world.spawn_actor(
                lane_invasion_bp,
                lane_invasion_transform,
                attach_to=self.vehicle
            )
            self.actor_list.append(self.lane_invasion_sensor)
            self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))
            logger.info("Sensor de invasión de carril adjuntado al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar el sensor de invasión de carril: {e}")
            raise


    def _on_camera_image(self, sensor_name, image):
        """
        Callback de la cámara: convierte la imagen en un array NumPy.
        """
        try:
            image.convert(carla.ColorConverter.Raw)
            frame_id = image.frame
            sensor_dir = os.path.join("dataset", sensor_name)
            os.makedirs(sensor_dir, exist_ok=True)
            filename = os.path.join(sensor_dir, f"{frame_id:06d}.png")
            image.save_to_disk(filename, carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4)) 
            # Convertir a RGB (descartando el canal alpha)
            rgb_array = array[:, :, :3][:, :, ::-1]  # BGR->RGB
            self.image_data = rgb_array
        except Exception as e:
            logger.error(f"Error al procesar la imagen de la cámara: {e}")

    def camera_depth_callback(self, sensor_name, image):
        frame_id = image.frame
        sensor_dir = os.path.join("dataset", sensor_name)
        os.makedirs(sensor_dir, exist_ok=True)
        image.convert(carla.ColorConverter.Depth)
        
        filename = os.path.join(sensor_dir, f"{frame_id:06d}.png")
        image.save_to_disk(filename)

    def camera_semantic_callback(self, sensor_name, image):
        frame_id = image.frame
        sensor_dir = os.path.join("dataset", sensor_name)
        os.makedirs(sensor_dir, exist_ok=True)
        
        filename = os.path.join(sensor_dir, f"{frame_id:06d}.png")
        image.save_to_disk(filename, carla.ColorConverter.CityScapesPalette)
    
    
    def process_lidar(self, sensor_name: str,point_cloud: carla.LidarMeasurement):
        frame_id = point_cloud.frame
        sensor_dir = os.path.join("dataset", sensor_name)
        os.makedirs(sensor_dir, exist_ok=True)
        
        filename_ply = os.path.join(sensor_dir, f"{frame_id:06d}.ply")
        
        points = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        points = np.reshape(points, (int(points.shape[0]/4), 4))

        self.save_to_ply(points, filename_ply)

    def save_to_ply(self, points, filename):
        num_points = points.shape[0]
        ply_header = f"""ply
    format ascii 1.0
    element vertex {num_points}
    property float x
    property float y
    property float z
    property float intensity
    end_header
    """
        with open(filename, 'w') as f:
            f.write(ply_header)
            for i in range(num_points):
                x, y, z, intensity = points[i]
                f.write(f"{x} {y} {z} {intensity}\n")


    def process_radar(self,sensor_name,radar_data):
        frame_id = radar_data.frame
        sensor_dir = os.path.join("dataset", sensor_name)
        os.makedirs(sensor_dir, exist_ok=True)
        
        filename = os.path.join(sensor_dir, f"{frame_id:06d}.json")
        
        detections_list = []
        for detection in radar_data:
            det_dict = {
                "depth": detection.depth,
                "azimuth": detection.azimuth,
                "altitude": detection.altitude,
                "velocity": detection.velocity
            }
            detections_list.append(det_dict)
        
        with open(filename, 'w') as f:
            json.dump(detections_list, f, indent=2)

    def process_gnss(self,sensor_name,gnss_data):
        frame_id = gnss_data.frame
        sensor_dir = os.path.join("dataset", sensor_name)
        os.makedirs(sensor_dir, exist_ok=True)
        
        filename = os.path.join(sensor_dir, f"{frame_id:06d}.csv")
        
        # Simple CSV con cabecera
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "latitude", "longitude", "altitude"])
            writer.writerow([frame_id, gnss_data.latitude, gnss_data.longitude, gnss_data.altitude])

    def process_imu(self,sensor_name,imu_data):
        frame_id = imu_data.frame
        sensor_dir = os.path.join("dataset", sensor_name)
        os.makedirs(sensor_dir, exist_ok=True)
        
        filename = os.path.join(sensor_dir, f"{frame_id:06d}.csv")
        
        ax, ay, az = imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z
        gx, gy, gz = imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z
        compass = imu_data.compass
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "accel_x", "accel_y", "accel_z",
                            "gyro_x", "gyro_y", "gyro_z", "compass"])
            writer.writerow([frame_id, ax, ay, az, gx, gy, gz, compass])

    def record_control_callback(self,vehicle: carla.Vehicle, frame_id: int):
        control = vehicle.get_control()
        
        sensor_name = "control"
        sensor_dir = os.path.join("dataset", sensor_name)
        os.makedirs(sensor_dir, exist_ok=True)
        
        filename = os.path.join(sensor_dir, f"{frame_id:06d}.json")
        data_dict = {
            "frame": frame_id,
            "steer": control.steer,
            "throttle": control.throttle,
            "brake": control.brake,
            "hand_brake": control.hand_brake,
            "reverse": control.reverse,
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2)

    def _on_collision(self, event):
        """
        Callback del sensor de colisión: marca la bandera de colisión.
        """
        self.collision_occurred = True

    def _on_lane_invasion(self, event):
        """
        Callback del sensor de invasión de carril: incrementa el contador si se cruza una línea sólida.
        """
        self.lane_invasion = True
        
        crossed_solid = False
        crossed_broken = False
        
        for marking in event.crossed_lane_markings:
            if marking.type in [
                carla.LaneMarkingType.Solid,
                carla.LaneMarkingType.SolidSolid,
                carla.LaneMarkingType.SolidBroken,
                carla.LaneMarkingType.Curb
            ]:
                crossed_solid = True
                logger.info("Invasión de carril sólida detectada!")
                break  # No es necesario continuar si ya se detectó una línea sólida
            elif marking.type in [
                carla.LaneMarkingType.Broken,
                carla.LaneMarkingType.BrokenBroken,
                carla.LaneMarkingType.BottsDots,
                carla.LaneMarkingType.BrokenSolid,
            ]:
                crossed_broken = True
                logger.info("Invasión de carril discontinua detectada!")
                # No se hace break aquí si deseas detectar múltiples invasiones de carril en una misma iteración

        # Actualizar contadores fuera del bucle
        if crossed_solid:
            self.crossed_solid_line_count += 1
        else:
            self.crossed_solid_line_count = 0  # Resetear si no se ha cruzado una línea sólida

        if crossed_broken:
            self.crossed_broken_line = True
        else:
            self.crossed_broken_line = False



    def _get_observation(self):
        """
        Construye la observación actual.
        - Imagen de la cámara
        - Velocidad
        - Distancia al próximo waypoint
        """
        # Imagen
        if self.image_data is not None:
            image = self.image_data
        else:
            # Si aún no hay imagen, devolvemos una en negro
            image = np.zeros((120, 240, 3), dtype=np.uint8)

        # Velocidad lineal del vehículo
        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Distancia al siguiente waypoint
        if self.current_route_index < len(self.route_points):
            next_wp_loc = self.route_points[self.current_route_index]
            car_loc = self.vehicle.get_transform().location
            dist = self._euclidean_distance(car_loc, next_wp_loc)
        else:
            dist = 0.0

        # === 4. Ángulo hacia el waypoint vs. orientación del coche ===
        #     4.1 Calcula el ángulo del vector (coche->waypoint)
        dx = next_wp_loc.x - car_loc.x
        dy = next_wp_loc.y - car_loc.y
        # atan2 da el ángulo en radianes de [-pi, pi], lo convertimos a grados [-180..180]
        angle_to_wp = math.degrees(math.atan2(dy, dx))

        #     4.2 Yaw actual del coche (en grados, [-180..180])
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw

        #     4.3 Diferencia de ángulos (normalizar a [-180..180])
        angle_diff = angle_to_wp - vehicle_yaw
        # Truco para “encajar” en [-180..180]
        angle_diff = (angle_diff + 180) % 360 - 180  

        #     4.4 Normalizar a [-1, 1]
        angle_diff_normalized = angle_diff / 180.0

        obs = {
            "image": image,
            "velocity": np.array([speed], dtype=np.float32),
            "distance_wp": np.array([dist], dtype=np.float32),
            "angle_diff": np.array([angle_diff_normalized], dtype=np.float32)
        }
        return obs
    
    def _euclidean_distance(self, loc1, loc2):
        return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.z - loc2.z)**2)

    def _compute_reward(self, action, obs):
        """
        Calcula la recompensa del step:
        - Incentiva el progreso hacia adelante.
        - Penaliza colisiones y desviaciones del carril.
        - Recompensa mantener una velocidad óptima.
        - Penaliza giros bruscos y comportamientos ineficientes.
        """
        reward = 0.0

        distance_to_center = self._distance_to_lane_center()  # Método que estimes
        if distance_to_center < self.distance_2lane_center_tol:
            # Cuanto más cerca, mayor la recompensa
            reward += REWARD_CENTER_LINE * (1 - distance_to_center / self.distance_2lane_center_tol)

        # Penalización por colisión
        if self.collision_occurred:
            reward += PENALTY_COLLISION  # Penalización fuerte
            logger.info("Colisión detectada.")
        
        # Penalización proporcional por cruzar líneas sólidas
        if self.crossed_solid_line_count > 0:
            reward += PENALTY_OFF_TRACK
        else:
            waypoint = self.world.get_map().get_waypoint(
                self.vehicle.get_location(), 
                project_to_road=False
            )
            if waypoint is not None:
                if carla.LaneType.Driving != waypoint.lane_type:
                    reward += PENALTY_OFF_TRACK
            else:
                reward += PENALTY_OFF_TRACK

        if self.crossed_broken_line:
            reward += PENALTY_OFF_TRACK / 10 

        dist_wp = obs["distance_wp"][0]
        if dist_wp < 2:  # Si está muy cerca del waypoint
            reward += REWARD_WAYPOINT
            logger.info("Siguiente waypoint.")
            # Avanzar al siguiente waypoint
            self.current_route_index += 1
            # Si alcanzó el último waypoint, recompensa extra
            if self.current_route_index >= len(self.route_points):
                reward += REWARD_FINAL
                logger.info("wAYPOINT FINAL.")

        # 4. Penalización por el tiempo transcurrido (cada step)
        reward += PENALTY_TIME_STEP

        # 5. Recompensa/penalización por velocidad
        speed = obs["velocity"][0]
        if speed <= MAX_SAFE_SPEED:
            reward += REWARD_SPEED_FACTOR * speed  # pequeño incentivo
        else:
            # penalizar si excede la velocidad segura
            reward -= REWARD_SPEED_FACTOR * (speed - MAX_SAFE_SPEED)

        # 6. Penalización por conducción errática
        steer = float(action[0])
        accel = float(action[1])  # throttle
        # Penalizamos la diferencia con la acción anterior
        reward -= PENALTY_ERRATIC_STEER * abs(steer - self.prev_steer)   

        prev_dist = self.prev_distance_wp
        current_dist = obs["distance_wp"][0]
        delta_dist = prev_dist - current_dist
        if delta_dist > 0:
            # Recompensa por acercarse
            reward += PROGRESS_FACTOR * delta_dist
        else:
            # Penalización si se aleja
            reward += PROGRESS_FACTOR * delta_dist  # será negativo

        self.prev_distance_wp = current_dist

        elapsed_time = time.time() - self.episode_start_time
        # 2) Si excede el tiempo máximo
        if elapsed_time > self.max_episode_time:
            reward -= 40.0


        return reward


    def _check_done(self):
        elapsed_time = time.time() - self.episode_start_time

        # 1) Si hay colisión
        if self.collision_occurred:
            logger.info("Episodio terminado por colisión.")
            return True

        # 2) Si excede el tiempo máximo
        if elapsed_time > self.max_episode_time:
            logger.info("Episodio terminado por tiempo máximo alcanzado.")
            return True

        if self.crossed_solid_line_count > 0:
            logger.info("Episodio terminado por invadir carril contrario.")
            return True
        else:
            waypoint = self.world.get_map().get_waypoint(
                self.vehicle.get_location(), 
                project_to_road=None
            )
            if waypoint is not None:
                if carla.LaneType.Driving != waypoint.lane_type:
                    logger.info("Episodio terminado por invadir espacio de peaton.")
                    return True
            else:
                logger.info("Episodio terminado por invadir espacio de peaton.")
                return True


        # 3) Si se han alcanzado todos los waypoints
        if self.current_route_index >= len(self.route_points):
            logger.info("Episodio terminado: todos los waypoints de la ruta han sido alcanzados.")
            return True

        return False

    def _destroy_actors(self):
        """Destruye todos los actores (vehículo, sensores, etc.)"""
        logger.info("Destruyendo actores existentes.")
        for actor in self.actor_list:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception as e:
                    logger.warning(f"Error al destruir actor {actor.type_id}: {e}")
        self.actor_list = []
        self.collision_occurred = False
        if hasattr(self, 'lane_invasion_occurred'):
            del self.lane_invasion_occurred

    def _update_spectator(self):
        """
        Actualiza la posición y orientación del espectador para seguir al vehículo.
        """
        if self.vehicle is not None:
            spectator = self.world.get_spectator()
            vehicle_transform = self.vehicle.get_transform()
            
            # Calcula una posición detrás y por encima del vehículo
            offset_distance = 10.0  # Distancia detrás del vehículo
            offset_height = 5.0     # Altura sobre el vehículo
            forward_vector = vehicle_transform.get_forward_vector()
            
            # Calcula la nueva ubicación del espectador
            spectator_location = vehicle_transform.location + forward_vector * -offset_distance
            spectator_location.z += offset_height
            
            # Configura la rotación del espectador para mirar hacia el vehículo
            spectator_rotation = carla.Rotation(pitch=-10, yaw=vehicle_transform.rotation.yaw, roll=0)
            
            # Crea el transform para el espectador
            spectator_transform = carla.Transform(spectator_location, spectator_rotation)
            
            # Aplica el transform al espectador
            spectator.set_transform(spectator_transform)

if __name__ == "__main__":
    env = CarlaEnv(map="Town01",seed=42)
    env.collect_data_autopilot(num_steps=1, dataset_path="dataset")
    env.close()
    env = CarlaEnv(map="Town02",seed=49)
    env.collect_data_autopilot(num_steps=4000, dataset_path="dataset")
    env.close()
    env = CarlaEnv(map="Town03",seed=49)
    env.collect_data_autopilot(num_steps=4000, dataset_path="dataset")
    env.close()
