import gymnasium as gym
import numpy as np
import cv2
import random
import time
import carla
from carla import Location, Rotation
import collections
import math

import torch

from gymnasium import spaces, ObservationWrapper
from gymnasium.spaces import Box, Dict
import os
import logging

import pygame  # Para la visualización
from pygame.locals import QUIT

# Opcional: ocultar mensajes de pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REWARD_WAYPOINT = 20.0
REWARD_FINAL = 100.0
PENALTY_COLLISION = -40.0
PENALTY_OFF_TRACK = -40.0
PENALTY_TIME_STEP = 0.1    
PENALTY_ERRATIC_STEER = 0.5
PENALTY_ERRATIC_ACCEL = 0.01
PROGRESS_FACTOR = 1.0
REWARD_CENTER_LINE = 0.01

# Ejemplo de velocidad segura
MAX_SAFE_SPEED = 8.0  
REWARD_SPEED_FACTOR = 0.1


class CarlaEnv(gym.Env):
    """
    Entorno Gym para controlar un vehículo en CARLA en modo síncrono.
    Ahora con 4 sensores: RGB, Depth, GNSS y LiDAR.
    """

    def __init__(
        self,
        host="localhost",
        port=2000,
        world_timeout=10.0,
        render_display=False,
        image_width=240,
        image_height=120,
        max_episode_time=300.0,
        reward_collision=-400.0,
        reward_survival=1,
        frame_skip=1,
        seed=42,
        spectator_display=True,
        camera_fov=110,
        vehicle_model="vehicle.tesla.model3",
        enable_lane_following=True,
    ):
        super(CarlaEnv, self).__init__()

        self.seed(seed)
        self.host = host
        self.port = port
        self.world_timeout = world_timeout
        self.render_display = render_display
        self.spectator_display = spectator_display
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
        self.current_route_index = 0
        self.route_threshold = 2.0
        self.prev_distance_wp = 0
        self.distance_2lane_center_tol = 1

        self.prev_steer = 0
        self.prev_accel = 0

        # Variables de sensores
        self.rgb_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.depth_image = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        # GNSS: lat, lon, alt
        self.gnss_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # LiDAR: almacenaremos hasta N puntos. Ajustar N según tu preferencia.
        self.max_lidar_points = 10000  
        self.lidar_data = np.zeros((self.max_lidar_points, 3), dtype=np.float32)
        self.num_lidar_points = 0

        # Espacio de acciones: [steer, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float32
        )

        # Espacio de observaciones como Dict
        # Ajusta los rangos y shapes según tu caso.
        self.observation_space = spaces.Dict({
            "rgb_front": spaces.Box(
                low=0, high=255,
                shape=(self.image_height, self.image_width, 3),
                dtype=np.uint8
            ),
            "depth_cam": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.image_height, self.image_width),  # o (self.image_height, self.image_width, 1)
                dtype=np.float32
            ),
            "gnss": spaces.Box(
                low=-1e7, high=1e7,
                shape=(3,),
                dtype=np.float32
            ),
            "lidar": spaces.Box(
                low=-1e5, high=1e5,
                shape=(self.max_lidar_points, 3),
                dtype=np.float32
            ),
        })

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
        self._enable_synchronous_mode()

        # Variables de estado
        self.actor_list = []
        self.vehicle = None
        self.rgb_camera = None
        self.depth_camera = None
        self.gnss_sensor = None
        self.lidar_sensor = None
        self.collision_sensor = None

        self.collision_occurred = False
        self.episode_start_time = time.time()
        self.current_velocity = 0.0
        self.action_history = collections.deque(maxlen=3)
        self.crossed_solid_line = False
        self.crossed_solid_line_count = 0


    def seed(self, seed=None):
        """Fija la semilla para reproducibilidad."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _connect_to_carla(self, max_retries=5, delay=2):
        """Conecta con el servidor CARLA con reintentos."""
        attempts = 0
        while attempts < max_retries:
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(self.world_timeout)
                self.world = self.client.load_world("Town04")
                if self.world is not None:
                    logger.info(f"Conectado a CARLA en {self.host}:{self.port}")
                    return
            except Exception as e:
                logger.warning(f"Intento {attempts + 1}/{max_retries} de conexión fallido: {e}")
                time.sleep(delay)
                attempts += 1
        raise ConnectionError(f"No se pudo conectar a CARLA después de {max_retries} intentos.")

    def _enable_synchronous_mode(self):
        """Activa el modo síncrono en el mundo de CARLA."""
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1  # ~10 FPS
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
            self._spawn_vehicle(spawn_transform=spawn_point)
        except Exception as e:
            logger.error(f"Error al spawnear el vehículo: {e}")
            time.sleep(0.5)
            self.spawn_area_free()

    def reset(self, seed=None, options=None):
        logger.info("Reiniciando el entorno.")
        if seed is not None:
            self.seed(seed)

        if self.route_points is None:
            self.route_points, spawn_point = self._generate_predefined_route()

        # Limpiar actores
        self._destroy_actors()
        logger.info(f"Paso {self.current_step}")

        # Reset de banderas y contadores
        self.collision_occurred = False
        self.crossed_solid_line = False
        self.crossed_solid_line_count = 0
        self.episode_start_time = time.time()
        self.current_route_index = 0

        if len(self.route_points) == 0:
            raise ValueError("No hay puntos de ruta definidos.")

        first_location = self.route_points[0]
        self.spawn_area_free()

        # Adjuntar sensores
        self._attach_rgb_camera()
        self._attach_depth_camera()
        self._attach_collision_sensor()
        self._attach_gnss_sensor()
        self._attach_lidar_sensor()
        if self.enable_lane_following:
            self._attach_lane_invasion_sensor()

        # Hacer unos ticks para que arranquen los sensores
        for _ in range(5):
            self.world.tick()

        # Reset de velocidad
        self.current_velocity = self.vehicle.get_velocity().length()

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        self.action_history.append(action)
        self.current_step += 1

        steer, throttle, brake = action

        if throttle > 0.0:
            if throttle > brake:
                throttle = throttle - brake
                brake = 0.0
                

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.hand_brake = False
        control.manual_gear_shift = False
        control.engine_on = True


        try:
            self.vehicle.apply_control(control)
        except Exception as e:
            logger.error(f"Error al aplicar control al vehículo: {e}")

        for _ in range(self.frame_skip):
            try:
                self.world.tick()
            except Exception as e:
                logger.error(f"Error durante el tick del mundo: {e}")
                break

        obs = self._get_observation()
        velocity = self.vehicle.get_velocity()
        self.current_velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        transform = self.vehicle.get_transform()


        # Comprobar avance por waypoints
        vehicle_loc = self.vehicle.get_transform().location
        if self.current_route_index < len(self.route_points):
            target_loc = self.route_points[self.current_route_index]
            dist_to_target = vehicle_loc.distance(target_loc)
            if dist_to_target < self.route_threshold:
                self.current_route_index += 1

        reward = self._compute_reward(action)
        done = self._check_done()
        truncated = False

        info = {
            "velocity": self.current_velocity,
            "collision": self.collision_occurred,
            "crossed_solid_line_count": self.crossed_solid_line_count,
            "elapsed_time": time.time() - self.episode_start_time
        }
        if self.enable_lane_following:
            info["lane_invasion"] = getattr(self, 'lane_invasion_occurred', False)

        if self.spectator_display:
            self._update_spectator()

        if self.render_display:
            self.render()

        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        if self.render_display:
            try:
                image = self.rgb_image
                image = cv2.resize(image, (self.image_width, self.image_height))
                surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
                self.screen.blit(surface, (0, 0))
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.close()
                self.clock.tick(60)
            except Exception as e:
                logger.error(f"Error al renderizar la imagen: {e}")

    def close(self):
        logger.info("Cerrando el entorno.")
        self._destroy_actors()
        self._disable_synchronous_mode()
        if self.render_display:
            pygame.quit()

    def _euclidean_distance(self, loc1, loc2):
        return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.z - loc2.z)**2)
    
    def _distance_to_lane_center(self) -> float:
        """
        Retorna la distancia (en metros) del vehículo al centro del carril más cercano.
        """
        if not self.vehicle:
            return 0.0  

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
    # ----------------------------------------
    #   GENERACIÓN DE RUTA Y SPAWN
    # ----------------------------------------
    def _generate_predefined_route(self):
        route = []
        carla_map = self.world.get_map()
        spawn_points = carla_map.get_spawn_points()
        start_spawn = self.np_random.choice(spawn_points)
        start_waypoint = carla_map.get_waypoint(start_spawn.location)
        distance_between_points = 3.0
        num_points = 100
        route.append(start_waypoint.transform.location)

        current_waypoint = start_waypoint
        for i in range(num_points - 1):
            next_wp = current_waypoint.next(distance_between_points)[0]
            route.append(next_wp.transform.location)
            current_waypoint = next_wp

        for i in range(len(route) - 1):
            self.world.debug.draw_line(
                route[i],
                route[i+1],
                thickness=0.1,
                color=carla.Color(r=255, g=0, b=0),
                life_time=300
            )

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
                    logger.info(f"SpawnPoint no válido en {candidate.location}")
            else:
                logger.info(f"Zona ocupada en {candidate.location}")
                self._destroy_actors()
        if self.vehicle is None:
            raise Exception("No se pudo spawnear el vehículo después de varios intentos.")

    # ----------------------------------------
    #   ADJUNTAR SENSORES
    # ----------------------------------------
    def _attach_rgb_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_width))
        camera_bp.set_attribute("image_size_y", str(self.image_height))
        camera_bp.set_attribute("fov", str(self.camera_fov))

        camera_transform = carla.Transform(carla.Location(x=1.6, y=0.0, z=1.7))
        try:
            self.rgb_camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            self.actor_list.append(self.rgb_camera)
            self.rgb_camera.listen(self._on_rgb_image)
            logger.info("Cámara RGB adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara RGB: {e}")
            raise

    def _attach_depth_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        depth_bp = blueprint_library.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", str(self.image_width))
        depth_bp.set_attribute("image_size_y", str(self.image_height))
        depth_bp.set_attribute("fov", str(self.camera_fov))

        depth_transform = carla.Transform(carla.Location(x=1.6, y=0.0, z=1.7))
        try:
            self.depth_camera = self.world.spawn_actor(
                depth_bp, depth_transform, attach_to=self.vehicle
            )
            self.actor_list.append(self.depth_camera)
            self.depth_camera.listen(self._on_depth_image)
            logger.info("Cámara de profundidad adjuntada al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar la cámara de profundidad: {e}")
            raise

    def _attach_gnss_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        gnss_bp = blueprint_library.find("sensor.other.gnss")
        gnss_bp.set_attribute("sensor_tick", "0.1")

        gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
        try:
            self.gnss_sensor = self.world.spawn_actor(
                gnss_bp, gnss_transform, attach_to=self.vehicle
            )
            self.actor_list.append(self.gnss_sensor)
            self.gnss_sensor.listen(self._on_gnss_event)
            logger.info("Sensor GNSS adjuntado al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar el sensor GNSS: {e}")
            raise

    def _attach_lidar_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        # Ajustar parámetros del LiDAR:
        lidar_bp.set_attribute("range", "50.0")
        lidar_bp.set_attribute("rotation_frequency", "10.0")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("points_per_second", "56000") 

        lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.5))
        try:
            self.lidar_sensor = self.world.spawn_actor(
                lidar_bp, lidar_transform, attach_to=self.vehicle
            )
            self.actor_list.append(self.lidar_sensor)
            self.lidar_sensor.listen(self._on_lidar_event)
            logger.info("Sensor LiDAR adjuntado al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar el sensor LiDAR: {e}")
            raise

    def _attach_collision_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        try:
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, collision_transform, attach_to=self.vehicle
            )
            self.actor_list.append(self.collision_sensor)
            self.collision_sensor.listen(self._on_collision)
            logger.info("Sensor de colisión adjuntado al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar el sensor de colisión: {e}")
            raise

    def _attach_lane_invasion_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        lane_invasion_bp = blueprint_library.find("sensor.other.lane_invasion")
        lane_invasion_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        try:
            self.lane_invasion_sensor = self.world.spawn_actor(
                lane_invasion_bp, lane_invasion_transform,
                attach_to=self.vehicle
            )
            self.actor_list.append(self.lane_invasion_sensor)
            self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))
            logger.info("Sensor de invasión de carril adjuntado al vehículo.")
        except Exception as e:
            logger.error(f"Error al adjuntar el sensor de invasión de carril: {e}")
            raise

    # ----------------------------------------
    #   CALLBACKS DE SENSORES
    # ----------------------------------------
    def _on_rgb_image(self, image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # BGRA
            rgb_array = array[:, :, :3][:, :, ::-1]  # BGR->RGB
            self.rgb_image = rgb_array
        except Exception as e:
            logger.error(f"Error al procesar la imagen de la cámara RGB: {e}")

    def _on_depth_image(self, image):
        """
        La cámara de profundidad en CARLA da valores en [0..1000] o similares
        (dependiendo de la configuración). Normalizamos a [0..1].
        """
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # BGRA
            depth_meters = array[:, :, 2] + array[:, :, 1] * 256 + array[:, :, 0] * 256 * 256
            depth_meters = depth_meters / 16777215.0  # Normalizar a [0..1]
            self.depth_image = depth_meters.astype(np.float32)
        except Exception as e:
            logger.error(f"Error al procesar la imagen de la cámara Depth: {e}")

    def _on_gnss_event(self, gnss):
        try:
            self.gnss_data = np.array(
                [gnss.latitude, gnss.longitude, gnss.altitude],
                dtype=np.float32
            )
        except Exception as e:
            logger.error(f"Error al procesar dato GNSS: {e}")

    def _on_lidar_event(self, lidar_data):
        """
        El LiDAR retorna una nube de puntos en formato float32 (x, y, z, intensidad).
        """
        try:
            pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
            pts = pts.reshape((-1, 4))[:,:3]  # [x, y, z, intensity]
            
            # Si hay más puntos que el máximo, recortamos
            max_pts = min(self.max_lidar_points, pts.shape[0])
            self.lidar_data[:max_pts] = pts[:max_pts]
            self.num_lidar_points = max_pts

            # Si hay menos puntos, rellenamos con ceros en la parte sobrante
            if max_pts < self.max_lidar_points:
                self.lidar_data[max_pts:] = 0.0

        except Exception as e:
            logger.error(f"Error al procesar nube de puntos LiDAR: {e}")

    def _on_collision(self, event):
        self.collision_occurred = True
        logger.info("¡Colisión detectada!")

    def _on_lane_invasion(self, event):
        crossed_solid = False
        for marking in event.crossed_lane_markings:
            if marking.type in [
                carla.LaneMarkingType.Solid,
                carla.LaneMarkingType.SolidSolid,
                carla.LaneMarkingType.SolidBroken,
                carla.LaneMarkingType.Curb
            ]:
                crossed_solid = True
                logger.info("Invasión de carril sólida detectada!")
                break

        if crossed_solid:
            self.crossed_solid_line_count = 3
            self.crossed_solid_line = True
        else:
            self.crossed_solid_line = False

    # ----------------------------------------
    #   OBTENER OBSERVACIÓN
    # ----------------------------------------
    def _get_observation(self):
        """
        Devuelve un diccionario con los cuatro sensores:
          - rgb_front
          - depth_cam
          - gnss
          - lidar
        """
        # Copia para evitar problemas si se modifica asincrónicamente
        rgb_copy = self.rgb_image.copy()
        depth_copy = self.depth_image.copy()
        gnss_copy = self.gnss_data.copy()

        # Para LiDAR, puedes quedarte con self.num_lidar_points
        # y recortar solo la parte válida
        lidar_copy = self.lidar_data.copy()

        return {
            "rgb_front": rgb_copy,
            "depth_cam": depth_copy,
            "gnss": gnss_copy,
            "lidar": lidar_copy,
        }

    # ----------------------------------------
    #   RECOMPENSA Y DONE
    # ----------------------------------------
    def _compute_reward(self, action):
        """
        Calcula la recompensa del step:
        - Incentiva el progreso hacia adelante.
        - Penaliza colisiones y desviaciones del carril.
        - Recompensa mantener una velocidad óptima.
        - Penaliza giros bruscos y comportamientos ineficientes.
        """
        reward = 0.0

        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Distancia al siguiente waypoint
        if self.current_route_index < len(self.route_points):
            next_wp_loc = self.route_points[self.current_route_index]
            car_loc = self.vehicle.get_transform().location
            dist = self._euclidean_distance(car_loc, next_wp_loc)
        else:
            dist = 0.0

        distance_to_center = self._distance_to_lane_center() 
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
            


        dist_wp = dist
        print("Distancia al waypoint: ", dist_wp)
        if dist_wp < 4:  # Si está muy cerca del waypoint
            reward += REWARD_WAYPOINT
            logger.info("Siguiente waypoint.")
            # Avanzar al siguiente waypoint
            self.current_route_index += 1
            # Si alcanzó el último waypoint, recompensa extra
            if self.current_route_index >= len(self.route_points):
                reward += REWARD_FINAL
                logger.info("wAYPOINT FINAL.")

        # 4. Penalización por el tiempo transcurrido (cada step)
        reward -= PENALTY_TIME_STEP

        # 5. Recompensa/penalización por velocidad
        speed = speed
        if speed <= MAX_SAFE_SPEED:
            reward += REWARD_SPEED_FACTOR * speed  # pequeño incentivo
        else:
            # penalizar si excede la velocidad segura
            reward -= REWARD_SPEED_FACTOR * (speed - MAX_SAFE_SPEED)

        if(speed < 0.8):
            reward -= 1.0

        # 6. Penalización por conducción errática
        steer = float(action[0])
        accel = float(action[1])  # throttle
        # Penalizamos la diferencia con la acción anterior
        reward -= PENALTY_ERRATIC_STEER * abs(steer - self.prev_steer)   

        self.prev_steer = steer
        self.prev_accel = accel

        prev_dist = self.prev_distance_wp
        current_dist = dist
        delta_dist = prev_dist - current_dist
        if delta_dist > 0:
            # Recompensa por acercarse
            reward += PROGRESS_FACTOR * delta_dist
        else:
            # Penalización si se aleja
            if delta_dist > 10:
                delta_dist = 10
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
        if self.vehicle is not None:
            spectator = self.world.get_spectator()
            vehicle_transform = self.vehicle.get_transform()
            offset_distance = 10.0
            offset_height = 5.0
            forward_vector = vehicle_transform.get_forward_vector()
            spectator_location = vehicle_transform.location + forward_vector * -offset_distance
            spectator_location.z += offset_height
            spectator_rotation = carla.Rotation(
                pitch=-10,
                yaw=vehicle_transform.rotation.yaw,
                roll=0
            )
            spectator_transform = carla.Transform(spectator_location, spectator_rotation)
            spectator.set_transform(spectator_transform)
