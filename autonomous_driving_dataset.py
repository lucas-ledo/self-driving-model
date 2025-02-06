import os
import json
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import open3d as o3d  # para cargar .ply (LiDAR)


class AutonomousDrivingDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sequence_length=3,
        transform=None,
        sensor_list=('depth_cam', 'rgb_front'),
        tabular_sensors=('gnss_sensor',),
        lidar_sensor='lidar_sensor',
        file_extension_dict=None,
        max_lidar_points=2048,
        step=5  # Paso entre índices
    ):
        """
        root_dir: carpeta raíz del dataset, con subcarpetas:
          - control/ (archivos JSON -> steer, throttle, brake)
          - gnss_sensor/ (archivos CSV)
          - imu_sensor/ (archivos CSV)
          - lidar_sensor/ (archivos .ply)
          - depth_cam/, rgb_front/ (archivos .png)
          etc.

        sequence_length: cuántos frames consecutivos cargar (para LSTM).
        transform: transformaciones para imágenes (Resize, ToTensor, etc.).
        sensor_list: lista con carpetas de imágenes.
        tabular_sensors: lista con carpetas de datos tabulares en CSV (ej. gnss_sensor, imu_sensor).
        lidar_sensor: nombre de la carpeta con nubes .ply.
        file_extension_dict: mapeo carpeta -> extensión de archivo.
        max_lidar_points: número máximo de puntos a muestrear de la nube LiDAR.
        step: incremento entre índices consecutivos en las secuencias.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.sensor_list = sensor_list
        self.tabular_sensors = tabular_sensors
        self.lidar_sensor = lidar_sensor
        self.max_lidar_points = max_lidar_points
        self.step = step  # Almacena el paso definido

        # Extensiones por defecto si no se reciben como parámetro
        if file_extension_dict is None:
            file_extension_dict = {
                'depth_cam': '.png',
                'rgb_front': '.png',
                'control': '.json',
                'gnss_sensor': '.csv',
                'lidar_sensor': '.ply'
            }
        self.file_extension_dict = file_extension_dict

        # 1) Obtener TODOS los índices que hay en la carpeta control
        control_path = os.path.join(self.root_dir, 'control')
        if not os.path.isdir(control_path):
            raise FileNotFoundError(f"No se encontró la carpeta 'control' en {control_path}")

        self.all_indices = []
        control_files = sorted(os.listdir(control_path))
        for f in control_files:
            if f.endswith(self.file_extension_dict['control']):
                # Ej: "000005.json" -> "000005" -> 5
                idx_str = f.replace(self.file_extension_dict['control'], '')
                self.all_indices.append(int(idx_str))
        self.all_indices = sorted(self.all_indices)

        if len(self.all_indices) == 0:
            raise ValueError(
                f"No se encontraron archivos de control con "
                f"extensión {self.file_extension_dict['control']} en {control_path}"
            )

        # 2) Encontrar sub-secuencias de índices con paso definido,
        #    de al menos self.sequence_length de longitud.
        self.sequences = []
        current_segment = [self.all_indices[0]]

        for i in range(1, len(self.all_indices)):
            prev_idx = self.all_indices[i - 1]
            current_idx = self.all_indices[i]

            if current_idx == prev_idx + self.step:
                # Son consecutivos según el paso definido
                current_segment.append(current_idx)
            else:
                # Se rompió la secuencia
                # Chequear si en 'current_segment' hay suficiente largo
                # para extraer sub-ventanas de longitud sequence_length
                if len(current_segment) >= self.sequence_length:
                    self._make_subwindows(current_segment)

                # Empezar un nuevo segmento
                current_segment = [current_idx]

        # 3) Manejar el último segmento tras salir del bucle
        if len(current_segment) >= self.sequence_length:
            self._make_subwindows(current_segment)

        # Si no se crearon sub-ventanas, se lanza una excepción para saberlo
        if len(self.sequences) == 0:
            raise ValueError(
                f"No fue posible formar secuencias de longitud {self.sequence_length} "
                "con el paso definido en los índices presentes en control/"
            )

    def _make_subwindows(self, segment):
        """
        Dado un segmento con índices espaciados por self.step, extrae todas las 'ventanas' de tamaño sequence_length.
        Ej: segment = [5,10,15,20], sequence_length=3 => sub-ventanas: [5,10,15], [10,15,20].
        Solo agrega la ventana si todos los archivos existen.
        """
        for start_i in range(len(segment) - self.sequence_length + 1):
            window = segment[start_i : start_i + self.sequence_length]
            if self._check_sequence_files_exist(window):
                self.sequences.append(window)
            else:
                # Opcional: Puedes imprimir o registrar las secuencias faltantes
                print(f"Secuencia {window} omitida por archivos faltantes.")

    def _check_sequence_files_exist(self, sequence_indices):
        """
        Verifica que todos los archivos necesarios existan para cada índice en la secuencia.
        """
        for idx in sequence_indices:
            # 1. Verificar archivos de imágenes
            for sensor in self.sensor_list:
                file_ext = self.file_extension_dict.get(sensor, '.png')
                file_name = f"{idx:06d}{file_ext}"
                img_path = os.path.join(self.root_dir, sensor, file_name)
                if not os.path.isfile(img_path):
                    return False

            # 2. Verificar archivos tabulares
            for sensor in self.tabular_sensors:
                ext = self.file_extension_dict.get(sensor, '.csv')
                file_name = f"{idx:06d}{ext}"
                path = os.path.join(self.root_dir, sensor, file_name)
                if not os.path.isfile(path):
                    print(f"Archivo faltante: {path}")
                    return False

            # 3. Verificar archivos LiDAR
            lidar_file_ext = self.file_extension_dict.get('lidar_sensor', '.ply')
            lidar_file_name = f"{idx:06d}{lidar_file_ext}"
            lidar_path = os.path.join(self.root_dir, self.lidar_sensor, lidar_file_name)
            if not os.path.isfile(lidar_path):
                return False

            # 4. Verificar archivos de control
            control_file_ext = self.file_extension_dict.get('control', '.json')
            control_file_name = f"{idx:06d}{control_file_ext}"
            control_path = os.path.join(self.root_dir, 'control', control_file_name)
            if not os.path.isfile(control_path):
                return False

        return True

    def __len__(self):
        """Número total de secuencias de longitud self.sequence_length."""
        return len(self.sequences)

    def _load_image(self, folder, idx):
        """Carga una imagen en formato .png y la convierte a tensor."""
        file_ext = self.file_extension_dict.get(folder, '.png')
        file_name = f"{idx:06d}{file_ext}"
        img_path = os.path.join(self.root_dir, folder, file_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"No existe la imagen: {img_path}")

        img = Image.open(img_path).convert('RGB')
        if self.transform and folder in self.transform:
            img = self.transform[folder](img)
        else:
            img = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        return img

    def _load_tabular(self, folder, idx):
        """Carga datos tabulares en formato CSV. Ignora la primera columna para gnss_sensor."""
        ext = self.file_extension_dict.get(folder, '.csv')
        file_name = f"{idx:06d}{ext}"
        path = os.path.join(self.root_dir, folder, file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No existe el CSV para {folder}: {path}")

        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames  # Obtiene el orden original de las columnas

            if fieldnames is None:
                raise ValueError(f"El CSV no tiene encabezados de columna: {path}")

            # Si es gnss_sensor, ignorar la primera columna
            if folder == 'gnss_sensor':
                fieldnames = fieldnames[1:]

            rows = list(reader)
            if len(rows) == 0:
                raise ValueError(f"El CSV está vacío: {path}")
            row = rows[0]

            vector = []
            for k in fieldnames:
                try:
                    vector.append(float(row[k]))
                except ValueError as e:
                    raise ValueError(f"Error al convertir la columna '{k}' a float en {path}: {e}")
                except KeyError as e:
                    raise KeyError(f"Columna faltante '{k}' en {path}: {e}")

        return torch.tensor(vector, dtype=torch.float32)



    def _load_lidar(self, idx):
        """Lee el archivo .ply y asegura que tenga max_lidar_points mediante muestreo o padding."""
        file_ext = self.file_extension_dict.get('lidar_sensor', '.ply')
        file_name = f"{idx:06d}{file_ext}"
        path = os.path.join(self.root_dir, self.lidar_sensor, file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No existe el archivo LiDAR: {path}")

        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)  # (N, 3)

        N = pts.shape[0]
        if N > self.max_lidar_points:
            # Muestrear aleatoriamente max_lidar_points
            choice = np.random.choice(N, self.max_lidar_points, replace=False)
            pts = pts[choice, :]
        elif N < self.max_lidar_points:
            # Padding con ceros si hay menos puntos
            padding = self.max_lidar_points - N
            pad_pts = np.zeros((padding, 3), dtype=np.float32)
            pts = np.vstack((pts, pad_pts))
        # Si N == max_lidar_points, no se hace nada

        return torch.tensor(pts, dtype=torch.float32)

    def _load_control(self, idx):
        """Lee el control .json con { 'steer':..., 'throttle':..., 'brake':... }."""
        file_ext = self.file_extension_dict.get('control', '.json')
        file_name = f"{idx:06d}{file_ext}"
        path = os.path.join(self.root_dir, 'control', file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No existe el control JSON: {path}")

        with open(path, 'r') as f:
            data = json.load(f)
        steer = float(data['steer'])
        throttle = float(data['throttle'])
        brake = float(data['brake'])
        return torch.tensor([steer, throttle, brake], dtype=torch.float32)

    def __getitem__(self, idx):
        """
        Carga la secuencia de índices consecutivos self.sequences[idx].
        Por ejemplo, si self.sequences[idx] = [5,10,15], se cargarán esos tres frames.
        """
        sequence_indices = self.sequences[idx]

        seq_imgs = {sensor: [] for sensor in self.sensor_list}
        seq_tabs = {sensor: [] for sensor in self.tabular_sensors}
        seq_lidar = []
        seq_controls = []

        for i in sequence_indices:
            # 1) Imágenes
            for sensor in self.sensor_list:
                try:
                    img = self._load_image(sensor, i)
                    seq_imgs[sensor].append(img)
                except FileNotFoundError:
                    # Saltar esta secuencia si falta alguna imagen
                    print(f"Imagen faltante para sensor {sensor} en índice {i}. Omitiendo secuencia.")
                    return None

            # 2) Datos tabulares
            for sensor in self.tabular_sensors:
                try:
                    tab = self._load_tabular(sensor, i)
                    seq_tabs[sensor].append(tab)
                except (FileNotFoundError, ValueError):
                    # Saltar esta secuencia si falta algún CSV o está vacío
                    print(f"Datos tabulares faltantes o vacíos para sensor {sensor} en índice {i}. Omitiendo secuencia.")
                    return None

            # 3) LiDAR
            try:
                lidar = self._load_lidar(i)
                seq_lidar.append(lidar)
            except FileNotFoundError:
                # Saltar esta secuencia si falta el archivo LiDAR
                print(f"Archivo LiDAR faltante en índice {i}. Omitiendo secuencia.")
                return None

            # 4) Control
            try:
                control = self._load_control(i)
                seq_controls.append(control)
            except FileNotFoundError:
                # Saltar esta secuencia si falta el archivo de control
                print(f"Archivo de control faltante en índice {i}. Omitiendo secuencia.")
                return None

        # Convertir listas a tensores
        for sensor in self.sensor_list:
            # (sequence_length, C, H, W)
            seq_imgs[sensor] = torch.stack(seq_imgs[sensor], dim=0)
        for sensor in self.tabular_sensors:
            # (sequence_length, num_features)
            seq_tabs[sensor] = torch.stack(seq_tabs[sensor], dim=0)
        # (sequence_length, max_lidar_points, 3)
        seq_lidar = torch.stack(seq_lidar, dim=0)
        # (sequence_length, 3)
        seq_controls = torch.stack(seq_controls, dim=0)

        return seq_imgs, seq_tabs, seq_lidar, seq_controls
