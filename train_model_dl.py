import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import r2_score
import torchvision.transforms as TT

# Asegúrate de importar tu modelo con las modificaciones anteriores
from multisensor_cross_attention_transformer import (
    MultiSensorCrossAttentionTransformer
)

from autonomous_driving_dataset import AutonomousDrivingDataset
from bigger_point_net import my_collate_fn  # Ajusta según tu estructura

torch.backends.cudnn.benchmark = True
############################
# EarlyStopping helper
############################
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Mejoró la pérdida de validación: {val_loss:.6f}')
        torch.save(model.state_dict(), 'checkpoint_chunks.pt')
        self.best_loss = val_loss


############################
# Función de entrenamiento
############################
def train_model_chunks(
    root_dir,
    num_epochs=200,
    batch_size=4,
    sequence_length=50,  
    chunk_size=16,       
    learning_rate=1e-4,
    weight_decay=1e-5,
    device='cuda',
    validation_split=0.2,
    patience=10
):
    # 1) Transforms por sensor
    transform_depth_cam = TT.Compose([
        TT.Resize((112, 112)),
        TT.Grayscale(num_output_channels=1),
        TT.ToTensor(),
    ])
    transform_rgb_front = TT.Compose([
        TT.Resize((112, 112)),
        TT.ToTensor(),
    ])

    # 2) Instanciar dataset
    dataset = AutonomousDrivingDataset(
        root_dir=root_dir,
        sequence_length=sequence_length,
        transform={
            'depth_cam': transform_depth_cam,
            'rgb_front': transform_rgb_front
        },
        sensor_list=('depth_cam','rgb_front'),
        tabular_sensors=('gnss_sensor',),
        lidar_sensor='lidar_sensor',
        max_lidar_points=2048
    )
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train size: {train_size}, Val size: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=my_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=my_collate_fn,
        pin_memory=True
    )

    # 3) Instanciar modelo
    tabular_input_dims = {'gnss_sensor': 3}
    model = MultiSensorCrossAttentionTransformer(
        sensor_list=('depth_cam','rgb_front'),
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
        temporal_num_layers=1,
        temporal_dim_feedforward=256,
        temporal_dropout=0.1,
        out_dim=3
    ).to(device)


    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # 4) Bucle de entrenamiento
    for epoch in range(num_epochs):
        start_time = time.time()

        # ---- Entrenamiento ----
        model.train()
        total_loss, total_r2, total_mae = 0.0, 0.0, 0.0
        count_batches = 0

        for batch_data in train_loader:
            if batch_data is None:
                continue
            batch_imgs, batch_tabs, batch_lidar, batch_ctrls = batch_data

            # Mover a GPU
            for sensor in batch_imgs:
                batch_imgs[sensor] = batch_imgs[sensor].to(device, non_blocking=True)
            for sensor in batch_tabs:
                batch_tabs[sensor] = batch_tabs[sensor].to(device, non_blocking=True)
            batch_ctrls = batch_ctrls.to(device, non_blocking=True)
            batch_lidar = torch.stack(batch_lidar, dim=0).to(device, non_blocking=True) 
            # => (B, T, num_points, 3)

            B, T, _ = batch_ctrls.shape  # => (B, T, 3)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                # ==> Forward completo, con chunking SOLO en la parte del transformer
                outputs = model(batch_imgs, batch_tabs, batch_lidar, chunk_size=chunk_size)
                loss = criterion(outputs, batch_ctrls)

            # Un único backward() por batch
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Métricas
            y_true = batch_ctrls.detach().cpu().numpy()
            y_pred = outputs.detach().cpu().numpy()
            batch_r2  = r2_score(y_true.reshape(-1, 3), y_pred.reshape(-1, 3))
            batch_mae = torch.mean(torch.abs(batch_ctrls - outputs)).item()

            total_loss += loss.item()
            total_r2   += batch_r2
            total_mae  += batch_mae
            count_batches += 1

        # Promedios
        avg_train_loss = total_loss / count_batches if count_batches > 0 else 0.0
        avg_train_r2   = total_r2   / count_batches if count_batches > 0 else 0.0
        avg_train_mae  = total_mae  / count_batches if count_batches > 0 else 0.0

        # ---- Validación ----
        model.eval()
        val_loss, val_r2, val_mae = 0.0, 0.0, 0.0
        val_batches = 0

        with torch.no_grad(), autocast(device_type='cuda'):
            for batch_data in val_loader:
                if batch_data is None:
                    continue
                batch_imgs, batch_tabs, batch_lidar, batch_ctrls = batch_data

                for sensor in batch_imgs:
                    batch_imgs[sensor] = batch_imgs[sensor].to(device, non_blocking=True)
                for sensor in batch_tabs:
                    batch_tabs[sensor] = batch_tabs[sensor].to(device, non_blocking=True)
                batch_ctrls = batch_ctrls.to(device, non_blocking=True)
                batch_lidar = torch.stack(batch_lidar, dim=0).to(device, non_blocking=True)

                outputs = model(batch_imgs, batch_tabs, batch_lidar, chunk_size=chunk_size)
                loss = criterion(outputs, batch_ctrls)

                y_true = batch_ctrls.detach().cpu().numpy()
                y_pred = outputs.detach().cpu().numpy()
                batch_r2  = r2_score(y_true.reshape(-1, 3), y_pred.reshape(-1, 3))
                batch_mae = torch.mean(torch.abs(batch_ctrls - outputs)).item()

                val_loss += loss.item()
                val_r2   += batch_r2
                val_mae  += batch_mae
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        avg_val_r2   = val_r2   / val_batches if val_batches > 0 else 0.0
        avg_val_mae  = val_mae  / val_batches if val_batches > 0 else 0.0

        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{num_epochs} - {elapsed:.1f}s] "
              f"Train Loss: {avg_train_loss:.4f} | R2: {avg_train_r2:.4f} | MAE: {avg_train_mae:.4f} || "
              f"Val Loss: {avg_val_loss:.4f} | R2: {avg_val_r2:.4f} | MAE: {avg_val_mae:.4f}")

        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("EarlyStopping activado")
            break

    # Cargar el mejor modelo
    model.load_state_dict(torch.load('checkpoint_chunks.pt'))
    print("Entrenamiento finalizado. Modelo óptimo cargado.")


if __name__ == "__main__":
    # Ejemplo de ejecución
    train_model_chunks(
        root_dir="./dataset",
        num_epochs=200,
        batch_size=4,
        sequence_length=30,
        chunk_size=8,
        learning_rate=1e-4,
        weight_decay=1e-5,
        device='cuda',
        validation_split=0.2,
        patience=20
    )
