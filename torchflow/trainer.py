"""
A Trainer class for managing the training and validation of PyTorch models.

Includes support for metrics, TensorBoard logging, and MLflow tracking.
@author: Muyiwa J. Obadara
@date: 2025-09-16
@license: MIT
@version: 1.0

"""
import os
import torch
import tqdm
import mlflow
import torchmetrics
from typing import Optional, Tuple
from pathlib import Path

# Create the models directory if it doesn't exist
Path("models").mkdir(parents=True, exist_ok=True)

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cpu',
                 metrics: Optional[torchmetrics.Metric] = None,
                 writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
                 mlflow_tracking: bool = False) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metrics = metrics
        self.writer = writer
        self.mlflow_tracking = mlflow_tracking

    
    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        running_loss = 0.0

        for inputs, targets in tqdm.tqdm(dataloader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.writer:
                step = self.optimizer.state_dict().get('step', 0)
                self.writer.add_scalar('Train/Loss', loss.item(), step)

            if self.mlflow_tracking:
                mlflow.log_metric('Train/Loss', loss.item(), step=step)
            
        avg_loss = running_loss / len(dataloader.dataset)
        return avg_loss
    

    def validate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, Optional[dict]]:
        self.model.eval()
        running_loss = 0.0

        if self.metrics:
            self.metrics.reset()

        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(dataloader, desc="Validation", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                if self.metrics:
                    self.metrics.update(outputs, targets)

        avg_loss = running_loss / len(dataloader.dataset)
        metric_results = self.metrics.compute() if self.metrics else None
        return avg_loss, metric_results
    

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            num_epochs: int = 5):
        for epoch in tqdm.tqdm(range(num_epochs), desc="Training", leave=False):
            train_loss = self.train_one_epoch(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            if val_loader:
                val_loss, val_metrics = self.validate(val_loader)
                print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
                if val_metrics:
                    for name, value in val_metrics.items():
                        print(f"    {name}: {value:.4f}")
                if self.writer:
                    step = self.optimizer.state_dict().get('step', 0)
                    self.writer.add_scalar('Val/Loss', val_loss, step)
                    if val_metrics:
                        for name, value in val_metrics.items():
                            self.writer.add_scalar(f'Val/{name}', value, step)
            if self.mlflow_tracking:
                step = self.optimizer.state_dict().get('step', 0)
                mlflow.log_metric('Train/Loss', train_loss, step=step)
                if val_loader:
                    mlflow.log_metric('Val/Loss', val_loss, step=step)
                    if val_metrics:
                        for name, value in val_metrics.items():
                            mlflow.log_metric(f'Val/{name}', value, step=step)
            self.save_model(f"models/model_epoch_{epoch+1}.pth")
        if self.writer:
            self.writer.flush()
            self.writer.close()

        return self

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        if self.mlflow_tracking:
            mlflow.pytorch.log_model(self.model, artifact_path=path)
            mlflow.log_artifact(path)
            mlflow.log_param("model_name", self.model.__class__.__name__)
            mlflow.log_param("model_version", self.model.__version__)
            mlflow.log_param("optimizer", self.optimizer.__class__.__name__)
            mlflow.log_param("criterion", self.criterion.__class__.__name__)
            if self.metrics:
                mlflow.log_param("metrics", [m.__class__.__name__ for m in self.metrics])
                for m in self.metrics:
                    mlflow.log_param(f"metric_{m.__class__.__name__}", m.__dict__)
            mlflow.log_param("device", self.device)
            if self.writer:
                mlflow.log_param("tensorboard_logging", True)
            else:
                mlflow.log_param("tensorboard_logging", False)
            mlflow.log_param("mlflow_tracking", self.mlflow_tracking)
            mlflow.log_param("model_parameters", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            mlflow.log_param("total_parameters", sum(p.numel() for p in self.model.parameters()))
            mlflow.log_param("optimizer_parameters", sum(p.numel() for p in self.optimizer.state_dict().values() if isinstance(p, torch.Tensor)))
            mlflow.log_param("criterion_parameters", sum(p.numel() for p in self.criterion.parameters() if p.requires_grad))
            mlflow.log_param("training_epochs", self.optimizer.state_dict().get('step', 0))
            mlflow.log_param("training_device", self.device)
            mlflow.log_param("training_batch_size", self.optimizer.state_dict().get('batch_size', 'N/A'))
            mlflow.log_param("training_learning_rate", self.optimizer.state_dict().get('lr', 'N/A'))
            mlflow.log_param("training_loss_function", self.criterion.__class__.__name__)
            mlflow.log_param("training_metrics", [m.__class__.__name__ for m in self.metrics] if self.metrics else 'N/A')
            mlflow.log_param("training_optimizer", self.optimizer.__class__.__name__)
