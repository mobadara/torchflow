"""
A Trainer class for managing the training and validation of PyTorch models.

Includes support for metrics, TensorBoard logging, and MLflow tracking.
"""

import torch
import tqdm
try:
    import mlflow
except Exception:
    mlflow = None
try:
    import torchmetrics
except Exception:
    torchmetrics = None
import logging
from typing import Optional, Tuple, Union, Any, Iterable
from pathlib import Path
try:
    from torch.utils.tensorboard.writer import SummaryWriter
except Exception:
    # Don't import tensorboard SummaryWriter here; Trainer only accepts a writer
    # instance via the `writer` argument and checks for add_scalar at runtime.
    SummaryWriter = None

# Create the models directory if it doesn't exist

Path("models").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Trainer:
    """
    Trainer class for managing the training and validation of PyTorch models.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str, optional): Device to run the training on. Defaults to 'cpu'.
        metrics (torchmetrics.Metric or list, optional): Metrics for evaluation. Can be a single metric or a list of metrics. Defaults to None.
        writer (SummaryWriter, optional): TensorBoard writer. Must implement the TensorBoard SummaryWriter API. Defaults to None.
        mlflow_tracking (bool, optional): Whether to use MLflow for tracking. Defaults to False.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cpu',
                 metrics: Optional[Any] = None,
                 writer: Optional[Any] = None,
                 mlflow_tracking: bool = False,
                 callbacks: Optional[Iterable[Any]] = None) -> None:
        """
        Initialize the Trainer with model, criterion, optimizer, device, metrics, and logging options.
        Args:
            model (torch.nn.Module): The model to be trained.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str, optional): Device to run the training on. Defaults to 'cpu'.
            metrics (torchmetrics.Metric or list, optional): Metrics for evaluation. Can be a single metric or a list of metrics. Defaults to None.
            writer (SummaryWriter, optional): TensorBoard writer. Must implement the TensorBoard SummaryWriter API. Defaults to None.
            mlflow_tracking (bool, optional): Whether to use MLflow for tracking. Defaults to False.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metrics = metrics
        self.writer = writer
        self.mlflow_tracking = mlflow_tracking
        # callbacks should be an iterable of callback instances
        self.callbacks = list(callbacks) if callbacks is not None else []
        # flag that can be set by callbacks to stop training early
        self.stop_training = False

    
    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): The data loader for the training data.

        Returns:
            float: The average loss for the epoch.

        Raises:
            RuntimeError: If there is an issue during training.

        Notes:
            This method sets the model to training mode, iterates over the data loader,
            computes the loss, performs backpropagation, and updates the model parameters.
        """
        self.model.train()
        train_loss = 0.0
        total_samples = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Training", leave=False)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            total_samples += batch_size

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            step = batch_idx
            if self.writer and hasattr(self.writer, 'add_scalar'):
                self.writer.add_scalar('Train/Loss', loss.item(), step)
            if self.mlflow_tracking:
                mlflow.log_metric('Train/Loss', loss.item(), step=step)

            # callbacks: end of batch
            for cb in self.callbacks:
                try:
                    cb.on_batch_end(self, batch_idx, {'loss': loss.item()})
                except Exception:
                    pass

            if getattr(self, 'stop_training', False):
                break

        avg_loss = train_loss / total_samples if total_samples > 0 else float('nan')
        return avg_loss


    def validate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, Optional[dict]]:
        """
        Validate the model.

        Args:
            dataloader (torch.utils.data.DataLoader): The data loader for the validation data.

        Returns:
            Tuple[float, Optional[dict]]: The average loss and optional metrics for the validation data.

        Raises:
            RuntimeError: If there is an issue during validation.

        Notes:
            This method sets the model to evaluation mode, iterates over the data loader,
            computes the loss and metrics without updating model parameters.
        """
        self.model.eval()
        val_loss = 0.0
        total_samples = 0
        # Support both single and multiple metrics
        metrics = self.metrics
        if metrics:
            if isinstance(metrics, list):
                for m in metrics:
                    m.reset()
            else:
                metrics.reset()

        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(dataloader, desc="Validation", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                total_samples += batch_size
                if metrics:
                    if isinstance(metrics, list):
                        for m in metrics:
                            m.update(outputs, targets)
                    else:
                        metrics.update(outputs, targets)

        # compute average loss and metrics
        avg_loss = val_loss / total_samples if total_samples > 0 else float('nan')
        val_metrics = None
        if metrics:
            if isinstance(metrics, list):
                val_metrics = {m.__class__.__name__: m.compute() for m in metrics}
            else:
                val_metrics = {metrics.__class__.__name__: metrics.compute()}

        logs = {'val_loss': avg_loss} if total_samples > 0 else {}
        if val_metrics:
            logs.update(val_metrics)

        for cb in self.callbacks:
            try:
                cb.on_validation_end(self, 0, logs)
            except Exception:
                pass
        if metrics:
            if isinstance(metrics, list):
                metric_results = {m.__class__.__name__: m.compute() for m in metrics}
            else:
                metric_results = {metrics.__class__.__name__: metrics.compute()}
        else:
            metric_results = None
        return avg_loss, metric_results

    def train(self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 5) -> Tuple[torch.nn.Module, dict, 'Trainer']:
        """
        Fit the model to the training data and validate on the validation data.

        Args:
            train_loader (torch.utils.data.DataLoader): The data loader for the training data.
            val_loader (Optional[torch.utils.data.DataLoader], optional): The data loader for the validation data. Defaults to None.
            num_epochs (int, optional): The number of epochs to train. Defaults to 5.

        Returns:
            Tuple[torch.nn.Module, dict, Trainer]: The trained model, training history, and the Trainer instance.
        
        Raises:
            RuntimeError: If there is an issue during training or validation.

        Notes:
            This method manages the overall training process, including logging to TensorBoard and MLflow if enabled.
        """
        history = {'train_loss': [], 'val_loss': []}
        # notify callbacks training is starting
        for cb in self.callbacks:
            try:
                cb.on_train_begin(self)
            except Exception:
                pass

        for epoch in tqdm.tqdm(range(num_epochs), desc="Training", leave=False):
            if getattr(self, 'stop_training', False):
                break
            # epoch begin callbacks
            for cb in self.callbacks:
                try:
                    cb.on_epoch_begin(self, epoch)
                except Exception:
                    pass

            train_loss = self.train_one_epoch(train_loader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            history['train_loss'].append(train_loss)
            val_loss = None
            val_metrics = None
            if val_loader:
                val_loss, val_metrics = self.validate(val_loader)
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
                history['val_loss'].append(val_loss)
                if val_metrics:
                    for name, value in val_metrics.items():
                        logging.info(f"    {name}: {value:.4f}")
                        history.setdefault('val_' + name, []).append(value)
                if self.writer and hasattr(self.writer, 'add_scalar'):
                    self.writer.add_scalar('Val/Loss', val_loss, epoch)
                    if val_metrics:
                        for name, value in val_metrics.items():
                            self.writer.add_scalar(f'Val/{name}', value, epoch)

            # epoch end callbacks
            logs = {'train_loss': train_loss}
            if val_loss is not None:
                logs['val_loss'] = val_loss
            if val_metrics:
                logs.update(val_metrics)
            for cb in self.callbacks:
                try:
                    cb.on_epoch_end(self, epoch, logs)
                except Exception:
                    pass

            if self.mlflow_tracking:
                mlflow.log_metric('Train/Loss', train_loss, step=epoch)
                if val_loader and val_loss is not None:
                    mlflow.log_metric('Val/Loss', val_loss, step=epoch)
                    if val_metrics:
                        for name, value in val_metrics.items():
                            mlflow.log_metric(f'Val/{name}', value, step=epoch)
            # allow callbacks to stop training after epoch
            if getattr(self, 'stop_training', False):
                break

        # notify callbacks training ended
        for cb in self.callbacks:
            try:
                cb.on_train_end(self)
            except Exception:
                pass
        if self.writer:
            if hasattr(self.writer, 'flush'):
                self.writer.flush()
            if hasattr(self.writer, 'close'):
                self.writer.close()
        return self.model, history, self

    def save_model(self, path: str) -> None:
        """
        Save the model to the specified path and log to MLflow if enabled.
        Args:
            path (str): The path to save the model.
        Returns:
            None
        """
        torch.save(self.model.state_dict(), path)
        if self.mlflow_tracking:
            # Only use mlflow.log_artifact for model file
            mlflow.log_artifact(path)
            mlflow.log_param("model_name", self.model.__class__.__name__)
            model_version = getattr(self.model, "__version__", "N/A")
            mlflow.log_param("model_version", model_version)
            mlflow.log_param("optimizer", self.optimizer.__class__.__name__)
            mlflow.log_param("criterion", self.criterion.__class__.__name__)
            if self.metrics:
                if isinstance(self.metrics, list):
                    mlflow.log_param("metrics", [m.__class__.__name__ for m in self.metrics])
                    for m in self.metrics:
                        mlflow.log_param(f"metric_{m.__class__.__name__}", str(m.__dict__))
                else:
                    mlflow.log_param("metrics", self.metrics.__class__.__name__)
                    mlflow.log_param(f"metric_{self.metrics.__class__.__name__}", str(self.metrics.__dict__))
            mlflow.log_param("device", self.device)
            mlflow.log_param("tensorboard_logging", bool(self.writer))
            mlflow.log_param("mlflow_tracking", self.mlflow_tracking)
            mlflow.log_param("model_parameters", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            mlflow.log_param("total_parameters", sum(p.numel() for p in self.model.parameters()))
            mlflow.log_param("optimizer_parameters", sum(p.numel() for p in self.optimizer.state_dict().values() if isinstance(p, torch.Tensor)))
            
            # Criterion parameters if available
            if hasattr(self.criterion, 'parameters'):
                mlflow.log_param("criterion_parameters", sum(p.numel() for p in self.criterion.parameters() if p.requires_grad))
            mlflow.log_param("training_loss_function", self.criterion.__class__.__name__)
            mlflow.log_param("training_optimizer", self.optimizer.__class__.__name__)

            # Log optimizer hyperparameters if available
            if hasattr(self.optimizer, 'defaults'):
                for k, v in self.optimizer.defaults.items():
                    mlflow.log_param(f"optimizer_{k}", v)
