"""
Callbacks for training lifecycle events.

Provides a base Callback class and a couple of useful implementations
like EarlyStopping and ModelCheckpoint. These are designed to be simple
and lightweight so they can be used in unit tests and examples.
"""
from __future__ import annotations
from typing import Optional, Callable, Any
import os
import torch
import csv
from typing import Iterable, Sequence
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def make_summary_writer(log_dir: Optional[str] = None) -> Optional[Any]:
    """Safely create a SummaryWriter if tensorboard is available.

    Returns None when tensorboard or its dependencies are not installed.
    """
    if SummaryWriter is None:
        return None
    try:
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return None


class Callback:
    """
    Base callback. Subclass and override the desired methods.

    Methods receive the trainer instance (self) or context information
    so callbacks can control training (for example, stopping early).
    """
    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[dict] = None) -> None:
        pass

    def on_batch_end(self, trainer: Any, batch: int, logs: Optional[dict] = None) -> None:
        pass

    def on_validation_end(self, trainer: Any, epoch: int, logs: Optional[dict] = None) -> None:
        pass


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Args:
        monitor: metric name to monitor (e.g. 'val_loss').
        patience: epochs with no improvement after which training will be stopped.
        min_delta: minimum change to qualify as improvement.
        mode: 'min' or 'max' depending whether lower is better.
    """
    def __init__(self, monitor: str = 'val_loss', patience: int = 3, min_delta: float = 0.0, mode: str = 'min'):
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.wait = 0

    def _is_improvement(self, current, best) -> bool:
        if best is None:
            return True
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta

    def on_validation_end(self, trainer: Any, epoch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        if self.monitor not in logs:
            return
        current = logs[self.monitor]
        if self._is_improvement(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # signal trainer to stop
                setattr(trainer, 'stop_training', True)


class ModelCheckpoint(Callback):
    """Save the model after every epoch or only when monitored metric improves."""
    def __init__(self, filepath: str, monitor: str = 'val_loss', save_best_only: bool = True, mode: str = 'min'):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = None

    def _is_improvement(self, current, best) -> bool:
        if best is None:
            return True
        if self.mode == 'min':
            return current < best
        return current > best

    def on_validation_end(self, trainer: Any, epoch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        if not os.path.isdir(os.path.dirname(self.filepath)) and os.path.dirname(self.filepath) != '':
            try:
                os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            except Exception:
                pass
        if self.monitor in logs:
            current = logs[self.monitor]
            if not self.save_best_only or self._is_improvement(current, self.best):
                self.best = current
                # delegate saving to trainer if available
                try:
                    trainer.save_model(self.filepath)
                except Exception:
                    # fallback: attempt to save state_dict directly
                    try:
                        torch.save(trainer.model.state_dict(), self.filepath)
                    except Exception:
                        pass


class LambdaCallback(Callback):
    """Create a callback from simple callables.

    Usage: LambdaCallback(on_epoch_end=lambda trainer, epoch, logs: ...)
    """
    def __init__(self,
                 on_train_begin: Optional[Callable] = None,
                 on_train_end: Optional[Callable] = None,
                 on_epoch_begin: Optional[Callable] = None,
                 on_epoch_end: Optional[Callable] = None,
                 on_batch_end: Optional[Callable] = None,
                 on_validation_end: Optional[Callable] = None):
        self._on_train_begin = on_train_begin
        self._on_train_end = on_train_end
        self._on_epoch_begin = on_epoch_begin
        self._on_epoch_end = on_epoch_end
        self._on_batch_end = on_batch_end
        self._on_validation_end = on_validation_end

    def on_train_begin(self, trainer: Any) -> None:
        if self._on_train_begin:
            self._on_train_begin(trainer)

    def on_train_end(self, trainer: Any) -> None:
        if self._on_train_end:
            self._on_train_end(trainer)

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        if self._on_epoch_begin:
            self._on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[dict] = None) -> None:
        if self._on_epoch_end:
            self._on_epoch_end(trainer, epoch, logs)

    def on_batch_end(self, trainer: Any, batch: int, logs: Optional[dict] = None) -> None:
        if self._on_batch_end:
            self._on_batch_end(trainer, batch, logs)

    def on_validation_end(self, trainer: Any, epoch: int, logs: Optional[dict] = None) -> None:
        if self._on_validation_end:
            self._on_validation_end(trainer, epoch, logs)


class LearningRateScheduler(Callback):
    """Adjust learning rate according to a schedule.

    You may provide either a PyTorch lr_scheduler instance (object with a
    .step(...) method) or a simple schedule function that accepts the epoch
    number and returns either a single lr or a list/tuple of lrs (one per
    param_group).

    By default the scheduler.step() will be called at the end of each epoch.
    """
    def __init__(self, scheduler: Optional[Any] = None, schedule_fn: Optional[Callable] = None, step_at: str = 'epoch'):
        if scheduler is None and schedule_fn is None:
            raise ValueError('Either scheduler or schedule_fn must be provided')
        self.scheduler = scheduler
        self.schedule_fn = schedule_fn
        self.step_at = step_at
        # detect whether using a torch scheduler
        self._use_torch_scheduler = scheduler is not None

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        if self._use_torch_scheduler:
            # torch schedulers sometimes accept a metric for step
            try:
                # prefer passing a metric if provided (ReduceLROnPlateau expects it)
                if 'val_loss' in logs:
                    self.scheduler.step(logs.get('val_loss'))
                else:
                    self.scheduler.step()
            except TypeError:
                # fallback when scheduler.step() doesn't accept arguments
                try:
                    self.scheduler.step()
                except Exception:
                    pass
        else:
            # schedule_fn returns either a single lr or sequence of lrs
            lr = self.schedule_fn(epoch)
            if isinstance(lr, (list, tuple)):
                for pg, v in zip(trainer.optimizer.param_groups, lr):
                    pg['lr'] = float(v)
            else:
                for pg in trainer.optimizer.param_groups:
                    pg['lr'] = float(lr)


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a monitored metric has stopped improving.

    A lightweight wrapper around the common behavior. If you already have
    a torch.optim.lr_scheduler.ReduceLROnPlateau instance you can use
    `LearningRateScheduler(scheduler=your_scheduler)` instead.
    """
    def __init__(self, monitor: str = 'val_loss', factor: float = 0.1, patience: int = 2,
                 min_lr: float = 0.0, mode: str = 'min', min_delta: float = 1e-8, cooldown: int = 0):
        self.monitor = monitor
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.cooldown = int(cooldown)

        self.best = None
        self.wait = 0
        self.cooldown_counter = 0

    def _is_improvement(self, current, best) -> bool:
        if best is None:
            return True
        if self.mode == 'min':
            return current < best - self.min_delta
        return current > best + self.min_delta

    def on_validation_end(self, trainer: Any, epoch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        if self.monitor not in logs:
            return
        current = logs[self.monitor]
        if self._is_improvement(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return
            self.wait += 1
            if self.wait > self.patience:
                # reduce learning rate for all parameter groups
                for pg in trainer.optimizer.param_groups:
                    old = float(pg.get('lr', 0.0))
                    new = max(old * self.factor, self.min_lr)
                    if new < old:
                        pg['lr'] = new
                self.cooldown_counter = self.cooldown
                self.wait = 0


class CSVLogger(Callback):
    """Log epoch-level metrics to a CSV file.

    Writes a header the first time and appends rows for each epoch. Accepts
    a logs dict (as passed to on_epoch_end) and writes key/value pairs.
    """
    def __init__(self, filename: str, append: bool = True, separator: str = ','):
        self.filename = filename
        self.append = append
        self.sep = separator
        self._written_header = False

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        row = {'epoch': epoch + 1}
        row.update(logs)

        # ensure directory exists
        d = os.path.dirname(self.filename)
        if d and not os.path.isdir(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass

        mode = 'a' if self.append else 'w'
        # write header if needed
        if not self._written_header:
            write_header = True
        else:
            write_header = False

        with open(self.filename, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=self.sep)
            if write_header:
                writer.writeheader()
                self._written_header = True
            writer.writerow(row)


class TensorBoardCallback(Callback):
    """Log metrics to TensorBoard SummaryWriter.

    Provide either an existing SummaryWriter instance or a log_dir to create one.
    The callback writes scalars for train/val losses and any metrics present in logs.
    """
    def __init__(self, writer: Optional[Any] = None, log_dir: Optional[str] = None):
        if writer is None and log_dir is None:
            raise ValueError('Either writer or log_dir must be provided')
        if writer is not None:
            self.writer = writer
        else:
            # use the safe factory which returns None on failure
            self.writer = make_summary_writer(log_dir)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[dict] = None) -> None:
        if self.writer is None:
            return
        logs = logs or {}
        for k, v in logs.items():
            try:
                # normalize numeric types
                self.writer.add_scalar(k, float(v), epoch)
            except Exception:
                pass

    def on_train_end(self, trainer: Any) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
