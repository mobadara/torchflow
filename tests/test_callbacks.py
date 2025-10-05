import pytest
pytest.importorskip('torch')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchflow.trainer import Trainer
from torchflow.callbacks import LearningRateScheduler, ReduceLROnPlateau, CSVLogger, LambdaCallback


def _tiny_trainer_with_optimizer(lr=0.1):
    x = torch.linspace(-1, 1, steps=20).unsqueeze(1)
    y = x.clone()
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)

    model = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
    crit = nn.MSELoss()
    opt = optim.SGD(model.parameters(), lr=lr)
    trainer = Trainer(model, crit, opt, device='cpu')
    return trainer, loader


def test_learning_rate_scheduler_function():
    # schedule function halves lr every 2 epochs
    def schedule(epoch):
        if epoch < 2:
            return 0.1
        return 0.05

    trainer, loader = _tiny_trainer_with_optimizer(lr=0.1)
    lrs = []
    cb = LearningRateScheduler(schedule_fn=schedule)
    # capture lr at epoch end
    lc = LambdaCallback(on_epoch_end=lambda t, e, logs: lrs.append(t.optimizer.param_groups[0]['lr']))
    trainer.callbacks = [cb, lc]
    trainer.train(loader, val_loader=loader, num_epochs=4)
    assert lrs[0] == 0.1
    assert lrs[2] == 0.05


def test_reduce_lr_on_plateau_and_csvlogger(tmp_path):
    trainer, loader = _tiny_trainer_with_optimizer(lr=0.1)
    # use ReduceLROnPlateau with patience 0 to force reduction quickly
    reduce_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=0)
    csv_file = tmp_path / "log.csv"
    csv_cb = CSVLogger(str(csv_file))
    trainer.callbacks = [reduce_cb, csv_cb]
    trainer.train(loader, val_loader=loader, num_epochs=3)

    # csv file should exist and contain header + rows
    text = csv_file.read_text()
    assert 'epoch' in text
    assert '\n' in text
