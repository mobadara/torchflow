"""Example demonstrating TensorBoardCallback with the safe factory."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchflow.trainer import Trainer
from torchflow.callbacks import TensorBoardCallback, make_summary_writer

# Prepare tiny dataset
x = torch.linspace(-1, 1, steps=100).unsqueeze(1)
y = x.clone()
loader = DataLoader(TensorDataset(x, y), batch_size=16)

model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
crit = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr=0.01)

# Use safe factory to create writer (may be None if tensorboard not installed)
writer = make_summary_writer('runs/example')
tb = TensorBoardCallback(writer=writer) if writer is not None else None

callbacks = [cb for cb in (tb,) if cb is not None]
trainer = Trainer(model, crit, opt, callbacks=callbacks)
model, history, t = trainer.train(loader, val_loader=loader, num_epochs=5)
print('Finished training; TensorBoard logging enabled:', writer is not None)
