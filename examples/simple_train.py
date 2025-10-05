"""Simple example showing how to use Trainer with callbacks."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchflow.trainer import Trainer
from torchflow.callbacks import LambdaCallback, EarlyStopping

# tiny dataset
x = torch.linspace(-1, 1, steps=100).unsqueeze(1)
y = x.clone()
loader = DataLoader(TensorDataset(x, y), batch_size=16)

# model, criterion, optimizer
model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
crit = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr=0.05)

# callbacks
events = {"epochs": 0}
cb = LambdaCallback(on_epoch_end=lambda t, e, logs: events.update({"epochs": events["epochs"] + 1}))
early = EarlyStopping(monitor='val_loss', patience=2)

trainer = Trainer(model, crit, opt, callbacks=[cb, early])
model, history, t = trainer.train(loader, val_loader=loader, num_epochs=20)
print('Done. epochs run =', len(history['train_loss']))
print('events:', events)
