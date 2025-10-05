"""Example showing LearningRateScheduler, ReduceLROnPlateau and CSVLogger."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchflow.trainer import Trainer
from torchflow.callbacks import LearningRateScheduler, ReduceLROnPlateau, CSVLogger

# tiny dataset
x = torch.linspace(-1, 1, steps=100).unsqueeze(1)
y = x.clone()
loader = DataLoader(TensorDataset(x, y), batch_size=16)

model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
crit = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.01)

# torch scheduler wrapped
torch_sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
lr_cb = LearningRateScheduler(scheduler=torch_sched)
reduce_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)
csv_cb = CSVLogger('examples/training_log.csv')

trainer = Trainer(model, crit, opt, callbacks=[lr_cb, reduce_cb, csv_cb])
model, history, t = trainer.train(loader, val_loader=loader, num_epochs=12)
print('Training finished. CSV log at examples/training_log.csv')
