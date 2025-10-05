import pytest
pytest.importorskip('torch')
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from torchflow.trainer import Trainer
from torchflow.callbacks import EarlyStopping, LambdaCallback


def test_callbacks_and_early_stopping():
	# tiny dataset: y = x
	x = torch.linspace(-1, 1, steps=20).unsqueeze(1)
	y = x.clone()
	ds = TensorDataset(x, y)
	loader = DataLoader(ds, batch_size=4)

	# tiny model
	model = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
	criterion = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=0.0)

	events = {'train_begin': 0, 'epoch_begin': 0, 'epoch_end': 0, 'train_end': 0, 'batches': 0}

	cb = LambdaCallback(
		on_train_begin=lambda trainer: events.update({'train_begin': events['train_begin'] + 1}),
		on_epoch_begin=lambda trainer, epoch: events.update({'epoch_begin': events['epoch_begin'] + 1}),
		on_epoch_end=lambda trainer, epoch, logs: events.update({'epoch_end': events['epoch_end'] + 1}),
		on_train_end=lambda trainer: events.update({'train_end': events['train_end'] + 1}),
		on_batch_end=lambda trainer, batch, logs: events.update({'batches': events['batches'] + 1}),
	)

	early = EarlyStopping(monitor='val_loss', patience=0, min_delta=1e-6)

	trainer = Trainer(model, criterion, optimizer, device='cpu', callbacks=[cb, early])

	# run for many epochs but expect early stopping to stop it quickly
	model, history, t = trainer.train(loader, val_loader=loader, num_epochs=50)

	assert events['train_begin'] == 1
	assert events['train_end'] == 1
	assert events['epoch_begin'] >= 1
	assert events['epoch_end'] >= 1
	assert events['batches'] >= 1
	# early stopping should have stopped before completing all 50 epochs
	assert len(history['train_loss']) < 50
