import torch
import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, device='cpu', metrics=None, writer=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metrics = metrics
        self.writer = writer

    
    def train_one_epoch(self, dataloader):
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
            
        avg_loss = running_loss / len(dataloader.dataset)
        return avg_loss
    

    def validate(self, dataloader):
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
    

    def fit(self, train_loader, val_loader=None, num_epochs=5):
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
            
