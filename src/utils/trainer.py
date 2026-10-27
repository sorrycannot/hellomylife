import torch
import torch.nn as nn
import numpy as np
import click
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

class EarlyStopping:
  def __init__(self, model_path, min_delta = 0, patience = 10, verbose = True):
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.min_delta = min_delta
    self.model_path = model_path
    self.best_score = float('inf')

  def __call__(self, val_loss, model):
    if val_loss + self.min_delta < self.best_score:
      self.best_score = val_loss
      self.counter = 0
      torch.save(model.state_dict(), self.model_path)
      if self.verbose:
        click.echo(f"Loss improved. Model saved.")
    else:
      self.counter += 1
      if self.verbose:
        click.echo(f"Early stopping counter: {self.counter} out of {self.patience}")
      if self.counter >= self.patience:
        if self.verbose:
          click.echo(f"Early stopping triggered. Best loss: {self.best_score:.6f}")
        return True
    return False


class Trainer:
  def __init__(self, model, criterion, optimizer, train_loader, val_loader, device, early_stopping):
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.device = device
    self.early_stopping = early_stopping
    self.model.to(self.device)

  def _train_epoch(self):
    self.model.train()
    total_loss = 0
    for X_tab, X_img, X_txt, y in self.train_loader:
      X_tab, X_img, X_txt, y = X_tab.to(self.device), X_img.to(self.device), X_txt.to(self.device), y.to(self.device)

      output = self.model(X_tab, X_img, X_txt)
      loss = self.criterion(output, y)
      total_loss += loss.item() * y.size(0)
      
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    return total_loss / len(self.train_loader.dataset)

  def _eval_epoch(self):
    self.model.eval()
    total_loss = 0
    with torch.no_grad():
      for X_tab, X_img, X_txt, y in self.val_loader:
        X_tab, X_img, X_txt, y = X_tab.to(self.device), X_img.to(self.device), X_txt.to(self.device), y.to(self.device)

        output = self.model(X_tab, X_img, X_txt)
        loss = self.criterion(output, y)
        total_loss += loss.item() * y.size(0)

    return total_loss / len(self.val_loader.dataset)

  def train(self, num_epochs):
    click.echo("Starting training...")
    for epoch in range(num_epochs):
      click.echo(f"----- Starting epoch {epoch+1}/{num_epochs} -----")
      train_loss = self._train_epoch()
      val_loss = self._eval_epoch()
      click.echo(f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
      if self.early_stopping(val_loss, self.model):
        break
    click.echo(f"Training finished")

    self.model.load_state_dict(torch.load(self.early_stopping.model_path, map_location=self.device))
    self.model.to(self.device)

  def predict(self, loader):
    click.echo("Predicting...")
    self.model.eval()
    preds = []
    with torch.no_grad():
      for X_tab, X_img, X_txt, y in loader:
        X_tab, X_img, X_txt, y = X_tab.to(self.device), X_img.to(self.device), X_txt.to(self.device), y.to(self.device)
        output = self.model(X_tab, X_img, X_txt)
        preds.append(output.cpu().numpy())
    return np.concatenate(preds, axis=0)

  def load_model(self):
    self.model.load_state_dict(torch.load(self.early_stopping.model_path, map_location=self.device))
    self.model.to(self.device)