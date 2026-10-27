import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

def evaluate_predictions(preds, y_true, task, threshold=None):
  if task == 'classification':
    probs = torch.sigmoid(torch.from_numpy(preds)).numpy()
    if threshold is None:
      # Find best threshold
      best_bacc, best_th = 0, 0
      for i in range(1,10,2):
        th = i / 10
        bacc = balanced_accuracy_score(y_true=y_true.flatten(), y_pred=(probs > th).astype(int))
        if bacc > best_bacc:
          best_bacc, best_th = bacc, th
      threshold = best_th
      bacc = best_bacc
    else:
      bacc = balanced_accuracy_score(y_true=y_true.flatten(), y_pred=(probs > threshold).astype(int))
    return {'bacc': bacc, 'threshold': threshold}
  else:
    mae = mean_absolute_error(y_true=y_true.flatten(), y_pred=preds.flatten())
    return {'mae': mae}