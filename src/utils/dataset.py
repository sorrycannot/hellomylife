import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

class SMILESDataset(Dataset):
  def __init__(self, X_tab, X_img, X_txt, labels):
    self.X_tab = X_tab
    self.X_img = X_img
    self.X_txt = X_txt
    self.labels = labels

  def __len__(self):
    return len(self.X_tab)

  def __getitem__(self, idx):
    X_tab = torch.tensor(self.X_tab[idx], dtype=torch.float)
    X_img = torch.tensor(self.X_img[idx], dtype=torch.float)
    X_txt = torch.tensor(self.X_txt[idx], dtype=torch.float)
    y = torch.tensor(self.labels[idx], dtype=torch.float)

    return X_tab, X_img, X_txt, y


def load_and_process_feature(BASE_FEATURES_PATH, feature_name, task):
  X_train = np.load(BASE_FEATURES_PATH.format("train", feature_name, task))
  X_val = np.load(BASE_FEATURES_PATH.format("val", feature_name, task))
  X_test = np.load(BASE_FEATURES_PATH.format("test", feature_name, task))

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)
  X_test = scaler.transform(X_test)

  return X_train, X_val, X_test
