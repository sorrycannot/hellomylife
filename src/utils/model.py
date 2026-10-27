import torch
import torch.nn as nn
import torch.nn.functional as F

class SMILESModel(nn.Module):
  def __init__(self, d_tab, d_img, d_txt, proj_dim, dropout):
    super().__init__()

    self.proj_tab = nn.Sequential(
      nn.LayerNorm(d_tab),
      nn.Linear(d_tab, proj_dim),
      nn.ReLU(),
      nn.Dropout(dropout)
    )

    self.proj_img = nn.Sequential(
      nn.LayerNorm(d_img),
      nn.Linear(d_img, proj_dim),
      nn.ReLU(),
      nn.Dropout(dropout)
    )

    self.proj_txt = nn.Sequential(
      nn.LayerNorm(d_txt),
      nn.Linear(d_txt, proj_dim),
      nn.ReLU(),
      nn.Dropout(dropout)
    )

    self.attention_pooling = nn.Sequential(
      nn.Linear(proj_dim, proj_dim),
      nn.Tanh(),
      nn.Linear(proj_dim, 1, bias=False)
    )

    self.classifier = nn.Sequential(
      nn.Linear(proj_dim, proj_dim), 
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(proj_dim, 1)
    )

  def forward(self, tab, img, txt, return_attention_weights=False):
    h_tab = self.proj_tab(tab)
    h_img = self.proj_img(img)
    h_txt = self.proj_txt(txt)

    embeddings = torch.stack([h_tab, h_img, h_txt], dim=1)
    scores = self.attention_pooling(embeddings)
    weights = F.softmax(scores, dim=1)
    weighted_embeddings = embeddings * weights
    pooled_embeddings = torch.sum(weighted_embeddings, dim=1)

    output = self.classifier(pooled_embeddings).squeeze(-1) 

    if return_attention_weights:
      return output, weights
    else:
      return output