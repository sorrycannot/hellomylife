"""Mechanism-conditioned additive attention, preserving TITAN's original head.

Attention: e_j = v^T tanh(W h_j + b + U m), alpha = softmax_j(e).
The shared context U m is placed BEFORE tanh; a shared scalar added after
scoring would cancel in softmax and would not condition the attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model import SMILESModel


class LateConcatModel(SMILESModel):
    """The previously used late-concatenation control: 2048+3 -> 2048 -> 1."""

    def __init__(self, d_tab, d_img, d_txt, proj_dim=2048, dropout=0.1):
        super().__init__(d_tab, d_img, d_txt, proj_dim, dropout)
        self.proj_dim = proj_dim
        old = self.classifier[0]
        if not isinstance(old, nn.Linear) or (old.in_features, old.out_features) != (proj_dim, proj_dim):
            raise ValueError("Unexpected TITAN classification head in src/utils/model.py")
        # Align all common initial weights and the following dropout/loader RNG.
        rng = torch.get_rng_state()
        widened = nn.Linear(proj_dim + 3, proj_dim)
        torch.set_rng_state(rng)
        with torch.no_grad():
            widened.weight[:, :proj_dim].copy_(old.weight)
            widened.weight[:, proj_dim:].zero_()
            widened.bias.copy_(old.bias)
        self.classifier[0] = widened

    @staticmethod
    def check_mechanism(mechanism, batch):
        if mechanism.ndim != 2 or mechanism.shape != (batch, 3):
            raise ValueError("mechanism must be (batch, 3), ordered efflux/influx/pampa")

    def embeddings(self, tab, img, txt):
        return torch.stack([self.proj_tab(tab), self.proj_img(img), self.proj_txt(txt)], dim=1)

    def forward(self, tab, img, txt, mechanism, return_attention_weights=False):
        self.check_mechanism(mechanism, tab.shape[0])
        h = self.embeddings(tab, img, txt)
        weights = F.softmax(self.attention_pooling(h), dim=1)
        pooled = torch.sum(h * weights, dim=1)
        output = self.classifier(torch.cat([pooled, mechanism], dim=1)).squeeze(-1)
        return (output, weights) if return_attention_weights else output


class MechanismAttentionModel(LateConcatModel):
    """Mechanisms condition modality attention AND remain in the late head."""

    def __init__(self, d_tab, d_img, d_txt, proj_dim=2048, dropout=0.1):
        super().__init__(d_tab, d_img, d_txt, proj_dim, dropout)
        a = self.attention_pooling
        if (len(a) != 3 or not isinstance(a[0], nn.Linear)
                or not isinstance(a[1], nn.Tanh) or not isinstance(a[2], nn.Linear)
                or a[0].out_features != proj_dim or a[2].out_features != 1):
            raise ValueError("Expected original TITAN Linear -> Tanh -> Linear attention")
        rng = torch.get_rng_state()
        self.mechanism_context = nn.Linear(3, proj_dim, bias=False)
        torch.set_rng_state(rng)
        # Start from the same function as late concat; U has a learnable gradient.
        nn.init.zeros_(self.mechanism_context.weight)

    def forward(self, tab, img, txt, mechanism, return_attention_weights=False,
                attention_mechanism=None):
        self.check_mechanism(mechanism, tab.shape[0])
        context_scores = mechanism if attention_mechanism is None else attention_mechanism
        self.check_mechanism(context_scores, tab.shape[0])
        h = self.embeddings(tab, img, txt)                       # (B, 3, 2048)
        projected = self.attention_pooling[0](h)                # W h_j + b
        context = self.mechanism_context(context_scores).unsqueeze(1)  # (B, 1, 2048)
        conditioned = self.attention_pooling[1](projected + context)
        scores = self.attention_pooling[2](conditioned)          # (B, 3, 1)
        weights = F.softmax(scores, dim=1)                      # across modalities
        pooled = torch.sum(h * weights, dim=1)                  # (B, 2048)
        fused = torch.cat([pooled, mechanism], dim=1)            # (B, 2051)
        output = self.classifier(fused).squeeze(-1)
        return (output, weights) if return_attention_weights else output
