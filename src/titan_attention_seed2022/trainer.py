#!/usr/bin/env python3
"""Shared TITAN training operations used by run_titan_attention.py.

This is a bundled internal module, called by run_titan_attention.py.

Uses the existing repository's utils/model.py and utils/utils.py. Other training
settings follow the supplied classification train.py: projection 2048, dropout
0.1, AdamW(lr=1e-4, default weight_decay=0.01), batch 32, weighted BCE, caller-provided seed,
maximum 1000 epochs, validation-loss patience 10, best checkpoint reload, then
validation BACC threshold selection on {0.1,0.3,0.5,0.7,0.9}. No test selection.
"""
from __future__ import annotations

import argparse
import math
import platform
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .common import (
    SPLITS, SCORE_COLUMNS, choose_threshold, classification_metrics,
    file_record, read_json, sha256, validate_scores, write_json,
)

MODALITIES = ("tabular", "image", "text")


def load_dependencies(root):
    global torch, nn, DataLoader, SMILESModel, SMILESMechanismModel, set_seed, original_model_module
    sys.path.insert(0, str(root / "src"))
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import utils.model as original_model_module
    from utils.model import SMILESModel
    from utils.utils import set_seed
    from .model import LateConcatModel as SMILESMechanismModel


def load_inputs(root, mechanism_dir):
    meta_path = mechanism_dir / "manifest.json"
    meta = read_json(meta_path)
    if meta.get("status") != "complete" or meta.get("score_columns") != SCORE_COLUMNS:
        raise ValueError("Mechanism manifest must be complete with column order efflux/influx/pampa")
    arrays, frames, provenance = {}, {}, {"mechanism_manifest": file_record(meta_path), "inputs": {}}
    scalers = {}
    for split in SPLITS:
        csv_path = root / "dataset" / f"{split}_classification.csv"
        if sha256(csv_path) != meta["dataset_files"][split]["sha256"]:
            raise ValueError(f"{split}: dataset changed after generating mechanism features")
        df = pd.read_csv(csv_path, usecols=["SMILES", "label"], dtype={"SMILES": str},
                         keep_default_na=False, encoding="utf-8-sig")
        y = pd.to_numeric(df.label, errors="raise").to_numpy()
        if not np.isfinite(y).all() or set(np.unique(y)) != {0, 1}:
            raise ValueError(f"{split}: requires binary labels and both classes")
        frames[split] = df
        score_path = mechanism_dir / f"{split}-mechanism-classification.npy"
        audit_path = mechanism_dir / f"{split}-mechanism-audit.csv"
        if sha256(score_path) != meta["splits"][split]["scores"]["sha256"]:
            raise ValueError(f"{split}: score array changed since generation")
        if sha256(audit_path) != meta["splits"][split]["audit"]["sha256"]:
            raise ValueError(f"{split}: mechanism row audit changed since generation")
        audit = pd.read_csv(audit_path, dtype={"original_smiles": str}, keep_default_na=False)
        if (len(audit) != len(df) or audit.original_smiles.tolist() != df.SMILES.tolist()
                or not np.array_equal(audit.record, np.arange(1, len(df) + 1))):
            raise ValueError(f"{split}: mechanism and TITAN row orders do not agree")
        scores = validate_scores(np.load(score_path, allow_pickle=False), len(df), split)
        if not np.allclose(scores, audit[SCORE_COLUMNS].to_numpy(dtype=float), rtol=1e-6, atol=1e-7):
            raise ValueError(f"{split}: score columns do not match saved row audit")
        arrays[split] = {"mechanism": scores.astype(np.float32), "labels": y.astype(np.float32)}
        provenance["inputs"][split] = {
            "dataset": file_record(csv_path), "mechanism": file_record(score_path),
            "row_audit": file_record(audit_path), "n": len(df),
        }
    # Exactly the supplied dataset.py policy: fit each scaler on TRAIN ONLY,
    # then transform val/test. Preserve native input dtype until StandardScaler.
    for modality in MODALITIES:
        raw, paths = {}, {}
        for split in SPLITS:
            path = root / "features" / f"{split}-{modality}-classification.npy"
            value = np.load(path, allow_pickle=False)
            if value.ndim != 2 or len(value) != len(frames[split]) or not np.isfinite(value).all():
                raise ValueError(f"{path}: bad shape, row count or nonfinite values")
            raw[split], paths[split] = value, path
        dims = {v.shape[1] for v in raw.values()}
        if len(dims) != 1:
            raise ValueError(f"{modality}: feature dimension differs across splits")
        scaler = StandardScaler()
        arrays["train"][modality] = scaler.fit_transform(raw["train"]).astype(np.float32)
        for split in ("val", "test"):
            arrays[split][modality] = scaler.transform(raw[split]).astype(np.float32)
        scalers[modality] = scaler
        for split in SPLITS:
            provenance["inputs"][split][modality] = file_record(paths[split])
            provenance["inputs"][split][modality]["shape"] = list(raw[split].shape)
    provenance["mechanism_scaling"] = "none; raw fixed auxiliary scores in [0,1]"
    provenance["existing_TITAN_split_identity_overlaps"] = meta["existing_TITAN_split_identity_overlaps"]
    return arrays, frames, scalers, provenance


class FusionDataset:
    # Map-style Dataset protocol; no torch import needed for file validation.
    def __init__(self, arrays):
        self.values = arrays

    def __len__(self):
        return len(self.values["labels"])

    def __getitem__(self, idx):
        return tuple(torch.tensor(self.values[key][idx], dtype=torch.float32)
                     for key in ("tabular", "image", "text", "mechanism", "labels"))


def forward_batch(model, inputs, mode):
    return model(*inputs[:3]) if mode == "baseline" else model(*inputs[:4])


def architecture_smoke_test(model, mode, dims, device):
    """Real Torch shape/backward check, automatically run on the server."""
    capture = []
    handle = model.classifier[0].register_forward_pre_hook(lambda layer, inputs: capture.append(int(inputs[0].shape[1])))
    model.eval()  # No dropout or random probe generation: preserve training RNG.
    inputs = [torch.zeros((2, d), dtype=torch.float32, device=device) for d in dims]
    inputs.append(torch.full((2, 3), 0.5, dtype=torch.float32, device=device))
    try:
        output = forward_batch(model, inputs, mode)
        expected = 2048 if mode == "baseline" else 2051
        if output.shape != (2,) or capture != [expected]:
            raise RuntimeError(f"Bad model shape: output={output.shape}, classifier widths={capture}")
        loss = nn.BCEWithLogitsLoss()(output, torch.tensor([0., 1.], device=device))
        loss.backward()
        grad = model.classifier[0].weight.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError("Classification-head backward test failed")
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train()
    return {"classifier_input": expected, "output_shape": [2], "backward_finite": True}


def run_epoch(model, loader, criterion, optimizer, mode, device):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            inputs = [v.to(device) for v in batch[:-1]]
            y = batch[-1].to(device)
            logits = forward_batch(model, inputs, mode)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite neural training/validation loss")
            total += loss.item() * y.size(0)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return total / len(loader.dataset)


def predict(model, loader, mode, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in loader:
            inputs = [v.to(device) for v in batch[:-1]]
            parts.append(forward_batch(model, inputs, mode).cpu().numpy())
    logits = np.concatenate(parts).reshape(-1)
    if not np.isfinite(logits).all():
        raise RuntimeError("Nonfinite neural predictions")
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return logits, probabilities


def train_one(mode, args, arrays, output, device):
    set_seed(args.seed)  # Reset separately for baseline and mechanism runs.
    folder = output / f"{mode}_seed{args.seed}"
    folder.mkdir()
    log_path = folder / "training.log"

    def log(text):
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    dims = [arrays["train"][m].shape[1] for m in MODALITIES]
    # Loader settings deliberately match original TITAN (num_workers=0).
    train_loader = DataLoader(FusionDataset(arrays["train"]), batch_size=32, shuffle=True)
    val_loader = DataLoader(FusionDataset(arrays["val"]), batch_size=32)
    model_type = SMILESModel if mode == "baseline" else SMILESMechanismModel
    model = model_type(d_tab=dims[0], d_img=dims[1], d_txt=dims[2], proj_dim=2048, dropout=0.1)
    model.to(device)
    architecture = architecture_smoke_test(model, mode, dims, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    y = arrays["train"]["labels"]
    spw = int((y == 0).sum()) / int((y == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([spw], dtype=torch.float32, device=device))
    checkpoint = folder / "best_model.pth"
    config = {
        "mode": mode, "seed": args.seed, "d_tab": dims[0], "d_img": dims[1], "d_txt": dims[2],
        "projection_dim": 2048, "pooled_dim": 2048, "classifier_input_dim": architecture["classifier_input"],
        "classifier_hidden_dim": 2048, "dropout": 0.1, "batch_size": 32,
        "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": optimizer.defaults["weight_decay"],
        "loss": "BCEWithLogitsLoss", "pos_weight_train_only": spw,
        "max_epochs": args.epochs, "patience": 10, "checkpoint_selection": "minimum validation weighted BCE",
        "threshold_selection": "validation BACC over 0.1,0.3,0.5,0.7,0.9, first tie",
        "mechanism_columns": SCORE_COLUMNS if mode == "mechanism" else [],
        "mechanism_scores_frozen": True, "titan_feature_extractors_frozen": True,
        "trainable_modules": "TITAN modality projections, attention pooling and classification head",
        "initial_extra_head_columns": "zeros" if mode == "mechanism" else None,
        "architecture_smoke_test": architecture,
    }
    write_json(folder / "model_config.json", config)
    history, counter, best_loss, best_epoch = [], 0, float("inf"), 0
    log(f"[{mode}] dims={dims}; pooled=2048; classifier input={architecture['classifier_input']}; seed={args.seed}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, mode, device)
        val_loss = run_epoch(model, val_loader, criterion, None, mode, device)
        improved = val_loss < best_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "improved": improved})
        log(f"[{mode}] epoch {epoch}/{args.epochs}: train loss={train_loss:.6f}, val loss={val_loss:.6f}")
        if improved:
            best_loss, best_epoch, counter = val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            counter += 1
            if counter >= 10:
                log(f"[{mode}] Early stopping: best epoch={best_epoch}, best val loss={best_loss:.6f}")
                break
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    if not math.isfinite(best_loss) or best_epoch == 0:
        raise RuntimeError(f"{mode}: no valid checkpoint saved")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    logits_val, probs_val = predict(model, val_loader, mode, device)
    threshold = choose_threshold(arrays["val"]["labels"], probs_val)
    val_metrics = classification_metrics(arrays["val"]["labels"], probs_val, threshold)
    np.save(folder / "val-classification.npy", probs_val)
    pd.DataFrame({"record": np.arange(1, len(probs_val) + 1), "y_true": arrays["val"]["labels"],
                  "logit": logits_val, "probability": probs_val,
                  "prediction": (probs_val > threshold).astype(int)}).to_csv(folder / "val_predictions.csv", index=False)
    lock = {"mode": mode, "seed": args.seed, "best_epoch": best_epoch, "best_val_loss": best_loss,
            "threshold": threshold, "checkpoint": file_record(checkpoint), "validation": val_metrics}
    # Save model/threshold selection BEFORE accessing the test loader or outcomes.
    write_json(folder / "selection_lock.json", lock)
    log(f"[{mode}] locked epoch={best_epoch}, threshold={threshold}; val AUROC={val_metrics['AUROC']:.6f}, BACC={val_metrics['BACC']:.6f}")
    del model, optimizer, criterion
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return folder, config, lock


def evaluate_test(folder, config, lock, arrays, device):
    # Called only after ALL requested neural models and thresholds are locked.
    mode = config["mode"]
    model_type = SMILESModel if mode == "baseline" else SMILESMechanismModel
    model = model_type(d_tab=config["d_tab"], d_img=config["d_img"], d_txt=config["d_txt"],
                       proj_dim=2048, dropout=0.1).to(device)
    checkpoint = folder / "best_model.pth"
    if sha256(checkpoint) != lock["checkpoint"]["sha256"]:
        raise RuntimeError(f"{mode}: checkpoint changed after validation selection")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    loader = DataLoader(FusionDataset(arrays["test"]), batch_size=32)
    logits, probabilities = predict(model, loader, mode, device)
    threshold = lock["threshold"]
    metrics = classification_metrics(arrays["test"]["labels"], probabilities, threshold)
    np.save(folder / "test-classification.npy", probabilities)
    pd.DataFrame({"record": np.arange(1, len(probabilities) + 1), "y_true": arrays["test"]["labels"],
                  "logit": logits, "probability": probabilities,
                  "prediction": (probabilities > threshold).astype(int)}).to_csv(folder / "test_predictions.csv", index=False)
    write_json(folder / "metrics.json", {"val": lock["validation"], "test": metrics})
    print(f"[{mode}] test AUROC={metrics['AUROC']:.6f}, BACC={metrics['BACC']:.6f}, "
          f"MCC={metrics['MCC']:.6f}, threshold={threshold}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics

