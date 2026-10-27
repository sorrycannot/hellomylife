"""File, alignment and evaluation utilities; no Torch or RDKit import here."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
)

TASKS = ("efflux", "influx", "pampa")
SPLITS = ("train", "val", "test")
SCORE_COLUMNS = [f"p_{t}" for t in TASKS]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def import_file(path, name):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_smiles(path):
    # Auxiliary score generation never requests TITAN labels.
    return pd.read_csv(path, usecols=["SMILES"], dtype=str,
                       keep_default_na=False, encoding="utf-8-sig")


def align_identity_rows(identity_map, split, smiles):
    needed = {"split", "record", "original_smiles", "canonical_smiles", "inchikey"}
    if not needed.issubset(identity_map.columns):
        raise ValueError(f"Identity map missing columns: {sorted(needed - set(identity_map.columns))}")
    rows = identity_map[identity_map["split"] == split].copy()
    record = pd.to_numeric(rows["record"], errors="raise").to_numpy()
    if not np.isfinite(record).all() or not np.equal(record, np.floor(record)).all():
        raise ValueError(f"{split}: invalid record numbers in identity map")
    rows["record"] = record.astype(int)
    rows = rows.sort_values("record", kind="stable").reset_index(drop=True)
    if len(rows) != len(smiles) or not np.array_equal(rows.record, np.arange(1, len(smiles) + 1)):
        raise ValueError(f"{split}: identity map must have each record 1..N exactly once")
    if rows.original_smiles.tolist() != smiles.SMILES.tolist():
        raise ValueError(f"{split}: dataset SMILES/order differ from the identity map; regenerate exclusion and auxiliary models")
    for col in ("canonical_smiles", "inchikey"):
        if rows[col].isna().any() or (rows[col].astype(str).str.strip() == "").any():
            raise ValueError(f"{split}: missing {col}")
    return rows


def validate_scores(values, n, name):
    values = np.asarray(values)
    if values.shape != (n, 3):
        raise ValueError(f"{name}: score shape {values.shape}, expected {(n, 3)}")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name}: scores must be finite and between 0 and 1")
    return values


def choose_threshold(y, probabilities):
    """Original TITAN grid and tie policy: maximize validation BACC, first tie."""
    y = np.asarray(y).reshape(-1)
    p = np.asarray(probabilities).reshape(-1)
    if len(y) != len(p) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold selection requires aligned binary validation labels with both classes")
    best_score, best_threshold = float("-inf"), 0.1
    for i in range(1, 10, 2):
        threshold = i / 10
        score = balanced_accuracy_score(y, (p > threshold).astype(int))
        if score > best_score:
            best_score, best_threshold = float(score), threshold
    return best_threshold


def classification_metrics(y, probabilities, threshold):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(probabilities).reshape(-1)
    if len(y) != len(p) or set(np.unique(y)) != {0, 1} or not np.isfinite(p).all():
        raise ValueError("Metrics require finite, aligned predictions and both binary classes")
    pred = (p > threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "n": len(y), "threshold": float(threshold), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "ACC": float(accuracy_score(y, pred)), "BACC": float(balanced_accuracy_score(y, pred)),
        "SN": tp / (tp + fn), "SP": tn / (tn + fp),
        "F1": float(f1_score(y, pred, zero_division=0)), "MCC": float(matthews_corrcoef(y, pred)),
        "AUROC": float(roc_auc_score(y, p)), "AUPRC_AP": float(average_precision_score(y, p)),
    }
