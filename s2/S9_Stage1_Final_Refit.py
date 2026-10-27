#!/usr/bin/env python
"""
final_refit_v3 - Option B Stage-1 full leak-safe refit + mechanistic score regeneration.

Does NOT overwrite originals. Does NOT train Stage-2.
"""
from __future__ import annotations

import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import compute_morgan, compute_physchem10  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
SCORE_DIR = ROOT / "scores"
S2_DIR = ROOT / "stage2_inputs"
LOG_DIR = PROJECT_ROOT / "revision" / "logs"
V2_OUT = ROOT.parent / "v2_corrected" / "outputs"

SEED = 2026
PHYSCHEM = [
    "MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "HeavyAtomCount",
    "FractionCSP3", "NumAromaticRings",
]

LGB_BASE = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 48,
    "max_depth": 5,
    "min_child_samples": 42,
    "bagging_fraction": 0.84,
    "bagging_freq": 1,
    "feature_fraction": 0.97,
    "lambda_l1": 0.003,
    "lambda_l2": 1e-05,
    "learning_rate": 0.031,
    "verbosity": -1,
    "seed": SEED,
    "feature_pre_filter": False,
}

EXPECTED_PROCESSED = {
    "efflux": ("3465a1123eff191bc551297183f650cd5f8a76dcfbb5670995ceeb83941bb708", 2457, 1505, 952),
    "influx": ("94e5c4e81e1d7bb84fa3a8c715c4d8ee27bb9126d3fb27bb65d948572d26ffc2", 886, 157, 729),
    "pampa": ("e25b9d094455c1a60280900eaa127915f5567f792e171f3a2384a6f82457393d", 1484, 1235, 249),
}

EXPECTED_LEAKSAFE = {
    "efflux": (2236, 1390, 846),
    "influx": (807, 134, 673),
    "pampa": (1442, 1209, 233),
}

CLAIMED_ITERS = {"efflux": 520, "influx": 538, "pampa": 637}

SPLIT_HASHES = {
    "train": ("splits/bbbp_scaffold_train.txt", 1365, "346f22a6990526b5f7af414d6909b89b224f4dc060f1d001da29fccfef46988c"),
    "val": ("splits/bbbp_scaffold_val.txt", 292, "9fb457a08d0f6134f8b8c7654bc756792b9cdb5b5f4467b00807440f44346f5c"),
    "test": ("splits/bbbp_scaffold_test.txt", 293, "335685e4318eb47bb57f5387141f79fb77fc55a0a3714e46d376accced0761ca"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_split_iks(subset: str) -> list[str]:
    path = PROJECT_ROOT / f"splits/bbbp_scaffold_{subset}.txt"
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def featurize(smiles: list[str]) -> np.ndarray:
    pc = compute_physchem10(smiles).values.astype(np.float64)
    fp = compute_morgan(smiles, radius=2, n_bits=2048).astype(np.float64)
    X = np.hstack([pc, fp])
    if X.shape[1] != 2058:
        raise RuntimeError(f"Expected 2058 features, got {X.shape[1]}")
    if np.isnan(pc).any():
        # Match historical Stage-1: NaN physchem rows - fill with 0 for score gen stability
        # but record count. Training data from processed CSVs should be valid.
        n_nan = int(np.isnan(pc).any(axis=1).sum())
        raise RuntimeError(f"NaN physchem in {n_nan} rows - blocker")
    return X


def verify_iterations() -> dict:
    verified = {}
    for task, claimed in CLAIMED_ITERS.items():
        path = V2_OUT / f"metrics_fair_scaffold_{task}.csv"
        if not path.exists():
            raise RuntimeError(f"Missing v2 metrics: {path}")
        row = pd.read_csv(path).iloc[0]
        got = int(row["best_iteration"])
        if got != claimed:
            raise RuntimeError(f"Iteration mismatch {task}: claimed {claimed}, got {got}")
        if str(row["protocol"]) != "fair_scaffold":
            raise RuntimeError(f"{task}: protocol not fair_scaffold")
        if str(row["early_stopping_set"]) != "inner_random_20pct_of_outer_train_only":
            raise RuntimeError(f"{task}: ES set not inner-only")
        if str(row["weighting_mechanism"]) != "scale_pos_weight_only":
            raise RuntimeError(f"{task}: weighting not scale_pos_weight_only")
        verified[task] = {
            "best_iteration": got,
            "protocol": str(row["protocol"]),
            "early_stopping_set": str(row["early_stopping_set"]),
            "weighting_mechanism": str(row["weighting_mechanism"]),
            "source_file": str(path),
            "source_sha256": sha256_file(path),
            "bbbp_or_b3db_used_for_selection": False,
            "note": "Selected on leak-safe scaffold outer-train inner ES only (v2).",
        }
    return verified


def train_final(task: str, X: np.ndarray, y: np.ndarray, n_rounds: int, spw: float):
    params = dict(LGB_BASE)
    params["scale_pos_weight"] = float(spw)
    # Explicit exclusions
    assert "class_weight" not in params
    assert "is_unbalance" not in params
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=int(n_rounds),
        callbacks=[lgb.log_evaluation(0)],
    )
    return model, params


def score_df(model, smiles: list[str]) -> np.ndarray:
    X = featurize(smiles)
    return model.predict(X)


def main():
    for d in (OUT, MODEL_DIR, SCORE_DIR, S2_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=" * 70)
    log("final_refit_v3 - Option B Stage-1 full refit + score regeneration")
    log(f"root={ROOT}")
    log("=" * 70)

    # --- Verify iterations ---
    try:
        iter_meta = verify_iterations()
    except RuntimeError as e:
        (OUT / "BLOCKER_iteration_mismatch.txt").write_text(str(e), encoding="utf-8")
        raise
    (OUT / "locked_iterations_provenance.json").write_text(json.dumps(iter_meta, indent=2), encoding="utf-8")
    log(f"Locked iterations verified: { {k: v['best_iteration'] for k,v in iter_meta.items()} }")

    # --- BBBP exclusion ---
    split_info = {}
    bbbp_all = set()
    for subset, (rel, n_exp, h_exp) in SPLIT_HASHES.items():
        path = PROJECT_ROOT / rel
        h = sha256_file(path)
        iks = load_split_iks(subset)
        if len(iks) != n_exp or h != h_exp:
            raise RuntimeError(f"Split mismatch {subset}: n={len(iks)} hash={h}")
        split_info[subset] = {"path": str(path), "n": len(iks), "sha256": h}
        bbbp_all |= set(iks)
    if len(bbbp_all) != 1950:
        raise RuntimeError(f"BBBP union size {len(bbbp_all)} != 1950")
    log(f"BBBP exclusion set: {len(bbbp_all)}")

    # --- Load BBBP table with splits ---
    bbbp = pd.read_csv(PROJECT_ROOT / "data_processed" / "bbbp.csv")
    bbbp_hash = sha256_file(PROJECT_ROOT / "data_processed" / "bbbp.csv")
    split_map = {}
    for subset in ("train", "val", "test"):
        for ik in load_split_iks(subset):
            split_map[ik] = subset
    bbbp["split"] = bbbp["inchikey"].map(split_map)
    if bbbp["split"].isna().any() or len(bbbp) != 1950 or bbbp["inchikey"].nunique() != 1950:
        raise RuntimeError("BBBP table/split alignment failed")
    vc = bbbp["split"].value_counts().to_dict()
    if vc.get("train") != 1365 or vc.get("val") != 292 or vc.get("test") != 293:
        raise RuntimeError(f"BBBP split counts wrong: {vc}")

    # Prefer smiles from bbbp; old scores used 'smiles' alias
    if "canonical_smiles" not in bbbp.columns:
        raise RuntimeError("bbbp missing canonical_smiles")

    model_records = []
    models = {}
    train_iks_by_task = {}

    for task in ("efflux", "influx", "pampa"):
        path = PROJECT_ROOT / "data_processed" / f"cornelissen_{task}.csv"
        digest = sha256_file(path)
        exp_h, exp_n, exp_pos, exp_neg = EXPECTED_PROCESSED[task]
        df = pd.read_csv(path)
        if digest != exp_h or len(df) != exp_n:
            raise RuntimeError(f"{task} processed file mismatch hash/n")
        if int((df.y == 1).sum()) != exp_pos or int((df.y == 0).sum()) != exp_neg:
            raise RuntimeError(f"{task} processed class counts mismatch")

        leak = df[~df["inchikey"].isin(bbbp_all)].copy().reset_index(drop=True)
        exp_ln, exp_lp, exp_lnn = EXPECTED_LEAKSAFE[task]
        n_pos = int((leak.y == 1).sum())
        n_neg = int((leak.y == 0).sum())
        if len(leak) != exp_ln or n_pos != exp_lp or n_neg != exp_lnn:
            raise RuntimeError(
                f"{task} leak-safe mismatch: got {len(leak)} ({n_pos}/{n_neg}), "
                f"expected {exp_ln} ({exp_lp}/{exp_lnn})"
            )
        # Confirm no BBBP in training
        if set(leak.inchikey) & bbbp_all:
            raise RuntimeError(f"{task}: BBBP InChIKeys remained in training")
        train_iks_by_task[task] = set(leak.inchikey.astype(str))
        leak.to_csv(OUT / f"leak_safe_train_{task}.csv", index=False)

        log(f"[{task}] featurizing N={len(leak)} ...")
        X = featurize(leak["canonical_smiles"].tolist())
        y = leak["y"].values.astype(int)
        spw = float(n_neg / max(n_pos, 1))
        n_rounds = CLAIMED_ITERS[task]
        log(f"[{task}] training final: rounds={n_rounds} scale_pos_weight={spw:.6f}")
        model, params = train_final(task, X, y, n_rounds, spw)
        if int(model.num_trees()) != n_rounds:
            # LightGBM may report trees == rounds for fixed train
            log(f"[{task}] WARNING num_trees={model.num_trees()} vs rounds={n_rounds}")

        model_path = MODEL_DIR / f"{task}_final_leak_safe_all.joblib"
        joblib.dump(model, model_path)
        model_hash = sha256_file(model_path)
        models[task] = model

        rec = {
            "task": task,
            "source_csv": str(path),
            "source_sha256": digest,
            "train_n": len(leak),
            "train_n_pos": n_pos,
            "train_n_neg": n_neg,
            "scale_pos_weight": spw,
            "n_features": 2058,
            "num_boost_round": n_rounds,
            "num_trees": int(model.num_trees()),
            "params": params,
            "weighting": "scale_pos_weight_only",
            "is_unbalance": False,
            "sample_weight": False,
            "class_weight": False,
            "model_path": str(model_path),
            "model_sha256": model_hash,
            "iteration_provenance": iter_meta[task],
            "bbbp_excluded_n": int(len(df) - len(leak)),
            "held_out_fraction_in_final_fit": 0.0,
            "note": "Full leak-safe refit; v2 scaffold holdout remains generalization evidence.",
        }
        model_records.append(rec)
        log(f"[{task}] saved {model_path.name} sha256={model_hash[:16]}...")

    (OUT / "final_model_manifest.json").write_text(json.dumps(model_records, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k != "params" and k != "iteration_provenance"} | {
        "params_json": json.dumps(r["params"]),
        "locked_iteration": r["num_boost_round"],
        "iteration_source": r["iteration_provenance"]["source_file"],
    } for r in model_records]).to_csv(OUT / "final_model_manifest.csv", index=False)

    # --- BBBP scores ---
    log("Generating BBBP mechanistic scores...")
    bbbp_smiles = bbbp["canonical_smiles"].tolist()
    # Ensure no train leakage
    for task, tiks in train_iks_by_task.items():
        inter = tiks & set(bbbp["inchikey"].astype(str))
        if inter:
            raise RuntimeError(f"Leakage: {task} train ∩ BBBP = {len(inter)}")

    score_table = bbbp[["inchikey", "canonical_smiles", "split", "y"]].copy()
    score_table = score_table.rename(columns={"y": "bbbp_y_audit_only"})
    col_map = {"efflux": "p_efflux", "influx": "p_influx", "pampa": "p_pampa"}
    model_hash_by_col = {}
    for task, col in col_map.items():
        probs = score_df(models[task], bbbp_smiles)
        score_table[col] = probs
        model_hash_by_col[col] = next(r["model_sha256"] for r in model_records if r["task"] == task)

    # QC
    if len(score_table) != 1950 or score_table["inchikey"].nunique() != 1950:
        raise RuntimeError("BBBP score row/unique IK failure")
    if score_table["split"].value_counts().to_dict() != {"train": 1365, "val": 292, "test": 293}:
        # value_counts order may differ
        vc2 = score_table["split"].value_counts().to_dict()
        if vc2.get("train") != 1365 or vc2.get("val") != 292 or vc2.get("test") != 293:
            raise RuntimeError(f"split counts {vc2}")
    for col in ("p_efflux", "p_influx", "p_pampa"):
        s = score_table[col]
        if s.isna().any() or not np.isfinite(s).all():
            raise RuntimeError(f"{col} has NA/non-finite")
        if (s < 0).any() or (s > 1).any():
            # LightGBM binary predict should be in [0,1]; soft-warn if tiny float noise
            if (s < -1e-9).any() or (s > 1 + 1e-9).any():
                raise RuntimeError(f"{col} outside [0,1]: min={s.min()} max={s.max()}")
            score_table[col] = s.clip(0, 1)

    bbbp_score_path = SCORE_DIR / "bbbp_mechanistic_probs_final_refit_v3.csv"
    score_table.to_csv(bbbp_score_path, index=False)
    bbbp_score_hash = sha256_file(bbbp_score_path)
    log(f"Wrote {bbbp_score_path} sha256={bbbp_score_hash}")

    # Also alias columns matching historical pEfflux naming for convenience
    alias = score_table.rename(columns={
        "canonical_smiles": "smiles",
        "p_efflux": "pEfflux",
        "p_influx": "pInflux",
        "p_pampa": "pPampa",
    })
    alias_path = SCORE_DIR / "bbbp_mechanistic_probs_final_refit_v3_legacy_colnames.csv"
    alias[["inchikey", "smiles", "split", "pEfflux", "pInflux", "pPampa"]].to_csv(alias_path, index=False)

    score_meta = {
        "path": str(bbbp_score_path),
        "sha256": bbbp_score_hash,
        "n_rows": 1950,
        "n_unique_inchikey": 1950,
        "split_counts": {"train": 1365, "val": 292, "test": 293},
        "columns": list(score_table.columns),
        "model_sha256_by_score_column": model_hash_by_col,
        "bbbp_labels_used_for_scoring": False,
        "bbbp_y_column": "bbbp_y_audit_only",
        "feature_dim": 2058,
        "bbbp_source_sha256": bbbp_hash,
        "full_bbbp_exclusion_confirmed": True,
    }
    (OUT / "bbbp_score_manifest.json").write_text(json.dumps(score_meta, indent=2), encoding="utf-8")

    # --- Old vs new score comparison ---
    old_path = PROJECT_ROOT / "data" / "features" / "bbbp_mechanistic_probs.csv"
    old = pd.read_csv(old_path)
    old_hash = sha256_file(old_path)
    merged = old.merge(
        score_table[["inchikey", "p_efflux", "p_influx", "p_pampa"]],
        on="inchikey",
        how="outer",
        indicator=True,
    )
    comp_rows = []
    for old_c, new_c in (("pEfflux", "p_efflux"), ("pInflux", "p_influx"), ("pPampa", "p_pampa")):
        both = merged[merged["_merge"] == "both"].copy()
        a = both[old_c].astype(float)
        b = both[new_c].astype(float)
        diff = b - a
        pr = pearsonr(a, b)
        sr = spearmanr(a, b)
        comp_rows.append({
            "score": new_c,
            "old_column": old_c,
            "n_matched": int(len(both)),
            "n_unmatched_old_only": int((merged["_merge"] == "left_only").sum()),
            "n_unmatched_new_only": int((merged["_merge"] == "right_only").sum()),
            "old_mean": float(a.mean()),
            "new_mean": float(b.mean()),
            "old_median": float(a.median()),
            "new_median": float(b.median()),
            "old_std": float(a.std(ddof=1)),
            "new_std": float(b.std(ddof=1)),
            "old_min": float(a.min()),
            "new_min": float(b.min()),
            "old_max": float(a.max()),
            "new_max": float(b.max()),
            "diff_mean": float(diff.mean()),
            "diff_std": float(diff.std(ddof=1)),
            "diff_min": float(diff.min()),
            "diff_max": float(diff.max()),
            "diff_median": float(diff.median()),
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "spearman_rho": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
            "old_file_sha256": old_hash,
            "new_file_sha256": bbbp_score_hash,
        })
        # save per-compound diffs
        both[["inchikey", old_c, new_c]].assign(diff=diff).to_csv(
            OUT / f"score_diff_{new_c}.csv", index=False
        )
    pd.DataFrame(comp_rows).to_csv(OUT / "old_vs_new_score_comparison.csv", index=False)
    log("Old vs new score comparison written")

    # --- B3DB 4610 compatibility scores (no BBB metrics) ---
    b3_path = PROJECT_ROOT / "results" / "external_b3db" / "b3db_properties.csv"
    b3 = pd.read_csv(b3_path)
    b3_hash = sha256_file(b3_path)
    if len(b3) != 4610:
        log(f"WARNING: b3db_properties N={len(b3)} (expected 4610)")
    b3_scores = b3[["inchikey", "canonical_smiles"]].copy()
    # keep y for audit only if present - do not use for decisions
    if "y" in b3.columns:
        b3_scores["b3db_y_audit_only"] = b3["y"].values
    for task, col in col_map.items():
        b3_scores[col] = score_df(models[task], b3["canonical_smiles"].tolist())
    b3_out = SCORE_DIR / "b3db_4610_mechanistic_scores_final_refit_v3.csv"
    b3_scores.to_csv(b3_out, index=False)
    b3_meta = {
        "path": str(b3_out),
        "sha256": sha256_file(b3_out),
        "n_rows": len(b3_scores),
        "source_properties": str(b3_path),
        "source_sha256": b3_hash,
        "filter_definition": "current 4610 = IK-filter then train-scaffold filter (not strict 4080)",
        "bbb_metrics_computed": False,
        "labels_used_for_fitting_or_iteration": False,
        "model_sha256_by_score_column": model_hash_by_col,
        "strict_4080_note": "Not written/verified in this run; do not treat 4610 as strict set",
    }
    (OUT / "b3db_4610_score_manifest.json").write_text(json.dumps(b3_meta, indent=2), encoding="utf-8")
    log(f"Wrote B3DB 4610 compatibility scores N={len(b3_scores)}")

    # --- Stage-2 input tables (no training) ---
    log("Building Stage-2 input tables...")
    # Features for all BBBP
    X_all = featurize(bbbp_smiles)
    X_pc = X_all[:, :10]
    X_fp = X_all[:, 10:]
    mech = score_table[["p_efflux", "p_influx", "p_pampa"]].values

    base_meta = score_table[["inchikey", "canonical_smiles", "split", "bbbp_y_audit_only"]].copy()

    # Model A: 2048 ECFP
    A = base_meta.copy()
    for i in range(2048):
        A[f"ECFP4_{i}"] = X_fp[:, i]
    assert A.shape[1] == 4 + 2048
    A_path = S2_DIR / "stage2_input_modelA_ecfp2048.csv"
    # Save compact: meta + npz for features to avoid huge CSV? User asked for verified table.
    # Write parquet-like via csv may be huge (~1950*2048). Use npz + meta csv.
    meta_path = S2_DIR / "stage2_input_meta.csv"
    base_meta.to_csv(meta_path, index=False)
    np.savez_compressed(
        S2_DIR / "stage2_features_modelA_ecfp2048.npz",
        X=X_fp.astype(np.float32),
        inchikey=base_meta["inchikey"].values.astype(str),
        feature_names=np.array([f"ECFP4_{i}" for i in range(2048)]),
    )
    np.savez_compressed(
        S2_DIR / "stage2_features_modelB_physchem_ecfp.npz",
        X=X_all.astype(np.float32),
        inchikey=base_meta["inchikey"].values.astype(str),
        feature_names=np.array(PHYSCHEM + [f"ECFP4_{i}" for i in range(2048)]),
    )
    X_c = np.hstack([X_all, mech]).astype(np.float32)
    assert X_c.shape == (1950, 2061)
    np.savez_compressed(
        S2_DIR / "stage2_features_modelC_physchem_ecfp_mech.npz",
        X=X_c,
        inchikey=base_meta["inchikey"].values.astype(str),
        feature_names=np.array(
            PHYSCHEM + [f"ECFP4_{i}" for i in range(2048)] + ["p_efflux", "p_influx", "p_pampa"]
        ),
    )
    # Also a single joined CSV with mech + physchem + split (manageable) and pointer to npz
    join = base_meta.copy()
    for i, name in enumerate(PHYSCHEM):
        join[name] = X_pc[:, i]
    join["p_efflux"] = mech[:, 0]
    join["p_influx"] = mech[:, 1]
    join["p_pampa"] = mech[:, 2]
    join_path = S2_DIR / "stage2_input_modelC_meta_physchem_mech.csv"
    join.to_csv(join_path, index=False)

    # Integrity checks
    if np.isnan(X_c).any():
        raise RuntimeError("NaNs in Stage-2 Model C features")
    # Label leakage check: mech scores must not be derived using bbbp_y
    # (already true by construction); confirm y not in feature matrices
    s2_manifest = {
        "meta_csv": str(meta_path),
        "meta_sha256": sha256_file(meta_path),
        "n_compounds": 1950,
        "split_counts": {"train": 1365, "val": 292, "test": 293},
        "model_A": {
            "npz": "ECFP4 only",
            "n_features": 2048,
            "path": str(S2_DIR / "stage2_features_modelA_ecfp2048.npz"),
            "sha256": sha256_file(S2_DIR / "stage2_features_modelA_ecfp2048.npz"),
        },
        "model_B": {
            "definition": "PhysChem10 + ECFP4",
            "n_features": 2058,
            "path": str(S2_DIR / "stage2_features_modelB_physchem_ecfp.npz"),
            "sha256": sha256_file(S2_DIR / "stage2_features_modelB_physchem_ecfp.npz"),
        },
        "model_C": {
            "definition": "PhysChem10 + ECFP4 + p_efflux/p_influx/p_pampa",
            "n_features": 2061,
            "feature_order_tail": ["p_efflux", "p_influx", "p_pampa"],
            "path": str(S2_DIR / "stage2_features_modelC_physchem_ecfp_mech.npz"),
            "sha256": sha256_file(S2_DIR / "stage2_features_modelC_physchem_ecfp_mech.npz"),
            "meta_physchem_mech_csv": str(join_path),
            "meta_physchem_mech_csv_sha256": sha256_file(join_path),
        },
        "stage1_score_file": str(bbbp_score_path),
        "stage1_score_sha256": bbbp_score_hash,
        "missing_values": False,
        "label_leakage_into_features": False,
        "stage2_trained": False,
        "ready_for_stage2_rebuild": True,
    }
    (OUT / "stage2_input_manifest.json").write_text(json.dumps(s2_manifest, indent=2), encoding="utf-8")
    (S2_DIR / "README.md").write_text(
        "# Stage-2 inputs (final_refit_v3)\n\n"
        "Feature matrices prepared from new Stage-1 scores. **No Stage-2 training was performed.**\n\n"
        "- Model A: `stage2_features_modelA_ecfp2048.npz` (2048)\n"
        "- Model B: `stage2_features_modelB_physchem_ecfp.npz` (2058)\n"
        "- Model C: `stage2_features_modelC_physchem_ecfp_mech.npz` (2061)\n"
        "- Meta: `stage2_input_meta.csv`\n",
        encoding="utf-8",
    )

    # --- Final report ---
    write_report(model_records, iter_meta, score_meta, comp_rows, s2_manifest, b3_meta)

    log_path = LOG_DIR / f"final_refit_v3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (OUT / log_path.name).write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log("DONE - Stage-2 not trained")


def write_report(model_records, iter_meta, score_meta, comp_rows, s2_manifest, b3_meta):
    lines = []
    lines.append("# final_refit_v3 RESULTS - Stage-1 full refit + score regeneration\n")
    lines.append("**Stage-2 trained?** No\n")
    lines.append("\n## 1. Final Stage-1 training counts and parameters\n")
    lines.append("| Task | N | pos/neg | scale_pos_weight | rounds | features |")
    lines.append("|------|--:|--------:|-----------------:|-------:|---------:|")
    for r in model_records:
        lines.append(
            f"| {r['task']} | {r['train_n']} | {r['train_n_pos']}/{r['train_n_neg']} | "
            f"{r['scale_pos_weight']:.6f} | {r['num_boost_round']} | {r['n_features']} |"
        )
    lines.append("\nNative params: see `final_model_manifest.json` (v2 schema + scale_pos_weight).\n")
    lines.append("\n## 2. Locked boosting iterations\n")
    for task, m in iter_meta.items():
        lines.append(
            f"- **{task}**: {m['best_iteration']} from `{m['source_file']}` "
            f"({m['protocol']}, {m['early_stopping_set']}, {m['weighting_mechanism']})\n"
        )
    lines.append("\n## 3. Model artifacts\n")
    for r in model_records:
        lines.append(f"- `{r['model_path']}` SHA256 `{r['model_sha256']}`\n")
    lines.append("\n## 4. BBBP score file\n")
    lines.append(
        f"- Path: `{score_meta['path']}`\n"
        f"- Shape: {score_meta['n_rows']} × columns {score_meta['columns']}\n"
        f"- SHA256: `{score_meta['sha256']}`\n"
        f"- Splits: {score_meta['split_counts']}\n"
    )
    lines.append("\n## 5. Full-BBBP exclusion\n")
    lines.append("Confirmed: no BBBP train∪val∪test InChIKey in any Stage-1 training pool; score gen did not use BBBP labels.\n")
    lines.append("\n## 6. Old vs new score comparison\n")
    lines.append("| Score | Pearson r | Spearman ρ | Δmean (new−old) | matched |")
    lines.append("|-------|----------:|-----------:|----------------:|--------:|")
    for c in comp_rows:
        lines.append(
            f"| {c['score']} | {c['pearson_r']:.4f} | {c['spearman_rho']:.4f} | "
            f"{c['diff_mean']:+.4f} | {c['n_matched']} |"
        )
    lines.append("\nDescriptive only - not used to choose models. Scores are model outputs, not calibrated probabilities or assay measurements.\n")
    lines.append("\n## 7. Stage-2 inputs (untrained)\n")
    lines.append(
        f"- Meta: `{s2_manifest['meta_csv']}`\n"
        f"- A 2048: `{s2_manifest['model_A']['path']}`\n"
        f"- B 2058: `{s2_manifest['model_B']['path']}`\n"
        f"- C 2061: `{s2_manifest['model_C']['path']}`\n"
        f"- Ready for Stage-2 rebuild: **{s2_manifest['ready_for_stage2_rebuild']}**\n"
    )
    lines.append("\n## 8. Failures / unresolved\n")
    lines.append(
        "- Strict B3DB N=4080 not produced (R1-6 pending). Compatibility scores written for current **4610** only.\n"
        "- Model C advantage vs B **not tested** in this run.\n"
    )
    lines.append("\n## 9. Can corrected Stage-2 rebuild begin?\n")
    lines.append("**Yes** - Stage-1 final models and BBBP Model C inputs are ready under `final_refit_v3/`. Awaiting separate approval to train Stage-2.\n")
    lines.append(
        f"\n## B3DB 4610 compatibility scores\n`{b3_meta['path']}` (N={b3_meta['n_rows']}); no BBB metrics.\n"
    )
    (OUT / "FINAL_REFIT_V3_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": model_records,
        "iterations": iter_meta,
        "bbbp_scores": score_meta,
        "old_vs_new": comp_rows,
        "stage2_inputs": s2_manifest,
        "b3db_4610": b3_meta,
        "stage2_trained": False,
        "ready_for_stage2_rebuild": True,
    }
    (OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "FATAL_ERROR.txt").write_text(err, encoding="utf-8")
        sys.exit(1)
