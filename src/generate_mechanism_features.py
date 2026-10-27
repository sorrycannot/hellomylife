#!/usr/bin/env python3
"""Generate N x 3 TITAN scores from verified final auxiliary models.

Uses cached standardized identities from the exclusion step, preserving original
TITAN row order. It neither trains a model nor reads TITAN outcome columns.
Run from TITAN-BBB-main/src:
  python generate_mechanism_features.py --project-root ..
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from mechanism_common import (
    TASKS, SPLITS, SCORE_COLUMNS, align_identity_rows, file_record,
    import_file, read_json, read_smiles, sha256, validate_scores, write_json,
)


def generate(args):
    root = args.project_root.resolve(strict=True)
    stage1 = (args.stage1_dir or root / "auxiliary_stage1").resolve(strict=True)
    auxiliary = (args.auxiliary_dir or root / "auxiliary_titan").resolve(strict=True)
    script = (args.stage1_script or root / "src" / "train_auxiliary_titan.py").resolve(strict=True)
    output = (args.output_dir or root / "mechanism_features").resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}; use a new output directory")
    manifest_path = stage1 / "run_manifest.json"
    stage_meta = read_json(manifest_path)
    if stage_meta.get("status") != "complete":
        raise ValueError("Auxiliary training run_manifest.json does not have status=complete")
    if sha256(script) != stage_meta["script"]["sha256"]:
        raise ValueError("Stage-1 script changed since fitting; feature definitions must match the fitted models")
    identity_path = auxiliary / "titan_identity_map.csv"
    if sha256(identity_path) != stage_meta["titan_identity_map"]["sha256"]:
        raise ValueError("Identity map differs from the map used to exclude TITAN before auxiliary fitting")
    ids = pd.read_csv(identity_path, dtype=str, keep_default_na=False, encoding="utf-8-sig",
                      usecols=["split", "record", "original_smiles", "canonical_smiles", "inchikey"])
    if set(ids["split"]) != set(SPLITS):
        raise ValueError("Identity map must contain exactly train, val and test splits")
    aligned, datasets = {}, {}
    for split in SPLITS:
        path = root / "dataset" / f"{split}_classification.csv"
        aligned[split] = align_identity_rows(ids, split, read_smiles(path))
        datasets[split] = file_record(path)
    union = set(ids.inchikey)
    if len(union) != int(stage_meta["titan_unique_excluded_identities"]):
        raise ValueError("TITAN identity union differs from auxiliary training metadata")

    # Reuse the exact feature functions in the script that produced these models.
    feature_module = import_file(script, "_titan_stage1_feature_source")
    feature_module.load_dependencies()
    rdkit_version = feature_module.rdBase.rdkitVersion
    expected = stage_meta["versions"]["rdkit"]
    if rdkit_version != expected:
        raise ValueError(f"RDKit version differs from auxiliary fitting: current={rdkit_version}, fit={expected}")
    lgb, joblib = feature_module.lgb, feature_module.joblib
    if lgb.__version__ != stage_meta["versions"]["lightgbm"]:
        raise ValueError("LightGBM version differs from auxiliary fitting")
    finals = {rec["task"]: rec for rec in stage_meta["final_models"]}
    if set(finals) != set(TASKS):
        raise ValueError("Expected exactly three final auxiliary models")
    models, model_records, pool_records = {}, {}, {}
    for task in TASKS:
        pool = auxiliary / f"cornelissen_{task}_titan_excluded.csv"
        if sha256(pool) != stage_meta["input_pools"][task]["sha256"]:
            raise ValueError(f"{task}: auxiliary training pool changed")
        pool_keys = set(pd.read_csv(pool, usecols=["inchikey"], dtype=str,
                                    encoding="utf-8-sig").inchikey)
        if pool_keys & union:
            raise ValueError(f"{task}: auxiliary training identities overlap TITAN")
        pool_records[task] = file_record(pool)
        path = stage1 / "final_models" / f"{task}.joblib"
        if sha256(path) != finals[task]["models"]["joblib"]["sha256"]:
            raise ValueError(f"{task}: model does not match the completed final-refit manifest")
        model = joblib.load(path)
        if int(model.num_feature()) != 2058:
            raise ValueError(f"{task}: expected a 2058-feature auxiliary model")
        if int(model.num_trees()) != int(finals[task]["models"]["num_trees"]):
            raise ValueError(f"{task}: model tree count differs from final-refit manifest")
        models[task], model_records[task] = model, file_record(path)

    output.mkdir(parents=True)
    meta = {
        "status": "running", "score_columns": SCORE_COLUMNS, "task": "classification",
        "script": file_record(__file__), "stage1_script": file_record(script),
        "stage1_manifest": file_record(manifest_path), "identity_map": file_record(identity_path),
        "auxiliary_models": model_records, "auxiliary_pools": pool_records,
        "dataset_files": datasets, "row_order": "exact original TITAN CSV order, record 1..N",
        "standardized_structures": "cached identities from TITAN exclusion, no restandardization",
        "titan_outcome_columns_read": [], "scores_are_calibrated_probabilities": False,
        "score_transform": "LightGBM binary predict, no second sigmoid, no scaling",
        "versions": {"python": platform.python_version(), "rdkit": rdkit_version,
                     "lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "splits": {},
    }
    keysets = {s: set(aligned[s].inchikey) for s in SPLITS}
    meta["existing_TITAN_split_identity_overlaps"] = {
        "train__val": len(keysets["train"] & keysets["val"]),
        "train__test": len(keysets["train"] & keysets["test"]),
        "val__test": len(keysets["val"] & keysets["test"]),
    }
    write_json(output / "manifest.json", meta)
    try:
        for split in SPLITS:
            rows = aligned[split]
            parts = []
            print(f"[{split}] generating scores for {len(rows)} rows ...", flush=True)
            for start in range(0, len(rows), args.batch_size):
                batch = rows.iloc[start:start + args.batch_size]
                X = feature_module.featurize(batch.canonical_smiles.tolist())
                raw = np.column_stack([models[t].predict(X, num_threads=args.num_threads) for t in TASKS])
                parts.append(validate_scores(raw, len(batch), split))
            scores = validate_scores(np.concatenate(parts, axis=0), len(rows), split)
            score_path = output / f"{split}-mechanism-classification.npy"
            np.save(score_path, scores.astype(np.float32))
            audit = rows[["record", "original_smiles", "canonical_smiles", "inchikey"]].copy()
            for col, values in zip(SCORE_COLUMNS, scores.T):
                audit[col] = values
            audit_path = output / f"{split}-mechanism-audit.csv"
            audit.to_csv(audit_path, index=False)
            meta["splits"][split] = {
                "n": len(rows), "shape": [len(rows), 3], "scores": file_record(score_path),
                "audit": file_record(audit_path), "column_min": scores.min(axis=0).tolist(),
                "column_max": scores.max(axis=0).tolist(), "column_mean": scores.mean(axis=0).tolist(),
            }
            write_json(output / "manifest.json", meta)
            print(f"[{split}] shape={(len(rows), 3)}, columns={SCORE_COLUMNS}, TITAN overlap with auxiliary pools=0", flush=True)
        meta["status"] = "complete"
        write_json(output / "manifest.json", meta)
        print(f"DONE: {output}", flush=True)
    except Exception as exc:
        meta.update(status="failed", error=str(exc))
        write_json(output / "manifest.json", meta)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--stage1-dir", type=Path)
    parser.add_argument("--auxiliary-dir", type=Path)
    parser.add_argument("--stage1-script", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_threads < 1:
        parser.error("batch size and thread count must be positive")
    return args


if __name__ == "__main__":
    generate(parse_args())
