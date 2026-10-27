#!/usr/bin/env python3
"""MechBBB S8-style auxiliary evaluation, then S9-style full-pool refitting.

Run inside TITAN-BBB-main/src (requires numpy/pandas/RDKit/sklearn/joblib/LightGBM):
  python train_auxiliary_titan.py --data-dir ../auxiliary_titan \
      --output-dir ../auxiliary_stage1 --num-threads 4

Inputs: cornelissen_{efflux,influx,pampa}_titan_excluded.csv and
titan_identity_map.csv from deduplicate_auxiliary_titan.py.
Only the identity map's inchikey column is read; no TITAN outcome is used.

Protocol: S8's random AND scaffold outer ~80:20 holdouts, each with inner
random 80:20 early stopping on OUTER TRAIN ONLY. Refit outer train at its
selected round and evaluate the untouched outer holdout. Like S9, lock the
round selected inside the SCAFFOLD outer train, then refit each entire pool.
The random/scaffold holdout scores NEVER select rounds or hyperparameters.

Feature functions and fixed hyperparameters are copied from the supplied
MechBBB features.py and S8/S9. No scaler or learned feature transform is used.
The original split_utils.py was NOT supplied. Default scaffold extraction is
explicit non-chiral RDKit Murcko, whose equality to that helper is unverified.
Pass --split-utils /path/to/original/split_utils.py to use compute_scaffold
from that file. Both policies retain empty Murcko molecules as singletons.

Output must be a NEW directory. Intermediate evaluation artifacts and final
models are separate. Holdout scores describe the evaluation models; final
all-pool models do not have a separate, untouched auxiliary holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TASKS = ("efflux", "influx", "pampa")
SEED = 2026
MAX_ROUNDS = 5000
PATIENCE = 100
PHYSCHEM_ORDER = [
    "MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "HeavyAtomCount", "FractionCSP3",
    "NumAromaticRings",
]
LGB_BASE_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "num_leaves": 48, "max_depth": 5, "min_child_samples": 42,
    "bagging_fraction": 0.84, "bagging_freq": 1, "feature_fraction": 0.97,
    "lambda_l1": 0.003, "lambda_l2": 0.00001, "learning_rate": 0.031,
    "verbosity": -1, "seed": SEED, "feature_pre_filter": False,
}
MEMBER_COLUMNS = [
    "inchikey", "canonical_smiles", "y", "scaffold_raw", "scaffold_group_id",
    "is_acyclic_empty_murcko",
]


def load_dependencies():
    # Defer imports so --help also works before dependencies are installed.
    global np, pd, lgb, joblib, Chem, AllChem, Descriptors, rdMolDescriptors
    global rdBase, MurckoScaffold, average_precision_score, roc_auc_score, sklearn
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import joblib
    import sklearn
    from sklearn.metrics import average_precision_score, roc_auc_score
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def class_counts(y):
    y = np.asarray(y, dtype=int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return {"n": len(y), "n_pos": pos, "n_neg": neg,
            "pos_frac": pos / len(y) if len(y) else None}


def require_two_classes(y, name):
    cc = class_counts(y)
    if not cc["n_pos"] or not cc["n_neg"]:
        raise ValueError(f"{name}: requires both classes, got {cc}; no seed redrawing")
    return cc


def scale_pos_weight_from_y(y):
    cc = require_two_classes(y, "fit subset")
    return cc["n_neg"] / cc["n_pos"]


def compute_physchem10(smiles_list):
    # Same calculations and order as the supplied MechBBB features.py.
    results = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append([np.nan] * 10)
                continue
            results.append([
                Descriptors.MolWt(mol), Descriptors.TPSA(mol),
                Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
                rdMolDescriptors.CalcNumRings(mol), Descriptors.HeavyAtomCount(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
            ])
        except Exception:
            results.append([np.nan] * 10)
    return pd.DataFrame(results, columns=PHYSCHEM_ORDER)


def compute_morgan(smiles_list, radius=2, n_bits=2048):
    # Keep the legacy RDKit call used by the supplied code; no fingerprint swap.
    fp_array = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            for j in range(n_bits):
                if fp.GetBit(j):
                    fp_array[i, j] = 1
        except Exception:
            pass
    return fp_array


def featurize(smiles):
    # Validate first: the historical feature helper otherwise silently zeroed
    # failed fingerprints. Bad input must not become an unreported training row.
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid structure at pool row {i + 1}: {s!r}")
    pc = compute_physchem10(smiles).values.astype(np.float64)
    # The legacy API emits one deprecation warning per molecule in newer RDKit.
    # Suppress those repeated RDKit logs without changing fingerprint settings.
    with rdBase.BlockLogs():
        fp = compute_morgan(smiles, radius=2, n_bits=2048).astype(np.float64)
    X = np.hstack([pc, fp])
    if X.shape != (len(smiles), 2058) or not np.isfinite(X).all():
        raise ValueError(f"Invalid feature matrix: shape={X.shape}; expected finite N x 2058")
    return X


def get_scaffold_function(helper_path):
    if helper_path:
        path = Path(helper_path).resolve(strict=True)
        name = "_provided_mechbbb_split_utils"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        fn = getattr(module, "compute_scaffold", None)
        if not callable(fn):
            raise ValueError(f"{path} has no callable compute_scaffold")
        return fn, {"policy": "provided_compute_scaffold", "source": file_record(path)}

    def compute_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES for scaffold: {smiles!r}")
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    return compute_scaffold, {
        "policy": "RDKit_MurckoScaffoldSmiles_includeChirality_False",
        "original_split_utils_supplied": False,
        "equivalence_to_missing_original_helper_verified": False,
    }


def annotate_scaffolds(df, compute_scaffold):
    out = df.copy()
    scaffolds = [compute_scaffold(s) for s in out["canonical_smiles"]]
    if any(not isinstance(s, str) for s in scaffolds):
        raise ValueError("compute_scaffold must return a string; invalid outputs are not empty scaffolds")
    out["scaffold_raw"] = scaffolds
    out["is_acyclic_empty_murcko"] = out["scaffold_raw"] == ""
    out["scaffold_group_id"] = [
        sc if sc != "" else f"__ACYCLIC_SINGLETON__{ik}"
        for sc, ik in zip(out["scaffold_raw"], out["inchikey"])
    ]
    return out


def random_split_indices(n, train_frac=0.8, seed=SEED):
    # Exactly S8: random permutation, not a stratified split.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_train = int(train_frac * n)
    return indices[:n_train], indices[n_train:]


def overlap_audit(train_df, eval_df):
    tr_raw, ev_raw = set(train_df.scaffold_raw), set(eval_df.scaffold_raw)
    return {
        "inchikey_overlap": len(set(train_df.inchikey) & set(eval_df.inchikey)),
        "group_id_overlap": len(set(train_df.scaffold_group_id) & set(eval_df.scaffold_group_id)),
        "nonempty_scaffold_overlap": len((tr_raw - {""}) & (ev_raw - {""})),
        "raw_scaffold_string_overlap_incl_empty": len(tr_raw & ev_raw),
        "empty_string_shared_across_splits": "" in tr_raw and "" in ev_raw,
        "n_empty_murcko_train": int(train_df.is_acyclic_empty_murcko.sum()),
        "n_empty_murcko_eval": int(eval_df.is_acyclic_empty_murcko.sum()),
    }


def scaffold_holdout_80_20(df, seed=SEED):
    # S8 greedy assignment, including its seeded equal-size tie handling and
    # preservation of original row order for the later inner random split.
    groups = defaultdict(list)
    for ik, gid in zip(df["inchikey"], df["scaffold_group_id"]):
        groups[gid].append(ik)
    scaffold_list = sorted(groups.items(), key=lambda x: (-len(x[1]), x[0]))
    rng = np.random.default_rng(seed)
    blocks, i = [], 0
    while i < len(scaffold_list):
        j = i
        size = len(scaffold_list[i][1])
        while j < len(scaffold_list) and len(scaffold_list[j][1]) == size:
            j += 1
        block = scaffold_list[i:j]
        blocks.extend([block[k] for k in rng.permutation(len(block))])
        i = j
    train_target = int(0.8 * len(df))
    eval_target = len(df) - train_target
    train_ik, eval_ik = [], []
    train_n = eval_n = 0
    for gid, iks in blocks:
        train_ratio = train_n / train_target if train_target else float("inf")
        eval_ratio = eval_n / eval_target if eval_target else float("inf")
        if train_ratio <= eval_ratio:
            train_ik.extend(iks)
            train_n += len(iks)
        else:
            eval_ik.extend(iks)
            eval_n += len(iks)
    train_df = df[df["inchikey"].isin(set(train_ik))].copy()
    eval_df = df[df["inchikey"].isin(set(eval_ik))].copy()
    return train_df, eval_df, overlap_audit(train_df, eval_df)


def load_pool(path, excluded_keys):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"canonical_smiles", "inchikey", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(df.columns))}")
    if df[list(required)].isna().any().any():
        raise ValueError(f"{path}: missing structure, identity or label")
    if not df.y.isin([0, 1]).all():
        raise ValueError(f"{path}: nonbinary auxiliary labels")
    for col in ("canonical_smiles", "inchikey"):
        df[col] = df[col].astype(str)
        if (df[col].str.strip() == "").any():
            raise ValueError(f"{path}: empty {col}")
        if (df[col] != df[col].str.strip()).any():
            raise ValueError(f"{path}: whitespace in {col}; use unchanged dedup output")
    if df.inchikey.duplicated().any():
        raise ValueError(f"{path}: duplicated full InChIKey")
    overlap = set(df.inchikey) & excluded_keys
    if overlap:
        raise ValueError(f"{path}: {len(overlap)} identities overlap TITAN")
    df["y"] = df.y.astype(int)
    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    require_two_classes(df.y, str(path))
    return df


def training_params(y, num_threads):
    params = dict(LGB_BASE_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight_from_y(y)
    params["num_threads"] = num_threads
    return params


def train_fixed(X, y, rounds, num_threads):
    return lgb.train(training_params(y, num_threads), lgb.Dataset(X, label=y),
                     num_boost_round=int(rounds), callbacks=[lgb.log_evaluation(0)])


def save_model(model, stem):
    stem = Path(stem)
    joblib_path = stem.with_suffix(".joblib")
    text_path = stem.with_suffix(".txt")
    joblib.dump(model, joblib_path)
    model.save_model(str(text_path))
    return {"joblib": file_record(joblib_path), "text": file_record(text_path),
            "num_trees": int(model.num_trees()), "n_features": int(model.num_feature())}


def evaluate_protocol(task, protocol, train_df, eval_df, X, output, threads, log):
    folder = output / "evaluation" / task / protocol
    folder.mkdir(parents=True)
    train_df[MEMBER_COLUMNS].assign(split="train_outer").to_csv(folder / "train_outer.csv", index=False)
    eval_df[MEMBER_COLUMNS].assign(split="eval_outer").to_csv(folder / "eval_outer.csv", index=False)
    audit = overlap_audit(train_df, eval_df)
    write_json(folder / "overlap_audit.json", audit)
    if audit["inchikey_overlap"]:
        raise ValueError(f"{task}/{protocol}: outer identity overlap")
    if protocol == "fair_scaffold" and (audit["group_id_overlap"] or audit["nonempty_scaffold_overlap"]):
        raise ValueError(f"{task}/{protocol}: outer scaffold group overlap")
    y_train = train_df.y.to_numpy(dtype=int)
    y_eval = eval_df.y.to_numpy(dtype=int)
    tr_cc, ev_cc = class_counts(y_train), class_counts(y_eval)
    result = {"task": task, "protocol": protocol, "usable": False,
              "train_n": tr_cc["n"], "train_n_pos": tr_cc["n_pos"], "train_n_neg": tr_cc["n_neg"],
              "eval_n": ev_cc["n"], "eval_n_pos": ev_cc["n_pos"], "eval_n_neg": ev_cc["n_neg"],
              "eval_pr_baseline": ev_cc["pos_frac"], "best_iteration": None,
              "auroc": None, "auprc": None}
    if ev_cc["n_pos"] < 5 or ev_cc["n_neg"] < 5:
        result["reason"] = "S8 criterion: outer eval requires >=5 positives AND >=5 negatives"
        pd.DataFrame([result]).to_csv(folder / "metrics.csv", index=False)
        log(f"[{task}/{protocol}] UNUSABLE: {result['reason']}; counts={ev_cc}")
        return result

    require_two_classes(y_train, f"{task}/{protocol} outer train")
    inner_tr, inner_es = random_split_indices(len(train_df))
    require_two_classes(y_train[inner_tr], f"{task}/{protocol} inner train")
    require_two_classes(y_train[inner_es], f"{task}/{protocol} inner ES")
    inner = pd.concat([
        train_df.iloc[inner_tr][MEMBER_COLUMNS].assign(split="inner_train"),
        train_df.iloc[inner_es][MEMBER_COLUMNS].assign(split="inner_es"),
    ], ignore_index=True)
    inner.to_csv(folder / "inner_membership.csv", index=False)
    X_train = X[train_df._row.to_numpy(dtype=int)]
    X_eval = X[eval_df._row.to_numpy(dtype=int)]
    params = training_params(y_train[inner_tr], threads)
    ds_tr = lgb.Dataset(X_train[inner_tr], label=y_train[inner_tr])
    ds_es = lgb.Dataset(X_train[inner_es], label=y_train[inner_es], reference=ds_tr)
    history = {}
    log(f"[{task}/{protocol}] outer train/eval={len(train_df)}/{len(eval_df)}; "
        f"inner train/ES={len(inner_tr)}/{len(inner_es)}; selecting rounds ...")
    model_es = lgb.train(
        params, ds_tr, num_boost_round=MAX_ROUNDS,
        valid_sets=[ds_es], valid_names=["inner_es"],
        callbacks=[lgb.early_stopping(PATIENCE, verbose=False),
                   lgb.log_evaluation(0), lgb.record_evaluation(history)],
    )
    rounds = int(model_es.best_iteration or model_es.current_iteration())
    if rounds < 1:
        raise RuntimeError(f"{task}/{protocol}: invalid best iteration {rounds}")
    write_json(folder / "inner_es_history.json", history)
    save_model(model_es, folder / "inner_es_model")
    model = train_fixed(X_train, y_train, rounds, threads)
    models = save_model(model, folder / "outer_train_model")
    p = np.asarray(model.predict(X_eval), dtype=float).reshape(-1)
    if len(p) != len(y_eval) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise RuntimeError(f"{task}/{protocol}: invalid predictions")
    pred = eval_df[MEMBER_COLUMNS].rename(columns={"y": "y_true"}).copy()
    pred["y_prob"] = p
    pred.to_csv(folder / "predictions.csv", index=False)
    saved = pd.read_csv(folder / "predictions.csv")
    # Reported metrics are recomputed from the actual saved per-molecule file.
    result.update({
        "usable": True, "best_iteration": rounds,
        "auroc": float(roc_auc_score(saved.y_true, saved.y_prob)),
        "auprc": float(average_precision_score(saved.y_true, saved.y_prob)),
        "early_stopping_set": "inner_random_20pct_of_outer_train_only",
        "early_stopping_metric": "binary_logloss", "refit_on_full_outer_train": True,
        "weighting_mechanism": "scale_pos_weight_only",
        "scale_pos_weight_inner_train": params["scale_pos_weight"],
        "scale_pos_weight_full_outer_train": scale_pos_weight_from_y(y_train),
        "inner_train_n": len(inner_tr), "inner_es_n": len(inner_es),
        "inner_train_n_pos": int(y_train[inner_tr].sum()),
        "inner_es_n_pos": int(y_train[inner_es].sum()),
        "es_evaluated_rounds": len(history.get("inner_es", {}).get("binary_logloss", [])),
        "metrics_verified_from_saved_predictions": True,
        "prediction_file": str((folder / "predictions.csv").resolve()),
        "evaluation_model": models,
    })
    pd.DataFrame([{k: v for k, v in result.items() if k != "evaluation_model"}]).to_csv(folder / "metrics.csv", index=False)
    write_json(folder / "result.json", result)
    log(f"[{task}/{protocol}] AUROC={result['auroc']:.6f}, "
        f"AUPRC(AP)={result['auprc']:.6f}, PR baseline={ev_cc['pos_frac']:.6f}, "
        f"best_iteration={rounds}")
    if result["es_evaluated_rounds"] == MAX_ROUNDS:
        log(f"[{task}/{protocol}] NOTICE: reached {MAX_ROUNDS}-round ES search cap; recorded in result")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--split-utils", type=Path, help="Optional original MechBBB split_utils.py containing compute_scaffold")
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    return args


def run(args):
    load_dependencies()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new output directory.")
    data = args.data_dir.resolve(strict=True)
    id_path = data / "titan_identity_map.csv"
    # Read ONLY identities, not the label_audit_only column in that file.
    ids = pd.read_csv(id_path, usecols=["inchikey"], dtype=str, encoding="utf-8-sig")
    if len(ids) == 0 or ids.inchikey.isna().any() or (ids.inchikey.str.strip() == "").any():
        raise ValueError("Empty/missing TITAN identities; cannot verify complete exclusion")
    exclusion = set(ids.inchikey)
    compute_scaffold, scaffold_meta = get_scaffold_function(args.split_utils)
    pools, sources = {}, {}
    # Validate all task tables before training or creating output.
    for task in TASKS:
        path = data / f"cornelissen_{task}_titan_excluded.csv"
        pools[task] = annotate_scaffolds(load_pool(path, exclusion), compute_scaffold)
        sources[task] = file_record(path)
    output.mkdir(parents=True)
    (output / "final_models").mkdir()
    log_path = output / "run.log"

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "script": file_record(__file__), "input_pools": sources,
        "titan_identity_map": file_record(id_path), "titan_identity_columns_read": ["inchikey"],
        "titan_unique_excluded_identities": len(exclusion),
        "titan_labels_used_for_auxiliary_model_selection": False,
        "extra_BBBP_exclusion": False,
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion,
                     "numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lgb.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "base_params": LGB_BASE_PARAMS, "num_threads": args.num_threads,
        "max_rounds": MAX_ROUNDS, "early_stopping_rounds": PATIENCE,
        "feature_count": 2058, "feature_names": PHYSCHEM_ORDER + [f"ECFP4_{i}" for i in range(2048)],
        "feature_dtype": "float64", "learned_feature_preprocessing": False,
        "scaffold": scaffold_meta, "empty_murcko_policy": "per_InChIKey_singleton",
        "final_iteration_source": "fair_scaffold_inner_random_ES_only",
        "protocol_or_seed_selected_by_outer_eval_score": False,
        "auprc_definition": "sklearn.average_precision_score (AP), not trapezoidal PR-AUC",
        "holdout_metrics_apply_to": "outer_train_refit_models; not final_all_pool_models",
        "scores_calibrated_as_probabilities": False,
        "evaluation": [], "locked_iterations": {}, "final_models": [],
    }
    manifest_path = output / "run_manifest.json"
    write_json(manifest_path, manifest)
    log(f"Output: {output}")
    log(f"TITAN exclusion identities={len(exclusion)}; TITAN labels are not read")
    log(f"Scaffold policy: {scaffold_meta['policy']}")
    try:
        # Evaluate every task before ANY final all-pool fit.
        for task in TASKS:
            df = pools[task]
            log(f"[{task}] pool={class_counts(df.y)}; TITAN overlap=0; computing 2058 features ...")
            X = featurize(df.canonical_smiles.tolist())
            annotated_path = output / f"pool_{task}_with_scaffolds.csv"
            df.drop(columns="_row").to_csv(annotated_path, index=False)
            train_i, eval_i = random_split_indices(len(df))
            train_s, eval_s, _ = scaffold_holdout_80_20(df)
            for protocol, tr, ev in (
                ("fair_random", df.iloc[train_i].copy(), df.iloc[eval_i].copy()),
                ("fair_scaffold", train_s, eval_s),
            ):
                res = evaluate_protocol(task, protocol, tr, ev, X, output, args.num_threads, log)
                manifest["evaluation"].append(res)
                flat = [{k: v for k, v in r.items() if k != "evaluation_model"} for r in manifest["evaluation"]]
                pd.DataFrame(flat).to_csv(output / "evaluation_summary.csv", index=False)
                if protocol == "fair_scaffold" and res["usable"]:
                    metrics = output / "evaluation" / task / protocol / "metrics.csv"
                    membership = metrics.parent / "inner_membership.csv"
                    manifest["locked_iterations"][task] = {
                        "rounds": res["best_iteration"], "protocol": protocol,
                        "selection_set": "inner_es_subset_of_scaffold_outer_train",
                        "metrics_source": file_record(metrics), "inner_membership": file_record(membership),
                        "source_pool": sources[task],
                    }
                write_json(manifest_path, manifest)
            del X

        missing = set(TASKS) - set(manifest["locked_iterations"])
        if missing:
            raise RuntimeError(f"No usable scaffold evaluation for {sorted(missing)}; "
                               "final refitting stopped. No fallback to random or old iteration numbers.")
        # Written before final fitting; performance comparisons do not alter it.
        write_json(output / "locked_iterations.json", manifest["locked_iterations"])
        log("Locked new rounds: " + str({t: manifest["locked_iterations"][t]["rounds"] for t in TASKS}))
        for task in TASKS:
            locked = manifest["locked_iterations"][task]
            for rec in (sources[task], locked["metrics_source"], locked["inner_membership"]):
                if sha256_file(rec["path"]) != rec["sha256"]:
                    raise RuntimeError(f"File changed during run: {rec['path']}")
            df = pools[task]
            X = featurize(df.canonical_smiles.tolist())
            y = df.y.to_numpy(dtype=int)
            rounds = locked["rounds"]
            log(f"[{task}] final fit: ALL {len(df)} auxiliary rows, rounds={rounds}, "
                f"scale_pos_weight={scale_pos_weight_from_y(y):.6f}")
            model = train_fixed(X, y, rounds, args.num_threads)
            artifacts = save_model(model, output / "final_models" / task)
            # Serialization check confirms the delivered model preserves scores.
            reloaded = joblib.load(artifacts["joblib"]["path"])
            probe = X[:min(32, len(X))]
            if not np.allclose(model.predict(probe), reloaded.predict(probe), rtol=0, atol=1e-12):
                raise RuntimeError(f"{task}: model serialization prediction mismatch")
            rec = {"task": task, "train_counts": class_counts(y), "requested_rounds": rounds,
                   "params": training_params(y, args.num_threads), "models": artifacts,
                   "iteration_provenance": locked, "held_out_fraction_in_final_fit": 0.0,
                   "serialization_prediction_check": True}
            manifest["final_models"].append(rec)
            if artifacts["num_trees"] != rounds:
                log(f"[{task}] NOTICE: requested {rounds} rounds, actual trees={artifacts['num_trees']}")
            write_json(manifest_path, manifest)
            del X, model, reloaded
        manifest["status"] = "complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        log("DONE: evaluation_summary.csv; locked_iterations.json; final_models/{efflux,influx,pampa}.joblib")
        log("Reported holdout metrics belong to outer-train models, not independent tests of final all-pool models.")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_json(manifest_path, manifest)
        (output / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log(f"FAILED: {exc}")
        raise


if __name__ == "__main__":
    try:
        run(parse_args())
    except ModuleNotFoundError as exc:
        sys.exit(f"Missing dependency: {exc.name}. Install it in the active titan_bbb environment, then rerun.")
