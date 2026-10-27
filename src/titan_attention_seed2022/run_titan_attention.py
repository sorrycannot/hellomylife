#!/usr/bin/env python3
"""TITAN classification, default seed=42: attention ablations + late control.

Place this folder under TITAN-BBB-main/src and run from src:
  python titan_attention_seed2022/run_titan_attention.py --project-root ..
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from titan_attention_seed2022 import trainer as core
from titan_attention_seed2022.common import (
    SPLITS, SCORE_COLUMNS, classification_metrics, file_record, read_json, sha256, write_json,
)

PLAN = {
    "baseline": None,
    "zero_control": (0, 0, 0),
    "efflux_only": (1, 0, 0),
    "influx_only": (0, 1, 0),
    "pampa_only": (0, 0, 1),
    "efflux_influx": (1, 1, 0),
    "efflux_pampa": (1, 0, 1),
    "influx_pampa": (0, 1, 1),
    "full_fusion": (1, 1, 1),
    "late_concat": (1, 1, 1),
}
GROUPS = tuple(PLAN)
ATTENTION_GROUPS = set(GROUPS) - {"baseline", "late_concat"}
EXPECTED_COUNTS = {"train": 7593, "val": 727, "test": 942}
EXPECTED_DIMS = {"tabular": 384, "image": 2048, "text": 768}
METRICS = ["AUROC", "AUPRC_AP", "ACC", "BACC", "SN", "SP", "F1", "MCC"]


def load_titan(root, mechanism_dir):
    """Use the original TITAN files and their verified existing score manifest."""
    meta = read_json(mechanism_dir / "manifest.json")
    if meta.get("task") != "classification":
        raise ValueError("Expected the original TITAN classification mechanism_features/manifest.json")
    arrays, frames, scalers, provenance = core.load_inputs(root, mechanism_dir)
    counts = {s: len(frames[s]) for s in SPLITS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Not the user's original TITAN partitions: {counts}; expected {EXPECTED_COUNTS}. "
                         "Use PROJECT_ROOT/dataset, not a DeepB3 result directory.")
    for s in SPLITS:
        for modality, dimension in EXPECTED_DIMS.items():
            if arrays[s][modality].shape != (counts[s], dimension):
                raise ValueError(f"{s}/{modality}: expected {counts[s]} x {dimension}")
            if not np.isfinite(arrays[s][modality]).all():
                raise ValueError(f"{s}/{modality}: scaling produced nonfinite values")
    # The original manifest binds the current CSV bytes, per-row SMILES and
    # raw score arrays to the previously TITAN-excluded auxiliary model run.
    provenance.update(dataset="original TITAN classification", label_mapping="unchanged TITAN labels",
                      counts=counts, partition_policy="original train/val/test; no resplit")
    return arrays, frames, scalers, provenance


def mechanism_summary(arrays):
    rows = []
    for split in SPLITS:
        x = arrays[split]["mechanism"]
        for i, column in enumerate(SCORE_COLUMNS):
            rows.append(dict(split=split, score=column, n=len(x), minimum=float(x[:, i].min()),
                             maximum=float(x[:, i].max()), mean=float(x[:, i].mean()),
                             std=float(x[:, i].std()), distinct=int(len(np.unique(x[:, i])))))
    return rows


def mask_arrays(arrays, group, splits):
    """Mask BOTH the attention context and late-head inputs via the same m.

    Keep all three column positions. Never drop/reorder features or mutate
    the original cached mechanism arrays. Baseline ignores m entirely.
    """
    if group not in PLAN:
        raise ValueError(f"Unknown experiment: {group}")
    mask = PLAN[group]
    result = {}
    for split in splits:
        source = arrays[split]
        result[split] = dict(source)
        result[split]["mechanism_raw"] = source["mechanism"]
        result[split]["mechanism"] = (np.zeros_like(source["mechanism"]) if mask is None else
            source["mechanism"]*np.asarray(mask, dtype=np.float32)[None, :])
    return result


def conditioned_smoke_test(base_smoke):
    """Real Torch gradient and attention-sensitivity checks on the server."""
    def check(model, mode, dims, device):
        info = base_smoke(model, mode, dims, device)
        if not hasattr(model, "mechanism_context"):
            return info
        torch = core.torch
        was_training = model.training
        original = model.mechanism_context.weight.detach().clone()
        model.eval()
        try:
            xs = [torch.sin(torch.arange(4*d, device=device, dtype=torch.float32)*.17).reshape(4, d)
                  for d in dims]
            m = torch.tensor([[.1, .3, .8], [.8, .2, .1], [.2, .9, .4], [.7, .4, .6]],
                             device=device, dtype=torch.float32)
            _, alpha = model(*xs, m, return_attention_weights=True)
            # Probe the ATTENTION path directly, not gradients through late concat.
            contrast = torch.tensor([1., 0., -1.], device=device)
            (alpha.squeeze(-1)*contrast).sum().backward()
            grad = model.mechanism_context.weight.grad
            grad_norm = float(grad.abs().sum().item()) if grad is not None else 0.
            if grad is None or not torch.isfinite(grad).all() or grad_norm <= 0:
                raise RuntimeError("Mechanism-to-attention path has no finite nonzero gradient")
            with torch.no_grad():
                w = model.mechanism_context.weight
                w.copy_(.3*torch.sin(torch.arange(w.numel(), device=device, dtype=w.dtype)).reshape_as(w))
                _, real = model(*xs, m, return_attention_weights=True)
                _, disabled = model(*xs, m, return_attention_weights=True,
                                    attention_mechanism=torch.zeros_like(m))
                delta = float((real-disabled).abs().max().item())
                if not torch.isfinite(real).all() or delta <= 1e-9:
                    raise RuntimeError("Changing only attention context did not change attention weights")
                if not torch.allclose(real.sum(dim=1), torch.ones((4, 1), device=device), atol=1e-6):
                    raise RuntimeError("Attention weights do not sum to 1 across the three modalities")
            info.update(attention_context_gradient_l1=grad_norm,
                        probe_attention_max_change=delta, probe_weights_restored=True)
        finally:
            with torch.no_grad():
                model.mechanism_context.weight.copy_(original)
            model.zero_grad(set_to_none=True)
            model.train(was_training)
        return info
    return check


def select_model(group):
    from titan_attention_seed2022.model import LateConcatModel, MechanismAttentionModel
    if group not in PLAN:
        raise ValueError(f"Unknown experiment: {group}")
    core.SMILESMechanismModel = MechanismAttentionModel if group in ATTENTION_GROUPS else LateConcatModel
    return "baseline" if group == "baseline" else "mechanism"


def build_model(group, config, checkpoint, device):
    mode = select_model(group)
    cls = core.SMILESModel if mode == "baseline" else core.SMILESMechanismModel
    model = cls(config["d_tab"], config["d_img"], config["d_txt"], 2048, .1).to(device)
    # These are this experiment's own, locally created state_dict checkpoints.
    model.load_state_dict(core.torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def infer_attention(model, group, arrays, device, attention_override=None):
    torch = core.torch
    loader = core.DataLoader(core.FusionDataset(arrays), batch_size=32, shuffle=False)
    logits, weights = [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            xs = [v.to(device) for v in batch[:-1]]
            kwargs = {"return_attention_weights": True}
            if attention_override is not None:
                kwargs["attention_mechanism"] = torch.as_tensor(
                    attention_override[offset:offset+len(xs[0])], dtype=torch.float32, device=device)
            z, alpha = model(*(xs[:3] if group == "baseline" else xs[:4]), **kwargs)
            logits.append(z.cpu().numpy())
            weights.append(alpha.squeeze(-1).cpu().numpy())
            offset += len(z)
    z, a = np.concatenate(logits), np.concatenate(weights)
    if not np.isfinite(z).all() or a.shape != (len(z), 3) or not np.isfinite(a).all():
        raise RuntimeError("Invalid saved attention inference")
    if not np.allclose(a.sum(axis=1), 1., atol=1e-6):
        raise RuntimeError("Invalid attention normalization")
    p = torch.sigmoid(torch.from_numpy(z)).numpy()
    return z, p, a


def export_attention(group, config, lock, folder, arrays, frames, device):
    model = build_model(group, config, folder / "best_model.pth", device)
    for split in ("val", "test"):
        z, p, a = infer_attention(model, group, arrays[split], device)
        saved = pd.read_csv(folder / f"{split}_predictions.csv")
        if not np.allclose(saved.probability, p, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"{group}/{split}: attention export disagrees with locked predictions")
        table = pd.DataFrame({"record": np.arange(1, len(p)+1), "SMILES": frames[split].SMILES,
                              "y_true": arrays[split]["labels"], "probability": p})
        for i, name in enumerate(("tabular", "image", "text")):
            table[f"attention_{name}"] = a[:, i]
        for i, name in enumerate(SCORE_COLUMNS):
            table[name] = arrays[split]["mechanism"][:, i]
            table[f"raw_{name}"] = arrays[split]["mechanism_raw"][:, i]
        table.to_csv(folder / f"{split}_attention.csv", index=False)
        if split == "val" and group in ATTENTION_GROUPS:
            real_z, real_a = z, a
            m = arrays[split]["mechanism"]
            permutation = np.random.default_rng(config["seed"]).permutation(len(m))
            diagnostics = []
            for name, override in (("actual_context", None),
                                   ("attention_context_disabled", np.zeros_like(m)),
                                   ("attention_context_permuted", m[permutation])):
                dz, dp, da = (z, p, a) if override is None else infer_attention(
                    model, group, arrays[split], device, override)
                diagnostics.append(dict(condition=name,
                    mean_abs_logit_change=float(np.abs(dz-real_z).mean()),
                    mean_attention_l1_change=float(np.abs(da-real_a).sum(axis=1).mean()),
                    **classification_metrics(arrays[split]["labels"], dp, lock["threshold"])))
            pd.DataFrame(diagnostics).to_csv(folder / "val_context_diagnostic.csv", index=False)
            w = model.mechanism_context.weight.detach().cpu().numpy()
            write_json(folder / "learned_context_summary.json", dict(
                columns=SCORE_COLUMNS, column_l2_norms=np.linalg.norm(w, axis=0).tolist(),
                attention_order=["tabular", "image", "text"],
                note="Diagnostics only; no model or threshold selected from these interventions."))
    del model
    if device.type == "cuda":
        core.torch.cuda.empty_cache()


def annotate_predictions(folder, split, frame, locked_threshold):
    p = folder / f"{split}_predictions.csv"
    df = pd.read_csv(p)
    if len(df) != len(frame) or not np.array_equal(df.record, np.arange(1, len(frame)+1)):
        raise ValueError("Prediction row order differs from original TITAN partition")
    if not np.array_equal(df.y_true.to_numpy(), frame.label.to_numpy()):
        raise ValueError("Prediction label coding differs from original TITAN labels")
    if "SMILES" in df and df.SMILES.tolist() != frame.SMILES.tolist():
        raise ValueError("Prediction SMILES changed")
    df["SMILES"] = frame.SMILES.to_numpy()
    df["locked_threshold"] = locked_threshold
    for threshold in (.5, .7):
        df[f"prediction_fixed_{str(threshold).replace('.', 'p')}"] = (df.probability > threshold).astype(int)
    df.to_csv(p, index=False)
    return df.probability.to_numpy()


def write_contrasts(output, selected, fixed):
    """Predefined descriptive contrasts, without test-based group selection."""
    pairs = [("zero_control", "baseline"), ("full_fusion", "baseline"),
             ("full_fusion", "zero_control"), ("full_fusion", "late_concat"),
             ("efflux_only", "zero_control"), ("influx_only", "zero_control"),
             ("pampa_only", "zero_control"),
             ("full_fusion", "influx_pampa"),  # remove efflux
             ("full_fusion", "efflux_pampa"),  # remove influx
             ("full_fusion", "efflux_influx")]
    rows = []
    for split in ("val", "test"):
        for policy, source in (("validation_selected", selected), ("fixed_0.5", fixed), ("fixed_0.7", fixed)):
            table = source[source.split.eq(split) & source.threshold_policy.eq(policy)].set_index("experiment")
            for a, b in pairs:
                if a not in table.index or b not in table.index:
                    continue
                aa, bb = table.loc[a], table.loc[b]
                rows.append(dict(split=split, threshold_policy=policy, experiment_a=a, experiment_b=b,
                    threshold_a=float(aa.threshold), threshold_b=float(bb.threshold),
                    **{f"delta_{m}": float(aa[m]-bb[m]) for m in METRICS}))
    if rows:
        pd.DataFrame(rows).to_csv(output / "contrast_summary.csv", index=False)
    seed = int(selected.seed.iloc[0])
    lines = [f"# TITAN mechanism-attention ablation, seed={seed}", "",
             "Main results: each model's threshold is selected on validation, then locked for test.",
             "All deltas in contrast_summary.csv are descriptive A minus B; no significance claim.", ""]
    for split in ("val", "test"):
        lines.extend([f"## {split}", "", "| Experiment | Threshold | " + " | ".join(METRICS) + " |",
                      "|---|---:|" + "---:|"*len(METRICS)])
        for _, r in selected[selected.split.eq(split)].iterrows():
            lines.append(f"| {r.experiment} | {r.threshold:.1f} | " + " | ".join(f"{r[m]:.6f}" for m in METRICS) + " |")
        lines.append("")
    (output / "ablation_table.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(args):
    root = args.project_root.resolve(strict=True)
    mechanism_dir = (args.mechanism_dir or root / "mechanism_features").resolve(strict=True)
    arrays, frames, scalers, provenance = load_titan(root, mechanism_dir)
    print(f"TITAN rows: {provenance['counts']}; labels unchanged; seed={args.seed}", flush=True)
    print("Mechanism order: p_efflux, p_influx, p_pampa; raw scores retained.", flush=True)
    if args.check_only:
        print(pd.DataFrame(mechanism_summary(arrays)).to_string(index=False))
        print("CHECK PASSED: TITAN inputs and score alignment verified; no training/output directory created.")
        return
    core.load_dependencies(root)
    torch = core.torch
    requested = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(requested)
    # Keep the original TITAN seed helper, optimizer, dropout, batch and losses.
    core.architecture_smoke_test = conditioned_smoke_test(core.architecture_smoke_test)
    output = (args.output_dir or root / f"titan_attention_ablation_seed{args.seed}_results").resolve()
    sources = [file_record(p) for p in sorted(HERE.glob("*.py"))]
    sources += [file_record(root / "src/utils/model.py"), file_record(root / "src/utils/utils.py")]
    definition = dict(seed=args.seed, groups=args.groups, plan=PLAN, epochs=args.epochs, device=requested,
                      data=provenance, sources=sources,
                      versions=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
                                    sklearn=sklearn.__version__, joblib=joblib.__version__,
                                    python=platform.python_version()),
                      threshold_policy="validation BACC grid 0.1/0.3/0.5/0.7/0.9; first tie",
                      mechanism_scaling="none; unchanged from the late-concat control")
    signature = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "run_manifest.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Output exists: {output}; use --resume or a new --output-dir")
        meta = read_json(manifest_path)
        if meta.get("signature") != signature:
            raise ValueError("Inputs, source, environment or experiment settings changed; use a new --output-dir")
    else:
        if args.resume:
            raise FileNotFoundError(f"Cannot resume missing output: {output}")
        output.mkdir(parents=True)
        meta = dict(status="running", signature=signature, definition=definition, fitted={},
                    dataset="TITAN classification", test_used_for_selection=False)
        joblib.dump(scalers, output / "titan_modality_scalers.joblib")
        pd.DataFrame(mechanism_summary(arrays)).to_csv(output / "mechanism_score_summary.csv", index=False)
    meta["status"] = "running"
    write_json(manifest_path, meta)
    print(f"Output: {output}; groups={args.groups}; device={device}", flush=True)
    try:
        fitted = []
        for group in args.groups:
            mode = select_model(group)
            trainval = mask_arrays(arrays, group, ("train", "val"))
            old = meta["fitted"].get(group)
            if old:
                folder = output / old["folder"]
                for key, name in (("config", "model_config.json"), ("lock", "selection_lock.json")):
                    if sha256(folder / name) != old[key]["sha256"]:
                        raise ValueError(f"{group}: saved {key} changed")
                config, lock = read_json(folder / "model_config.json"), read_json(folder / "selection_lock.json")
                if sha256(folder / "best_model.pth") != lock["checkpoint"]["sha256"]:
                    raise ValueError(f"{group}: checkpoint changed")
                print(f"[{group}] reuse completed validation selection", flush=True)
            else:
                group_dir = output / "experiments" / group
                group_dir.mkdir(parents=True, exist_ok=True)
                attempt = 1
                while (group_dir / f"attempt_{attempt:03d}").exists():
                    attempt += 1
                attempt_dir = group_dir / f"attempt_{attempt:03d}"
                attempt_dir.mkdir()
                print(f"[{group}] training from scratch, seed={args.seed}", flush=True)
                folder, config, lock = core.train_one(mode,
                    SimpleNamespace(seed=args.seed, epochs=args.epochs), trainval, attempt_dir, device)
                config.update(experiment=group, dataset="TITAN original classification", task="classification",
                    mechanism_mask=PLAN[group], mask_applied_to="attention context AND late prediction head",
                    attention_conditioned_on_mechanisms=group in ATTENTION_GROUPS,
                    attention_equation=("v^T tanh(W h_j+b+U m)" if group in ATTENTION_GROUPS else "v^T tanh(W h_j+b)"),
                    mechanism_scaling="none; raw frozen scores", late_concat=group != "baseline",
                    initial_attention_context="zero" if group in ATTENTION_GROUPS else None,
                    trainable_modules=("TITAN projections, attention, prediction head" +
                                       (", mechanism context U" if group in ATTENTION_GROUPS else "")))
                write_json(folder / "model_config.json", config)
                meta["fitted"][group] = dict(folder=str(folder.relative_to(output)),
                    config=file_record(folder / "model_config.json"), lock=file_record(folder / "selection_lock.json"))
                write_json(manifest_path, meta)
            fitted.append((group, folder, config, lock))
        # EVERY requested model/epoch/threshold is locked before ANY test scoring.
        write_json(output / "all_selections_locked.json", meta["fitted"])
        selected, fixed = [], []
        for group, folder, config, lock in fitted:
            select_model(group)
            masked = mask_arrays(arrays, group, SPLITS)
            test = core.evaluate_test(folder, config, lock, {"test": masked["test"]}, device)
            for split, metrics in (("val", lock["validation"]), ("test", test)):
                p = annotate_predictions(folder, split, frames[split], lock["threshold"])
                prefix = dict(experiment=group, seed=args.seed, split=split, best_epoch=lock["best_epoch"])
                selected.append(dict(**prefix, threshold_policy="validation_selected", **metrics))
                for threshold in (.5, .7):
                    fixed.append(dict(**prefix, threshold_policy=f"fixed_{threshold}",
                        **classification_metrics(arrays[split]["labels"], p, threshold)))
            export_attention(group, config, lock, folder, masked, frames, device)
        selected, fixed = pd.DataFrame(selected), pd.DataFrame(fixed)
        selected.to_csv(output / "comparison.csv", index=False)
        for split in ("val", "test"):
            s, f = selected[selected.split.eq(split)], fixed[fixed.split.eq(split)]
            s.to_csv(output / f"{split}_comparison.csv", index=False)
            for threshold in (.5, .7):
                f[f.threshold.eq(threshold)].to_csv(output / f"{split}_fixed_threshold{int(threshold*10):02d}.csv", index=False)
            merged = pd.concat([s, f], ignore_index=True).drop_duplicates(["experiment", "threshold"], keep="first")
            merged.to_csv(output / f"{split}_all_thresholds.csv", index=False)
        write_contrasts(output, selected, fixed)
        meta.update(status="complete", comparison=file_record(output / "comparison.csv"))
        write_json(manifest_path, meta)
        print(selected[selected.split.eq("test")][["experiment", "threshold", *METRICS]].to_string(index=False))
        print(f"DONE: {output}/test_comparison.csv", flush=True)
    except Exception as exc:
        meta.update(status="failed", error=str(exc))
        write_json(manifest_path, meta)
        (output / "error_traceback.txt").write_text(traceback.format_exc())
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--mechanism-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--groups", nargs="+", choices=(*GROUPS, "mechanism_attention"), default=list(GROUPS),
                        help="Default: 9-group attention ablation plus the old late-concat control")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.groups = ["full_fusion" if g == "mechanism_attention" else g for g in args.groups]
    if args.epochs < 1 or not 0 <= args.seed < 2**32 or len(set(args.groups)) != len(args.groups):
        parser.error("Use positive epochs, seed in [0, 2**32), and distinct groups")
    return args


if __name__ == "__main__":
    run(parse_args())
