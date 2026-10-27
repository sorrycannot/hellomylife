#!/usr/bin/env python3
"""Adapt MechBBB S2 ingestion and S8/S9 full-InChIKey exclusion to TITAN.

Requires RDKit, plus the supplied MechBBB chem.py. No training is performed.
Only Cornelissen efflux/influx/PAMPA labels are used; BBBP is not excluded.
Both sources are identified by the same standardize_and_identify() function.

Example from TITAN-BBB-main/src, with this script and chem.py in src:
  python deduplicate_auxiliary_titan.py \
    --source ../dataset/20220103_table.csv \
    --titan-dir ../dataset --output-dir ../auxiliary_titan

S2 originally imports src.utils_chem. This adapter uses the supplied chem.py,
whose docstring says it matches revision utils_chem; byte identity to the
unprovided utils_chem is not independently verified. The original S2/S8/S9
label conflict, keep-first and complete-InChIKey exclusion policies are used.
Extra guards reject nonbinary labels and unavailable standardization support.
Output must be a NEW directory. Input files are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path


TASKS = {"efflux": "Status Efflux", "influx": "Status Influx", "pampa": "Status PAMPA"}
SPLITS = ("train", "val", "test")
DATA_COLUMNS = ["canonical_smiles", "inchikey", "y", "source_dataset"]
AUDIT_COLUMNS = [
    "source_record", "original_smiles", "original_label", "canonical_smiles",
    "inchikey", "y", "decision", "matched_titan_splits",
]


def read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path}: malformed CSV row / extra columns")
    return rows


def write_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_missing(value):
    return value is None or str(value).strip().lower() in {"", "nan", "na", "null", "none"}


def binary_label(value):
    if is_missing(value):
        return None
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Nonbinary label: {value!r}") from None
    if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise ValueError(f"Nonbinary label: {value!r}")
    return int(numeric)


def load_standardizer(path):
    # Import the provided file explicitly; do not accidentally import another chem.
    path = Path(path).resolve(strict=True)
    name = "_provided_mechbbb_chem"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not getattr(module, "_HAS_STANDARDIZE", False):
        raise RuntimeError("RDKit standardization support is unavailable; refusing to skip cleanup.")
    from rdkit import rdBase

    @lru_cache(maxsize=None)
    def normalize(smiles):
        if is_missing(smiles):
            return None
        return module.standardize_and_identify(str(smiles).strip(), None)

    return normalize, rdBase.rdkitVersion


def load_titan(titan_dir, normalize):
    identities, by_split, input_paths = [], {}, []
    for split in SPLITS:
        path = Path(titan_dir) / f"{split}_classification.csv"
        rows = read_csv(path, ["SMILES", "label"])
        if not rows:
            raise ValueError(f"Empty TITAN split: {path}")
        input_paths.append(path)
        by_split[split] = set()
        for row_number, row in enumerate(rows, start=1):
            mid = normalize(row.get("SMILES"))
            if mid is None:
                # Skipping target molecules would make exclusion incomplete.
                raise ValueError(f"TITAN {split}, data record {row_number}: structure identification failed")
            by_split[split].add(mid.inchikey)
            identities.append({
                "split": split, "record": row_number,
                "original_smiles": row["SMILES"], "label_audit_only": row["label"],
                "supplied_inchikey": row.get("InChI_Key", ""),
                "canonical_smiles": mid.canonical_smiles, "inchikey": mid.inchikey,
            })
        print(f"TITAN {split}: {len(rows)} rows, {len(by_split[split])} standardized identities", flush=True)
    return identities, by_split, input_paths


def clean_task(rows, task, normalize, titan_by_split):
    column = TASKS[task]
    groups, audit = defaultdict(list), []
    counts = Counter()
    for record, row in enumerate(rows, start=1):
        raw_label = row.get(column)
        if is_missing(raw_label):
            counts["missing_label_rows"] += 1
            continue
        item = {
            "source_record": record, "original_smiles": row.get("ParentSmiles", ""),
            "original_label": raw_label, "canonical_smiles": "", "inchikey": "",
            "y": "", "decision": "", "matched_titan_splits": "",
        }
        audit.append(item)
        try:
            label = binary_label(raw_label)
        except ValueError:
            item["decision"] = "invalid_label"
            counts["invalid_label_rows"] += 1
            continue
        counts["rows_with_binary_label"] += 1
        counts[f"raw_label_{label}"] += 1
        mid = normalize(row.get("ParentSmiles"))
        if mid is None:
            item["decision"] = "invalid_structure"
            counts["invalid_structure_rows"] += 1
            continue
        item.update(canonical_smiles=mid.canonical_smiles, inchikey=mid.inchikey, y=label)
        groups[mid.inchikey].append(item)

    processed, remaining = [], []
    for key, members in groups.items():
        # Same policy as S2: remove the ENTIRE identity if labels disagree.
        if len({m["y"] for m in members}) > 1:
            counts["conflict_identity_groups_removed"] += 1
            counts["conflict_rows_removed"] += len(members)
            for item in members:
                item["decision"] = "conflicting_labels"
            continue

        representative = members[0]  # S2: keep first, preserving source order.
        data = {key_: representative[key_] for key_ in ("canonical_smiles", "inchikey", "y")}
        data["source_dataset"] = "Cornelissen"
        processed.append(data)
        matched = [split for split in SPLITS if key in titan_by_split[split]]
        for item in members:
            item["matched_titan_splits"] = ";".join(matched)
        for item in members[1:]:
            item["decision"] = "duplicate_same_label"
            counts["same_label_duplicate_rows_removed"] += 1

        if matched:
            representative["decision"] = "overlap_with_titan"
            counts["titan_overlap_identity_groups_removed"] += 1
        else:
            representative["decision"] = "retained"
            remaining.append(data)

    all_titan = set().union(*titan_by_split.values())
    kept_keys = [r["inchikey"] for r in remaining]
    if len(kept_keys) != len(set(kept_keys)) or set(kept_keys) & all_titan:
        raise RuntimeError(f"{task}: final identity uniqueness / exclusion check failed")
    if len(audit) != sum(bool(a["decision"]) for a in audit):
        raise RuntimeError(f"{task}: incomplete row audit")
    # Account for every raw binary-labeled record across sequential filtering.
    accounted = (
        counts["invalid_structure_rows"] + counts["conflict_rows_removed"]
        + counts["same_label_duplicate_rows_removed"] + len(processed)
    )
    if accounted != counts["rows_with_binary_label"]:
        raise RuntimeError(f"{task}: row reconciliation failed")

    def balance(data):
        return {str(v): sum(r["y"] == v for r in data) for v in (0, 1)}

    prefixes = {key.split("-")[0] for key in all_titan}
    report = dict(counts)
    report.update({
        "task": task, "label_column": column,
        "processed_before_titan_n": len(processed),
        "processed_before_titan_class_balance": balance(processed),
        "remaining_n": len(remaining), "remaining_class_balance": balance(remaining),
        "remaining_full_inchikey_overlap": len(set(kept_keys) & all_titan),
        "remaining_with_shared_connectivity_prefix": sum(k.split("-")[0] in prefixes for k in kept_keys),
        "connectivity_note": "Prefix overlap is reported only; it does not trigger removal.",
        "both_classes_remaining": all(balance(remaining)[str(v)] > 0 for v in (0, 1)),
    })
    return processed, remaining, audit, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="Cornelissen raw CSV")
    parser.add_argument("--titan-dir", required=True, type=Path, help="Directory with the three classification CSVs")
    parser.add_argument("--chem", type=Path, default=Path(__file__).with_name("chem.py"), help="Provided MechBBB chem.py")
    parser.add_argument("--output-dir", required=True, type=Path, help="New output directory; existing paths are refused")
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"Output already exists: {args.output_dir}. Choose a new directory name.")
    source = args.source.resolve(strict=True)
    rows = read_csv(source, ["ParentSmiles", *TASKS.values()])
    normalize, rdkit_version = load_standardizer(args.chem)
    titan, by_split, titan_paths = load_titan(args.titan_dir, normalize)
    products = {}
    for task in TASKS:
        print(f"Cleaning and excluding TITAN: {task} ...", flush=True)
        products[task] = clean_task(rows, task, normalize, by_split)

    input_paths = [source, *titan_paths, args.chem.resolve(), Path(__file__).resolve()]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "rdkit_version": rdkit_version,
        "input_files": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in input_paths],
        "source_records_including_empty": len(rows),
        "source_all_fields_empty_records": sum(not any(str(v or "").strip() for v in r.values()) for r in rows),
        "policy": {
            "identity": "full InChIKey recomputed after identical standardization on both sources",
            "standardizer": "supplied chem.py: Cleanup -> FragmentParent -> Uncharger",
            "conflicts": "S2: remove all rows for an identity with multiple task labels",
            "duplicates": "S2: keep first same-label identity in original source order",
            "external_exclusion": "S8/S9 full InChIKey match, using TITAN train+val+test instead of BBBP",
            "bbbp_excluded_separately": False, "titan_labels_used_for_filtering": False,
            "extra_guards": "require standardization module; reject nonbinary labels; abort on invalid TITAN structures",
            "provenance_limit": "S2 imports src.utils_chem; supplied chem.py claims equivalence, not byte-verified here",
        },
        "titan_rows": len(titan), "titan_unique_standardized_identities": len(set().union(*by_split.values())),
        "titan_split_identity_overlaps": {
            f"{a}__{b}": len(by_split[a] & by_split[b]) for a, b in combinations(SPLITS, 2)
        },
        "tasks": {task: product[3] for task, product in products.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "titan_identity_map.csv", titan, list(titan[0]))
    for task, (processed, remaining, audit, stats) in products.items():
        write_csv(args.output_dir / f"cornelissen_{task}_before_titan.csv", processed, DATA_COLUMNS)
        write_csv(args.output_dir / f"cornelissen_{task}_titan_excluded.csv", remaining, DATA_COLUMNS)
        write_csv(args.output_dir / f"{task}_row_audit.csv", audit, AUDIT_COLUMNS)
        print(
            f"{task}: labeled={stats.get('rows_with_binary_label', 0)}, "
            f"cleaned={stats['processed_before_titan_n']}, "
            f"TITAN overlap={stats.get('titan_overlap_identity_groups_removed', 0)}, "
            f"remaining={stats['remaining_n']} "
            f"(positive={stats['remaining_class_balance']['1']}, negative={stats['remaining_class_balance']['0']})",
            flush=True,
        )
        if not stats["both_classes_remaining"]:
            print(f"WARNING: {task} has no remaining samples in at least one class.", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE. Output: {args.output_dir.resolve()}", flush=True)
    print("No models were trained; original inputs and TITAN partitions are unchanged.")


if __name__ == "__main__":
    main()
