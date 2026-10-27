#!/usr/bin/env python3
"""Run the supplied original S2 unchanged, then verify S9 EXPECTED_PROCESSED.

This is a verification step BEFORE TITAN/BBBP exclusion and before training.
Requires pandas and RDKit. No network connection, GPU or LightGBM is needed.

Example (all five files in the current directory):
  python verify_cornelissen_s2.py --source 20220103_table.csv \
    --s2 S2_Cornelissen_Ingestion.py --s9 S9_Stage1_Final_Refit.py \
    --chem chem.py --output-dir s2_verification

--chem should ideally be S2's actual src/utils_chem.py. If only the supplied
deployment chem.py is available, it can be used, but any mismatch must also
be investigated against that dependency. The script records the exact file.

The verifier copies inputs to a NEW isolated directory, preserving original
S2 code and its pandas CSV serialization. S9 is parsed as text via ast; S9
training code is NOT executed. Existing output directories are refused.

A PASS verifies byte-identical processed efflux/influx/PAMPA CSVs against
S9's recorded SHA256 values. It does not prove that the original raw CSVs
were byte-identical: different raw formatting can yield identical outputs.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import platform
import runpy
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


TASKS = ("efflux", "influx", "pampa")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_s9_expected(path):
    """Read only a literal assignment; never execute S9 or its imports."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_PROCESSED"
            for target in node.targets
        ):
            matches.append(ast.literal_eval(node.value))
    if len(matches) != 1:
        raise ValueError("S9 must contain exactly one literal EXPECTED_PROCESSED assignment")
    expected = matches[0]
    for task in TASKS:
        value = expected.get(task)
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"Invalid EXPECTED_PROCESSED entry: {task}")
        digest, n, n_pos, n_neg = value
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid SHA256 in S9: {task}")
        if any(type(v) is not int or v < 0 for v in (n, n_pos, n_neg)) or n_pos + n_neg != n:
            raise ValueError(f"Invalid class counts in S9: {task}")
    return expected


def verify_processed(directory, expected):
    import pandas as pd

    rows = []
    for task in TASKS:
        exp_hash, exp_n, exp_pos, exp_neg = expected[task]
        path = Path(directory) / f"cornelissen_{task}.csv"
        record = {
            "task": task, "file": str(path), "expected_sha256": exp_hash,
            "expected_n": exp_n, "expected_positive": exp_pos, "expected_negative": exp_neg,
        }
        if not path.is_file():
            record.update(status="MISSING_OUTPUT", passed=False)
            rows.append(record)
            continue
        actual_hash = sha256_file(path)
        record["actual_sha256"] = actual_hash
        record["sha256_match"] = actual_hash == exp_hash
        try:
            frame = pd.read_csv(path)
            expected_columns = ["canonical_smiles", "inchikey", "y", "source_dataset"]
            schema_ok = list(frame.columns) == expected_columns
            numeric_y = pd.to_numeric(frame["y"], errors="raise")
            n_pos = int((numeric_y == 1).sum())
            n_neg = int((numeric_y == 0).sum())
            valid_labels = n_pos + n_neg == len(frame)
            counts_match = (len(frame), n_pos, n_neg) == (exp_n, exp_pos, exp_neg)
            no_missing_id = not frame[["canonical_smiles", "inchikey"]].isna().any().any()
            unique_ids = frame["inchikey"].nunique() == len(frame)
            source_ok = frame["source_dataset"].eq("Cornelissen").all()
            integrity = bool(schema_ok and valid_labels and no_missing_id and unique_ids and source_ok)
            record.update(
                actual_n=len(frame), actual_positive=n_pos, actual_negative=n_neg,
                counts_match=bool(counts_match), schema_match=bool(schema_ok),
                identity_integrity=integrity,
            )
            passed = bool(counts_match and integrity and actual_hash == exp_hash)
            record["passed"] = passed
            if passed:
                record["status"] = "PASS"
            elif counts_match and integrity:
                record["status"] = "COUNTS_MATCH_HASH_DIFF"
            else:
                record["status"] = "DATA_MISMATCH"
        except Exception as error:
            record.update(status="UNREADABLE_OR_INVALID_SCHEMA", passed=False, error=str(error))
        rows.append(record)
    return rows


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="Your original 20220103_table.csv")
    parser.add_argument("--s2", type=Path, default=Path("S2_Cornelissen_Ingestion.py"))
    parser.add_argument("--s9", type=Path, default=Path("S9_Stage1_Final_Refit.py"))
    parser.add_argument("--chem", type=Path, default=Path("chem.py"), help="Prefer original utils_chem.py when available")
    parser.add_argument("--output-dir", required=True, type=Path, help="NEW verification directory")
    args = parser.parse_args()
    paths = {name: getattr(args, name).resolve(strict=True) for name in ("source", "s2", "s9", "chem")}
    expected = read_s9_expected(paths["s9"])
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new directory name")

    # Imports are deferred so --help and nonchemical verification tests work
    # without executing the chemistry or any LightGBM code.
    import pandas as pd
    from rdkit import rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: F401

    output.mkdir(parents=True, exist_ok=False)
    (output / "data_raw").mkdir()
    (output / "src").mkdir()
    (output / "scripts").mkdir()
    # S2's own parent.parent root calculation now points to this isolated tree.
    raw_copy = output / "data_raw" / "cornelissen_table.csv"
    s2_copy = output / "scripts" / "S2_Cornelissen_Ingestion.py"
    chem_copy = output / "src" / "utils_chem.py"
    shutil.copy2(paths["source"], raw_copy)
    shutil.copy2(paths["s2"], s2_copy)
    shutil.copy2(paths["chem"], chem_copy)
    (output / "src" / "__init__.py").write_text("", encoding="utf-8")
    if sha256_file(raw_copy) != sha256_file(paths["source"]) or sha256_file(s2_copy) != sha256_file(paths["s2"]):
        raise RuntimeError("Input copy verification failed")

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "pandas": pd.__version__, "rdkit": rdBase.rdkitVersion,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "verifier_sha256": sha256_file(__file__),
        "s2_code_modified": False, "s9_executed": False,
        "titan_or_bbbp_exclusion_performed": False,
        "standardizer_note": (
            "The --chem file is copied unchanged as src/utils_chem.py to satisfy S2's original import. "
            "If deployment chem.py is supplied, equivalence to the original training utils_chem.py "
            "is an assumption until processed file hashes match."
        ),
        "expected_processed": expected,
    }
    print(f"Verification directory: {output}", flush=True)
    print(f"RDKit {rdBase.rdkitVersion}; pandas {pd.__version__}", flush=True)
    print(f"Standardization implementation: {paths['chem']}", flush=True)
    print("Running the original S2 unchanged. No auxiliary models will be trained.", flush=True)
    old_cwd, old_path = Path.cwd(), sys.path[:]
    # Remove no user files; isolate any prior Python module bindings as well.
    old_src_modules = {key: value for key, value in sys.modules.items() if key == "src" or key.startswith("src.")}
    for key in old_src_modules:
        del sys.modules[key]
    try:
        os.chdir(output)
        sys.path.insert(0, str(output))
        import src.utils_chem as chemistry
        if not callable(getattr(chemistry, "standardize_and_identify", None)):
            raise RuntimeError("Provided chemistry file lacks standardize_and_identify")
        if getattr(chemistry, "_HAS_STANDARDIZE", True) is False:
            raise RuntimeError("Chemistry module would silently skip standardization; stopped")
        with (output / "s2_run.log").open("w", encoding="utf-8") as logfile:
            with contextlib.redirect_stdout(Tee(sys.stdout, logfile)):
                runpy.run_path(str(s2_copy), run_name="__main__")
    except Exception:
        provenance.update(status="S2_FAILED", error=traceback.format_exc())
        (output / "verification_report.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for key in list(sys.modules):
            if key == "src" or key.startswith("src."):
                del sys.modules[key]
        sys.modules.update(old_src_modules)

    checks = verify_processed(output / "data_processed", expected)
    passed = all(row["passed"] for row in checks)
    provenance.update(status="PASS" if passed else "NOT_MATCHED", checks=checks)
    (output / "verification_report.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== S9 EXPECTED_PROCESSED comparison =====")
    for row in checks:
        print(f"\n{row['task']}: {row['status']}")
        print(f"  Expected N/positive/negative: {row['expected_n']}/{row['expected_positive']}/{row['expected_negative']}")
        print(f"  Actual   N/positive/negative: {row.get('actual_n', '?')}/{row.get('actual_positive', '?')}/{row.get('actual_negative', '?')}")
        print(f"  Expected SHA256: {row['expected_sha256']}")
        print(f"  Actual   SHA256: {row.get('actual_sha256', 'MISSING')}")
    if passed:
        print("\nPASS: all three processed auxiliary CSVs match S9 byte for byte.")
        print("This verifies the processed inputs, not byte identity of the original raw tables.")
    else:
        print("\nNOT MATCHED: review verification_report.json. Counts alone do not establish equality.")
        print("A hash mismatch may reflect content, ordering, serialization or dependency differences.")
    print(f"Report: {output / 'verification_report.json'}")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
