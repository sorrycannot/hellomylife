#!/usr/bin/env python3
"""Independent four-model, two-group, three-objective Stage-2 Bayesian search.

Upload this complete .py file to TITAN-BBB-main/src. Keep the existing
src/titan_priority_bayes package, original priority search and 36-arm ablation
validation outputs. No previous follow-up .py is imported.

Install the only newly required package in the existing environment:
  python -m pip install optuna==4.5.0
Run everything (search, lock four models/controls, then test):
  python run_grouped_multiobjective_bayes.py --project-root ..
Inspect only, without fitting or test data:
  python run_grouped_multiobjective_bayes.py --project-root .. --stage inspect
Search/validation only, without test data:
  python run_grouped_multiobjective_bayes.py --project-root .. --stage search
Resume (budget and sampler settings remain fixed):
  python run_grouped_multiobjective_bayes.py --project-root .. --resume <attempt>
Test a completed locked search only:
  python run_grouped_multiobjective_bayes.py --project-root .. --resume <attempt> --stage test

Default input search: titan_priority_bayes_seed42_results/attempt_bayes003.
Default source ablation: <search>/ablation_evaluation/attempt_001; override with
--ablation-run if the completed 36-arm validation outputs have another path.
Default output: titan_grouped_multiobjective_seed42_results/attempt_bayes001.

A: V01,V03 share a. B: V02,V04,V05,V06,V08,V12 share b. Both true classes are
weighted. Within-group union, cross-group MAX, no multiplication or normalization.
The previous V01=V03=1.10 reference is NEW a=1.10,b=1.00, not a=b=1.10.

Four independent native LightGBM heads: full_fusion, efflux_influx,
efflux_pampa, influx_pampa. Frozen original H2048 + fixed raw mechanism slots
(efflux,influx,pampa), inactive slots zeroed, unchanged TITAN labels/partitions,
seed42 and original LightGBM/class-weight/early-stopping protocol.

Per model: 200 UNIQUE pairs a,b in [1,3] step .05; 40 initial unique points
(4 anchors + 36 random), followed by multi-objective TPE. Directions maximize
validation AUROC, BACC at .5, MCC at .5. Exact-grid integer indices avoid float
step drift. MOTPE retains its fitted density models and maximizes its log-density-ratio
acquisition over ALL UNTRIED exact-grid pairs. No duplicate fits or random
fallback is used. This private acquisition hook requires Optuna exactly 4.5.0.

Final choice: Pareto point with all three metrics >= its OWN unweighted
reference (numerical tolerance 1e-12); maximize fixed score
S=.50*AUROC+.25*BACC+.25*(MCC+1)/2; exact ties weaker max/sum/core/reserve.
The unweighted anchor ensures a valid choice even if no weighting helps.

All proposals, native model files, validation vectors, metrics and sampler
events are saved. Deterministic ask/tell replay restores the sampler without
pickle. Pending fits recover from committed checkpoints without duplicate fits.
A global file lock prevents concurrent writers into one attempt.

After four winners are locked, matched core-only (same a,b=1) and H2048-only
(same a,b) controls are saved separately and are NOT used to reselect winners.
Only then are test data opened. Fixed .5 and the validation-BACC-selected
threshold from .1,.3,.5,.7,.9 are both exported. No test parameter selection.
All original 7593/727/942 rows remain; structural eligibility is unchanged.

These structure candidates originated in FULL-FUSION validation errors; using
four independent searches does not make their discovery independent. The
original TITAN test set has been examined in earlier development; report this
as a reused-test development comparison, not independent confirmation.

Algorithm references:
https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html
https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.study.Study.html
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import html
import importlib
import platform
from pathlib import Path
import sys
import numpy as np
import pandas as pd

DEFAULT_SEARCH='titan_priority_bayes_seed42_results/attempt_bayes003'
NAMES=('reference_saved_unweighted','control_retrained_unweighted','bayes_selected')
POLICIES=('fixed_0p5','validation_selected')
SCORES=('AUROC','AUPRC_AP','ACC','BACC','SN','SP','F1','MCC')


def core_modules(root):
    package = root/'src/titan_priority_bayes'
    if not (package/'priority_bayes/inputs.py').is_file():
        raise FileNotFoundError(f'Missing {package}/priority_bayes/inputs.py. '
                                'Keep the previously installed titan_priority_bayes folder.')
    sys.path.insert(0, str(package))
    return {name: importlib.import_module('priority_bayes.'+name)
            for name in ('inputs', 'protocol', 'training', 'report', 'chemistry')}


def resolve(value, root):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root/p).resolve()


def move_path(value, root, old_root):
    p = Path(value).expanduser()
    if p.is_absolute():
        try:
            candidate = root/p.relative_to(old_root)
            if candidate.exists():
                return candidate.resolve()
        except ValueError:
            pass
    return resolve(p, root)


def tracked(path, files, core, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing locked file: {path}. No alternate file was guessed.')
    value = core['protocol'].sha(path)
    if expected is not None and value != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    if str(path) in files and files[str(path)] != value:
        raise ValueError(f'File changed during evaluation: {path}')
    files[str(path)] = value
    return path


def original_digest(path, mapping, root, old_root):
    path = Path(path).resolve()
    for old_path, digest in mapping.items():
        if move_path(old_path, root, old_root) == path:
            return digest
    raise ValueError(f'File is absent from the completed search input lock: {path}')


def load_split(root, source, original, split, files, core):
    """Read only the named split; the test CSV loader uses TEST aliases."""
    old_root = Path(original['datasets']['val']['path']).parent.parent
    ds = root/'dataset'/f'{split}_classification.csv'
    tracked(ds, files, core, original['datasets'][split]['sha256'])
    f = pd.read_csv(ds, dtype=str, keep_default_na=False)
    if not {'SMILES', 'label'}.issubset(f):
        raise ValueError(f'{split}: dataset needs SMILES and original label columns')
    y = pd.to_numeric(f.label, errors='raise').to_numpy(dtype=float)
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Need unchanged TITAN binary labels; BBB+ = 1')
    f['label'] = y.astype(int)
    expected_n = {'train': 7593, 'val': 727, 'test': 942}[split]
    if len(f) != expected_n:
        raise ValueError(f'{split}: expected {expected_n} original rows, got {len(f)}')
    records, selected = {}, {}
    default_order = original.get('original_mechanism_order', original.get('mechanism_order'))
    for kind in ('H', 'M'):
        rec = dict(original['selected_arrays'][split][kind])
        rec['path'] = str(core['inputs'].relocated(rec['path'], root, old_root, source))
        tracked(rec['path'], files, core, rec['sha256'])
        if kind == 'H':
            a = core['inputs'].array(rec)
            alignment = 'historical_exporter_order; original bytes verified'
        else:
            order = list(rec.get('mechanism_order', default_order) or [])
            canonical = ['efflux', 'influx', 'pampa']
            if len(order) != 3 or sorted(order) != sorted(canonical):
                raise ValueError('Original mechanism order must identify efflux,influx,pampa')
            path = Path(rec['path'])
            if path.suffix.lower() == '.csv':
                columns = rec.get('columns')
                if not columns or len(columns) != 3 or 'label' in columns:
                    raise ValueError('Mechanism CSV needs the three recorded score columns')
                usecols = list(columns) + [rec[k] for k in
                    ('smiles_column', 'row_index_column', 'split_column') if rec.get(k)]
                if rec.get('split_column'):
                    aliases = ('val', 'valid', 'validation') if split == 'val' else (split,)
                    parts = pd.read_csv(path, usecols=[rec['split_column']],
                                        dtype=str, keep_default_na=False)
                    keep = set(np.flatnonzero(parts[rec['split_column']].str.lower().isin(aliases)))
                    scores = pd.read_csv(path, usecols=list(dict.fromkeys(usecols)),
                        keep_default_na=False, skiprows=lambda n: n > 0 and n-1 not in keep)
                else:
                    scores = pd.read_csv(path, usecols=list(dict.fromkeys(usecols)),
                                         keep_default_na=False)
                scores, alignment = core['inputs'].align_frame(scores, rec, f.SMILES.tolist())
                a = scores[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
            else:
                a = core['inputs'].array(rec)
                alignment = 'historical_exporter_order; original bytes verified'
            a = np.asarray(a)
            if a.shape != (len(f), 3) or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
                raise ValueError(f'{split}: expected finite original raw mechanism probabilities')
            a = a[:, [order.index(name) for name in canonical]]
        width = 2048 if kind == 'H' else 3
        if np.asarray(a).shape != (len(f), width) or not np.isfinite(a).all():
            raise ValueError(f'{split}/{kind}: wrong shape or nonfinite input')
        if rec.get('rows_csv'):
            witness = core['inputs'].relocated(rec['rows_csv'], root, old_root, source)
            tracked(witness, files, core, rec.get('rows_csv_sha256'))
            rows = pd.read_csv(witness, usecols=['SMILES'], keep_default_na=False)
            if rows.SMILES.tolist() != f.SMILES.tolist():
                raise ValueError(f'{split}/{kind}: row witness differs from original dataset')
            alignment = 'exact original SMILES sequence in row witness'
        selected[kind] = a
        records[kind] = dict(**rec, evaluation_row_alignment=alignment)
    X = np.empty((len(f), 2051), dtype=np.float32)
    X[:, :2048] = selected['H']; X[:, 2048:] = selected['M']
    return f, X, dict(dataset=str(ds), selected_arrays=records,
        mechanism_order=['efflux', 'influx', 'pampa'], shape=list(X.shape),
        row_limit='Bare-array historical row order is inherited, not independently proved by a shape or hash.')


def read_witness(path, frame, files, core):
    tracked(path, files, core)
    f = pd.read_csv(path, dtype=str, keep_default_na=False)
    ids = pd.to_numeric(f.original_row_index, errors='raise').to_numpy(dtype=float)
    if (not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all()
            or sorted(ids.tolist()) != list(range(len(frame)))):
        raise ValueError(f'Prediction row indices do not cover the original split: {path}')
    f = f.iloc[np.argsort(ids)].reset_index(drop=True)
    if f.SMILES.tolist() != frame.SMILES.tolist() or not np.array_equal(
            pd.to_numeric(f.label).to_numpy(), frame.label.to_numpy()):
        raise ValueError(f'Prediction SMILES/labels do not align with original rows: {path}')
    p = pd.to_numeric(f.positive_probability, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f'Invalid saved probabilities: {path}')
    return p


def verify_record(folder, signature, files, core):
    record = core['protocol'].read_json(tracked(folder/'COMPLETE.json', files, core))
    if record['run_signature'] != signature or record['trial_id'] != folder.name:
        raise ValueError('Trial record does not belong to the locked completed search')
    for name, digest in record['files_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unexpected relative path in trial file record')
        tracked(folder/name, files, core, digest)
    for required in ('model.txt', 'val_probability.npy', 'PROPOSAL.json'):
        if required not in record['files_sha256']:
            raise ValueError('Incomplete original trial artifact record')
    proposal = core['protocol'].read_json(folder/'PROPOSAL.json')
    if proposal['indices'] != record['indices'] or proposal['run_signature'] != signature:
        raise ValueError('Stored trial proposal and completion record differ')
    return record


def prepare_locked(root, search, files, core):
    read = core['protocol'].read_json
    lock = read(tracked(search/'protocol_lock.json', files, core))
    fields = ('settings', 'versions', 'input_paths_sha256', 'rules_sha256', 'code_sha256')
    if core['protocol'].object_sha({k: lock[k] for k in fields}) != lock['run_signature']:
        raise ValueError('Original protocol signature is inconsistent')
    done = read(tracked(search/'COMPLETE.json', files, core))
    selection = read(tracked(search/'selected_model/selection_lock.json', files, core,
                             done['selection_lock_sha256']))
    if (done['run_signature'] != lock['run_signature']
            or selection['run_signature'] != lock['run_signature']
            or done['selected_trial'] != selection['selected_trial']
            or done['completed_unique_configurations'] != selection['completed_unique_configurations']
            or done['requested_budget'] != done['completed_unique_configurations']
            or done['requested_budget'] != selection['budget']):
        raise ValueError('Search must be completed and the selected model already locked')
    settings = lock['settings']
    if settings['seed'] != 42 or settings['active_rules'] != ['V01', 'V03']:
        raise ValueError('Expected the seed42 V01/V03 priority search')
    old_root = Path(settings['project_root'])
    source = move_path(settings['source_run'], root, old_root)
    metadata = {}
    for name in ('input_audit.json', 'reference_audit.json'):
        p = source/name
        expected = original_digest(p, lock['input_paths_sha256'], root, old_root)
        metadata[name] = read(tracked(p, files, core, expected))
    original, ref = metadata['input_audit.json'], metadata['reference_audit.json']
    original_root = Path(original['datasets']['val']['path']).parent.parent
    reference_path = core['inputs'].relocated(ref['path'], root, original_root, source)
    tracked(reference_path, files, core, ref['sha256'])
    best = verify_record(search/'trials'/selection['selected_trial'], lock['run_signature'], files, core)
    summary = read(tracked(search/'best_trial.json', files, core))
    if summary != best or best['indices'] != selection['indices']:
        raise ValueError('Selected trial, best_trial.json and selection lock disagree')
    weights = core['protocol'].pair_weights(best['indices'])
    if dict(zip(('V01', 'V03'), weights)) != selection['weights']:
        raise ValueError('Selected structural weights are inconsistent')
    control = verify_record(search/'trials/trial_0001', lock['run_signature'], files, core)
    if control['indices'] != [0, 0]:
        raise ValueError('First completed trial must be the unweighted refit control')
    chosen_path = tracked(search/'selected_model/best_model.txt', files, core,
                           selection['model_sha256'])
    if selection['model_sha256'] != best['files_sha256']['model.txt']:
        raise ValueError('Selected model differs from its completed trial checkpoint')
    paths = dict(zip(NAMES, (reference_path, search/'trials/trial_0001/model.txt', chosen_path)))
    models = {name: core['training'].load_reference(p) for name, p in paths.items()}
    for name, record in ((NAMES[1], control), (NAMES[2], best)):
        if models[name].current_iteration() != record['best_iteration']:
            raise ValueError(f'{name}: checkpoint iteration differs from its training record')
    fv, Xv, validation_audit = load_split(root, source, original, 'val', files, core)
    table_path = tracked(search/'validation_comparison.csv', files, core)
    table = pd.read_csv(table_path)
    thresholds, validation_rows = {}, []
    records = {NAMES[1]: control, NAMES[2]: best}
    for name in NAMES:
        predicted = np.asarray(models[name].predict(Xv), dtype=float)
        if name in records:
            saved = np.load(search/'trials'/records[name]['trial_id']/'val_probability.npy', allow_pickle=False)
            if not np.allclose(predicted, saved, rtol=0, atol=1e-10):
                raise ValueError(f'{name}: checkpoint does not reproduce its saved validation vector')
        thresholds[name] = {}
        for policy in POLICIES:
            rows = table.loc[table.experiment.eq(name) & table.threshold_policy.eq(policy)]
            if len(rows) != 1:
                raise ValueError(f'Need exactly one saved validation row for {name}/{policy}')
            threshold = float(rows.iloc[0]['threshold'])
            if threshold not in (.1, .3, .5, .7, .9) or (policy == 'fixed_0p5' and threshold != .5):
                raise ValueError('Invalid locked threshold')
            if name in records and threshold != records[name]['validation_metrics'][policy]['threshold']:
                raise ValueError('Saved table threshold contradicts the completed trial')
            if name == NAMES[2] and threshold != selection['thresholds'][policy]:
                raise ValueError('Selected threshold contradicts selection_lock.json')
            if policy == 'validation_selected' and threshold != core['protocol'].choose_threshold(fv.label.to_numpy(), predicted):
                raise ValueError('Archived threshold no longer follows the original validation rule')
            witness = read_witness(search/'predictions'/name/policy/'val_all.csv', fv, files, core)
            if not np.allclose(predicted, witness, rtol=0, atol=1e-10):
                raise ValueError(f'{name}: locked checkpoint does not reproduce its validation export')
            got = core['protocol'].metrics(fv.label.to_numpy(), predicted, threshold)
            for metric in (*SCORES, 'FN', 'FP'):
                if abs(got[metric] - float(rows.iloc[0][metric])) > 1e-10:
                    raise ValueError(f'Validation audit mismatch: {name}/{policy}/{metric}')
            if name in records:
                for metric in (*SCORES, 'FN', 'FP'):
                    if abs(got[metric] - records[name]['validation_metrics'][policy][metric]) > 1e-10:
                        raise ValueError('Validation record and actual checkpoint metrics disagree')
            thresholds[name][policy] = threshold
            validation_rows.append(dict(experiment=name, split='val', threshold_policy=policy, **got))
    return dict(lock=lock, selection=selection, source=source, original=original,
        model_paths=paths, models=models, thresholds=thresholds, weights=weights,
        validation=pd.DataFrame(validation_rows), validation_audit=validation_audit)


def outcome_frame(frame, probability, threshold):
    f = frame.copy().reset_index(drop=True)
    reserved = ('original_row_index', 'csv_row_number', 'positive_probability',
                'threshold', 'predicted_label', 'outcome')
    if set(reserved) & set(f.columns):
        raise ValueError('Original CSV contains reserved prediction column names')
    f.insert(0, 'original_row_index', np.arange(len(f)))
    f.insert(1, 'csv_row_number', np.arange(len(f)) + 2)
    f['positive_probability'] = np.asarray(probability, dtype=float)
    f['threshold'] = threshold
    f['predicted_label'] = (f.positive_probability > threshold).astype(int)
    y, pred = f.label.to_numpy(), f.predicted_label.to_numpy()
    f['outcome'] = np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 0),
                            (y == 0) & (pred == 1)], ['TP', 'TN', 'FP'], default='FN')
    return f


def paired_changes(frame, reference_p, best_p):
    ref = outcome_frame(frame, reference_p, .5)
    best = outcome_frame(frame, best_p, .5)
    result = ref.drop(columns=['positive_probability', 'threshold', 'predicted_label', 'outcome']).copy()
    result['reference_probability'] = reference_p
    result['selected_probability'] = best_p
    result['reference_outcome'] = ref.outcome
    result['selected_outcome'] = best.outcome
    result['transition'] = ref.outcome + '->' + best.outcome
    result['corrected_error'] = ref.outcome.isin(['FN', 'FP']) & best.outcome.isin(['TP', 'TN'])
    result['new_error'] = ref.outcome.isin(['TP', 'TN']) & best.outcome.isin(['FN', 'FP'])
    return result


def confusion_svg(path, name, m):
    values = [[m['TN'], m['FP']], [m['FN'], m['TP']]]
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="690" viewBox="0 0 850 690">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#243B53">',
        f'<text x="425" y="44" text-anchor="middle" font-size="27" font-weight="bold">{html.escape(name)}</text>',
        f'<text x="425" y="77" text-anchor="middle" font-size="18">TITAN test: n={m["n"]}; threshold={m["threshold"]:.2f}; BBB+ = 1</text>']
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for row in range(2):
        for col in range(2):
            x, y = 245 + col*225, 165 + row*200
            fill = '#D9EAF7' if row == col else '#F9E4D4'
            fraction = values[row][col]/sum(values[row])
            pieces += [f'<rect x="{x}" y="{y}" width="215" height="190" rx="8" fill="{fill}"/>',
                f'<text x="{x+107.5}" y="{y+48}" text-anchor="middle" font-size="24">{labels[row][col]}</text>',
                f'<text x="{x+107.5}" y="{y+105}" text-anchor="middle" font-size="43" font-weight="bold">{values[row][col]}</text>',
                f'<text x="{x+107.5}" y="{y+148}" text-anchor="middle" font-size="22">{fraction:.1%}</text>',
                f'<text x="{x+107.5}" y="{y+175}" text-anchor="middle" font-size="15">within actual class</text>']
    pieces += ['<text x="352" y="141" text-anchor="middle" font-size="22">Predicted BBB-</text>',
        '<text x="577" y="141" text-anchor="middle" font-size="22">Predicted BBB+</text>',
        '<text x="225" y="267" text-anchor="end" font-size="22">Actual BBB-</text>',
        '<text x="225" y="467" text-anchor="end" font-size="22">Actual BBB+</text>',
        f'<text x="425" y="615" text-anchor="middle" font-size="20">AUROC {m["AUROC"]:.6f} | BACC {m["BACC"]:.6f} | MCC {m["MCC"]:.6f}</text>',
        '<text x="425" y="656" text-anchor="middle" font-size="16">Locked comparison on a previously examined test set</text>', '</g></svg>']
    path.write_text('\n'.join(pieces), encoding='utf-8')


def make_model_input(X2051, mechanism_mask):
    """Keep frozen H2048 intact; disable mechanisms by zeroing their fixed slots."""
    if mechanism_mask is None:
        return np.ascontiguousarray(X2051[:, :2048], dtype=np.float32)
    X = np.array(X2051, dtype=np.float32, copy=True, order='C')
    X[:, 2048:] *= np.asarray(mechanism_mask, dtype=np.float32)
    return X


def runtime_versions():
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm', 'rdkit', 'joblib', 'openpyxl'):
        try:
            versions[name] = importlib.import_module(name).__version__
        except ImportError as exc:
            raise RuntimeError(f'Missing {name}; activate conda titan_bbb first.') from exc
    return versions


def read_saved_flags(frame, column):
    values = frame[column].astype(str).str.lower()
    if not values.isin(['true', 'false', '1', '0']).all():
        raise ValueError(f'Invalid saved Boolean structure column: {column}')
    return values.isin(['true', '1']).to_numpy()


def training_inputs(root, search, locked, files, core):
    frames, X, audits = {}, {}, {}
    for split in ('train', 'val'):
        frames[split], X[split], audits[split] = load_split(
            root, locked['source'], locked['original'], split, files, core)
    rules_path = root/'src/titan_priority_bayes/priority_bayes/reference_rules.json'
    reference = core['protocol'].read_json(tracked(
        rules_path, files, core, locked['lock']['rules_sha256']))
    matches, structure_audit = core['chemistry'].prepare(reference, frames)
    if not structure_audit['training_coverage_matches_reported_audit']:
        raise ValueError('V01/V03 training coverage differs from the original search. No new fit started.')
    # Confirm the exact row masks used in the completed Bayesian search.
    for split in ('train', 'val'):
        saved_path = tracked(search/f'{split}_structure_matches.csv', files, core)
        saved = pd.read_csv(saved_path, dtype=str, keep_default_na=False)
        if saved.SMILES.tolist() != frames[split].SMILES.tolist() or not np.array_equal(
                pd.to_numeric(saved.label).to_numpy(), frames[split].label.to_numpy()):
            raise ValueError(f'{split}: archived structure-match rows have changed')
        for rule in ('V01', 'V03'):
            if not np.array_equal(read_saved_flags(saved, rule), matches[split][rule].to_numpy(dtype=bool)):
                raise ValueError(f'{split}/{rule}: exact training/validation mask differs from the search')
    return frames, X, matches, dict(inputs=audits, structure=structure_audit)


# Two grouped weights; four independent three-objective studies.
CORE_RULES = ('V01', 'V03')
RESERVE_RULES = ('V02', 'V04', 'V05', 'V06', 'V08', 'V12')
MODEL_MASKS = {
    'full_fusion': [1, 1, 1],
    'efflux_influx': [1, 1, 0],
    'efflux_pampa': [1, 0, 1],
    'influx_pampa': [0, 1, 1],
}
GRID = tuple((i, j) for i in range(41) for j in range(41))
ANCHORS = ((0, 0), (2, 0), (0, 2), (2, 2))
EXPECTED_RESERVE_COVERAGE = {
    'V02': dict(units=36, positive=21, negative=15, connectivity_groups=24),
    'V04': dict(units=123, positive=73, negative=50, connectivity_groups=71),
    'V05': dict(units=67, positive=30, negative=37, connectivity_groups=44),
    'V06': dict(units=111, positive=65, negative=46, connectivity_groups=85),
    'V08': dict(units=329, positive=215, negative=114, connectivity_groups=215),
    'V12': dict(units=1037, positive=391, negative=646, connectivity_groups=505),
}


def pair_weights(pair):
    if len(pair) != 2 or any(int(k) != k or not 0 <= k <= 40 for k in pair):
        raise ValueError('Group-weight indices must be integers in 0..40')
    return tuple((20 + int(k))/20 for k in pair)


def initial_plan(startup=40):
    if not 1 <= startup <= len(GRID):
        raise ValueError('Initial unique configurations must fit the active grid')
    anchors = [pair for pair in ANCHORS if pair in GRID][:startup]
    available = [p for p in GRID if p not in anchors]
    order = np.random.default_rng(42).permutation(len(available))
    return anchors + [available[int(i)] for i in order[:startup-len(anchors)]]


def grouped_weights(matches, pair):
    a, b = pair_weights(pair)
    return np.maximum(np.where(matches.core_group.to_numpy(dtype=bool), a, 1.),
                      np.where(matches.reserve_group.to_numpy(dtype=bool), b, 1.))


def objective_vector(metric):
    vector = np.array([metric['AUROC'], metric['BACC'], metric['MCC']], dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError('All three validation objectives must be finite')
    return vector


def balanced_score(metric):
    auc, bacc, mcc = objective_vector(metric)
    return float(.50*auc + .25*bacc + .25*(mcc+1)/2)


def selection_flags(records):
    if not records or tuple(records[0]['indices']) != (0, 0):
        raise ValueError('The first unique trial must be this model\'s unweighted reference')
    v = np.array([objective_vector(r['validation_metrics']['fixed_0p5']) for r in records])
    eligible = np.all(v >= v[0]-1e-12, axis=1)
    # True Pareto dominance uses exact recorded values, not display rounding.
    pareto = np.array([not np.any(np.all(v >= row, axis=1) & np.any(v > row, axis=1))
                       for row in v], dtype=bool)
    return eligible, pareto


def choose_grouped_best(records):
    eligible, pareto = selection_flags(records)
    candidates = [r for r, ok, front in zip(records, eligible, pareto) if ok and front]
    if not candidates:
        raise ValueError('No admissible Pareto point; the unweighted anchor should ensure one exists')
    return min(candidates, key=lambda r: (
        -balanced_score(r['validation_metrics']['fixed_0p5']),
        max(pair_weights(r['indices'])), sum(pair_weights(r['indices'])),
        *pair_weights(r['indices'])))


def load_model_record(folder, signature, core, expected_pair=None, expected_mode=None, yv=None):
    read, sha = core['protocol'].read_json, core['protocol'].sha
    r = read(folder/'COMPLETE.json')
    if r['run_signature'] != signature:
        raise ValueError(f'Model record belongs to another run: {folder}')
    if expected_pair is not None and r['indices'] != list(expected_pair):
        raise ValueError('Saved model has different group weights')
    if expected_mode is not None and r['mode'] != expected_mode:
        raise ValueError('Saved model has different mechanism inputs')
    for name, digest in r['files_sha256'].items():
        if Path(name).name != name or sha(folder/name) != digest:
            raise ValueError(f'Saved model artifact changed: {folder/name}')
    if not {'model.txt', 'val_probability.npy', 'fit_history.json', 'PROPOSAL.json'} <= set(r['files_sha256']):
        raise ValueError('Incomplete model artifact checksum registry')
    if yv is not None:
        p = np.load(folder/'val_probability.npy', allow_pickle=False)
        for policy, old in r['validation_metrics'].items():
            actual = core['protocol'].metrics(yv, p, r['thresholds'][policy])
            if core['protocol'].object_sha(actual) != core['protocol'].object_sha(old):
                raise ValueError('Validation metrics disagree with saved probabilities')
        if r['thresholds'] != dict(fixed_0p5=.5,
                validation_selected=core['protocol'].choose_threshold(yv, p)):
            raise ValueError('Saved thresholds violate the validation-only policy')
        if abs(r['balanced_score']-balanced_score(r['validation_metrics']['fixed_0p5'])) > 1e-14:
            raise ValueError('Saved balanced score changed')
    return r


class MOTPESession:
    """Ask/tell MOTPE with an auditable deterministic event replay.

    No pickle/database state is trusted. On Optuna 4.5.0, a narrowly scoped
    TPESampler subclass evaluates the existing MOTPE log-density-ratio
    acquisition on ALL UNTRIED exact-grid pairs, replacing only its internal
    random acquisition-candidate pool. No random fallback or duplicate trials.
    Initial queued trials are without replacement. Pinning 4.5.0 is required
    because the acquisition hook is a private Optuna API.
    """
    def __init__(self, folder, mode, signature, startup, core, validate_record):
        import optuna
        import warnings
        self.optuna, self.folder, self.mode = optuna, Path(folder), mode
        self.signature, self.startup, self.core = signature, startup, core
        self.validate_record = validate_record
        self.records, self.seen, self.events = [], set(), []
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder/'events').mkdir(exist_ok=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        owner = self
        class UniqueGridTPESampler(optuna.samplers.TPESampler):
            def _compute_acquisition_func(self, samples, mpe_below, mpe_above):
                if set(samples) != {'core_index', 'reserve_index'}:
                    raise ValueError('Expected a joint two-parameter MOTPE acquisition; no independent fallback was used')
                pool = np.asarray([pair for pair in GRID if pair not in owner.seen], dtype=float)
                if not len(pool):
                    raise ValueError('All exact-grid pairs have already been tried')
                # TPESampler._sample passes this SAME dict to _compare after
                # calling this hook. Replacing it preserves the fitted MOTPE
                # density models and exact acquisition, while restricting the
                # comparison to valid unseen pairs. Tested against 4.5.0.
                samples.clear()
                samples.update(core_index=pool[:, 0], reserve_index=pool[:, 1])
                values = super()._compute_acquisition_func(samples, mpe_below, mpe_above)
                if not np.isfinite(values).all():
                    raise ValueError('MOTPE acquisition produced nonfinite values')
                return values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
            sampler = UniqueGridTPESampler(seed=42, n_startup_trials=startup,
                n_ei_candidates=64, multivariate=True, group=False,
                constant_liar=False, warn_independent_sampling=False)
        self.study = optuna.create_study(study_name=mode+'_seed42', sampler=sampler,
            directions=['maximize', 'maximize', 'maximize'])
        for path in sorted((self.folder/'events').glob('event_*.json')):
            event = self.core['protocol'].read_json(path)
            if event['event_number'] != len(self.events)+1 or path.name != f'event_{len(self.events)+1:06d}.json':
                raise ValueError('Sampler event history has gaps or renamed files')
            trial, actual = self._ask_next()
            expected = {k: event[k] for k in ('run_signature', 'event_number', 'optuna_number', 'indices', 'method', 'unique_number')}
            if actual != expected:
                raise ValueError('MOTPE replay differs from saved proposals; use the original code and software versions')
            self._tell_event(trial, event, replay=True)
            self.events.append(event)
        # A completed model may exist while its event was not yet committed.
        # Pending proposal recovery will reuse that checkpoint, never fit twice.

    def _ask_next(self):
        forced, method = None, 'MOTPE_full_untried_grid'
        if len(self.records) < self.startup:
            forced = initial_plan(self.startup)[len(self.records)]
            method = 'fixed_anchor' if len(self.records) < len([p for p in ANCHORS if p in GRID]) else 'initial_random'
        if forced is not None:
            self.study.enqueue_trial(dict(core_index=forced[0], reserve_index=forced[1]))
        trial = self.study.ask()
        pair = [trial.suggest_int('core_index', 0, 40), trial.suggest_int('reserve_index', 0, 40)]
        event = dict(run_signature=self.signature, event_number=len(self.events)+1,
            optuna_number=trial.number, indices=pair, method=method, unique_number=len(self.records)+1)
        return trial, event

    def _tell_event(self, trial, event, replay=False):
        pair = tuple(event['indices'])
        if event['state'] != 'COMPLETE' or pair in self.seen:
            raise ValueError('Invalid completed sampler event or duplicate model fit')
        folder = self.folder/'trials'/event['trial_id']
        if self.core['protocol'].sha(folder/'COMPLETE.json') != event['model_record_sha256']:
            raise ValueError('Committed sampler event points to a modified model record')
        record = self.validate_record(folder, pair)
        values = objective_vector(record['validation_metrics']['fixed_0p5']).tolist()
        if event['values'] != values or record['unique_number'] != len(self.records)+1:
            raise ValueError('Sampler objectives disagree with completed validation trial')
        self.study.tell(trial, values)
        self.records.append(record); self.seen.add(pair)

    def commit_event(self, trial, event):
        """Persist before changing the in-memory study so interruptions replay safely."""
        target = self.folder/'events'/f'event_{event["event_number"]:06d}.json'
        self.core['protocol'].write_json(target, event)
        self._tell_event(trial, event)
        self.events.append(event)

    def next_unique(self):
        read, write = self.core['protocol'].read_json, self.core['protocol'].write_json
        pending = self.folder/'PENDING.json'
        while True:
            trial, event = self._ask_next()
            if pending.exists():
                old = read(pending)
                if old != event:
                    # Crash after event commit but before removing PENDING.
                    if self.events and old == {k: self.events[-1][k] for k in event}:
                        pending.unlink()
                    else:
                        raise ValueError('Pending sampler proposal disagrees with deterministic replay')
            if tuple(event['indices']) in self.seen:
                raise ValueError('Unique-grid MOTPE unexpectedly proposed a previously fitted pair')
            write(pending, event)
            return trial, event

    def complete_unique(self, trial, event, record):
        folder = self.folder/'trials'/record['trial_id']
        committed = dict(**event, state='COMPLETE', trial_id=record['trial_id'],
            values=objective_vector(record['validation_metrics']['fixed_0p5']).tolist(),
            model_record_sha256=self.core['protocol'].sha(folder/'COMPLETE.json'))
        self.commit_event(trial, committed)
        pending = self.folder/'PENDING.json'
        if pending.exists():
            pending.unlink()



def grouped_runtime():
    versions = runtime_versions()
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError('Missing optuna. In conda titan_bbb run: python -m pip install optuna==4.5.0') from exc
    versions['optuna'] = optuna.__version__
    if optuna.__version__ != '4.5.0':
        raise ValueError('The exact-grid MOTPE acquisition hook requires optuna==4.5.0. '
                         'In conda titan_bbb run: python -m pip install optuna==4.5.0')
    return versions


def group_masks(root, search, locked, files, core):
    # Original all-17 validation calibration and core training-coverage checks.
    frames, X, matches, audit = training_inputs(root, search, locked, files, core)
    reference = core['protocol'].read_json(root/'src/titan_priority_bayes/priority_bayes/reference_rules.json')
    matcher = core['chemistry'].Matcher(reference)
    candidates, groups = [], []
    for split, frame in frames.items():
        m = matches[split]
        eligible = m.eligibility_status.isin(['primary_representative', 'consistent_duplicate']).to_numpy()
        hits = [set(matcher.graph(s)['hits']) if ok else set()
                for s, ok in zip(frame.SMILES, eligible)]
        for rule in RESERVE_RULES:
            m[rule] = [rule in h for h in hits]
        m['core_group'] = m[list(CORE_RULES)].any(axis=1)
        m['reserve_group'] = m[list(RESERVE_RULES)].any(axis=1)
        m['eligible_for_structure_weight'] = eligible
        for rule in CORE_RULES+RESERVE_RULES:
            mask = m[rule].to_numpy(dtype=bool)
            primary = mask & m.primary_unit.to_numpy(dtype=bool)
            candidates.append(dict(split=split, candidate=rule, raw_rows=int(mask.sum()),
                units=int(primary.sum()), positive=int((primary & frame.label.eq(1).to_numpy()).sum()),
                negative=int((primary & frame.label.eq(0).to_numpy()).sum()),
                connectivity_groups=int(m.loc[primary, 'connectivity_group'].nunique())))
        a, b = m.core_group.to_numpy(), m.reserve_group.to_numpy()
        for cohort, mask in dict(core=a, reserve=b, core_only=a&~b, reserve_only=b&~a,
                                 intersection=a&b, union=a|b, neither=~(a|b)).items():
            primary = mask & m.primary_unit.to_numpy(dtype=bool)
            groups.append(dict(split=split, cohort=cohort, raw_rows=int(mask.sum()),
                units=int(primary.sum()), positive=int((primary & frame.label.eq(1).to_numpy()).sum()),
                negative=int((primary & frame.label.eq(0).to_numpy()).sum()),
                raw_positive=int((mask & frame.label.eq(1).to_numpy()).sum()),
                raw_negative=int((mask & frame.label.eq(0).to_numpy()).sum()),
                connectivity_groups=int(m.loc[primary, 'connectivity_group'].nunique())))
    coverage = pd.DataFrame(candidates)
    actual = coverage.loc[coverage.split.eq('train')].set_index('candidate')
    comparison = {rule: all(int(actual.loc[rule, k]) == v for k, v in expected.items())
                  for rule, expected in EXPECTED_RESERVE_COVERAGE.items()}
    if not all(comparison.values()):
        bad = [k for k, ok in comparison.items() if not ok]
        raise ValueError('Extended-rule training coverage differs from the previously reported audit: '
                         + ', '.join(bad) + '. No optimization was started.')
    audit.update(core_rules=list(CORE_RULES), reserve_rules=list(RESERVE_RULES),
        reserve_coverage_matches_reported_audit=comparison,
        expected_reserve_coverage=EXPECTED_RESERVE_COVERAGE,
        overlap_rule='max(a,b); within-group union, no repeated stacking',
        known_reference='old V01=V03=1.10 is NEW (core=1.10,reserve=1.00)',
        no_test_structures_or_labels_used=True)
    return frames, X, matches, audit, coverage, pd.DataFrame(groups)


def source_ablation(root, path, search, locked, Xv, yv, files, core):
    """Verify only pre-existing validation records, never old test reports."""
    read = core['protocol'].read_json
    protocol = read(tracked(path/'ablation_protocol_lock.json', files, core))
    payload = {k: v for k, v in protocol.items() if k != 'run_signature'}
    if core['protocol'].object_sha(payload) != protocol['run_signature']:
        raise ValueError('Source ablation protocol signature is inconsistent')
    registry = read(tracked(path/'validation_lock.json', files, core))
    if (registry['run_signature'] != protocol['run_signature'] or registry['completed_models'] != 36
            or protocol['source_search_signature'] != locked['lock']['run_signature']):
        raise ValueError('--ablation-run must identify the completed 36-arm validation registry from the original priority search')
    for key in ('params', 'max_rounds', 'patience', 'class_weight'):
        reference = (locked['lock']['settings']['source_common_params'] if key == 'params' else
                     locked['lock']['class_weight'] if key == 'class_weight' else locked['lock']['settings'][key])
        if protocol[key] != reference:
            raise ValueError(f'Source ablation {key} differs from the fixed training protocol')
    warm = {}
    for mode in list(MODEL_MASKS)+['H_only']:
        mechanism = 'baseline' if mode == 'H_only' else mode
        for pair, policy in [((0, 0), 'unweighted'), ((2, 0), 'V01_V03')]:
            name = policy+'__'+mechanism; folder = path/'models'/name
            record = read(tracked(folder/'COMPLETE.json', files, core))
            spec = record['spec']
            if (record['run_signature'] != protocol['run_signature'] or spec['experiment_id'] != name
                    or spec['experiment'] != mechanism or spec['structure_policy'] != policy
                    or spec['V01_weight'] != pair_weights(pair)[0]
                    or spec['V03_weight'] != pair_weights(pair)[0]
                    or spec['mechanism_mask'] != (None if mode == 'H_only' else MODEL_MASKS[mode])):
                raise ValueError('Source ablation arm does not match its declared features and structural weights')
            for fname, digest in record['files_sha256'].items():
                if Path(fname).name != fname:
                    raise ValueError('Unexpected relative path in source ablation registry')
                tracked(folder/fname, files, core, digest)
            saved = registry['models'][name]
            if (saved['model_sha256'] != record['files_sha256']['model.txt']
                    or saved['thresholds'] != record['thresholds'] or saved['best_iteration'] != record['best_iteration']):
                raise ValueError('Source ablation checkpoint contradicts the validation lock')
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(folder/'model.txt'))
            mask = None if mode == 'H_only' else MODEL_MASKS[mode]
            expected_dim = 2048 if mode == 'H_only' else 2051
            if model.num_feature() != expected_dim or model.current_iteration() != record['best_iteration']:
                raise ValueError('Source ablation checkpoint dimension/round mismatch')
            probability = np.asarray(model.predict(make_model_input(Xv, mask)), dtype=float)
            archived = np.load(folder/'val_probability.npy', allow_pickle=False)
            if not np.allclose(probability, archived, rtol=0, atol=1e-10):
                raise ValueError('Source ablation model does not reproduce its validation predictions')
            thresholds = dict(fixed_0p5=.5, validation_selected=core['protocol'].choose_threshold(yv, probability))
            if thresholds != record['thresholds']:
                raise ValueError('Source ablation threshold no longer matches validation-only selection')
            for policy, threshold in thresholds.items():
                actual = core['protocol'].metrics(yv, probability, threshold)
                for k, value in actual.items():
                    old = record['validation_metrics'][policy][k]
                    if value is None and old is not None or value is not None and (old is None or abs(value-old) > 1e-10):
                        raise ValueError('Source ablation validation metrics cannot be reproduced')
            warm[mode, pair] = dict(path=folder/'model.txt', model=model,
                model_sha256=record['files_sha256']['model.txt'], thresholds=thresholds,
                original_group_weights='V01 and V03 each equal '+str(pair_weights(pair)[0]),
                new_group_weights=pair_weights(pair))
    return warm


def fit_grouped(folder, mode, pair, proposal, ctx, unique_number=0, phase='search'):
    """One native LightGBM fit, or an exact verified matching warm checkpoint."""
    import time
    import shutil
    core, frames, X = ctx['core'], ctx['frames'], ctx['X']
    signature = ctx['signature']; yv = frames['val'].label.to_numpy()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        return load_model_record(folder, signature, core, pair, mode, yv)
    before = dict(run_signature=signature, mode=mode, indices=list(pair), proposal=proposal,
                  unique_number=unique_number, phase=phase)
    if (folder/'PROPOSAL.json').exists():
        if core['protocol'].read_json(folder/'PROPOSAL.json') != before:
            raise ValueError('Unfinished fit conflicts with the resumed proposal')
    else:
        core['protocol'].write_json(folder/'PROPOSAL.json', before)
    a, b = pair_weights(pair); started = time.monotonic()
    mask = None if mode == 'H_only' else MODEL_MASKS[mode]
    Xv = make_model_input(X['val'], mask)
    w = grouped_weights(ctx['matches']['train'], pair)
    warm = ctx['warm'].get((mode, tuple(pair)))
    print(f'[{mode}/{folder.name}] core={a:.2f}, reserve={b:.2f}; {phase}', flush=True)
    if warm is not None:
        model = warm['model']; shutil.copyfile(warm['path'], folder/'model.txt')
        history = dict(reused_verified_model=str(warm['path']), model_sha256=warm['model_sha256'])
        fit_mode = 'reused_exact_validation_verified_checkpoint'
    else:
        model, history = core['training'].fit_head(
            make_model_input(X['train'], mask), frames['train'].label.to_numpy(),
            ctx['params'], ctx['max_rounds'], Xv, yv,
            weights=None if np.equal(w, 1.).all() else w,
            patience=ctx['patience'], scale_pos_weight=ctx['class_weight'], verbose=False)
        model.save_model(str(folder/'model.txt')); fit_mode = 'new_train_only_fit'
    p = np.asarray(model.predict(Xv), dtype=float)
    thresholds = dict(fixed_0p5=.5, validation_selected=core['protocol'].choose_threshold(yv, p))
    metrics = {policy: core['protocol'].metrics(yv, p, threshold) for policy, threshold in thresholds.items()}
    np.save(folder/'val_probability.npy', p, allow_pickle=False)
    core['protocol'].write_json(folder/'fit_history.json', history)
    record = dict(**before, trial_id=folder.name, core_weight=a, reserve_weight=b,
        model_relpath=str((folder/'model.txt').relative_to(ctx['out'])),
        model_folder_relpath=str(folder.relative_to(ctx['out'])),
        input_dim=2048 if mask is None else 2051, mechanism_mask=mask,
        thresholds=thresholds, validation_metrics=metrics,
        balanced_score=balanced_score(metrics['fixed_0p5']), fit_mode=fit_mode,
        best_iteration=int(model.best_iteration if model.best_iteration > 0 else model.current_iteration()),
        seconds=time.monotonic()-started,
        sample_weight_summary=dict(n_upweighted=int((w > 1).sum()), min=float(w.min()), max=float(w.max()),
            positive_sum=float(w[frames['train'].label.to_numpy() == 1].sum()),
            negative_sum=float(w[frames['train'].label.to_numpy() == 0].sum())),
        files_sha256={name: core['protocol'].sha(folder/name) for name in
            ('PROPOSAL.json', 'model.txt', 'val_probability.npy', 'fit_history.json')})
    core['protocol'].write_json(folder/'COMPLETE.json', record)
    m = metrics['fixed_0p5']
    print(f'  val AUROC={m["AUROC"]:.6f}, BACC={m["BACC"]:.6f}, MCC={m["MCC"]:.6f}, '
          f'S={record["balanced_score"]:.6f}; rounds={record["best_iteration"]}; {fit_mode}', flush=True)
    return record


def history_frame(records, mode):
    eligible, front = selection_flags(records)
    rows = []
    for record, ok, pareto in zip(records, eligible, front):
        for policy, metric in record['validation_metrics'].items():
            rows.append(dict(mode=mode, trial_id=record['trial_id'], unique_number=record['unique_number'],
                core_weight=record['core_weight'], reserve_weight=record['reserve_weight'],
                threshold_policy=policy, phase=record['proposal']['method'],
                S_fixed05=record['balanced_score'], admissible_vs_own_unweighted=bool(ok),
                pareto_fixed05=bool(pareto), best_iteration=record['best_iteration'],
                fit_mode=record['fit_mode'], seconds=record['seconds'], **metric))
    return pd.DataFrame(rows)


def update_study_reports(ctx, mode, records, events):
    core, out = ctx['core'], ctx['out']; folder = out/'studies'/mode
    f = history_frame(records, mode)
    core['report'].csv(folder/'history_validation.csv', f)
    core['report'].csv(folder/'pareto_validation.csv', f.loc[f.pareto_fixed05])
    core['report'].csv(folder/'sampler_events.csv', pd.DataFrame(events))
    best = choose_grouped_best(records)
    core['protocol'].write_json(folder/'best_so_far.json', dict(
        completed_unique=len(records), selected_trial=best['trial_id'], weights=[best['core_weight'],best['reserve_weight']],
        metrics=best['validation_metrics']['fixed_0p5'], S=best['balanced_score']))
    if len(records) % 10 == 0:
        core['report'].workbook(folder/'excel/search_progress.xlsx', dict(trials=f,
            pareto=f.loc[f.pareto_fixed05], sampler_events=pd.DataFrame(events)))


def optimize_mode(ctx, mode, budget, startup):
    folder = ctx['out']/'studies'/mode
    validate = lambda path, pair: load_model_record(path, ctx['signature'], ctx['core'], pair,
                                                   mode, ctx['frames']['val'].label.to_numpy())
    session = MOTPESession(folder, mode, ctx['signature'], startup, ctx['core'], validate)
    if len(session.records) > budget:
        raise ValueError('Completed unique trials exceed the locked budget')
    while len(session.records) < budget:
        trial, event = session.next_unique()
        n = len(session.records)+1
        record = fit_grouped(folder/'trials'/f'trial_{n:04d}', mode, event['indices'],
                             event, ctx, unique_number=n)
        session.complete_unique(trial, event, record)
        update_study_reports(ctx, mode, session.records, session.events)
    update_study_reports(ctx, mode, session.records, session.events)
    return session.records


def select_and_lock(ctx, histories):
    import shutil
    core, out = ctx['core'], ctx['out']; selected, comparisons = {}, {}
    all_history = []; val_rows = []
    for mode, records in histories.items():
        best = choose_grouped_best(records)
        selected[mode] = best
        key = tuple(best['indices']); folder = out/'selected_models'/mode; folder.mkdir(parents=True, exist_ok=True)
        source = out/best['model_relpath']; shutil.copyfile(source, folder/'best_model.txt')
        matched = next((r for r in records if r['indices'] == [key[0], 0]), None)
        if matched is None:
            matched = fit_grouped(out/'matched_controls'/mode/'core_only', mode, (key[0],0),
                dict(method='predeclared_matched_core_control; excluded_from_search_selection'), ctx,
                phase='post_selection_control')
        hcontrol = fit_grouped(out/'matched_controls'/mode/'H_only', 'H_only', key,
            dict(method='predeclared_same_weights_H2048_control; excluded_from_search_selection'), ctx,
            phase='post_selection_control')
        comparisons[mode] = dict(unweighted=records[0], core_only=matched, selected=best, H_only=hcontrol)
        eligibility, pareto = selection_flags(records)
        chosen = records.index(best)
        lock = dict(run_signature=ctx['signature'], mode=mode, selected_trial=best['trial_id'],
            indices=list(key), core_weight=best['core_weight'], reserve_weight=best['reserve_weight'],
            core_rules=list(CORE_RULES), reserve_rules=list(RESERVE_RULES),
            selection='Pareto + all three validation objectives >= own unweighted reference within 1e-12; '
                      'maximize S=.50*AUROC+.25*BACC+.25*(MCC+1)/2; exact ties weaker max/sum/core/reserve weights',
            validation_metrics=best['validation_metrics'], thresholds=best['thresholds'],
            S=best['balanced_score'], admissible=bool(eligibility[chosen]), pareto=bool(pareto[chosen]),
            model_sha256=core['protocol'].sha(folder/'best_model.txt'),
            completed_unique_configurations=len(records),
            supplementary_controls_excluded_from_search_selection=True)
        if (folder/'selection_lock.json').exists() and core['protocol'].read_json(folder/'selection_lock.json') != lock:
            raise ValueError('An existing per-model selection lock would change')
        core['protocol'].write_json(folder/'selection_lock.json', lock)
        model_rows = pd.concat([ctx['frames']['train'].reset_index(drop=True),
                               ctx['matches']['train'].drop(columns='original_row_index').reset_index(drop=True)],axis=1)
        model_rows['sample_weight'] = grouped_weights(ctx['matches']['train'], key)
        model_rows['effective_loss_weight'] = model_rows.sample_weight.to_numpy()*np.where(
            model_rows.label.to_numpy() == 1, ctx['class_weight'], 1.)
        core['report'].csv(folder/'selected_train_weights.csv', model_rows)
        all_history.append(history_frame(records, mode))
        for condition, record in comparisons[mode].items():
            for policy, metric in record['validation_metrics'].items():
                val_rows.append(dict(mode=mode, condition=condition, split='val', threshold_policy=policy,
                    core_weight=record['core_weight'], reserve_weight=record['reserve_weight'],
                    best_iteration=record['best_iteration'], S_fixed05=record['balanced_score'], **metric))
    registry = dict(run_signature=ctx['signature'],
        selection_locks={mode:dict(path=str((out/'selected_models'/mode/'selection_lock.json').relative_to(out)),
            sha256=core['protocol'].sha(out/'selected_models'/mode/'selection_lock.json')) for mode in selected},
        comparisons={mode:{condition:dict(
            record_path=str((out/record['model_folder_relpath']/'COMPLETE.json').relative_to(out)),
            record_sha256=core['protocol'].sha(out/record['model_folder_relpath']/'COMPLETE.json'),
            model_path=record['model_relpath'], model_sha256=record['files_sha256']['model.txt'],
            indices=record['indices'], thresholds=record['thresholds'], input_dim=record['input_dim'],
            mechanism_mask=record['mechanism_mask'], best_iteration=record['best_iteration'])
            for condition, record in group.items()} for mode, group in comparisons.items()},
        selection_complete_before_test=True, test_data_opened_at_lock=False,
        all_source_groups_and_metrics_predeclared=True)
    if (out/'global_selection_lock.json').exists() and core['protocol'].read_json(out/'global_selection_lock.json') != registry:
        raise ValueError('Previously locked four-model test plan would change')
    core['protocol'].write_json(out/'global_selection_lock.json', registry)
    core['report'].csv(out/'search_history_validation.csv', pd.concat(all_history, ignore_index=True))
    val = pd.DataFrame(val_rows)
    core['report'].csv(out/'validation_comparison.csv', val)
    core['report'].csv(out/'selected_validation.csv', val.loc[val.condition.eq('selected')])
    return registry, val



def verify_registry(ctx):
    core, out = ctx['core'], ctx['out']; read = core['protocol'].read_json
    registry = read(out/'global_selection_lock.json')
    if registry['run_signature'] != ctx['signature'] or set(registry['comparisons']) != set(MODEL_MASKS):
        raise ValueError('Incomplete or incompatible four-model selection registry')
    yv = ctx['frames']['val'].label.to_numpy()
    for mode in MODEL_MASKS:
        rows, seen = [], set()
        for path in sorted((out/'studies'/mode/'events').glob('event_*.json')):
            event = read(path)
            if event['run_signature'] != ctx['signature']:
                raise ValueError('Historical sampler event belongs to another run')
            if event['state'] != 'COMPLETE':
                continue
            folder = out/'studies'/mode/'trials'/event['trial_id']
            if core['protocol'].sha(folder/'COMPLETE.json') != event['model_record_sha256']:
                raise ValueError('Historical model record changed')
            r = load_model_record(folder, ctx['signature'], core, event['indices'], mode, yv)
            if tuple(r['indices']) in seen or r['unique_number'] != len(rows)+1:
                raise ValueError('Historical unique trial registry is duplicated or has gaps')
            if event['values'] != objective_vector(r['validation_metrics']['fixed_0p5']).tolist():
                raise ValueError('Historical objective feedback changed')
            rows.append(r); seen.add(tuple(r['indices']))
        if len(rows) != ctx['budget']:
            raise ValueError('Finish all four locked search budgets before test evaluation')
        expected = choose_grouped_best(rows)
        item = registry['selection_locks'][mode]
        if core['protocol'].sha(out/item['path']) != item['sha256']:
            raise ValueError('Per-model selection lock changed')
        selection = read(out/item['path'])
        if (selection['selected_trial'] != expected['trial_id'] or selection['indices'] != expected['indices']
                or selection['thresholds'] != expected['thresholds']
                or selection['validation_metrics'] != expected['validation_metrics']):
            raise ValueError('Locked choice disagrees with the predeclared validation selection rule')
        if core['protocol'].sha(out/'selected_models'/mode/'best_model.txt') != selection['model_sha256']:
            raise ValueError('Copied selected checkpoint changed')
        for condition, control in registry['comparisons'][mode].items():
            path = out/control['record_path']
            if core['protocol'].sha(path) != control['record_sha256']:
                raise ValueError('A matched-control record changed')
            actual_mode = 'H_only' if condition == 'H_only' else mode
            record = load_model_record(path.parent, ctx['signature'], core,
                                       control['indices'], actual_mode, yv)
            for key in ('thresholds', 'input_dim', 'mechanism_mask', 'best_iteration'):
                if record[key] != control[key]:
                    raise ValueError('Matched control contradicts the locked test registry')
            if core['protocol'].sha(out/control['model_path']) != control['model_sha256']:
                raise ValueError('A matched-control checkpoint changed')
    return registry


def generic_table_svg(path, table, title, label_column):
    columns = ['threshold', *SCORES, 'FN', 'FP']
    left, first, cell = 40, 390, 135
    width = left*2+first+cell*len(columns); height = 165+len(table)*60+90
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>', '<g font-family="Arial, sans-serif" fill="#243B53">',
        f'<text x="40" y="45" font-size="30" font-weight="bold">{html.escape(title)}</text>',
        '<text x="40" y="80" font-size="19">Four independent validation-selected models | core / reserve grouped weights | seed=42</text>',
        f'<rect x="40" y="110" width="{width-80}" height="52" rx="7" fill="#243B53"/>',
        '<text x="55" y="145" font-size="21" fill="white">Model / condition</text>']
    for j, col in enumerate(columns):
        label = 'AUPRC (AP)' if col == 'AUPRC_AP' else ('Threshold' if col == 'threshold' else col)
        parts.append(f'<text x="{left+first+cell*(j+.5)}" y="145" text-anchor="middle" font-size="19" fill="white">{label}</text>')
    for i, (_, row) in enumerate(table.iterrows()):
        y = 162+i*60
        if i%2 == 0:
            parts.append(f'<rect x="40" y="{y}" width="{width-80}" height="60" fill="#F2F6FA"/>')
        parts.append(f'<text x="55" y="{y+39}" font-size="21">{html.escape(str(row[label_column]))}</text>')
        for j, col in enumerate(columns):
            value = float(row[col]); color = '#243B53'; bold = 'normal'
            if col != 'threshold':
                ordered = sorted(set(float(v) for v in table[col]), reverse=col not in ('FN', 'FP'))
                rank = ordered.index(value)
                if rank < (1 if len(table) <= 3 else 3):
                    color = ('#1467C7', '#20854E', '#B88600')[rank]; bold = 'bold'
            label = str(int(value)) if col in ('FN', 'FP') else f'{value:.6f}'
            parts.append(f'<text x="{left+first+cell*(j+.5)}" y="{y+39}" text-anchor="middle" font-size="21" fill="{color}" font-weight="{bold}">{label}</text>')
    foot = 162+len(table)*60
    parts += [f'<text x="40" y="{foot+35}" font-size="17">Blue / green / yellow: ranks 1 / 2 / 3; ties share color. Lower FN/FP is better.</text>',
              f'<text x="40" y="{foot+65}" font-size="17">Weights and checkpoints selected on validation only; the original TITAN test set was previously examined.</text>', '</g></svg>']
    path.write_text('\n'.join(parts), encoding='utf-8')


def export_search_workbook(ctx):
    core, out = ctx['core'], ctx['out']
    history = pd.read_csv(out/'search_history_validation.csv')
    val = pd.read_csv(out/'validation_comparison.csv')
    sheets = dict(selected_validation=val.loc[val.condition.eq('selected')],
        validation_comparison=val, all_trials_validation=history,
        pareto_validation=history.loc[history.pareto_fixed05],
        candidate_coverage=ctx['coverage'], group_coverage=ctx['group_coverage'])
    core['report'].workbook(out/'excel/multiobjective_validation.xlsx', sheets)
    for policy in POLICIES:
        generic_table_svg(out/'figures'/f'selected_validation_{policy}.svg',
            val.loc[val.condition.eq('selected') & val.threshold_policy.eq(policy)],
            f'TITAN validation: selected models ({policy})', 'mode')
    return sheets


def evaluate_grouped_test(ctx):
    import lightgbm as lgb
    core, out = ctx['core'], ctx['out']; write = core['protocol'].write_json
    registry = verify_registry(ctx)
    # This registry has already been written after all four studies and matched
    # controls completed. No test dataframe is opened until after verification.
    models = {}
    for mode, conditions in registry['comparisons'].items():
        for condition, spec in conditions.items():
            model = lgb.Booster(model_file=str(out/spec['model_path']))
            if model.num_feature() != spec['input_dim'] or model.current_iteration() != spec['best_iteration']:
                raise ValueError('A locked checkpoint has the wrong input dimension or iteration')
            # Reproduce committed validation vectors again with the saved native model.
            r = core['protocol'].read_json(out/spec['record_path'])
            probability = np.asarray(model.predict(make_model_input(ctx['X']['val'], spec['mechanism_mask'])), dtype=float)
            archived = np.load(out/r['model_folder_relpath']/'val_probability.npy', allow_pickle=False)
            if not np.allclose(probability, archived, rtol=0, atol=1e-10):
                raise ValueError('A locked checkpoint no longer reproduces its validation probabilities')
            models[mode, condition] = model
    write(out/'test_evaluation_lock.json', dict(run_signature=ctx['signature'],
        selection_registry_sha256=core['protocol'].sha(out/'global_selection_lock.json'),
        model_plan=registry['comparisons'], all_four_models_locked=True,
        test_data_opened_at_lock=False, model_selection_on_test=False))
    frame, X, test_audit = load_split(ctx['root'], ctx['locked']['source'], ctx['locked']['original'],
                                    'test', ctx['files'], core)
    rows, probabilities, paired_summaries = [], {}, []
    y = frame.label.to_numpy()
    for mode, conditions in registry['comparisons'].items():
        for condition, spec in conditions.items():
            p = np.asarray(models[mode, condition].predict(make_model_input(X, spec['mechanism_mask'])), dtype=float)
            probabilities[mode, condition] = p
            a,b = pair_weights(spec['indices'])
            for policy, threshold in spec['thresholds'].items():
                m = core['protocol'].metrics(y, p, threshold)
                rows.append(dict(mode=mode, condition=condition, split='test', threshold_policy=policy,
                    core_weight=a, reserve_weight=b, best_iteration=spec['best_iteration'], **m))
                result = outcome_frame(frame, p, threshold)
                folder = out/'predictions'/mode/condition/policy
                core['report'].csv(folder/'test_all.csv', result)
                for outcome in ('TP','TN','FP','FN'):
                    core['report'].csv(folder/f'test_{outcome}.csv', result.loc[result.outcome.eq(outcome)])
                confusion_svg(folder/'confusion_matrix.svg', mode+' / '+condition, m)
        # All paired transitions use the same fixed .5 decision rule.
        for reference in ('unweighted','core_only','H_only'):
            changes = paired_changes(frame, probabilities[mode, reference], probabilities[mode,'selected'])
            folder = out/'paired_errors'/mode/reference
            core['report'].csv(folder/'paired_all_fixed05.csv', changes)
            core['report'].csv(folder/'corrected_errors_fixed05.csv', changes.loc[changes.corrected_error])
            core['report'].csv(folder/'new_errors_fixed05.csv', changes.loc[changes.new_error])
            transitions = changes.groupby(['reference_outcome','selected_outcome','transition']).size().rename('n').reset_index()
            core['report'].csv(folder/'transition_counts_fixed05.csv', transitions)
            paired_summaries.append(dict(mode=mode, reference=reference,
                corrected_errors=int(changes.corrected_error.sum()), new_errors=int(changes.new_error.sum()),
                net_corrected=int(changes.corrected_error.sum()-changes.new_error.sum())))
    table = pd.DataFrame(rows); deltas = []
    for mode in MODEL_MASKS:
        for policy in POLICIES:
            group = table.loc[table['mode'].eq(mode) & table.threshold_policy.eq(policy)].set_index('condition')
            for reference in ('unweighted','core_only','H_only'):
                new, old = group.loc['selected'], group.loc[reference]
                deltas.append(dict(mode=mode, reference=reference, threshold_policy=policy,
                    selected_threshold=float(new.threshold), reference_threshold=float(old.threshold),
                    **{'delta_'+k:float(new[k]-old[k]) for k in (*SCORES,'FN','FP')}))
    delta = pd.DataFrame(deltas)
    core['report'].csv(out/'test_comparison.csv',table)
    core['report'].csv(out/'selected_test.csv',table.loc[table.condition.eq('selected')])
    core['report'].csv(out/'test_deltas.csv',delta)
    core['report'].csv(out/'paired_error_summary.csv',pd.DataFrame(paired_summaries))
    sheets = export_search_workbook(ctx)
    sheets.update(selected_test=table.loc[table.condition.eq('selected')], test_comparison=table,
        test_deltas=delta, paired_errors=pd.DataFrame(paired_summaries))
    core['report'].workbook(out/'excel/multiobjective_results.xlsx',sheets)
    for policy in POLICIES:
        generic_table_svg(out/'figures'/f'selected_test_{policy}.svg',
            table.loc[table.condition.eq('selected') & table.threshold_policy.eq(policy)],
            f'TITAN test: four validation-selected models ({policy})','mode')
        for mode in MODEL_MASKS:
            generic_table_svg(out/'figures'/f'{mode}_test_{policy}.svg',
                table.loc[table['mode'].eq(mode) & table.threshold_policy.eq(policy)],
                f'TITAN test: {mode} ({policy})','condition')
    write(out/'test_input_audit.json',dict(inputs=test_audit,source_files_sha256=ctx['files'],
        all_original_942_rows_retained=True,no_test_model_weight_or_threshold_selection=True,
        no_test_data_used_to_discover_structure_rules=True,
        limitation='Development comparison on a previously examined test set; not independent external confirmation.'))
    for path,digest in ctx['files'].items():
        if core['protocol'].sha(path)!=digest:
            raise ValueError(f'Pinned source changed during evaluation: {path}')
    write(out/'COMPLETE.json',dict(run_signature=ctx['signature'],four_models_complete=True,
        unique_search_budget_per_model=ctx['budget'],model_selection_on_test=False,
        files_sha256={str(p.relative_to(out)):core['protocol'].sha(p) for p in sorted(out.rglob('*'))
                      if p.is_file() and p!=out/'COMPLETE.json' and p.name!='run.lock'}))
    print('\nTEST RESULTS; four models independently selected using validation AUROC/BACC/MCC:',flush=True)
    cols=['mode','condition','threshold_policy','core_weight','reserve_weight','threshold',*SCORES,'FN','FP']
    print(table.loc[table.condition.eq('selected'),cols].to_string(index=False,float_format=lambda x:f'{x:.6f}'),flush=True)
    print(f'\nDONE: {out}\nExcel: {out/"excel/multiobjective_results.xlsx"}',flush=True)
    return table


def make_grouped_attempt(base):
    base.mkdir(parents=True,exist_ok=True)
    for k in range(1,10000):
        out=base/f'attempt_bayes{k:03d}'
        try:
            out.mkdir();return out
        except FileExistsError:
            pass
    raise RuntimeError('No available grouped multiobjective attempt directory')


def run(args):
    from contextlib import contextmanager
    import fcntl
    root=Path(args.project_root).expanduser().resolve();core=core_modules(root)
    read,write,sha=core['protocol'].read_json,core['protocol'].write_json,core['protocol'].sha
    versions=grouped_runtime();files={}
    if args.resume:
        out=resolve(args.resume,root);previous=read(out/'protocol_lock.json')
        old_root=Path(previous['project_root'])
        search=move_path(previous['search_run'],root,old_root)
        ablation=move_path(previous['ablation_run'],root,old_root)
        budget=previous['budget'];startup=previous['startup']
        if args.budget is not None and args.budget!=budget or args.startup is not None and args.startup!=startup:
            raise ValueError('Budget/startup are fixed on resume; start a fresh attempt to change them')
        for supplied,actual in [(args.search_run,search),(args.ablation_run,ablation)]:
            if supplied and resolve(supplied,root)!=actual:
                raise ValueError('Source path conflicts with the resumed protocol')
    else:
        if args.stage=='test':
            raise ValueError('--stage test requires --resume pointing to this script\'s completed validation search')
        search=resolve(args.search_run or DEFAULT_SEARCH,root)
        ablation=resolve(args.ablation_run,root) if args.ablation_run else search/'ablation_evaluation/attempt_001'
        budget=args.budget if args.budget is not None else 200
        startup=args.startup if args.startup is not None else 40
        out=make_grouped_attempt(resolve(args.output,root));previous=None
    if not 4<=startup<=budget<=1681:
        raise ValueError('Need 4 <= startup <= per-model unique budget <= 1681')
    @contextmanager
    def exclusive():
        with (out/'run.lock').open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError('Another process is using this attempt') from exc
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)
    print('Output:',out,flush=True)
    with exclusive():
        try:
            locked=prepare_locked(root,search,files,core)
            for name in ('numpy','pandas','sklearn','lightgbm','rdkit'):
                if versions[name]!=locked['lock']['versions'][name]:
                    raise ValueError(f'{name} changed since the original search; activate the original titan_bbb environment')
            for name in ('protocol','training','chemistry'):
                tracked(core[name].__file__,files,core,locked['lock']['code_sha256'][name+'.py'])
            frames,X,matches,audit,coverage,group_coverage=group_masks(root,search,locked,files,core)
            warm=source_ablation(root,ablation,search,locked,X['val'],frames['val'].label.to_numpy(),files,core)
            params=dict(locked['lock']['settings']['source_common_params'])
            if params.get('seed')!=42 or float(params.get('scale_pos_weight',1.))!=locked['lock']['class_weight']:
                raise ValueError('Source seed or class-weight policy is inconsistent')
            payload=dict(project_root=str(root),search_run=str(search),ablation_run=str(ablation),
                source_search_signature=locked['lock']['run_signature'],versions=versions,
                self_code_sha256=sha(Path(__file__)),source_files_sha256=files.copy(),
                core_rules=list(CORE_RULES),reserve_rules=list(RESERVE_RULES),
                expected_reserve_coverage=EXPECTED_RESERVE_COVERAGE,
                model_masks=MODEL_MASKS,budget=budget,startup=startup,seed=42,
                params=params,max_rounds=locked['lock']['settings']['max_rounds'],
                patience=locked['lock']['settings']['patience'],class_weight=locked['lock']['class_weight'],
                sampler=dict(name='Optuna MOTPE',n_ei_candidates=64,multivariate=True,group=False,
                    constant_liar=False,acquisition='MOTPE log l(x)-log g(x) maximized over ALL UNTRIED exact-grid pairs',
                    internal_candidate_pool_replaced=True,required_optuna_version='4.5.0',
                    duplicate_policy='forbidden; no random fallback'),
                objective=['validation AUROC','validation BACC at fixed .5','validation MCC at fixed .5'],
                selection='Pareto + all three >= same-model unweighted within 1e-12; '
                    'S=.50*AUROC+.25*BACC+.25*(MCC+1)/2; exact ties weaker max/sum/core/reserve',
                weight_range=[1.,3.],weight_step=.05,overlap='max; within-group union; both true labels; no normalization',
                threshold_policy='fixed .5 plus validation BACC first maximum in .1,.3,.5,.7,.9; strict >',
                data_limit='Rules came from full-fusion validation error analysis; no per-model rediscovery. Original test set was previously examined.')
            signature=core['protocol'].object_sha(payload)
            if previous is not None:
                if previous['run_signature']!=signature:
                    raise ValueError('Resume signature mismatch: inputs, versions, code, groups or parameters changed; start a fresh attempt')
            else:
                write(out/'protocol_lock.json',dict(run_signature=signature,**payload))
                write(out/'group_definition.json',dict(core=list(CORE_RULES),reserve=list(RESERVE_RULES),
                    nested_alternatives_not_activated=['V07','V09','V10','V11'],
                    old_reference_maps_to_new_group_weights=dict(core=1.10,reserve=1.00),overlap='max'))
                write(out/'train_val_input_audit.json',audit)
                core['report'].csv(out/'candidate_coverage.csv',coverage)
                core['report'].csv(out/'group_coverage.csv',group_coverage)
                core['report'].csv(out/'initial_plan.csv',pd.DataFrame([
                    dict(position=i+1,core_weight=pair_weights(pair)[0],reserve_weight=pair_weights(pair)[1])
                    for i,pair in enumerate(initial_plan(startup))]))
                for split in ('train','val'):
                    core['report'].csv(out/f'{split}_structure_matches.csv',pd.concat([
                        frames[split].reset_index(drop=True),matches[split].drop(columns='original_row_index')],axis=1))
            (out/'figures').mkdir(exist_ok=True)
            ctx=dict(root=root,out=out,core=core,locked=locked,files=files,signature=signature,
                frames=frames,X=X,matches=matches,warm=warm,params=params,
                max_rounds=payload['max_rounds'],patience=payload['patience'],class_weight=payload['class_weight'],
                budget=budget,startup=startup,coverage=coverage,group_coverage=group_coverage)
            if args.stage=='inspect':
                core['report'].workbook(out/'excel/input_inspection.xlsx',dict(candidates=coverage,groups=group_coverage))
                print('INSPECT COMPLETE: all-17 rule calibration, train coverage and 10 warm checkpoints verified; no new fit or test data loaded.',flush=True)
                return out
            if args.stage!='test':
                histories={mode:optimize_mode(ctx,mode,budget,startup) for mode in MODEL_MASKS}
                registry,val=select_and_lock(ctx,histories)
                export_search_workbook(ctx)
                print('\nFOUR VALIDATION-SELECTED MODELS:',flush=True)
                cols=['mode','core_weight','reserve_weight','threshold_policy','threshold','AUROC','BACC','MCC','S_fixed05']
                print(val.loc[val.condition.eq('selected'),cols].to_string(index=False,float_format=lambda x:f'{x:.6f}'),flush=True)
                if args.stage=='search':
                    print(f'SEARCH COMPLETE: all four models and controls locked before test.\n{out}',flush=True)
                    return out
            evaluate_grouped_test(ctx)
            return out
        except BaseException as exc:
            write(out/'INTERRUPTED.json' if isinstance(exc,KeyboardInterrupt) else out/'FAILED.json',dict(
                type=type(exc).__name__,error=str(exc),UTC=datetime.now(timezone.utc).isoformat(),
                resume='Use --resume with this attempt. Completed models/events are verified; pending fits recover without duplication.'))
            raise


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--project-root',default='..')
    p.add_argument('--search-run',help='original completed priority search; default '+DEFAULT_SEARCH)
    p.add_argument('--ablation-run',help='completed 36-arm validation registry; default <search-run>/ablation_evaluation/attempt_001')
    p.add_argument('--output',default='titan_grouped_multiobjective_seed42_results')
    p.add_argument('--resume',help='existing grouped-multiobjective attempt, relative to project-root or absolute')
    p.add_argument('--budget',type=int,help='unique group-weight configurations PER model; default 200; fixed on resume')
    p.add_argument('--startup',type=int,help='initial unique points including four anchors; default 40; fixed on resume')
    p.add_argument('--stage',choices=['all','inspect','search','test'],default='all')
    p.add_argument('--debug',action='store_true')
    return p


def main():
    args=parser().parse_args()
    try:run(args)
    except KeyboardInterrupt:
        print('\nINTERRUPTED: model fits and sampler events are saved; resume the same attempt.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        if args.debug:raise
        print(f'\nSTOPPED: {exc}',file=sys.stderr)
        raise SystemExit(1)


if __name__=='__main__':
    main()
