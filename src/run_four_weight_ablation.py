#!/usr/bin/env python3
"""Four fixed grouped-weight regimes x nine mechanism ablations, seed=42.

Place this file beside run_grouped_multiobjective_bayes.py in TITAN-BBB-main/src.
Keep the existing titan_priority_bayes package and completed grouped search.

Run all 36 arms, lock validation checkpoints/thresholds, then evaluate test:
  python run_four_weight_ablation.py --project-root ..
Fit/validation only:
  python run_four_weight_ablation.py --project-root .. --stage fit
Resume an interrupted attempt (also permits --stage test after --stage fit):
  python run_four_weight_ablation.py --project-root .. --resume <attempt>

The four weight pairs are read from this run's validation selection locks.
Indices and declared weights must agree on the exact grid. Historical weight
values are reference information only. No weights are selected using test data.

Each weight regime is applied to ALL nine heads: baseline H2048;
zero_control H2048+[0,0,0]; three single mechanisms; three mechanism pairs;
full_fusion. Disabled mechanism slots remain zero in fixed efflux,influx,pampa
positions. Baseline means no mechanism scores, NOT no structure weighting.

Only Stage-2 LightGBM heads may be refitted. All upstream features, Stage-1
teachers, original TITAN rows/labels, parameters, class weighting, seed,
validation class-weighted BCE early stopping and maximum rounds are inherited.
Training weights are the union within each group, max(core,reserve) across
groups, for both true labels; validation and test rows receive no such weights.
Existing exact checkpoints are verified on validation and reused without
selection by their validation/test performance. Otherwise fit on train only.

All 36 validation checkpoints and both thresholds (.5 and the first maximum
validation BACC in .1,.3,.5,.7,.9; strict p>threshold) are locked before opening
the test dataframe. No weight, mechanism, model, or threshold is selected using
test. Per-arm test metrics, all original compound rows, TP/TN/FP/FN files,
paired errors, Excel workbooks, ranked SVG tables and SVG confusion matrices
are exported. No Optuna optimization is run by this script.

This is a fixed-weight transfer ablation, NOT independent weight optimization
of all 36 heads. Rules and weights used validation in earlier development.
The TITAN test set has already been examined: results are development
comparisons, not new independent confirmation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import html
import importlib.util
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd


DEFAULT_GROUPED_RUN = 'titan_grouped_multiobjective_seed42_results/attempt_bayes001'
HISTORICAL_WEIGHTS = {
    'full_fusion': (2.35, 2.30),
    'efflux_influx': (1.35, 1.60),
    'efflux_pampa': (1.05, 1.00),
    'influx_pampa': (1.25, 1.30),
}
# Kept as a mode-name compatibility alias for archived replay. Do not use its
# numerical values to constrain new selections or label current-run tables.
EXPECTED_WEIGHTS = HISTORICAL_WEIGHTS
MECHANISMS = {
    'baseline': None,
    'zero_control': [0, 0, 0],
    'efflux_only': [1, 0, 0],
    'influx_only': [0, 1, 0],
    'pampa_only': [0, 0, 1],
    'efflux_influx': [1, 1, 0],
    'efflux_pampa': [1, 0, 1],
    'influx_pampa': [0, 1, 1],
    'full_fusion': [1, 1, 1],
}
POLICIES = ('fixed_0p5', 'validation_selected')
SCORES = ('AUROC', 'AUPRC_AP', 'ACC', 'BACC', 'SN', 'SP', 'F1', 'MCC')


def helper_module(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Put the original '
            'run_grouped_multiobjective_bayes.py beside this script in src, '
            'or pass --pipeline-script with its exact path.')
    spec = importlib.util.spec_from_file_location('_four_ablation_original_pipeline', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan_from_locks(selections, gm):
    if set(selections) != set(gm.MODEL_MASKS):
        raise ValueError('Need exactly the four current validation-selected models')
    result = []
    for group in gm.MODEL_MASKS:
        lock = selections[group]
        if lock.get('mode') != group:
            raise ValueError(f'{group}: selection lock has an incorrect mode')
        if lock.get('core_rules') != list(gm.CORE_RULES) or lock.get('reserve_rules') != list(gm.RESERVE_RULES):
            raise ValueError(f'{group}: selection lock uses different structure groups')
        pair = tuple(lock['indices'])
        actual = gm.pair_weights(pair)
        if (lock['core_weight'], lock['reserve_weight']) != actual:
            raise ValueError(f'{group}: selected indices and declared weights disagree')
        for experiment, mask in MECHANISMS.items():
            result.append(dict(experiment_id=group+'__'+experiment,
                weight_group=group, source_selected_mode=group,
                core_weight=actual[0], reserve_weight=actual[1], indices=list(pair),
                experiment=experiment, mechanism_mask=mask,
                input_dim=2048 if mask is None else 2051,
                is_source_selected_arm=experiment == group))
    return result


def source_context(root, source, gm, core, versions):
    """Validate original inputs/chemistry and selected history; read no test tables."""
    files = {}
    read, sha = core['protocol'].read_json, core['protocol'].sha
    proto = read(gm.tracked(source/'protocol_lock.json', files, core))
    if core['protocol'].object_sha({k: v for k, v in proto.items()
                                   if k != 'run_signature'}) != proto['run_signature']:
        raise ValueError('Grouped source protocol signature changed')
    if sha(Path(gm.__file__)) != proto['self_code_sha256']:
        raise ValueError('The grouped-search helper differs from the script that created this run')
    for name, value in versions.items():
        if proto['versions'].get(name) != value:
            raise ValueError(f'{name} changed since the grouped search; use the original titan_bbb environment')
    if (proto['seed'] != 42 or proto['core_rules'] != list(gm.CORE_RULES)
            or proto['reserve_rules'] != list(gm.RESERVE_RULES)
            or proto['model_masks'] != gm.MODEL_MASKS):
        raise ValueError('Grouped source does not use the confirmed seed/groups/mechanism masks')
    old_root = Path(proto['project_root'])
    for name, digest in proto['source_files_sha256'].items():
        gm.tracked(gm.move_path(name, root, old_root), files, core, digest)
    search = gm.move_path(proto['search_run'], root, old_root)
    locked = gm.prepare_locked(root, search, files, core)
    for name in ('protocol', 'training', 'chemistry'):
        gm.tracked(core[name].__file__, files, core, locked['lock']['code_sha256'][name+'.py'])
    frames, X, matches, audit, coverage, groups = gm.group_masks(root, search, locked, files, core)
    # Reproduce ALL active grouped masks, not just the two original core rules.
    columns = [*gm.CORE_RULES, *gm.RESERVE_RULES, 'core_group', 'reserve_group',
               'eligible_for_structure_weight', 'primary_unit']
    for split in ('train', 'val'):
        saved = pd.read_csv(gm.tracked(source/f'{split}_structure_matches.csv', files, core),
                            dtype=str, keep_default_na=False)
        if saved.SMILES.tolist() != frames[split].SMILES.tolist() or not np.array_equal(
                pd.to_numeric(saved.label).to_numpy(), frames[split].label.to_numpy()):
            raise ValueError(f'{split}: grouped-search rows no longer align')
        for column in columns:
            if not np.array_equal(gm.read_saved_flags(saved, column),
                                  matches[split][column].to_numpy(dtype=bool)):
                raise ValueError(f'{split}/{column}: original grouped training mask changed')
    for key in ('params', 'max_rounds', 'patience', 'class_weight'):
        original = (locked['lock']['settings']['source_common_params'] if key == 'params'
            else locked['lock']['class_weight'] if key == 'class_weight'
            else locked['lock']['settings'][key])
        if proto[key] != original:
            raise ValueError(f'Grouped source {key} differs from the original training protocol')
    source_ctx = dict(out=source, signature=proto['run_signature'], core=core,
                      frames=frames, budget=proto['budget'])
    registry = gm.verify_registry(source_ctx)
    if not registry['selection_complete_before_test'] or registry['test_data_opened_at_lock']:
        raise ValueError('Grouped source was not locked before its original test evaluation')
    gm.tracked(source/'global_selection_lock.json', files, core)
    selections = {}
    for mode, record in registry['selection_locks'].items():
        path = gm.tracked(source/record['path'], files, core, record['sha256'])
        selections[mode] = read(path)
    plan = plan_from_locks(selections, gm)
    needed_pairs = {tuple(s['indices']) for s in plan}
    # Only exact pre-existing configurations are eligible; first archive order,
    # never the best validation/test score among different configurations.
    candidates = []
    for mode in gm.MODEL_MASKS:
        candidates.extend(sorted((source/'studies'/mode/'trials').glob('*/COMPLETE.json')))
    for conditions in registry['comparisons'].values():
        candidates.extend(source/spec['record_path'] for spec in conditions.values())
    warm = {}
    import lightgbm as lgb
    yv = frames['val'].label.to_numpy()
    for path in candidates:
        preliminary = read(path)
        key = (preliminary['mode'], tuple(preliminary['indices']))
        if key in warm or key[1] not in needed_pairs:
            continue
        record = gm.load_model_record(path.parent, proto['run_signature'], core,
                                     key[1], key[0], yv)
        gm.tracked(path, files, core)
        for name, digest in record['files_sha256'].items():
            gm.tracked(path.parent/name, files, core, digest)
        mask = None if key[0] == 'H_only' else gm.MODEL_MASKS[key[0]]
        if record['mechanism_mask'] != mask or record['input_dim'] != (2048 if mask is None else 2051):
            raise ValueError('Archived warm model input does not match its declared mask')
        model = lgb.Booster(model_file=str(path.parent/'model.txt'))
        probability = np.asarray(model.predict(gm.make_model_input(X['val'], mask)), dtype=float)
        saved = np.load(path.parent/'val_probability.npy', allow_pickle=False)
        if (model.num_feature() != record['input_dim']
                or model.current_iteration() != record['best_iteration']
                or not np.allclose(probability, saved, rtol=0, atol=1e-10)):
            raise ValueError('Warm checkpoint does not reproduce the archived validation predictions')
        warm[key] = dict(path=path.parent/'model.txt', record=record, model=model,
                         record_path=path)
    # Each diagonal arm must reuse the EXACT selected native model, even if
    # another record with matching settings exists earlier in the archive.
    for group, item in registry['comparisons'].items():
        selected = item['selected']
        key = (group, tuple(selected['indices']))
        if key not in warm:
            raise ValueError(f'Missing exact selected checkpoint for {group}')
        path = source/selected['model_path']
        if sha(path) != selected['model_sha256']:
            raise ValueError('Source-selected checkpoint changed')
        if sha(warm[key]['path']) != sha(path):
            raise ValueError('Duplicate exact configuration produced different native checkpoints')
    return dict(root=root, source=source, gm=gm, core=core, files=files,
        source_protocol=proto, locked=locked, frames=frames, X=X, matches=matches,
        audit=audit, coverage=coverage, group_coverage=groups,
        params=dict(proto['params']), max_rounds=proto['max_rounds'],
        patience=proto['patience'], class_weight=proto['class_weight'],
        source_registry=registry, selections=selections, warm=warm, plan=plan)


def thresholds_and_metrics(yv, probability, core):
    thresholds = dict(fixed_0p5=.5,
        validation_selected=core['protocol'].choose_threshold(yv, probability))
    return thresholds, {policy: core['protocol'].metrics(yv, probability, threshold)
                        for policy, threshold in thresholds.items()}


def verify_arm(ctx, spec):
    gm, core = ctx['gm'], ctx['core']
    folder = ctx['out']/'models'/spec['experiment_id']
    r = core['protocol'].read_json(folder/'COMPLETE.json')
    if r['run_signature'] != ctx['signature'] or r['spec'] != spec:
        raise ValueError('Ablation checkpoint belongs to another run or weight/mask specification')
    for name, digest in r['files_sha256'].items():
        if Path(name).name != name or core['protocol'].sha(folder/name) != digest:
            raise ValueError(f'Ablation artifact changed: {folder/name}')
    if set(r['files_sha256']) != {'PLAN.json', 'model.txt', 'val_probability.npy', 'fit_history.json'}:
        raise ValueError('Ablation checkpoint file registry is incomplete')
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(folder/'model.txt'))
    probability = np.asarray(model.predict(gm.make_model_input(ctx['X']['val'],
                             spec['mechanism_mask'])), dtype=float)
    saved = np.load(folder/'val_probability.npy', allow_pickle=False)
    if (model.num_feature() != spec['input_dim'] or model.current_iteration() != r['best_iteration']
            or not np.allclose(probability, saved, rtol=0, atol=1e-10)):
        raise ValueError('Ablation checkpoint dimension, rounds or validation predictions changed')
    thresholds, metrics = thresholds_and_metrics(ctx['frames']['val'].label.to_numpy(), probability, core)
    if thresholds != r['thresholds'] or core['protocol'].object_sha(metrics) != core['protocol'].object_sha(r['validation_metrics']):
        raise ValueError('Ablation thresholds/metrics contradict validation-only selection')
    if core['protocol'].read_json(folder/'PLAN.json') != dict(run_signature=ctx['signature'], spec=spec):
        raise ValueError('Ablation PLAN.json contradicts the locked experiment')
    return r, model


def fit_arm(ctx, spec):
    gm, core = ctx['gm'], ctx['core']
    folder = ctx['out']/'models'/spec['experiment_id']
    if (folder/'COMPLETE.json').is_file():
        return verify_arm(ctx, spec)[0]
    folder.mkdir(parents=True, exist_ok=True)
    write = core['protocol'].write_json
    before = dict(run_signature=ctx['signature'], spec=spec)
    if (folder/'PLAN.json').exists() and core['protocol'].read_json(folder/'PLAN.json') != before:
        raise ValueError('Interrupted arm has a different plan')
    write(folder/'PLAN.json', before)
    started = time.monotonic()
    mask = spec['mechanism_mask']
    Xv = gm.make_model_input(ctx['X']['val'], mask)
    yv = ctx['frames']['val'].label.to_numpy()
    w = gm.grouped_weights(ctx['matches']['train'], spec['indices'])
    key = ('H_only' if spec['experiment'] == 'baseline' else spec['experiment'], tuple(spec['indices']))
    warm = ctx['warm'].get(key)
    print(f'[{spec["experiment_id"]}] core={spec["core_weight"]:.2f}, '
          f'reserve={spec["reserve_weight"]:.2f}; input={spec["input_dim"]}', flush=True)
    if warm is not None:
        model = warm['model']
        shutil.copyfile(warm['path'], folder/'model.txt')
        history = dict(reused_model=str(warm['path']), no_new_fit=True,
                       original_record=str(warm['record_path']))
        fit_mode = 'reused_exact_validation_verified_grouped_checkpoint'
    else:
        model, history = core['training'].fit_head(
            gm.make_model_input(ctx['X']['train'], mask), ctx['frames']['train'].label.to_numpy(),
            ctx['params'], ctx['max_rounds'], Xv, yv,
            weights=None if np.equal(w, 1.).all() else w,
            patience=ctx['patience'], scale_pos_weight=ctx['class_weight'], verbose=False)
        model.save_model(str(folder/'model.txt'))
        fit_mode = 'new_train_only_fit; validation_class_weighted_BCE_early_stopping'
    probability = np.asarray(model.predict(Xv), dtype=float)
    thresholds, metrics = thresholds_and_metrics(yv, probability, core)
    if warm is not None and (thresholds != warm['record']['thresholds'] or
            core['protocol'].object_sha(metrics) != core['protocol'].object_sha(warm['record']['validation_metrics'])):
        raise ValueError('Reused model no longer reproduces its original validation metrics')
    np.save(folder/'val_probability.npy', probability, allow_pickle=False)
    write(folder/'fit_history.json', history)
    yt = ctx['frames']['train'].label.to_numpy()
    record = dict(**before, thresholds=thresholds, validation_metrics=metrics,
        fit_mode=fit_mode, reused_model_sha256=core['protocol'].sha(warm['path']) if warm else None,
        best_iteration=int(model.best_iteration if model.best_iteration > 0 else model.current_iteration()),
        seconds=time.monotonic()-started,
        sample_weight_summary=dict(n_upweighted=int((w > 1).sum()), min=float(w.min()), max=float(w.max()),
            positive_sum=float(w[yt == 1].sum()), negative_sum=float(w[yt == 0].sum())),
        files_sha256={name: core['protocol'].sha(folder/name) for name in
                      ('PLAN.json', 'model.txt', 'val_probability.npy', 'fit_history.json')})
    write(folder/'COMPLETE.json', record)
    print(f'  val AUROC={metrics["fixed_0p5"]["AUROC"]:.9f}; '
          f'rounds={record["best_iteration"]}; threshold={thresholds["validation_selected"]:.1f}; {fit_mode}', flush=True)
    return record


def metric_table(records, split, core, probabilities=None, y=None):
    rows = []
    for r in records:
        # Lists remain in JSON manifests, not Excel cells.
        spec = {k: v for k, v in r['spec'].items() if k not in ('indices', 'mechanism_mask')}
        for policy, threshold in r['thresholds'].items():
            metric = (r['validation_metrics'][policy] if split == 'val'
                      else core['protocol'].metrics(y, probabilities[spec['experiment_id']], threshold))
            rows.append(dict(**spec, split=split, threshold_policy=policy,
                             best_iteration=r['best_iteration'], fit_mode=r['fit_mode'], **metric))
    return pd.DataFrame(rows)


def contrasts(table):
    rows = []
    for group in EXPECTED_WEIGHTS:
        for policy in POLICIES:
            sub = table.loc[table.weight_group.eq(group) & table.threshold_policy.eq(policy)].set_index('experiment')
            for reference in ('baseline', 'zero_control', group):
                old = sub.loc[reference]
                for experiment, new in sub.iterrows():
                    if experiment == reference:
                        continue
                    rows.append(dict(weight_group=group, split=str(new['split']), threshold_policy=policy,
                        reference=reference, experiment=experiment, core_weight=float(new.core_weight),
                        reserve_weight=float(new.reserve_weight), reference_threshold=float(old.threshold),
                        candidate_threshold=float(new.threshold),
                        **{'delta_'+k: float(new[k]-old[k]) for k in (*SCORES, 'FN', 'FP')}))
    return pd.DataFrame(rows)


def table_svg(path, table, title, subtitle):
    cols = ['threshold', *SCORES, 'FN', 'FP']
    left, first, cell = 40, 340, 135
    width = left*2+first+cell*len(cols)
    height = 250+59*len(table)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>', '<g font-family="Arial, sans-serif" fill="#243B53">',
        f'<text x="40" y="43" font-size="29" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="40" y="79" font-size="18">{html.escape(subtitle)}</text>',
        f'<rect x="40" y="106" width="{width-80}" height="52" rx="7" fill="#243B53"/>',
        '<text x="55" y="141" font-size="20" fill="white">Mechanism configuration</text>']
    for j, col in enumerate(cols):
        name = 'AUPRC (AP)' if col == 'AUPRC_AP' else 'Threshold' if col == 'threshold' else col
        parts.append(f'<text x="{left+first+cell*(j+.5)}" y="141" text-anchor="middle" font-size="19" fill="white">{name}</text>')
    for i, (_, row) in enumerate(table.iterrows()):
        y = 158+i*59
        if i % 2 == 0:
            parts.append(f'<rect x="40" y="{y}" width="{width-80}" height="59" fill="#F2F6FA"/>')
        parts.append(f'<text x="55" y="{y+38}" font-size="21">{html.escape(str(row.experiment))}</text>')
        for j, col in enumerate(cols):
            value = float(row[col]); color = '#243B53'; bold = 'normal'
            if col != 'threshold':
                ranks = sorted(set(float(v) for v in table[col]), reverse=col not in ('FN', 'FP'))
                rank = ranks.index(value)
                if rank < (1 if len(table) <= 3 else 3):
                    color = ('#1467C7', '#20854E', '#B88600')[rank]; bold = 'bold'
            label = str(int(value)) if col in ('FN', 'FP') else f'{value:.6f}'
            parts.append(f'<text x="{left+first+cell*(j+.5)}" y="{y+38}" text-anchor="middle" font-size="21" fill="{color}" font-weight="{bold}">{label}</text>')
    end = 158+59*len(table)
    parts += [f'<text x="40" y="{end+34}" font-size="17">Blue / green / yellow: ranks 1 / 2 / 3; ties share color. Lower FN/FP is better.</text>',
        f'<text x="40" y="{end+64}" font-size="17">Baseline has the SAME structure weights. All original rows retained; reused TITAN test is a development comparison.</text>',
        '</g></svg>']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(parts), encoding='utf-8')


def export_tables(ctx, table):
    out, core = ctx['out'], ctx['core']
    split = str(table.split.iloc[0])
    core['report'].csv(out/f'all_{split}_comparison.csv', table)
    delta = contrasts(table)
    core['report'].csv(out/f'{split}_deltas.csv', delta)
    for group in EXPECTED_WEIGHTS:
        panel = table.loc[table.weight_group.eq(group)]
        weights = panel[['core_weight','reserve_weight']].drop_duplicates()
        if len(weights) != 1:
            raise ValueError(f'{group}: all nine heads must use the same selected weights')
        a,b = map(float,weights.iloc[0])
        folder = out/'groups'/group
        core['report'].csv(folder/f'{split}_comparison.csv', panel)
        for policy in POLICIES:
            sub = panel.loc[panel.threshold_policy.eq(policy)]
            core['report'].csv(folder/f'{split}_{policy}.csv', sub)
            table_svg(folder/f'{split}_{policy}.svg', sub, f'TITAN {split}: weights selected for {group}',
                f'Core={a:.2f} | Reserve={b:.2f} | {policy} | seed=42 | Fixed weights across all nine heads')
    return delta


def lock_validation(ctx, records):
    core, out = ctx['core'], ctx['out']
    registry = dict(run_signature=ctx['signature'], completed_models=36,
        model_or_weight_selected_by_test=False, test_data_opened_at_lock=False,
        models={r['spec']['experiment_id']: dict(spec=r['spec'],
            record_sha256=core['protocol'].sha(out/'models'/r['spec']['experiment_id']/'COMPLETE.json'),
            model_sha256=r['files_sha256']['model.txt'], thresholds=r['thresholds'],
            best_iteration=r['best_iteration']) for r in records})
    path = out/'validation_lock.json'
    if path.exists() and core['protocol'].read_json(path) != registry:
        raise ValueError('Previously locked 36-arm validation registry would change')
    core['protocol'].write_json(path, registry)
    table = metric_table(records, 'val', core)
    delta = export_tables(ctx, table)
    sheets = dict(all_validation=table, validation_deltas=delta,
        weights=ctx['weights_table'], candidate_coverage=ctx['coverage'], group_coverage=ctx['group_coverage'])
    for i, group in enumerate(EXPECTED_WEIGHTS, 1):
        sheets[f'group{i}_validation'] = table.loc[table.weight_group.eq(group)]
    core['report'].workbook(out/'excel/four_weight_validation.xlsx', sheets)
    return registry, table


def evaluate_test(ctx, records, registry, val):
    core, gm, out = ctx['core'], ctx['gm'], ctx['out']
    read, write = core['protocol'].read_json, core['protocol'].write_json
    if read(out/'validation_lock.json') != registry or len(records) != 36:
        raise ValueError('All 36 models must be locked before test')
    models = {}
    for r in records:
        spec = r['spec']; name = spec['experiment_id']
        if core['protocol'].sha(out/'models'/name/'COMPLETE.json') != registry['models'][name]['record_sha256']:
            raise ValueError('An arm record changed after validation lock')
        checked, model = verify_arm(ctx, spec)
        if checked != r:
            raise ValueError('An arm record changed after checkpoint verification')
        models[name] = model
    test_lock = dict(run_signature=ctx['signature'], completed_models=36,
        validation_registry_sha256=core['protocol'].sha(out/'validation_lock.json'),
        all_models_and_thresholds_locked=True, test_data_opened_at_lock=False,
        no_test_model_weight_threshold_selection=True)
    if (out/'test_evaluation_lock.json').exists() and read(out/'test_evaluation_lock.json') != test_lock:
        raise ValueError('Existing test plan differs from the validation lock')
    write(out/'test_evaluation_lock.json', test_lock)
    frame, X, audit = gm.load_split(ctx['root'], ctx['locked']['source'], ctx['locked']['original'],
                                   'test', ctx['files'], core)
    probabilities = {}
    for r in records:
        spec = r['spec']; name = spec['experiment_id']
        p = np.asarray(models[name].predict(gm.make_model_input(X, spec['mechanism_mask'])), dtype=float)
        probabilities[name] = p
        for policy, threshold in r['thresholds'].items():
            result = gm.outcome_frame(frame, p, threshold)
            folder = out/'predictions'/spec['weight_group']/spec['experiment']/policy
            core['report'].csv(folder/'test_all.csv', result)
            for outcome in ('TP', 'TN', 'FP', 'FN'):
                core['report'].csv(folder/f'test_{outcome}.csv', result.loc[result.outcome.eq(outcome)])
            gm.confusion_svg(folder/'confusion_matrix.svg', name,
                             core['protocol'].metrics(frame.label.to_numpy(), p, threshold))
    table = metric_table(records, 'test', core, probabilities, frame.label.to_numpy())
    delta = export_tables(ctx, table)
    paired = []
    for group in EXPECTED_WEIGHTS:
        for reference in ('zero_control', group):
            for experiment in MECHANISMS:
                if experiment == reference:
                    continue
                name = group+'__'+experiment
                changes = gm.paired_changes(frame, probabilities[group+'__'+reference], probabilities[name])
                folder = out/'paired_errors'/group/('vs_'+reference)/experiment
                core['report'].csv(folder/'paired_all_fixed05.csv', changes)
                core['report'].csv(folder/'corrected_errors_fixed05.csv', changes.loc[changes.corrected_error])
                core['report'].csv(folder/'new_errors_fixed05.csv', changes.loc[changes.new_error])
                transitions = changes.groupby(['reference_outcome', 'selected_outcome', 'transition']).size().rename('n').reset_index()
                core['report'].csv(folder/'transition_counts_fixed05.csv', transitions)
                paired.append(dict(weight_group=group, reference=reference, experiment=experiment,
                    corrected_errors=int(changes.corrected_error.sum()), new_errors=int(changes.new_error.sum()),
                    net_corrected=int(changes.corrected_error.sum()-changes.new_error.sum())))
    paired = pd.DataFrame(paired)
    core['report'].csv(out/'paired_error_summary.csv', paired)
    diagonal = table.loc[table.is_source_selected_arm]
    core['report'].csv(out/'four_current_selected_test.csv', diagonal)
    core['report'].csv(out/'four_original_selected_test.csv', diagonal)
    source_reproduction = []
    for group in EXPECTED_WEIGHTS:
        name = group+'__'+group
        source_spec = ctx['source_registry']['comparisons'][group]['selected']
        current_sha = core['protocol'].sha(out/'models'/name/'model.txt')
        if current_sha != source_spec['model_sha256']:
            raise ValueError('An original selected checkpoint was not preserved exactly')
        source_reproduction.append(dict(weight_group=group,
            original_model_sha256=source_spec['model_sha256'], ablation_model_sha256=current_sha,
            byte_identical=True, thresholds_identical=registry['models'][name]['thresholds'] == source_spec['thresholds']))
    reproduction = pd.DataFrame(source_reproduction)
    core['report'].csv(out/'source_checkpoint_reproduction.csv', reproduction)
    sheets = dict(all_test=table, all_validation=val, test_deltas=delta,
        validation_deltas=contrasts(val), original_selected_test=diagonal,
        paired_errors=paired, source_reproduction=reproduction, weights=ctx['weights_table'],
        candidate_coverage=ctx['coverage'], group_coverage=ctx['group_coverage'])
    for i, group in enumerate(EXPECTED_WEIGHTS, 1):
        sheets[f'group{i}_test'] = table.loc[table.weight_group.eq(group)]
        sheets[f'group{i}_validation'] = val.loc[val.weight_group.eq(group)]
    core['report'].workbook(out/'excel/four_weight_ablation_results.xlsx', sheets)
    write(out/'test_input_audit.json', dict(inputs=audit, source_files_sha256=ctx['files'],
        original_rows_retained=942, model_weight_threshold_selected_on_test=False,
        limitation='Previously examined TITAN test; development comparison, not independent confirmation.'))
    for path, digest in ctx['files'].items():
        if core['protocol'].sha(path) != digest:
            raise ValueError(f'Pinned source changed while running: {path}')
    write(out/'COMPLETE.json', dict(run_signature=ctx['signature'], completed_models=36,
        reused_models=sum(r['fit_mode'].startswith('reused_') for r in records),
        model_selected_by_test=False, files_sha256={str(p.relative_to(out)): core['protocol'].sha(p)
            for p in sorted(out.rglob('*')) if p.is_file() and p.name not in
            ('COMPLETE.json', 'run.lock', 'FAILED.json', 'INTERRUPTED.json')}))
    cols = ['experiment', 'core_weight', 'reserve_weight', 'threshold', *SCORES, 'FN', 'FP']
    for group in EXPECTED_WEIGHTS:
        print(f'\nTEST ABLATION: weights selected for {group}; fixed .5:', flush=True)
        print(table.loc[table.weight_group.eq(group) & table.threshold_policy.eq('fixed_0p5'), cols]
              .to_string(index=False, float_format=lambda v: f'{v:.6f}'), flush=True)
    print(f'\nDONE: {out}\nExcel: {out/"excel/four_weight_ablation_results.xlsx"}\n'
          'Both fixed .5 and validation-selected threshold results are saved.', flush=True)
    return table


def new_attempt(base):
    base.mkdir(parents=True, exist_ok=True)
    for k in range(1, 10000):
        out = base/f'attempt_{k:03d}'
        try:
            out.mkdir()
            return out
        except FileExistsError:
            pass
    raise RuntimeError('No free attempt directory')


@contextmanager
def exclusive(out):
    import fcntl
    with (out/'run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process is using this ablation attempt') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run(args):
    root = Path(args.project_root).expanduser().resolve()
    pipeline = Path(args.pipeline_script).expanduser().resolve() if args.pipeline_script else Path(__file__).with_name('run_grouped_multiobjective_bayes.py')
    gm = helper_module(pipeline)
    core = gm.core_modules(root)
    read, write = core['protocol'].read_json, core['protocol'].write_json
    previous = None
    if args.resume:
        out = gm.resolve(args.resume, root)
        previous = read(out/'ablation_protocol_lock.json')
        source = gm.move_path(previous['grouped_run'], root, Path(previous['project_root']))
        if args.grouped_run and gm.resolve(args.grouped_run, root) != source:
            raise ValueError('--grouped-run conflicts with the resumed attempt')
    else:
        if args.stage == 'test':
            raise ValueError('--stage test requires --resume with a completed 36-arm validation attempt')
        source = gm.resolve(args.grouped_run or DEFAULT_GROUPED_RUN, root)
        base = gm.resolve(args.output, root) if args.output else source/'four_weight_ablation'
        out = new_attempt(base)
    print('Output:', out, flush=True)
    with exclusive(out):
        try:
            versions = gm.runtime_versions()
            ctx = source_context(root, source, gm, core, versions)
            payload = dict(project_root=str(root), grouped_run=str(source),
                grouped_signature=ctx['source_protocol']['run_signature'], seed=42,
                versions=versions, self_code_sha256=core['protocol'].sha(Path(__file__)),
                pipeline_code_sha256=core['protocol'].sha(pipeline), source_files_sha256=ctx['files'].copy(),
                plan=ctx['plan'], core_rules=list(gm.CORE_RULES), reserve_rules=list(gm.RESERVE_RULES),
                params=ctx['params'], max_rounds=ctx['max_rounds'], patience=ctx['patience'],
                class_weight=ctx['class_weight'], overlap='within-group union; max across groups; no stacking or normalization',
                thresholds='fixed .5 and validation BACC first maximum in .1,.3,.5,.7,.9; strict >',
                ablation='Four archived fixed weight regimes x nine mechanism heads; no further weight optimization',
                upstream='Original frozen H2048 plus fixed raw M3; no upstream refit',
                limitation='Rules and weights developed using validation; original TITAN test previously examined.')
            signature = core['protocol'].object_sha(payload)
            if previous is not None and previous != dict(run_signature=signature, **payload):
                raise ValueError('Resume signature mismatch: source, code, versions, weights, inputs or parameters changed')
            ctx.update(out=out, signature=signature)
            weights = pd.DataFrame([dict(weight_group=g, core_weight=lock['core_weight'], reserve_weight=lock['reserve_weight'],
                core_rules=','.join(gm.CORE_RULES), reserve_rules=','.join(gm.RESERVE_RULES))
                for g, lock in ctx['selections'].items()])
            ctx['weights_table'] = weights
            if previous is None:
                write(out/'ablation_protocol_lock.json', dict(run_signature=signature, **payload))
                write(out/'experiment_plan.json', ctx['plan'])
                write(out/'train_val_input_audit.json', ctx['audit'])
                core['report'].csv(out/'fixed_weights.csv', weights)
                core['report'].csv(out/'candidate_coverage.csv', ctx['coverage'])
                core['report'].csv(out/'group_coverage.csv', ctx['group_coverage'])
                for group in EXPECTED_WEIGHTS:
                    spec = next(s for s in ctx['plan'] if s['weight_group'] == group)
                    rows = pd.concat([ctx['frames']['train'].reset_index(drop=True),
                        ctx['matches']['train'].drop(columns='original_row_index').reset_index(drop=True)], axis=1)
                    rows['sample_weight'] = gm.grouped_weights(ctx['matches']['train'], spec['indices'])
                    rows['effective_loss_weight'] = rows.sample_weight.to_numpy()*np.where(
                        rows.label.to_numpy() == 1, ctx['class_weight'], 1.)
                    core['report'].csv(out/'groups'/group/'train_sample_weights.csv', rows)
            if args.stage == 'inspect':
                print('INSPECT COMPLETE: 4 archived weight pairs, 36 arms, original inputs/masks and reusable checkpoints verified. No new fit or test dataframe read.', flush=True)
                return out
            records = []
            for spec in ctx['plan']:
                if args.stage == 'test':
                    records.append(verify_arm(ctx, spec)[0])
                else:
                    records.append(fit_arm(ctx, spec))
            registry, val = lock_validation(ctx, records)
            if args.stage == 'fit':
                print(f'VALIDATION COMPLETE: 36 models/thresholds locked; no test dataframe read.\n{out}', flush=True)
                return out
            evaluate_test(ctx, records, registry, val)
            return out
        except BaseException as exc:
            write(out/('INTERRUPTED.json' if isinstance(exc, KeyboardInterrupt) else 'FAILED.json'),
                dict(type=type(exc).__name__, error=str(exc), UTC=datetime.now(timezone.utc).isoformat(),
                    resume='Use --resume with this ablation attempt; verified completed arms are reused.'))
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--project-root', default='..')
    p.add_argument('--grouped-run', help='completed grouped search; default '+DEFAULT_GROUPED_RUN)
    p.add_argument('--pipeline-script', help='exact path to the unchanged run_grouped_multiobjective_bayes.py; default beside this file')
    p.add_argument('--output', help='optional output base relative to project-root; default <grouped-run>/four_weight_ablation')
    p.add_argument('--resume', help='exact existing ablation attempt, relative to project-root or absolute')
    p.add_argument('--stage', choices=['all', 'inspect', 'fit', 'test'], default='all')
    p.add_argument('--debug', action='store_true')
    return p


def main():
    args = parser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print('\nINTERRUPTED: completed arms are saved; resume the same attempt.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        if args.debug:
            raise
        print(f'\nSTOPPED: {exc}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
