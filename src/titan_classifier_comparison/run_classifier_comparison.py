#!/usr/bin/env python3
"""Four controlled TITAN classifier experiments on ONE frozen representation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import joblib
import numpy as np
import pandas as pd
import sklearn

from titan_classifier_comparison import data as source
from titan_classifier_comparison import training as backend
from titan_classifier_comparison.common import (SPLITS, classification_metrics, file_record,
                                               read_json, write_json)

GROUPS, METRICS = backend.GROUPS, backend.METRICS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--project-root', type=Path, default=HERE.parents[1])
    p.add_argument('--baseline-run', type=Path, help='Results root containing the ORIGINAL 2048-D baseline and its manifest')
    p.add_argument('--mechanism-dir', type=Path, help='Default: PROJECT/mechanism_features')
    p.add_argument('--output-dir', type=Path, help='Default: PROJECT/titan_classifier_comparison_seed42_results')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:0')
    p.add_argument('--num-threads', type=int, default=4)
    p.add_argument('--feature-batch-size', type=int, default=32)
    p.add_argument('--nn-epochs', type=int, default=1000)
    p.add_argument('--nn-patience', type=int, default=10)
    p.add_argument('--lgbm-rounds', type=int, default=2000)
    p.add_argument('--lgbm-patience', type=int, default=50)
    p.add_argument('--check-only', action='store_true', help='Read-only provenance/alignment checks, no Torch/LightGBM import')
    p.add_argument('--resume', action='store_true', help='Reuse verified completed features and groups; no epoch-level resume')
    a = p.parse_args(argv)
    for name in ('num_threads','feature_batch_size','nn_epochs','nn_patience','lgbm_rounds','lgbm_patience'):
        if getattr(a,name) < 1:
            p.error(f'--{name.replace("_","-")} must be positive')
    if not 0 <= a.seed <= 2**31-1:
        p.error('--seed must be in 0..2147483647')
    return a


def next_attempt(parent):
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while (parent/f'attempt_{number:03d}').exists():
        number += 1
    return parent/f'attempt_{number:03d}'


def verify_group(output, group, record, seed):
    folder = (output/record['folder']).resolve(strict=True)
    # Do not silently load a folder outside this new run on resume.
    if not folder.is_relative_to(output):
        raise ValueError('Completed classifier path escaped the results folder')
    source.verify(folder/'model_config.json',record['config'],f'{group} config')
    source.verify(folder/'selection_lock.json',record['lock'],f'{group} selection lock')
    config, lock = read_json(folder/'model_config.json'), read_json(folder/'selection_lock.json')
    if config['experiment'] != group or lock['experiment'] != group or lock['seed'] != seed:
        raise ValueError('Saved classifier identity/seed mismatch')
    source.verify(folder/'model_config.json',lock['config'],f'{group} locked config')
    source.verify(folder/Path(lock['checkpoint']['path']).name,lock['checkpoint'],f'{group} checkpoint')
    source.verify(folder/'val_probability.npy',lock['val_probability'],f'{group} locked validation probabilities')
    return folder, config, lock


def save_predictions(path, frame, p, threshold):
    backend.check_probabilities(p, len(frame))
    pd.DataFrame(dict(record=np.arange(1,len(frame)+1), SMILES=frame.SMILES, y_true=frame.label,
                      probability=p, locked_threshold=threshold, prediction=(p>threshold).astype(int),
                      prediction_fixed_0p5=(p>.5).astype(int),
                      prediction_fixed_0p7=(p>.7).astype(int))).to_csv(path,index=False)


def export_tables(output, probabilities, arrays, locks, reference_report, reference, frames):
    tables = {}
    policies = ('validation_selected','fixed_0.5','fixed_0.7')
    for split in ('val','test'):
        policies_rows = {}
        for policy in policies:
            rows = []
            for group in GROUPS:
                threshold = locks[group]['threshold'] if policy == 'validation_selected' else float(policy.split('_')[1])
                row = dict(experiment=group, split=split, threshold_policy=policy,
                           **classification_metrics(arrays[split]['labels'],probabilities[group][split],threshold))
                rows.append(row)
            columns = ['experiment','split','threshold_policy','threshold',*METRICS,'n','TN','FP','FN','TP']
            policies_rows[policy] = pd.DataFrame(rows)[columns]
        tables[split] = policies_rows
        policies_rows['validation_selected'].to_csv(output/f'{split}_comparison.csv',index=False)
        policies_rows['fixed_0.5'].to_csv(output/f'{split}_fixed_threshold05.csv',index=False)
        policies_rows['fixed_0.7'].to_csv(output/f'{split}_fixed_threshold07.csv',index=False)
        pd.concat(policies_rows.values(),ignore_index=True).to_csv(output/f'{split}_all_thresholds.csv',index=False)
    pd.concat([tables[s]['validation_selected'] for s in ('val','test')],ignore_index=True).to_csv(output/'comparison.csv',index=False)
    contrasts = [
        ('mechanism_gain_neural','nn_mechanism','nn_baseline'),
        ('mechanism_gain_lightgbm','lightgbm_mechanism','lightgbm_baseline'),
        ('classifier_change_without_mechanisms','lightgbm_baseline','nn_baseline'),
        ('classifier_change_with_mechanisms','lightgbm_mechanism','nn_mechanism')]
    delta = []
    for split in ('val','test'):
        for policy in policies:
            tab = tables[split][policy].set_index('experiment')
            for name, a, b in contrasts:
                delta.append(dict(contrast=name, split=split, threshold_policy=policy, model_a=a, model_b=b,
                                  **{f'delta_{m}':float(tab.loc[a,m]-tab.loc[b,m]) for m in METRICS}))
    pd.DataFrame(delta).to_csv(output/'contrast_summary.csv',index=False)
    # Separate reference: the original end-to-end TITAN head is NOT one of the
    # newly trained, matched frozen-feature heads above.
    refs = []
    original_threshold = float(reference['lock']['threshold'])
    for split in ('val','test'):
        path = Path(reference_report['files'][split]['original_probability']['path'])
        source.verify(path,reference_report['files'][split]['original_probability'],f'{split} baseline reference')
        p = backend.check_probabilities(np.load(path,allow_pickle=False),len(frames[split]))
        refs.append(dict(experiment='original_end_to_end_baseline_reference_only',split=split,
                         **classification_metrics(arrays[split]['labels'],p,original_threshold)))
    pd.DataFrame(refs).to_csv(output/'original_baseline_reference_only.csv',index=False)
    with (output/'comparison_table.md').open('w',encoding='utf-8') as f:
        f.write('# Frozen TITAN classifier comparison\n\nSame frozen baseline vectors; seed='+str(reference['config']['seed'])+'.\n\n')
        for split in ('val','test'):
            for policy in policies:
                f.write(f'## {split}: {policy}\n\n')
                cols = ['experiment','threshold',*METRICS]
                f.write('| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n')
                for _,row in tables[split][policy].iterrows():
                    f.write('| '+' | '.join(str(row[c]) if c=='experiment' else f'{row[c]:.6f}' for c in cols)+' |\n')
                f.write('\n')
    print('\nTEST; each threshold selected on validation:',flush=True)
    print(tables['test']['validation_selected'][['experiment','threshold',*METRICS]].to_string(index=False,float_format=lambda x:f'{x:.6f}'))


def run(args):
    root = args.project_root.resolve(strict=True)
    mechanism = (args.mechanism_dir or root/'mechanism_features').resolve(strict=True)
    output = (args.output_dir or root/f'titan_classifier_comparison_seed{args.seed}_results').resolve()
    reference = source.locate_reference(root,args.baseline_run,args.seed)
    original, frames, fingerprints = source.load_inputs(root,mechanism,reference)
    print(f'Reference baseline: {reference["folder"]}',flush=True)
    print(f'TITAN labels unchanged; split sizes: '+str({s:len(frames[s]) for s in SPLITS}),flush=True)
    print('Saved baseline modality scalers reused; no fit on val/test; mechanism order=efflux,influx,pampa.',flush=True)
    if args.check_only:
        print('CHECK PASSED: hashes, original labels, splits and mechanism row alignment.\n'
              'No training or files written; numerical baseline reproduction is checked during the real run.')
        return
    device, versions = backend.load_backends(args.device,args.num_threads)
    # Record software and this exact implementation as well as inputs/model.
    versions.update(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                    sklearn=sklearn.__version__,joblib=joblib.__version__)
    definition = dict(seed=args.seed,groups=list(GROUPS), reference=reference['fingerprints'], inputs=fingerprints,
                      sources=[file_record(p) for p in sorted(HERE.glob('*.py'))],versions=versions,
                      device=str(device),num_threads=args.num_threads, feature_batch_size=args.feature_batch_size,
                      nn_epochs=args.nn_epochs,nn_patience=args.nn_patience,
                      lgbm_rounds=args.lgbm_rounds,lgbm_patience=args.lgbm_patience,
                      encoder='original baseline frozen, eval mode; classifier input captured before first Linear',
                      additional_scaling='none', mechanisms='raw fixed scores, appended after attention only',
                      selection='validation weighted BCE for epoch/round; validation BACC for threshold',
                      pretrained_reference_used_TITAN_train_labels=True,
                      test_policy='all four selections locked before any test metrics',
                      search='one predefined configuration per classifier, two feature settings')
    signature = hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
    protected = [root/'dataset',root/'features',root/'src',mechanism,reference['run']]
    if output == root or any(p == output or p.is_relative_to(output) or output.is_relative_to(p) for p in protected):
        raise ValueError('Choose a new results folder outside datasets, sources, mechanisms and the reference run')
    manifest_path = output/'run_manifest.json'
    if output.exists():
        if not args.resume:
            raise FileExistsError(f'{output} already exists. Use --resume or a new --output-dir; nothing was overwritten')
        manifest = read_json(manifest_path)
        if manifest['signature'] != signature:
            raise ValueError('Resume inputs, code, settings, versions or reference changed. Choose a new --output-dir')
    else:
        output.mkdir(parents=True)
        manifest = dict(status='running',signature=signature,definition=definition,preparation=None,fitted={})
        write_json(manifest_path,manifest)
    try:
        if manifest.get('preparation'):
            prep = manifest['preparation']
            folder = output/prep['folder']
            source.verify(folder/'preparation.json',prep['report'],'frozen feature preparation report')
            report = read_json(folder/'preparation.json')
            if report['status'] != 'complete':
                raise ValueError('Incomplete frozen features')
        else:
            folder = next_attempt(output/'features')
            report = source.extract_features(reference,original,frames,folder,device,args.feature_batch_size)
            manifest['preparation'] = dict(folder=str(folder.relative_to(output)),report=file_record(folder/'preparation.json'))
            write_json(manifest_path,manifest)
        arrays = source.read_cached_features(folder,report,original)
        del original
        fitted = {}
        for group in GROUPS:
            if group in manifest['fitted']:
                fitted[group] = verify_group(output,group,manifest['fitted'][group],args.seed)
                print(f'[{group}] reuse verified completed model',flush=True)
            else:
                model_folder = next_attempt(output/'models'/group)
                config, lock = backend.fit_group(group,args,{s:arrays[s] for s in ('train','val')},model_folder,device)
                manifest['fitted'][group] = dict(folder=str(model_folder.relative_to(output)),
                    config=file_record(model_folder/'model_config.json'),lock=file_record(model_folder/'selection_lock.json'))
                write_json(manifest_path,manifest)
                fitted[group] = verify_group(output,group,manifest['fitted'][group],args.seed)
        write_json(output/'all_selections_locked.json',dict(signature=signature,groups=manifest['fitted'],
                   note='Written after all classifier and threshold selection, before any test metrics'))
        probabilities, locks = {}, {}
        for group in GROUPS:
            model_folder, config, lock = fitted[group]
            locks[group] = lock
            val = backend.check_probabilities(np.load(model_folder/'val_probability.npy',allow_pickle=False),len(frames['val']))
            test = backend.predict_group(group,model_folder,config,lock,arrays['test'],device)
            probabilities[group] = dict(val=val,test=test)
            for split in ('val','test'):
                save_predictions(model_folder/f'{split}_predictions.csv',frames[split],probabilities[group][split],lock['threshold'])
                np.save(model_folder/f'{split}-classification.npy',probabilities[group][split])
        export_tables(output,probabilities,arrays,locks,report,reference,frames)
        manifest['status'] = 'complete'
        manifest.pop('error',None)
        manifest['result_files'] = [file_record(p) for p in sorted(output.glob('*.csv'))]
        write_json(manifest_path,manifest)
        print(f'\nDONE: {output}/test_comparison.csv',flush=True)
        print('Original inputs/checkpoints unchanged. Completed models can be reused with --resume.',flush=True)
    except BaseException as error:
        manifest['status'] = 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        write_json(manifest_path,manifest)
        raise


if __name__ == '__main__':
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print('\nInterrupted. Use the identical command plus --resume to reuse completed groups.',file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
