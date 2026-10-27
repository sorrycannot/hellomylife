"""Sequential, resumable, validation-only Bayesian sample-weight search."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import platform
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
RULES_PATH = HERE/'reference_rules.json'
DEFAULT_SOURCE = 'titan_structure_weighting_seed42_results/attempt_003'


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--project-root',default='..')
    p.add_argument('--source-run',help='original run containing input_audit.json, reference_audit.json and saved UNWEIGHTED reference validation probabilities')
    p.add_argument('--output',default='titan_priority_bayes_seed42_results')
    p.add_argument('--resume',help='existing attempt directory, relative to project-root or absolute')
    p.add_argument('--stage',choices=['inspect','all'],default='all')
    p.add_argument('--budget',type=int,default=200,help='TOTAL unique configurations, including the four anchors; not the number of extra trials on resume')
    p.add_argument('--startup',type=int,default=None,help='initial total including four anchors; default 40 (4 fixed + 36 random)')
    p.add_argument('--threads',type=int,default=None,help='default 4; must remain unchanged on resume')
    p.add_argument('--seed',type=int,choices=[42],default=42)
    p.add_argument('--debug',action='store_true')
    return p


def check_runtime():
    result={'python':platform.python_version()}
    for name in ('numpy','pandas','scipy','sklearn','lightgbm','rdkit','joblib','openpyxl'):
        try:m=importlib.import_module(name)
        except ImportError as exc:
            raise RuntimeError(f'Missing {name}. Activate conda titan_bbb. This script needs no optuna; install only missing dependencies from requirements.txt.') from exc
        result[name]=getattr(m,'__version__','unknown')
    return result


def make_attempt(base):
    base.mkdir(parents=True,exist_ok=True)
    for k in range(1,10000):
        p=base/f'attempt_bayes{k:03d}'
        try:p.mkdir();return p
        except FileExistsError:pass
    raise RuntimeError('No available attempt directory')


@contextmanager
def run_lock(out):
    # Kernel releases this lock even after an interrupted/killed process.
    import fcntl
    with (out/'run.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another process is using this attempt. Do not launch concurrent searches into one directory.') from exc
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def atomic_npy(path,array):
    import numpy as np
    tmp=Path(path).with_name(Path(path).name+'.tmp')
    with tmp.open('wb') as f:np.save(f,array,allow_pickle=False)
    os.replace(tmp,path)


def read_records(out,signature,yv):
    import numpy as np
    from .protocol import read_json,sha,metrics,pair_weights,choose_threshold
    records=[]
    for folder in sorted((out/'trials').glob('trial_*')) if (out/'trials').is_dir() else []:
        if not (folder/'COMPLETE.json').is_file():continue
        r=read_json(folder/'COMPLETE.json')
        if r['run_signature']!=signature:raise ValueError('Stored trial belongs to a different locked run')
        if r['trial_number']!=len(records)+1 or folder.name!=r['trial_id']:
            raise ValueError('Stored trial registry is not contiguous')
        pair_weights(r['indices'])
        for name,digest in r['files_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError(f'Stored trial artifact changed: {folder/name}')
        p=np.load(folder/'val_probability.npy',allow_pickle=False)
        proposal=read_json(folder/'PROPOSAL.json')
        if proposal['indices']!=r['indices'] or proposal['run_signature']!=signature:
            raise ValueError('Stored proposal and completed trial disagree')
        if r['validation_metrics']['fixed_0p5']['threshold']!=.5 or r['validation_metrics']['validation_selected']['threshold']!=choose_threshold(yv,p):
            raise ValueError('Stored threshold no longer follows the locked selection rule')
        for policy,m in r['validation_metrics'].items():
            got=metrics(yv,p,m['threshold'])
            for key,value in got.items():
                old=m[key]
                if value is None:
                    if old is not None:raise ValueError('Stored validation metric changed')
                elif old is None or abs(float(old)-float(value))>1e-12:
                    raise ValueError(f'Stored validation metric mismatch: {r["trial_id"]}/{policy}/{key}')
        records.append(r)
    pairs=[tuple(r['indices']) for r in records]
    if len(pairs)!=len(set(pairs)):raise ValueError('Duplicate completed weight combinations')
    return records


# CONTROL_GATE_V2: numerical consistency, not bitwise reproduction.
def control_reproduction_check(y, probability, reference_probability,
                               refit_rounds, reference_rounds):
    import numpy as np
    from .protocol import metrics, THRESHOLDS
    p = np.asarray(probability, dtype=float)
    ref = np.asarray(reference_probability, dtype=float)
    truth = np.asarray(y, dtype=int)
    # metrics() also rejects nonfinite, misaligned and non-probability inputs.
    current = metrics(truth, p, .5)
    previous = metrics(truth, ref, .5)
    if current['AUROC'] is None or previous['AUROC'] is None:
        raise ValueError('Control comparison requires both validation classes')
    tolerance = 1e-7
    difference = float(np.max(np.abs(p - ref)))
    thresholds = {f'{t:.1f}': bool(np.array_equal(p > t, ref > t))
                  for t in THRESHOLDS}
    positive, negative = truth == 1, truth == 0
    # Preserve every positive-negative ranking relation, including ties.
    # Equal aggregate AUROC alone could hide cancelling rank reversals.
    current_order = np.sign(p[positive, None] - p[negative][None, :])
    reference_order = np.sign(ref[positive, None] - ref[negative][None, :])
    changed_pairs = int(np.count_nonzero(current_order != reference_order))
    checks = dict(
        probability_within_absolute_tolerance=bool(np.allclose(p, ref, rtol=0, atol=tolerance)),
        best_iteration_identical=bool(int(refit_rounds) == int(reference_rounds)
                                     and int(reference_rounds) > 0),
        AUROC_identical=bool(abs(current['AUROC'] - previous['AUROC']) <= 1e-12),
        AUPRC_AP_identical=bool(abs(current['AUPRC_AP'] - previous['AUPRC_AP']) <= 1e-12),
        positive_negative_ranks_identical=changed_pairs == 0,
        all_candidate_threshold_decisions_identical=all(thresholds.values()))
    return dict(
        policy='CONTROL_GATE_V2; numerical consistency, not bitwise reproduction',
        tolerance=tolerance, relative_tolerance=0, previous_tolerance=1e-10,
        max_absolute_probability_difference=difference,
        strict_1e_10_passed=bool(np.allclose(p, ref, rtol=0, atol=1e-10)),
        fixed05_decisions_identical=thresholds['0.5'],
        candidate_threshold_decisions_identical=thresholds,
        n_changed_positive_negative_pairs=changed_pairs,
        reference_rounds=int(reference_rounds), refit_rounds=int(refit_rounds),
        reference_metrics=previous, refit_metrics=current,
        checks=checks, failed_checks=[name for name, ok in checks.items() if not ok],
        passed=all(checks.values()),
        interpretation='Passing establishes numerical and decision/ranking consistency on the locked validation rows; it is not proof of identical training trajectories.')


def trial(out,number,pair,proposal,signature,X,frames,audits,params,max_rounds,patience,spw,reference_p):
    import numpy as np
    from .protocol import build_weights,pair_weights,write_json,read_json,sha,metrics,choose_threshold
    from .training import fit_head
    folder=out/'trials'/f'trial_{number:04d}';folder.mkdir(parents=True,exist_ok=True)
    pending=dict(run_signature=signature,trial_number=number,indices=list(pair),proposal=proposal)
    if (folder/'PROPOSAL.json').exists():
        previous=read_json(folder/'PROPOSAL.json')
        if previous['run_signature']!=signature or previous['indices']!=list(pair):
            raise ValueError('Unfinished trial conflicts with resumed proposal')
    else:write_json(folder/'PROPOSAL.json',pending)
    w=build_weights(audits['train'].V01,audits['train'].V03,pair)
    yt=frames['train'].label.to_numpy();yv=frames['val'].label.to_numpy()
    w1,w3=pair_weights(pair)
    print(f'[{number:03d}] V01={w1:.2f}, V03={w3:.2f}; {proposal["method"]}; fitting...',flush=True)
    started=time.monotonic()
    model,history=fit_head(X['train'],yt,params,max_rounds,X['val'],yv,
        weights=None if pair==(0,0) else w,patience=patience,scale_pos_weight=spw,verbose=False)
    p=np.asarray(model.predict(X['val']),dtype=float)
    seconds=time.monotonic()-started
    if pair==(0,0):
        from .training import load_reference
        original = load_reference(read_json(out/'input_audit.json')['original_model_path'])
        refit_rounds = int(model.best_iteration if model.best_iteration > 0
                           else model.current_iteration())
        check = control_reproduction_check(yv, p, reference_p, refit_rounds,
                                           original.current_iteration())
        write_json(out/'control_reproduction_check.json', check)
        print(f'[control] max probability difference={check["max_absolute_probability_difference"]:.3g}; '
              f'atol={check["tolerance"]:.1g}; ranks/thresholds/rounds/metrics passed={check["passed"]}',
              flush=True)
        if not check['passed']:
            # Preserve the failed refit for diagnosis, without committing a trial.
            model.save_model(str(folder/'FAILED_CONTROL_MODEL.txt'))
            atomic_npy(folder/'FAILED_CONTROL_val_probability.npy', p)
            write_json(folder/'FAILED_CONTROL_fit_history.json', history)
            raise ValueError('Unweighted control consistency check failed: '
                             + ', '.join(check['failed_checks'])
                             + '. See control_reproduction_check.json. No weighted fit was started.')
    auto=choose_threshold(yv,p)
    validation_metrics={'fixed_0p5':metrics(yv,p,.5),'validation_selected':metrics(yv,p,auto)}
    model.save_model(str(folder/'model.tmp.txt'));os.replace(folder/'model.tmp.txt',folder/'model.txt')
    atomic_npy(folder/'val_probability.npy',p)
    write_json(folder/'fit_history.json',history)
    record=dict(**pending,trial_id=folder.name,fit_seconds=seconds,
        best_iteration=int(model.best_iteration if model.best_iteration>0 else model.current_iteration()),
        validation_metrics=validation_metrics,
        weight_summary=dict(n_upweighted=int((w>1).sum()),min=float(w.min()),max=float(w.max()),
            positive_sum=float(w[yt==1].sum()),negative_sum=float(w[yt==0].sum()),
            effective_positive_sum_with_class_weight=float(w[yt==1].sum()*spw),
            effective_negative_sum_with_class_weight=float(w[yt==0].sum())),
        files_sha256={n:sha(folder/n) for n in ('PROPOSAL.json','model.txt','val_probability.npy','fit_history.json')})
    # The commit is last. An incomplete trial is safely rerun, not counted twice.
    write_json(folder/'COMPLETE.json',record)
    print(f'[{number:03d}] val AUROC={validation_metrics["fixed_0p5"]["AUROC"]:.9f}; best_iteration={record["best_iteration"]}; seconds={seconds:.1f}',flush=True)
    return record


def finalize(out,records,frames,audits,reference_p,coverage,reference_metrics,spw,signature,budget):
    import numpy as np
    import pandas as pd
    from .protocol import choose_best,build_weights,pair_weights,write_json,sha
    from . import report
    best=choose_best(records);folder=out/'trials'/best['trial_id']
    best_p=np.load(folder/'val_probability.npy',allow_pickle=False)
    selected=out/'selected_model';selected.mkdir(exist_ok=True)
    shutil.copyfile(folder/'model.txt',selected/'best_model.txt')
    write_json(selected/'selection_lock.json',dict(run_signature=signature,
        selected_trial=best['trial_id'],indices=best['indices'],weights=dict(zip(('V01','V03'),pair_weights(best['indices']))),
        selection='maximum validation AUROC only; exact ties: lower max weight, then sum, then V01 and V03',
        completed_unique_configurations=len(records),budget=budget,
        model_sha256=sha(selected/'best_model.txt'),thresholds={k:m['threshold'] for k,m in best['validation_metrics'].items()},
        decision='probability > threshold',threshold_selection='validation BACC, first maximum in .1,.3,.5,.7,.9',
        no_test_dataset_or_predictions_loaded=True,no_upstream_model_refitted=True,
        scope='Development selection; validation supplied structural hypotheses and is not an independent confirmation set.'))
    write_json(out/'best_trial.json',best)
    archive=out/'selection_history';archive.mkdir(exist_ok=True)
    write_json(archive/f'best_after_{len(records):04d}_unique.json',best)
    weight=build_weights(audits['train'].V01,audits['train'].V03,best['indices'])
    wa=pd.concat([frames['train'].reset_index(drop=True),audits['train'].reset_index(drop=True)],axis=1)
    wa['sample_weight']=weight
    wa['effective_loss_weight']=weight*np.where(frames['train'].label.to_numpy()==1,spw,1.)
    report.csv(out/'selected_train_weights.csv',wa)
    report.csv(out/'selected_upweighted_train_rows.csv',wa[wa.sample_weight>1])
    summary=report.export_model(out,'reference_saved_unweighted',frames['val'],reference_p,audits['val'])
    named={tuple(r['indices']):r for r in records}
    for pair,name in [((0,0),'control_retrained_unweighted'),((10,0),'V01_fixed1p5'),((0,10),'V03_fixed1p5'),((10,10),'V01_V03_fixed1p5')]:
        if pair in named:
            p=np.load(out/'trials'/named[pair]['trial_id']/'val_probability.npy',allow_pickle=False)
            summary+=report.export_model(out,name,frames['val'],p,audits['val'])
    summary+=report.export_model(out,'bayes_selected',frames['val'],best_p,audits['val'])
    comparison=pd.DataFrame(summary);report.csv(out/'validation_comparison.csv',comparison)
    changes=report.paired(frames['val'],reference_p,best_p,audits['val'])
    report.csv(out/'validation_paired_changes_fixed05.csv',changes)
    report.csv(out/'validation_corrected_errors_fixed05.csv',changes[changes.corrected_error])
    report.csv(out/'validation_new_errors_fixed05.csv',changes[changes.new_error])
    transitions=changes.groupby(['reference_outcome','best_outcome','transition']).size().rename('n').reset_index()
    report.csv(out/'validation_transition_counts_fixed05.csv',transitions)
    sub=pd.DataFrame(report.subgroup(frames['val'],audits['val'],reference_p,'reference_saved_unweighted')+
        report.subgroup(frames['val'],audits['val'],best_p,'bayes_selected'))
    report.csv(out/'validation_subgroups_primary_fixed05.csv',sub)
    report.update(out,records,coverage,reference_metrics,
        {'selected_comparison':comparison,'subgroups_primary':sub,'error_transitions':transitions})
    write_json(out/'COMPLETE.json',dict(run_signature=signature,completed_unique_configurations=len(records),
        requested_budget=budget,selected_trial=best['trial_id'],selection_lock_sha256=sha(selected/'selection_lock.json'),
        excel_sha256=sha(out/'excel/priority_weight_search.xlsx'),scope='train/validation development only'))
    print('\nVALIDATION DEVELOPMENT RESULTS; no test model selection:',flush=True)
    cols=['experiment','threshold_policy','threshold','AUROC','AUPRC_AP','ACC','BACC','SN','SP','F1','MCC','FN','FP']
    print(comparison[cols].to_string(index=False,float_format=lambda x:f'{x:.6f}'),flush=True)
    print(f'BEST: {best["trial_id"]}; weights={pair_weights(best["indices"])}; val AUROC={best["validation_metrics"]["fixed_0p5"]["AUROC"]:.9f}',flush=True)
    print(f'DONE: {out}\nExcel: {out/"excel/priority_weight_search.xlsx"}',flush=True)


def run(args):
    versions=check_runtime()
    import numpy as np
    import pandas as pd
    from . import inputs,chemistry,report
    from .protocol import read_json,write_json,object_sha,sha,metrics,next_pair,choose_best
    from .training import common_params
    root=Path(args.project_root).expanduser().resolve()
    if args.resume:
        out=inputs.resolve(args.resume,root)
        previous=read_json(out/'protocol_lock.json')
        old_settings=previous['settings']
        source=inputs.resolve(args.source_run or old_settings['source_run'],root)
        startup=args.startup if args.startup is not None else old_settings['startup']
        threads=args.threads if args.threads is not None else old_settings['threads']
    else:
        source=inputs.resolve(args.source_run or DEFAULT_SOURCE,root)
        startup=args.startup if args.startup is not None else 40
        threads=args.threads if args.threads is not None else 4
        out=make_attempt(inputs.resolve(args.output,root));previous=None
    print('Output:',out,flush=True)
    if not 4<=args.budget<=1681 or not 4<=startup<=1681 or threads<1:
        raise ValueError('budget and startup must be in 4..1681; threads must be positive')
    with run_lock(out):
        try:
            write_json(out/'last_invocation.json',dict(arguments=vars(args),versions=versions,UTC=datetime.now(timezone.utc).isoformat()))
            reference=read_json(RULES_PATH)
            frames,X,model,reference_p,input_audit,source_ref=inputs.load(root,source)
            expected=np.asarray(reference['baseline_validation']['positive_probability'],dtype=float)
            if not np.allclose(reference_p,expected,rtol=0,atol=1e-10):
                raise ValueError('Reference probabilities do not reproduce the exact unweighted model used to discover V01/V03')
            params,removed,spw=common_params(model,args.seed,threads)
            # Recover the same explicit protocol, never infer training budget
            # from the saved checkpoint's truncated best iteration.
            if 'refit_max_rounds' not in source_ref or 'refit_patience' not in source_ref:
                raise ValueError('Source reference audit must record refit_max_rounds and refit_patience')
            max_rounds,patience=int(source_ref['refit_max_rounds']),int(source_ref['refit_patience'])
            if max_rounds<1 or patience<1:raise ValueError('Invalid locked round budget or patience')
            input_audit['original_validation_pointwise_reproduction_passed']=True
            preflight=out/'resume_preflight' if previous else out
            preflight.mkdir(exist_ok=True)
            write_json(preflight/'input_audit.json',input_audit)
            audits,chem_audit=chemistry.prepare(reference,frames)
            write_json(preflight/'structure_audit.json',chem_audit)
            for split,a in audits.items():
                report.csv(preflight/f'{split}_structure_matches.csv',pd.concat([frames[split].reset_index(drop=True),a],axis=1))
            report.csv(preflight/'train_val_structure_coverage.csv',pd.DataFrame(chem_audit['coverage']))
            if not chem_audit['training_coverage_matches_reported_audit']:
                raise ValueError('V01/V03 training coverage differs from the reported 1142/530-unit audit. See structure_audit.json; no weighted model was fitted.')
            print('Validation matching calibration passed for all 17 original rules.',flush=True)
            for c in chem_audit['coverage']:print(f'[{c["split"]}/{c["candidate"]}] units={c["units"]}; positive={c["positive"]}; negative={c["negative"]}; connectivity_groups={c["connectivity_groups"]}',flush=True)
            settings=dict(project_root=str(root),source_run=str(source),seed=args.seed,startup=startup,threads=threads,
                max_rounds=max_rounds,patience=patience,EI_xi=.0001,weight_range=[1,3],step=.05,
                surrogate='GaussianProcessRegressor Constant*Matern(nu2.5)+WhiteKernel; normalize_y; one restart',
                acquisition='stable log expected improvement; search all UNTRIED grid coordinates; zero-variance fallback max posterior sd',
                active_rules=['V01','V03'],weight_overlap='maximum; no multiplication; both true label classes weighted',
                normalization='none; preserve original scale_pos_weight',selection='validation AUROC only',
                tie_break='lower maximum weight, then sum, then V01, then V03',source_common_params=params,
                control_gate='CONTROL_GATE_V2; atol=1e-7, rtol=0; same rounds, AUROC/AP, all positive-negative ranks and .1/.3/.5/.7/.9 decisions')
            code_hashes={p.name:sha(p) for p in sorted(HERE.glob('*.py'))}
            payload=dict(settings=settings,versions=versions,input_paths_sha256=input_audit['input_paths_sha256'],
                rules_sha256=sha(RULES_PATH),code_sha256=code_hashes)
            signature=object_sha(payload)
            lock=dict(run_signature=signature,**payload,initial_requested_budget=args.budget,
                data_usage='No test dataset, test feature array or test predictions are loaded. This validation set already supplied structural hypotheses.',
                class_weight=spw,parameter_aliases_removed=removed)
            if previous:
                if previous['run_signature']!=signature:
                    raise ValueError('Resume signature mismatch: input bytes, rule definitions, settings, software versions or code changed. Start a fresh attempt; no existing trial was reused.')
                for name in ('input_audit.json','structure_audit.json','train_structure_matches.csv','val_structure_matches.csv','train_val_structure_coverage.csv'):
                    shutil.copyfile(preflight/name,out/name)
            else:write_json(out/'protocol_lock.json',lock)
            reference_metrics=[dict(experiment='reference_saved_unweighted',split='val',threshold_policy='fixed_0p5',**metrics(frames['val'].label.to_numpy(),reference_p,.5))]
            records=read_records(out,signature,frames['val'].label.to_numpy())
            if len(records)>args.budget:raise ValueError('--budget is total configurations and cannot be below the already completed count')
            if args.stage=='inspect':
                report.workbook(out/'excel/input_inspection.xlsx',{'coverage':pd.DataFrame(chem_audit['coverage']),'reference_validation':pd.DataFrame(reference_metrics)})
                print('INSPECT COMPLETE: calibrated rules and pinned original train/val inputs; no new model fitted.',flush=True)
                return out
            budget_history=read_json(out/'budget_history.json') if (out/'budget_history.json').exists() else []
            budget_history.append(dict(requested_total=args.budget,previous_completed=len(records),UTC=datetime.now(timezone.utc).isoformat()))
            write_json(out/'budget_history.json',budget_history)
            print(f'Unique budget={args.budget}; initial plan={startup} (4 anchors + {startup-4} random); seed=42; AUROC selection; scale_pos_weight={spw:.9g}; early stopping={patience}; max_rounds={max_rounds}',flush=True)
            while len(records)<args.budget:
                pair,proposal=next_pair(records,args.seed,startup,settings['EI_xi'])
                r=trial(out,len(records)+1,pair,proposal,signature,X,frames,audits,params,max_rounds,patience,spw,reference_p)
                records.append(r)
                # CSV + SVG checkpoint each trial; Excel every 10 and at the end.
                report.csv(out/'search_history.csv',report.flat_trials(records))
                report.csv(out/'search_progress.csv',report.progress(records))
                report.progress_svg(out/'search_progress.svg',records)
                if len(records)%10==0:
                    report.update(out,records,chem_audit['coverage'],reference_metrics)
            for path,digest in input_audit['input_paths_sha256'].items():
                if sha(path)!=digest:raise ValueError(f'Source file changed while the search was running: {path}')
            finalize(out,records,frames,audits,reference_p,chem_audit['coverage'],reference_metrics,spw,signature,args.budget)
            return out
        except BaseException as exc:
            write_json(out/'INTERRUPTED.json' if isinstance(exc,KeyboardInterrupt) else out/'FAILED.json',
                dict(type=type(exc).__name__,error=str(exc),UTC=datetime.now(timezone.utc).isoformat(),
                     resume='Use --resume with this attempt; committed configurations are verified and not retrained.'))
            raise


def main():
    args=parser().parse_args()
    try:run(args)
    except KeyboardInterrupt:
        print('\nINTERRUPTED: committed configurations are saved. Resume the same attempt.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        if args.debug:raise
        print(f'\nSTOPPED: {exc}',file=sys.stderr)
        raise SystemExit(1)
