"""Read-only checks of the original TITAN baseline, splits and cached features."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .common import SPLITS, SCORE_COLUMNS, file_record, read_json, sha256, validate_scores, write_json

COUNTS = {'train': 7593, 'val': 727, 'test': 942}
DIMS = {'tabular': 384, 'image': 2048, 'text': 768}


def verify(path, record, label):
    path = Path(path).resolve(strict=True)
    if sha256(path) != record['sha256']:
        raise ValueError(f'{label} changed: {path}')
    return path


def locate_reference(root, requested, seed):
    if requested:
        run = requested.resolve(strict=True)
    else:
        choices = [root/f'titan_attention_ablation_seed{seed}_results',
                   root/f'titan_attention_seed{seed}_results', root/'titan_attention_seed2022_results',
                   root/'titan_mechanism_2051_results']
        run = next((p for p in choices if (p/'run_manifest.json').is_file()), None)
        if run is None:
            raise FileNotFoundError('No completed TITAN baseline run found. Supply --baseline-run /path/to/results')
    meta = read_json(run/'run_manifest.json')
    if meta.get('status') != 'complete':
        raise ValueError(f'Baseline reference run is not complete: {run}')
    definition = meta.get('definition', {})
    reference_seed = definition.get('seed', meta.get('seed'))
    if reference_seed != seed:
        raise ValueError(f'Baseline checkpoint seed={reference_seed}, requested seed={seed}; use a matching --baseline-run')
    if 'baseline' in meta.get('fitted', {}):
        rec = meta['fitted']['baseline']
        folder = (run/rec['folder']).resolve(strict=True)
        verify(folder/'model_config.json', rec['config'], 'baseline configuration')
        verify(folder/'selection_lock.json', rec['lock'], 'baseline selection lock')
    else:
        folder = run/f'baseline_seed{seed}'
    config = read_json(folder/'model_config.json')
    lock = read_json(folder/'selection_lock.json')
    if (config.get('mode') != 'baseline' or config.get('seed') != seed
            or config.get('classifier_input_dim') != 2048 or config.get('pooled_dim') != 2048):
        raise ValueError('A genuine 2048-D original baseline checkpoint is required, not a mechanism model')
    if config.get('attention_conditioned_on_mechanisms', False):
        raise ValueError('Mechanism-conditioned checkpoints cannot supply the shared baseline representation')
    checkpoint = verify(folder/'best_model.pth', lock['checkpoint'], 'baseline checkpoint')
    provenance = definition.get('data', meta.get('data'))
    if not provenance or 'inputs' not in provenance:
        raise ValueError('Reference run lacks original input fingerprints')
    source_records = definition.get('sources', [])
    source = next((r for r in source_records if str(r['path']).replace('\\','/').endswith('/src/utils/model.py')),
                  meta.get('original_TITAN_model'))
    if not source:
        raise ValueError('Reference run lacks the original TITAN model source hash')
    model_source = verify(root/'src/utils/model.py', source, 'original TITAN model source')
    scaler_path = run/'titan_modality_scalers.joblib'
    if 'scalers' in meta:
        verify(scaler_path, meta['scalers'], 'baseline scalers')
    scalers = joblib.load(scaler_path)
    if set(scalers) != set(DIMS):
        raise ValueError('Expected the saved tabular/image/text scalers from the baseline run')
    return dict(run=run, folder=folder, config=config, lock=lock, checkpoint=checkpoint,
                provenance=provenance, scalers=scalers, scaler_path=scaler_path, model_source=model_source,
                fingerprints=dict(run_manifest=file_record(run/'run_manifest.json'),
                    config=file_record(folder/'model_config.json'), lock=file_record(folder/'selection_lock.json'),
                    checkpoint=file_record(checkpoint), scalers=file_record(scaler_path),
                    model_source=file_record(model_source),
                    reference_validation_predictions=file_record(folder/'val_predictions.csv')))


def load_inputs(root, mechanism_dir, reference):
    meta = read_json(mechanism_dir/'manifest.json')
    if (meta.get('status') != 'complete' or meta.get('task') != 'classification'
            or meta.get('score_columns') != SCORE_COLUMNS):
        raise ValueError('Need the complete original TITAN classification mechanism_features manifest')
    frames, data, fingerprints = {}, {}, {'mechanism_manifest': file_record(mechanism_dir/'manifest.json'), 'splits': {}}
    for split in SPLITS:
        source = reference['provenance']['inputs'][split]
        path = root/'dataset'/f'{split}_classification.csv'
        verify(path, source['dataset'], f'{split} baseline dataset')
        verify(path, meta['dataset_files'][split], f'{split} score-generation dataset')
        frame = pd.read_csv(path, usecols=['SMILES','label'], dtype={'SMILES': str}, keep_default_na=False,
                            encoding='utf-8-sig')
        if len(frame) != COUNTS[split]:
            raise ValueError(f'Expected original TITAN {split} n={COUNTS[split]}, got {len(frame)}')
        y = pd.to_numeric(frame.label, errors='raise').to_numpy(dtype=np.float32)
        if not np.isfinite(y).all() or set(np.unique(y)) != {0, 1}:
            raise ValueError(f'{split}: unchanged TITAN binary labels and both classes are required')
        score_path = mechanism_dir/f'{split}-mechanism-classification.npy'
        audit_path = mechanism_dir/f'{split}-mechanism-audit.csv'
        verify(score_path, meta['splits'][split]['scores'], f'{split} mechanism scores')
        verify(audit_path, meta['splits'][split]['audit'], f'{split} score row audit')
        verify(score_path, source['mechanism'], f'{split} baseline-run scores')
        audit = pd.read_csv(audit_path, dtype={'original_smiles': str}, keep_default_na=False)
        if (len(audit) != len(frame) or audit.original_smiles.tolist() != frame.SMILES.tolist()
                or not np.array_equal(audit.record.to_numpy(), np.arange(1,len(frame)+1))):
            raise ValueError(f'{split}: mechanism and TITAN row orders differ')
        m = validate_scores(np.load(score_path, allow_pickle=False), len(frame), split).astype(np.float32)
        if not np.allclose(m, audit[SCORE_COLUMNS].to_numpy(dtype=float), rtol=1e-6, atol=1e-7):
            raise ValueError(f'{split}: mechanism column order/values disagree with audit')
        data[split] = dict(labels=y, mechanism=m)
        frames[split] = frame
        fingerprints['splits'][split] = dict(dataset=file_record(path), mechanism=file_record(score_path),
                                             audit=file_record(audit_path), modalities={})
        for modality, d in DIMS.items():
            p = root/'features'/f'{split}-{modality}-classification.npy'
            verify(p, source[modality], f'{split}/{modality} original feature array')
            raw = np.load(p, allow_pickle=False)
            if raw.shape != (len(frame),d) or not np.isfinite(raw).all():
                raise ValueError(f'{split}/{modality}: invalid original feature matrix')
            scaler = reference['scalers'][modality]
            if (int(scaler.n_features_in_) != d
                    or not np.all(np.asarray(scaler.n_samples_seen_) == COUNTS['train'])):
                raise ValueError(f'{modality}: saved scaler dimensions or train-only sample count differ')
            scaled = scaler.transform(raw).astype(np.float32)
            if not np.isfinite(scaled).all():
                raise ValueError(f'{split}/{modality}: nonfinite transformed values')
            data[split][modality] = scaled
            fingerprints['splits'][split]['modalities'][modality] = file_record(p)
    fingerprints['existing_TITAN_identity_overlaps'] = meta.get('existing_TITAN_split_identity_overlaps', {})
    return data, frames, fingerprints


def extract_features(reference, data, frames, output, device, batch_size):
    """Capture the input to the TRUE baseline's first classifier Linear layer."""
    import torch
    from .common import import_file
    original = import_file(reference['model_source'], '_frozen_classifier_original_titan_model')
    c = reference['config']
    model = original.SMILESModel(c['d_tab'],c['d_img'],c['d_txt'],proj_dim=2048,dropout=c['dropout']).to(device)
    try:
        state = torch.load(reference['checkpoint'], map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(reference['checkpoint'], map_location=device)
    if tuple(state['classifier.0.weight'].shape) != (2048,2048):
        raise ValueError('Checkpoint does not have the required original 2048-D head')
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captures = []
    handle = model.classifier[0].register_forward_pre_hook(lambda _, inputs: captures.append(inputs[0].detach()))
    output.mkdir(parents=True, exist_ok=False)
    files, checks = {}, {}
    try:
        for split in SPLITS:
            zparts, hparts = [], []
            for start in range(0,len(frames[split]),batch_size):
                captures.clear()
                tensors = [torch.as_tensor(data[split][m][start:start+batch_size],device=device)
                           for m in DIMS]
                with torch.no_grad():
                    z = model(*tensors)
                if len(captures) != 1 or captures[0].shape[1] != 2048:
                    raise ValueError('Failed to capture the baseline attention-pooled 2048-D vector')
                hparts.append(captures[0].cpu().numpy())
                zparts.append(z.cpu().numpy())
            h, z = np.concatenate(hparts).astype(np.float32), np.concatenate(zparts).reshape(-1)
            if h.shape != (len(frames[split]),2048) or not np.isfinite(h).all() or not np.isfinite(z).all():
                raise ValueError(f'{split}: invalid frozen pooled features')
            p = torch.sigmoid(torch.from_numpy(z)).numpy()
            if split == 'val':
                old = pd.read_csv(reference['folder']/'val_predictions.csv')
                if (len(old) != len(p) or not np.array_equal(old.record,np.arange(1,len(p)+1))
                        or not np.array_equal(old.y_true.to_numpy(),data[split]['labels'])
                        or not np.allclose(old.probability.to_numpy(),p,rtol=1e-4,atol=1e-5)):
                    raise ValueError('Baseline validation predictions do not reproduce: check source, weights and saved scalers')
                checks['val_reference_prediction_max_abs_diff'] = float(np.max(np.abs(old.probability.to_numpy()-p)))
            hp, pp = output/f'{split}-pooled.npy', output/f'{split}-original-baseline-probability.npy'
            np.save(hp,h); np.save(pp,p)
            files[split] = dict(pooled=file_record(hp), original_probability=file_record(pp))
            print(f'[frozen] {split}: {h.shape}; no encoder gradients',flush=True)
    finally:
        handle.remove()
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report = dict(status='complete', files=files, checks=checks, pooled_dim=2048,
                  source_checkpoint=file_record(reference['checkpoint']), frozen=True,
                  mechanism_scores_entered_encoder=False, additional_scaling='none')
    write_json(output/'preparation.json',report)
    return report


def read_cached_features(folder, report, data):
    result = {}
    for split in SPLITS:
        p = verify(folder/f'{split}-pooled.npy', report['files'][split]['pooled'], f'{split} pooled cache')
        verify(folder/f'{split}-original-baseline-probability.npy',
               report['files'][split]['original_probability'], f'{split} original baseline prediction cache')
        h = np.load(p,allow_pickle=False)
        if h.shape != (len(data[split]['labels']),2048) or not np.isfinite(h).all():
            raise ValueError(f'{split}: bad pooled cache')
        result[split] = dict(pooled=h,labels=data[split]['labels'],mechanism=data[split]['mechanism'])
    return result
