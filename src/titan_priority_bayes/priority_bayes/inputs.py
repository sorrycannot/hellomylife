"""Load only the original locked train/validation inputs. No discovery by performance."""
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .protocol import sha, read_json

SPLITS = ('train', 'val')
MECHANISMS = ('efflux', 'influx', 'pampa')


def resolve(value, root):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(root)/p).resolve()


def relocated(value, root, old_root, source):
    p = Path(value).expanduser()
    candidates = []
    if p.is_absolute():
        try:
            candidates.append(Path(root)/p.relative_to(old_root))
        except ValueError:
            pass
        candidates.append(p)
    else:
        candidates += [Path(root)/p, Path(source)/p]
    existing = [p.resolve() for p in candidates if p.is_file()]
    if not existing:
        raise FileNotFoundError(f'Locked input is missing: {value}. Restore the file or update the source audit explicitly; no replacement was guessed.')
    return existing[0]


def check_sha(path, expected, label):
    if not isinstance(expected, str) or not re.fullmatch('[a-fA-F0-9]{64}', expected):
        raise ValueError(f'{label}: the source audit needs a SHA256 record')
    actual = sha(path)
    if actual.lower() != expected.lower():
        raise ValueError(f'{label}: SHA256 mismatch: {path}. No transformation or alternate input was selected.')
    return actual


def array(rec):
    p = Path(rec['path'])
    if p.suffix.lower() == '.npy':
        return np.load(p, allow_pickle=False)
    if p.suffix.lower() == '.npz':
        if not rec.get('key'):
            raise ValueError('Locked NPZ record needs its original key')
        with np.load(p, allow_pickle=False) as z:
            return np.array(z[rec['key']], copy=True)
    raise ValueError(f'Need NPY/NPZ: {p}')


def align_frame(f, rec, smiles):
    rowcol, smcol = rec.get('row_index_column'), rec.get('smiles_column')
    if len(f) != len(smiles):
        raise ValueError('Score table length differs from the original partition')
    if rowcol:
        idx = pd.to_numeric(f[rowcol], errors='raise').to_numpy(dtype=float)
        idx -= int(rec.get('row_index_base', 0))
        if not np.isfinite(idx).all() or not np.equal(idx, np.floor(idx)).all() or sorted(idx.tolist()) != list(range(len(smiles))):
            raise ValueError('Score row indices must identify each original row exactly once')
        f = f.iloc[np.argsort(idx)].reset_index(drop=True)
        mode = 'explicit_original_row_index'
        if smcol and f[smcol].astype(str).tolist() != smiles:
            raise ValueError('Score SMILES contradict original row indices')
    elif smcol:
        have = f[smcol].astype(str).tolist()
        if have == smiles:
            mode = 'exact_SMILES_sequence'
        elif len(set(have)) == len(have) and len(set(smiles)) == len(smiles) and set(have) == set(smiles):
            index = {s:i for i,s in enumerate(have)}
            f = f.iloc[[index[s] for s in smiles]].reset_index(drop=True)
            mode = 'unique_SMILES_join'
        else:
            raise ValueError('Reordered duplicate SMILES need explicit original row indices')
    else:
        mode = 'original_exporter_order_assumed_and_bytes_pinned'
    return f, mode


def mechanism(rec, frame, split, default_order):
    path = Path(rec['path'])
    order = list(rec.get('mechanism_order', default_order))
    if len(order) != 3 or sorted(order) != sorted(MECHANISMS):
        raise ValueError('Mechanism order must contain efflux,influx,pampa exactly once')
    if path.suffix.lower() == '.csv':
        columns = rec.get('columns')
        if not columns or len(columns) != 3:
            raise ValueError('Locked CSV mechanism record needs its original three column names')
        usecols = list(columns) + [rec[k] for k in ('smiles_column','row_index_column','split_column') if rec.get(k)]
        if rec.get('split_column'):
            aliases = ('val','valid','validation') if split == 'val' else ('train',)
            # Inspect partition names only, then skip other partitions before
            # their structure/score columns enter a DataFrame.
            parts = pd.read_csv(path,usecols=[rec['split_column']],dtype=str,keep_default_na=False)
            keep = set(np.flatnonzero(parts[rec['split_column']].str.lower().isin(aliases)))
            f = pd.read_csv(path,usecols=list(dict.fromkeys(usecols)),keep_default_na=False,
                skiprows=lambda n:n > 0 and n-1 not in keep)
        else:
            f = pd.read_csv(path,usecols=list(dict.fromkeys(usecols)),keep_default_na=False)
        f, mode = align_frame(f, rec, frame.SMILES.astype(str).tolist())
        a = f[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    else:
        a = array(rec); mode = 'original_exporter_order_assumed_and_bytes_pinned'
    if a.shape != (len(frame), 3) or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError(f'{split}: mechanism inputs must be original raw probabilities, shape {(len(frame),3)}')
    return np.asarray(a[:, [order.index(m) for m in MECHANISMS]], dtype=float), mode


def load(root, source):
    from .training import load_reference
    root, source = Path(root), Path(source)
    for name in ('input_audit.json', 'reference_audit.json'):
        if not (source/name).is_file():
            raise FileNotFoundError(f'Missing {source/name}. --source-run must identify the original structural run containing its saved UNWEIGHTED reference and input audit.')
    old = read_json(source/'input_audit.json')
    ref_audit = read_json(source/'reference_audit.json')
    old_root = Path(old['datasets']['val']['path']).parent.parent
    original_order = old.get('original_mechanism_order', old.get('mechanism_order'))
    frames, X, paths, records = {}, {}, {}, {}
    for split in SPLITS:
        ds = root/'dataset'/f'{split}_classification.csv'
        check_sha(ds, old['datasets'][split]['sha256'], split+' dataset')
        f = pd.read_csv(ds, dtype=str, keep_default_na=False)
        if not {'SMILES','label'}.issubset(f):
            raise ValueError('Dataset needs SMILES and original label columns')
        y = pd.to_numeric(f.label, errors='raise').to_numpy(dtype=float)
        if not np.isin(y, [0,1]).all() or len(np.unique(y)) != 2:
            raise ValueError('Need unchanged original TITAN binary labels; BBB+=1')
        f['label'] = y.astype(int); frames[split] = f
        records[split] = {}; selected = {}
        for kind in ('H','M'):
            rec = dict(old['selected_arrays'][split][kind])
            rec['path'] = str(relocated(rec['path'],root,old_root,source))
            check_sha(rec['path'],rec.get('sha256'),f'{split}/{kind}')
            paths[rec['path']] = sha(rec['path'])
            if kind == 'H':
                a = np.asarray(array(rec))
                if a.shape != (len(f),2048) or not np.isfinite(a).all():
                    raise ValueError(f'{split}: need unchanged finite 2048-dimensional H')
                alignment = 'original_exporter_order_assumed_and_bytes_pinned'
            else:
                a, alignment = mechanism(rec,f,split,original_order)
            if rec.get('rows_csv'):
                rp = relocated(rec['rows_csv'],root,old_root,source)
                if rec.get('rows_csv_sha256'):
                    check_sha(rp,rec['rows_csv_sha256'],f'{split}/{kind} row witness')
                sf = pd.read_csv(rp,usecols=['SMILES'],keep_default_na=False)
                if sf.SMILES.astype(str).tolist() != f.SMILES.tolist():
                    raise ValueError('Array row witness differs from original dataset order')
                paths[str(rp)] = sha(rp);alignment = 'explicit_original_SMILES_sequence'
            selected[kind] = a
            records[split][kind] = {**rec,'row_alignment_check':alignment}
        # This reproduces the existing head refit cast/order exactly.
        X[split] = np.empty((len(f),2051),dtype=np.float32)
        X[split][:,:2048] = selected['H']
        X[split][:,2048:] = selected['M']
        paths[str(ds)] = sha(ds)
    if {s:len(f) for s,f in frames.items()} != {'train':7593,'val':727}:
        raise ValueError('This protocol requires the original TITAN train=7593, val=727')
    model_path = relocated(ref_audit['path'],root,old_root,source)
    check_sha(model_path,ref_audit.get('sha256'),'original unweighted full-fusion model')
    model = load_reference(model_path)
    # Only validation reference predictions are opened. No test prediction path is used.
    saved = source/'predictions/reference_saved_full_fusion/fixed_0p5/val_all.csv'
    if not saved.is_file():
        raise FileNotFoundError(f'Missing original unweighted validation prediction witness: {saved}')
    sf = pd.read_csv(saved,dtype=str,keep_default_na=False)
    idx = pd.to_numeric(sf.original_row_index,errors='raise').to_numpy(dtype=float)
    if not np.isfinite(idx).all() or sorted(idx.tolist()) != list(range(len(frames['val']))):
        raise ValueError('Reference validation row indices are invalid')
    sf = sf.iloc[np.argsort(idx)].reset_index(drop=True)
    if sf.SMILES.tolist() != frames['val'].SMILES.tolist() or not np.array_equal(pd.to_numeric(sf.label).to_numpy(),frames['val'].label.to_numpy()):
        raise ValueError('Original reference validation witness row alignment failed')
    p = np.asarray(model.predict(X['val']),dtype=float)
    saved_p = pd.to_numeric(sf.positive_probability,errors='raise').to_numpy(dtype=float)
    if not np.allclose(p,saved_p,rtol=0,atol=1e-10):
        raise ValueError('Original reference model does not reproduce saved validation probabilities with these inputs')
    for path in (source/'input_audit.json',source/'reference_audit.json',model_path,saved):
        paths[str(path)] = sha(path)
    audit = dict(source_run=str(source),source_metadata_sha256={str(p):sha(p) for p in (source/'input_audit.json',source/'reference_audit.json')},
        datasets={s:old['datasets'][s] for s in SPLITS},selected_arrays=records,
        mechanism_order=list(MECHANISMS),data_dimension=2051,original_model_path=str(model_path),
        original_model_sha256=sha(model_path),input_paths_sha256=paths,
        row_alignment_limit='Historical bare-array row order is inherited from the original exporter. Byte pinning is not independent proof that the original exporter aligned training identities correctly.',
        data_usage='train and validation only; no test dataset, test array key or test prediction is loaded')
    return frames,X,model,p,audit,ref_audit
