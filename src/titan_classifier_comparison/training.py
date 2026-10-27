"""Train ONLY the downstream classifier; upstream TITAN features are arrays."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (SCORE_COLUMNS, choose_threshold, classification_metrics,
                     file_record, read_json, write_json)

GROUPS = ('nn_baseline', 'nn_mechanism', 'lightgbm_baseline', 'lightgbm_mechanism')
METRICS = ('AUROC', 'AUPRC_AP', 'ACC', 'BACC', 'SN', 'SP', 'F1', 'MCC')


def load_backends(device_name, threads):
    global torch, nn, lgb
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import torch.nn as nn
    import lightgbm as lgb
    torch.set_num_threads(threads)
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; choose --device cpu')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use cpu or cuda for this TITAN experiment')
    return device, {'torch': torch.__version__, 'lightgbm': lgb.__version__}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def features(arrays, group):
    if group not in GROUPS:
        raise ValueError(f'Unknown group: {group}')
    h = arrays['pooled']
    return np.concatenate([h, arrays['mechanism']], axis=1) if group.endswith('_mechanism') else h


def weighted_bce(y, p, positive_weight):
    """Mean over N, matching BCEWithLogitsLoss(pos_weight=..., reduction='mean')."""
    y, p = np.asarray(y).reshape(-1), np.asarray(p, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid binary probabilities')
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(positive_weight * y * np.log(p) + (1-y) * np.log1p(-p)))


def make_head(mechanism, pooled_dim=2048, hidden_dim=2048):
    """Common weights/RNG match; three extra input columns start at zero."""
    head = nn.Sequential(nn.Linear(pooled_dim, hidden_dim), nn.ReLU(), nn.Dropout(.1),
                         nn.Linear(hidden_dim, 1))
    if mechanism:
        rng = torch.get_rng_state()
        first = nn.Linear(pooled_dim + 3, hidden_dim)
        with torch.no_grad():
            first.weight[:, :pooled_dim].copy_(head[0].weight)
            first.weight[:, pooled_dim:].zero_()
            first.bias.copy_(head[0].bias)
        head[0] = first
        torch.set_rng_state(rng)
    return head


def load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Older Torch used in some original environments.
        return torch.load(path, map_location=device)


def neural_probabilities(head, x, device, batch_size=256):
    head.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = head(torch.as_tensor(x[start:start+batch_size], device=device)).reshape(-1)
            parts.append(torch.sigmoid(logits).cpu().numpy())
    return check_probabilities(np.concatenate(parts), len(x))


def check_probabilities(p, n):
    p = np.asarray(p).reshape(-1)
    if p.shape != (n,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Classifier returned invalid probabilities')
    return p


def train_neural(group, args, subset, folder, device, config, log):
    seed_everything(args.seed)
    x, vx = features(subset['train'], group), features(subset['val'], group)
    y, vy = subset['train']['labels'], subset['val']['labels']
    head = make_head(group.endswith('_mechanism')).to(device)
    # A real Torch backward/shape check on the server, before the optimizer exists.
    head.eval()
    probe = head(torch.as_tensor(x[:2], device=device)).reshape(-1)
    if probe.shape != (2,):
        raise ValueError('Unexpected neural head output shape')
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['scale_pos_weight']], device=device))
    criterion(probe, torch.as_tensor(y[:2], device=device)).backward()
    if head[0].weight.grad is None or not torch.isfinite(head[0].weight.grad).all():
        raise ValueError('Neural-head backward check failed')
    head.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=32, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)),
        batch_size=32, shuffle=False, num_workers=0)
    best, best_epoch, stale, history = float('inf'), 0, 0, []
    checkpoint = folder/'best_model.pth'
    for epoch in range(1, args.nn_epochs + 1):
        losses = []
        for current_loader, training in ((loader, True), (val_loader, False)):
            head.train(training)
            total = 0.
            with torch.set_grad_enabled(training):
                for bx, by in current_loader:
                    bx, by = bx.to(device), by.to(device)
                    loss = criterion(head(bx).reshape(-1), by)
                    if not torch.isfinite(loss):
                        raise ValueError(f'{group}: nonfinite neural loss')
                    total += float(loss.item()) * len(by)
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            losses.append(total / len(current_loader.dataset))
        train_loss, val_loss = losses
        improved = val_loss < best
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, improved=improved))
        log(f'epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}')
        if improved:
            best, best_epoch, stale = val_loss, epoch, 0
            torch.save(head.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.nn_patience:
                break
    pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
    if best_epoch < 1:
        raise ValueError('No finite best neural checkpoint')
    head.load_state_dict(load_state(checkpoint, device), strict=True)
    p = neural_probabilities(head, vx, device)
    del head, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return checkpoint, p, dict(best_epoch=best_epoch, best_validation_loss=float(best),
                               runtime_backward_check=True)


def lightgbm_parameters(seed, threads, positive_weight):
    return dict(objective='binary', boosting_type='gbdt', metric='None', learning_rate=.03,
                num_leaves=31, max_depth=5, min_data_in_leaf=40, feature_fraction=.9,
                bagging_fraction=.8, bagging_freq=1, lambda_l2=1.,
                scale_pos_weight=positive_weight, seed=seed, feature_fraction_seed=seed,
                bagging_seed=seed, data_random_seed=seed, num_threads=threads,
                deterministic=True, force_col_wise=True, verbosity=-1)


def train_lightgbm(group, args, subset, folder, config, log):
    x, vx = features(subset['train'], group), features(subset['val'], group)
    names = [f'pooled_{i+1:04d}' for i in range(2048)]
    if group.endswith('_mechanism'):
        names += SCORE_COLUMNS
    train = lgb.Dataset(x, label=subset['train']['labels'], feature_name=names)
    val = lgb.Dataset(vx, label=subset['val']['labels'], reference=train, feature_name=names)
    history = {}
    def metric(predictions, dataset):
        return ('weighted_bce', weighted_bce(dataset.get_label(), predictions,
                                             config['scale_pos_weight']), False)
    booster = lgb.train(config['lightgbm_parameters'], train, num_boost_round=args.lgbm_rounds,
                        valid_sets=[val], valid_names=['val'], feval=metric,
                        callbacks=[lgb.early_stopping(args.lgbm_patience, first_metric_only=True, verbose=False),
                                   lgb.record_evaluation(history), lgb.log_evaluation(period=50)])
    rounds = int(booster.best_iteration)
    if rounds < 1:
        raise ValueError('LightGBM did not report a best iteration')
    checkpoint = folder/'best_model.txt'
    booster.save_model(str(checkpoint), num_iteration=rounds)
    pd.DataFrame(dict(iteration=np.arange(1, len(history['val']['weighted_bce'])+1),
                      val_weighted_bce=history['val']['weighted_bce'])).to_csv(folder/'history.csv', index=False)
    # Reload the saved best model: exported importance and predictions refer to exactly it.
    booster = lgb.Booster(model_file=str(checkpoint))
    p = check_probabilities(booster.predict(vx), len(vx))
    pd.DataFrame(dict(feature=names, gain=booster.feature_importance('gain'),
                      split=booster.feature_importance('split'))).to_csv(folder/'feature_importance.csv', index=False)
    loss = weighted_bce(subset['val']['labels'], p, config['scale_pos_weight'])
    log(f'best iteration={rounds}; val weighted BCE={loss:.6f}')
    return checkpoint, p, dict(best_iteration=rounds, best_validation_loss=loss)


def fit_group(group, args, subset, folder, device):
    if set(subset) != {'train', 'val'}:
        raise ValueError('Classifier training receives train and val only')
    folder.mkdir(parents=True, exist_ok=False)
    def log(message):
        line = f'[{group}] {message}'
        print(line, flush=True)
        with (folder/'training.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')
    y = subset['train']['labels']
    spw = int((y == 0).sum()) / int((y == 1).sum())
    config = dict(experiment=group, seed=args.seed, classifier='neural' if group.startswith('nn_') else 'lightgbm',
                  input_dim=2051 if group.endswith('_mechanism') else 2048,
                  pooled_dim=2048, mechanism_columns=SCORE_COLUMNS if group.endswith('_mechanism') else [],
                  scale_pos_weight=spw, encoder_frozen=True, attention_frozen=True,
                  additional_scaling='none', selection='minimum validation weighted BCE; no test selection',
                  threshold_selection='validation BACC grid 0.1,0.3,0.5,0.7,0.9; first tie; strict >')
    if group.startswith('nn_'):
        config.update(hidden_dim=2048, dropout=.1, learning_rate=1e-4, weight_decay=.01,
                      optimizer='AdamW', batch_size=32, max_epochs=args.nn_epochs,
                      patience=args.nn_patience, common_initial_weights=True,
                      initial_extra_columns='zeros' if group.endswith('_mechanism') else None)
    else:
        config.update(lightgbm_parameters=lightgbm_parameters(args.seed,args.num_threads,spw),
                      max_rounds=args.lgbm_rounds, patience=args.lgbm_patience)
    write_json(folder/'model_config.json', config)
    log(f'input={config["input_dim"]}; train n={len(y)}; seed={args.seed}; fixed upstream features')
    if group.startswith('nn_'):
        checkpoint, p, selection = train_neural(group,args,subset,folder,device,config,log)
    else:
        checkpoint, p, selection = train_lightgbm(group,args,subset,folder,config,log)
    threshold = choose_threshold(subset['val']['labels'], p)
    np.save(folder/'val_probability.npy', p)
    lock = dict(experiment=group, seed=args.seed, threshold=threshold, **selection,
                checkpoint=file_record(checkpoint), config=file_record(folder/'model_config.json'),
                val_probability=file_record(folder/'val_probability.npy'),
                validation=classification_metrics(subset['val']['labels'],p,threshold))
    write_json(folder/'selection_lock.json', lock)
    log(f'LOCKED threshold={threshold}; val AUROC={lock["validation"]["AUROC"]:.6f}')
    return config, lock


def predict_group(group, folder, config, lock, arrays, device):
    x = features(arrays, group)
    if group.startswith('nn_'):
        # Loading for evaluation consumes RNG but never influences an already locked fit.
        head = make_head(group.endswith('_mechanism')).to(device)
        head.load_state_dict(load_state(folder/Path(lock['checkpoint']['path']).name,device), strict=True)
        p = neural_probabilities(head, x, device)
        del head
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return p
    booster = lgb.Booster(model_file=str(folder/Path(lock['checkpoint']['path']).name))
    if booster.num_feature() != config['input_dim']:
        raise ValueError('LightGBM saved model has wrong feature count')
    return check_probabilities(booster.predict(x), len(x))
