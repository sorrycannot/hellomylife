"""Training functions retained from the existing full-fusion LightGBM refit protocol."""

from pathlib import Path

import numpy as np

def get_booster(value):
    import lightgbm as lgb
    if isinstance(value, lgb.Booster):
        return value
    if isinstance(value, lgb.LGBMClassifier):
        if value.get_params().get('class_weight') is not None:
            raise ValueError('Reference uses class_weight. Supply its explicit native Booster '
                             'and preserve that class-weight policy before using this experiment.')
        return value.booster_
    if isinstance(value, dict):
        found = []
        for k in ('model', 'classifier', 'booster', 'lgbm', 'estimator'):
            if k in value:
                try:
                    found.append(get_booster(value[k]))
                except TypeError:
                    pass
        if len(found) == 1:
            return found[0]
    raise TypeError('Not a saved LightGBM classifier/Booster or recognized one-model bundle')

def load_reference(path):
    import lightgbm as lgb
    path = Path(path)
    if path.suffix.lower() in ('.txt', '.model', '.lgb', '.lgbm'):
        booster = lgb.Booster(model_file=str(path))
    else:
        import joblib
        booster = get_booster(joblib.load(path))
    if booster.num_feature() != 2051:
        raise ValueError(f'Reference must be full_fusion H2048+M3, got {booster.num_feature()} features')
    if not str(booster.params.get('objective', '')).startswith('binary'):
        raise ValueError('Reference must use native binary objective; custom objectives are unsupported')
    return booster

def common_params(booster, seed=42, threads=4):
    p = dict(booster.params)
    # Remove saved training budgets and aliases which would override arguments.
    removed = {}
    keys = ('num_iterations', 'num_iteration', 'n_estimators', 'num_boost_round',
            'num_round', 'num_rounds', 'nrounds', 'n_iter', 'max_iter', 'num_trees', 'n_trees',
            'early_stopping_round', 'early_stopping_rounds', 'early_stopping',
            'n_iter_no_change', 'metric', 'metrics', 'metric_types',
            'random_state', 'n_jobs', 'device', 'device_type', 'force_row_wise',
            'force_col_wise', 'num_thread', 'nthread', 'nthreads',
            'verbose', 'verbosity', 'class_weight', 'importance_type')
    for k in keys:
        if k in p:
            removed[k] = p.pop(k)
    if str(p.get('boosting', p.get('boosting_type', 'gbdt'))) == 'dart':
        raise ValueError('DART does not support this early-stopping protocol')
    if p.get('is_unbalance', False) not in (False, 'false', 'False', 0):
        raise ValueError('Reference uses is_unbalance; this script requires explicit scale_pos_weight')
    if p.get('categorical_feature') not in (None, [], '', 'auto'):
        raise ValueError('Categorical feature metadata is not supported for the dense frozen vector')
    p.update(objective='binary', metric='None', seed=seed,
             feature_fraction_seed=seed, bagging_seed=seed, data_random_seed=seed,
             extra_seed=seed, drop_seed=seed,
             deterministic=True, force_col_wise=True, device_type='cpu',
             num_threads=threads, verbosity=-1)
    spw = float(p.get('scale_pos_weight', 1.0))
    if not np.isfinite(spw) or spw <= 0:
        raise ValueError('Invalid reference scale_pos_weight')
    if float(p.get('sigmoid', 1)) != 1:
        raise ValueError('This protocol assumes binary sigmoid=1')
    return p, removed, spw

def fit_head(X, y, params, rounds, Xv=None, yv=None, weights=None,
             patience=50, scale_pos_weight=1., verbose=False):
    import lightgbm as lgb
    train = lgb.Dataset(X, label=y, weight=weights, free_raw_data=True)
    callbacks = [lgb.log_evaluation(50 if verbose else 0)]
    history = {}
    kwargs = {}
    if Xv is not None:
        valid = lgb.Dataset(Xv, label=yv, reference=train, free_raw_data=True)

        def weighted_bce(prob, data):
            truth = data.get_label()
            prob = np.clip(prob, 1e-12, 1 - 1e-12)
            # Same class-weighted BCE convention as torch BCEWithLogitsLoss
            # with mean reduction. No structure weights on validation data.
            loss = -(scale_pos_weight * truth * np.log(prob) +
                     (1 - truth) * np.log1p(-prob)).mean()
            return 'weighted_bce', float(loss), False

        callbacks += [lgb.early_stopping(patience, first_metric_only=True, verbose=verbose),
                      lgb.record_evaluation(history)]
        kwargs.update(valid_sets=[valid], valid_names=['val'], feval=weighted_bce)
    model = lgb.train(params, train, num_boost_round=int(rounds),
                      callbacks=callbacks, **kwargs)
    return model, history
