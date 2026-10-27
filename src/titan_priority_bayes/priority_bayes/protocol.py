"""The search space, objective and weighting policy; never consume held-out test results."""
from pathlib import Path
import hashlib
import json
import os
import warnings

import numpy as np

ACTIVE = ('V01', 'V03')
THRESHOLDS = (.1, .3, .5, .7, .9)
ANCHORS = ((0, 0), (10, 0), (0, 10), (10, 10))
GRID = tuple((i, j) for i in range(41) for j in range(41))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pair_weights(pair):
    if len(pair) != 2 or any(int(i) != i or i < 0 or i > 40 for i in pair):
        raise ValueError('Weight indices must be integers in 0..40')
    return tuple((20 + int(i)) / 20 for i in pair)


def initial_plan(seed=42, startup=40):
    if not 4 <= startup <= len(GRID):
        raise ValueError('startup must be in 4..1681')
    available = [p for p in GRID if p not in ANCHORS]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(available))
    return list(ANCHORS) + [available[int(i)] for i in order[:startup-4]]


def build_weights(v01, v03, pair):
    a, b = np.asarray(v01, dtype=bool), np.asarray(v03, dtype=bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError('Rule masks must be aligned one-dimensional arrays')
    w1, w3 = pair_weights(pair)
    return np.maximum(1., np.maximum(np.where(a, w1, 0.), np.where(b, w3, 0.)))


def metrics(y, probability, threshold):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
        accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef)
    y, p = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite(p).all():
        raise ValueError('Wrong probability shape or nonfinite predictions')
    if not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Need binary TITAN labels and probabilities in [0,1]')
    pred = p > threshold
    tp = int(((y == 1) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    n, both = len(y), len(np.unique(y)) == 2
    return dict(n=n, n_positive=tp+fn, n_negative=tn+fp, threshold=float(threshold),
        AUROC=float(roc_auc_score(y, p)) if both else None,
        AUPRC_AP=float(average_precision_score(y, p)) if both else None,
        ACC=float(accuracy_score(y, pred)) if n else None,
        BACC=float(balanced_accuracy_score(y, pred)) if both else None,
        SN=tp/(tp+fn) if tp+fn else None, SP=tn/(tn+fp) if tn+fp else None,
        F1=float(f1_score(y, pred, zero_division=0)) if n else None,
        MCC=float(matthews_corrcoef(y, pred)) if n else None,
        TN=tn, FP=fp, FN=fn, TP=tp)


def choose_threshold(y, p):
    # The original ascending grid, strict >, first maximum BACC.
    return max(THRESHOLDS, key=lambda t: metrics(y, p, t)['BACC'])


def choose_best(records):
    if not records:
        raise ValueError('No completed configurations')
    for r in records:
        if not np.isfinite(r['validation_metrics']['fixed_0p5']['AUROC']):
            raise ValueError('Missing finite validation AUROC')
    # AUROC only. Exact ties prefer weaker overall intervention, then V01/V03.
    return min(records, key=lambda r: (
        -r['validation_metrics']['fixed_0p5']['AUROC'],
        max(pair_weights(r['indices'])), sum(pair_weights(r['indices'])),
        *pair_weights(r['indices'])))


def log_expected_improvement(improvement, sd):
    """Stable log EI, including Gaussian negative tails where raw EI underflows.

    For z<-8 use the Mills-ratio asymptotic expansion of
    phi(z)+z*Phi(z). This preserves ranking of extremely small EI values.
    """
    from scipy.special import log_ndtr
    improvement,sd=np.asarray(improvement,dtype=float),np.asarray(sd,dtype=float)
    answer=np.full_like(sd,-np.inf)
    valid=sd>1e-15
    z=improvement[valid]/sd[valid]
    log_phi=-.5*z*z-.5*np.log(2*np.pi)
    v=np.empty_like(z)
    positive=z>=0
    with np.errstate(divide='ignore'):
        v[positive]=np.logaddexp(log_phi[positive],np.log(z[positive])+log_ndtr(z[positive]))
    middle=(z<0)&(z>=-8)
    ratio_log=np.log(-z[middle])+log_ndtr(z[middle])-log_phi[middle]
    v[middle]=log_phi[middle]+np.log(-np.expm1(ratio_log))
    tail=z<-8
    a=-z[tail];q=1/(a*a)
    correction=1-3*q+15*q**2-105*q**3+945*q**4-10395*q**5+135135*q**6
    v[tail]=log_phi[tail]-2*np.log(a)+np.log(correction)
    answer[valid]=np.log(sd[valid])+v
    return answer


def next_pair(records, seed=42, startup=40, xi=0.0001):
    """Exact finite-grid maximization of GP expected improvement over unseen pairs."""
    seen = {tuple(r['indices']) for r in records}
    if len(seen) != len(records):
        raise ValueError('Completed configuration registry contains duplicates')
    plan = initial_plan(seed, startup)
    for k, pair in enumerate(plan):
        if pair not in seen:
            return pair, dict(method='fixed_anchor' if k < 4 else 'initial_random',
                             startup_position=k+1)
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    untried = np.asarray([p for p in GRID if p not in seen], dtype=int)
    if not len(untried):
        raise ValueError('All 1681 combinations have already been evaluated')
    X = np.asarray([r['indices'] for r in records], dtype=float) / 40
    y = np.asarray([r['validation_metrics']['fixed_0p5']['AUROC'] for r in records])
    kernel = ConstantKernel(1., (0.01, 100.)) * Matern(
        length_scale=[.25, .25], length_scale_bounds=(.025, 4.), nu=2.5)
    kernel += WhiteKernel(1e-5, (1e-8, .1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-8,
        n_restarts_optimizer=1, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        gp.fit(X, y)
    mean, sd = gp.predict(untried / 40., return_std=True)
    if not np.isfinite(mean).all() or not np.isfinite(sd).all():
        raise ValueError('GP produced nonfinite predictions')
    improvement = mean - y.max() - xi
    log_ei=log_expected_improvement(improvement,sd)
    if not np.isfinite(log_ei).any():
        index = int(np.argmax(sd)); method = 'GP_uncertainty_zero_variance'
    else:
        index = int(np.argmax(log_ei)); method = 'GP_log_expected_improvement'
    pair = tuple(map(int, untried[index]))
    return pair, dict(method=method, expected_improvement=float(np.exp(log_ei[index])),
        log_expected_improvement=float(log_ei[index]) if np.isfinite(log_ei[index]) else None,
        predicted_AUROC=float(mean[index]), predicted_sd=float(sd[index]), xi=xi,
        fitted_kernel=str(gp.kernel_), warnings=[str(w.message) for w in caught])
