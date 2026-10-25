"""Patient-level metrics and paired, class-stratified bootstrap inference."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def validate_predictions(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("Require one label and C2 probability per patient")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Both C1 and C2 reference classes are required")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid C2 probabilities")
    return y.astype(int), p


def classification_metrics(labels, probabilities):
    y, p = validate_predictions(labels, probabilities)
    # Argmax over [1-p, p]: ties are assigned to C1, as in torch.argmax.
    pred = (p > .5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {"n": len(y), "C1": int((y == 0).sum()), "C2": int((y == 1).sum()),
            "auroc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision_C2": precision_score(y, pred, zero_division=0),
            "recall_C2": recall_score(y, pred), "f1_C2": f1_score(y, pred),
            "C1_correct": int(cm[0, 0]), "C1_as_C2": int(cm[0, 1]),
            "C2_as_C1": int(cm[1, 0]), "C2_correct": int(cm[1, 1])}


def stratified_draw(y, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == c), int((y == c).sum()),
                                     replace=True) for c in (0, 1)])


def paired_auc_bootstrap(labels, first, second, n_resamples=1000, seed=123):
    y, a = validate_predictions(labels, first)
    _, b = validate_predictions(labels, second)
    if n_resamples < 100:
        raise ValueError("Use at least 100 patient bootstrap resamples")
    observed = roc_auc_score(y, a) - roc_auc_score(y, b)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = stratified_draw(y, rng)
        draws[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    low, high = np.quantile(draws, [.025, .975])
    # Approximate null distribution centred at zero. Bootstrap replicates are
    # NOT treated as independent patients in a t-test. Add-one correction keeps
    # finite-Monte-Carlo P values nonzero and records their resolution.
    p = (1 + np.count_nonzero(np.abs(draws - observed) >= abs(observed))) / (n_resamples + 1)
    return {"difference": float(observed), "ci_low": float(low), "ci_high": float(high),
            "p_two_sided": float(p), "n_patients": len(y), "n_resamples": n_resamples,
            "p_resolution": 1 / (n_resamples + 1),
            "method": "paired class-stratified percentile CI; centred bootstrap two-sided P"}


def metric_intervals(labels, probabilities, n_resamples=1000, seed=123):
    y, p = validate_predictions(labels, probabilities)
    keys = ("auroc", "average_precision", "balanced_accuracy", "f1_C2")
    rng, values = np.random.default_rng(seed), []
    for _ in range(n_resamples):
        idx = stratified_draw(y, rng)
        row = classification_metrics(y[idx], p[idx])
        values.append([row[k] for k in keys])
    limits = np.quantile(np.asarray(values), [.025, .975], axis=0)
    return {f"{k}_ci_{side}": float(limits[j, i])
            for i, k in enumerate(keys) for j, side in enumerate(("low", "high"))}


def align_prediction_tables(first, second):
    needed = {"sample_id", "true", "pred_prob_C2"}
    for frame in (first, second):
        if not needed <= set(frame) or frame.sample_id.duplicated().any():
            raise ValueError("Prediction tables need unique sample_id, true and pred_prob_C2")
    if set(first.sample_id) != set(second.sample_id):
        raise ValueError("Paired comparisons require exactly the same patients")
    b = second.set_index("sample_id").loc[first.sample_id]
    if not np.array_equal(first.true.to_numpy(), b.true.to_numpy()):
        raise ValueError("Reference labels disagree between models")
    return first.true.to_numpy(), first.pred_prob_C2.to_numpy(), b.pred_prob_C2.to_numpy()
