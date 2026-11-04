import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score


def probabilities(logits, temperature=1.0):
    return torch.tensor(logits, dtype=torch.float64).div(temperature).softmax(-1).numpy()


def fit_temperature(logits, labels):
    # Fixed validation-only search grid; no test-driven optimization.
    candidates = np.exp(np.linspace(np.log(0.2), np.log(5.0), 81))
    nll = [
        -np.log(probabilities(logits, t)[np.arange(len(labels)), labels].clip(1e-12)).mean()
        for t in candidates
    ]
    return float(candidates[np.argmin(nll)])


def select_threshold(probs, labels, target=0.95, min_accepted=30):
    conf, pred = probs.max(1), probs.argmax(1)
    selected = 1.01
    for threshold in np.unique(conf):
        accept = conf >= threshold
        if accept.sum() >= min_accepted and (pred[accept] == labels[accept]).mean() >= target:
            selected = float(threshold)
            break  # lowest admissible threshold gives greatest validation coverage
    return selected


def metrics(probs, labels, threshold=0.0, repeats=200):
    labels = np.asarray(labels)
    classes = np.arange(probs.shape[1])
    pred, conf = probs.argmax(1), probs.max(1)
    correct = pred == labels
    ece = 0.0
    for left, right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (conf > left) & (conf <= right)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    accept = conf >= threshold
    rng = np.random.default_rng(9)
    # Stratified bootstrap prevents a rare resample from dropping a class.
    boot = []
    groups = [np.flatnonzero(labels == c) for c in classes]
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups if len(g)])
        boot.append(
            f1_score(labels[idx], pred[idx], labels=classes, average="macro", zero_division=0)
        )
    return {
        "n": len(labels),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(labels, pred, labels=classes, average="macro", zero_division=0)),
        "macro_f1_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "confusion": confusion_matrix(labels, pred, labels=classes).tolist(),
        "nll": float(-np.log(probs[np.arange(len(labels)), labels].clip(1e-12)).mean()),
        "brier": float(((probs - np.eye(len(classes))[labels]) ** 2).sum(1).mean()),
        "ece_10_bins": float(ece),
        "coverage": float(accept.mean()),
        "accepted_accuracy": float(correct[accept].mean()) if accept.any() else None,
    }
