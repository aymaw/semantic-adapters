import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def probabilities(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if np.ndim(logits) != 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be a finite two-dimensional array")
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


def metrics(probs, labels, threshold=0.0, repeats=200, groups=None):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if (
        probs.ndim != 2
        or not len(probs)
        or len(labels) != len(probs)
        or not np.isfinite(probs).all()
        or (probs < 0).any()
        or (probs > 1).any()
        or not np.allclose(probs.sum(1), 1.0, atol=1e-6)
        or not np.issubdtype(labels.dtype, np.integer)
        or (labels < 0).any()
        or (labels >= probs.shape[1]).any()
        or repeats < 1
    ):
        raise ValueError("Expected nonempty normalized probabilities and valid class indices")
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
    by_label = [np.flatnonzero(labels == c) for c in classes]
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in by_label if len(g)])
        boot.append(
            f1_score(labels[idx], pred[idx], labels=classes, average="macro", zero_division=0)
        )
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred, labels=classes, zero_division=0
    )
    report = {
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
        "per_class": {
            str(c): {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            }
            for c in classes
        },
    }
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != labels.shape:
            raise ValueError("groups must align with labels")
        units = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
        if any(len(np.unique(labels[idx])) != 1 for idx in units.values()):
            raise ValueError("Template groups must each have one intent label")
        strata = [[g for g, idx in units.items() if labels[idx[0]] == c] for c in classes]
        cluster_rng = np.random.default_rng(9)
        clustered = []
        for _ in range(repeats):
            selected = [
                g
                for stratum in strata
                if stratum
                for g in cluster_rng.choice(stratum, len(stratum), replace=True)
            ]
            idx = np.concatenate([units[g] for g in selected])
            clustered.append(
                f1_score(labels[idx], pred[idx], labels=classes, average="macro", zero_division=0)
            )
        report["macro_f1_template_bootstrap_95"] = np.quantile(clustered, [0.025, 0.975]).tolist()
        report["per_template_accuracy"] = {
            str(g): float(correct[idx].mean()) for g, idx in units.items()
        }
    return report
