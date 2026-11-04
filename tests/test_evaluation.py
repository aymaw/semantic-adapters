import numpy as np
import pytest

from semadapt.data import fingerprint, generate
from semadapt.evaluate import fit_temperature, metrics, probabilities, select_threshold


def test_split_family_disjointness_and_no_duplicates():
    data = generate(per_family=4)
    assert fingerprint(data) == fingerprint(generate(per_family=4))
    families = [set(r["family"] for r in data[k]) for k in ("train", "validation", "test")]
    assert not (families[0] & families[1] or families[0] & families[2] or families[1] & families[2])
    sets = [set(r["text"] for r in rows) for rows in data.values()]
    assert sum(map(len, sets)) == len(set.union(*sets))


def test_perfect_metrics_and_confusion():
    report = metrics(np.array([[0.99, 0.01], [0.01, 0.99]]), [0, 1], 0.9)
    assert report["macro_f1"] == 1
    assert report["confusion"] == [[1, 0], [0, 1]]
    assert report["ece_10_bins"] == pytest.approx(0.01)
    assert report["brier"] == pytest.approx(0.0002)


def test_temperature_reduces_validation_nll():
    logits = np.array([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]])
    labels = np.array([0, 1, 1, 1])
    temp = fit_temperature(logits, labels)
    assert (
        metrics(probabilities(logits, temp), labels)["nll"]
        <= metrics(probabilities(logits), labels)["nll"]
    )


def test_no_admissible_threshold_abstains():
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    threshold = select_threshold(probs, np.array([1, 0]), min_accepted=1)
    result = metrics(probs, [1, 0], threshold)
    assert threshold > 1
    assert result["coverage"] == 0
    assert result["accepted_accuracy"] is None


def test_threshold_uses_maximum_admissible_coverage():
    p = np.array([[0.6, 0.4], [0.8, 0.2], [0.9, 0.1]])
    assert select_threshold(p, np.array([1, 0, 0]), target=1.0, min_accepted=2) == 0.8
