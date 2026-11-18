"""Shared metrics so every model is scored the same way."""

import time
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Highest error rate we accept on messages routed without a human.
# Shared by all models so their review thresholds are comparable.
MAX_AUTO_ERROR_RATE = 0.05


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Macro-F1 (main metric: every intent weighs the same) and accuracy.

    Averaging only over the true intents means an invalid prediction (e.g. -1 for an LLM
    answer outside the label list) counts as an error, not as an extra class.
    """
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def coverage_error_curve(
    y_true: Sequence[int], y_pred: Sequence[int], scores: Sequence[float]
) -> list[dict[str, float]]:
    """For each threshold, the share of messages routed automatically (score >= threshold)
    and the error rate among those routed messages. The rest go to human review."""
    correct = np.asarray(y_true) == np.asarray(y_pred)
    scores = np.asarray(scores)
    curve = []
    for threshold in np.round(np.arange(0.0, 1.0, 0.01), 2):
        auto = scores >= threshold
        n_auto = int(auto.sum())
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": n_auto / len(scores),
                "error_rate": float(1 - correct[auto].mean()) if n_auto else 0.0,
            }
        )
    return curve


def choose_threshold(curve: list[dict[str, float]], max_error_rate: float) -> dict[str, float]:
    """Pick the lowest threshold whose error rate on auto-routed messages is within budget,
    which is the one that sends the fewest messages to human review.
    ``meets_target`` is False when no threshold reaches the budget (the highest one is returned)."""
    for point in curve:  # thresholds ascend, so coverage descends
        if point["error_rate"] <= max_error_rate:
            return {**point, "meets_target": True}
    return {**curve[-1], "meets_target": False}


def worst_classes(
    y_true: Sequence[int], y_pred: Sequence[int], label_names: list[str], n: int = 10
) -> list[dict[str, object]]:
    """The n intents with the lowest F1, each with the label it is most often confused with."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(label_names)))
    f1s = f1_score(y_true, y_pred, labels=labels, average=None)
    rows = []
    for label in np.argsort(f1s)[:n]:
        wrong = y_pred[(y_true == label) & (y_pred != label)]
        valid = wrong[wrong >= 0]  # invalid predictions are errors but have no label
        top = np.bincount(valid, minlength=len(labels)).argmax() if len(valid) else None
        rows.append(
            {
                "intent": label_names[label],
                "f1": float(f1s[label]),
                "n_errors": int(len(wrong)),
                "most_confused_with": label_names[top] if top is not None else None,
            }
        )
    return rows


def bootstrap_macro_f1(
    y_true: Sequence[int], preds: dict[str, Sequence[int]], n_boot: int = 1000, seed: int = 0
) -> dict[str, np.ndarray]:
    """Macro-F1 of each model on n_boot resamples of the test set. The same resampled rows
    are used for every model, so differences between models can be compared pairwise."""
    y_true = np.asarray(y_true)
    labels = np.unique(y_true)
    rng = np.random.default_rng(seed)
    out = {name: np.empty(n_boot) for name in preds}
    for b in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        for name, y_pred in preds.items():
            out[name][b] = f1_score(
                y_true[idx], np.asarray(y_pred)[idx], labels=labels, average="macro"
            )
    return out


def measure_latency(
    predict_one: Callable[[str], object], texts: Sequence[str], warmup: int = 20
) -> dict[str, float]:
    """Time single-message predictions, as an API would serve them, in milliseconds."""
    for text in texts[:warmup]:
        predict_one(text)
    timings = []
    for text in texts:
        start = time.perf_counter()
        predict_one(text)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "latency_median_ms": float(np.median(timings)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
    }
