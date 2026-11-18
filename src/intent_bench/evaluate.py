"""Shared metrics so every model is scored the same way."""

import time
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Macro-F1 (main metric: every intent weighs the same) and accuracy."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


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
