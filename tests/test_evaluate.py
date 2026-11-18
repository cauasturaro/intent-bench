"""Checks for the shared review-threshold helpers."""

from intent_bench.evaluate import choose_threshold, coverage_error_curve


def test_curve_and_threshold():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]  # the wrong prediction has the lowest score
    scores = [0.9, 0.3, 0.8, 0.7]
    curve = coverage_error_curve(y_true, y_pred, scores)
    assert curve[0] == {"threshold": 0.0, "coverage": 1.0, "error_rate": 0.25}
    best = choose_threshold(curve, max_error_rate=0.0)
    assert best["threshold"] == 0.31  # first threshold that excludes the 0.3 error
    assert best["coverage"] == 0.75
