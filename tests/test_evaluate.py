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


def test_invalid_prediction_counts_as_error_not_as_class():
    from intent_bench.evaluate import classification_metrics

    metrics = classification_metrics([0, 1], [0, -1])
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5  # class 0: F1 1, class 1: F1 0; -1 is not averaged in


def test_llm_prompt_and_parsing():
    from intent_bench.llm import INVALID, build_system_prompt, parse_label

    names = ["card_arrival", "Refund_not_showing_up", "reverted_card_payment?"]
    assert parse_label(" Refund_not_showing_up\n", names) == 1
    assert parse_label("reverted_card_payment?", names) == 2
    assert parse_label("refund_not_showing_up", names) == INVALID  # case must match
    assert parse_label("The intent is card_arrival", names) == INVALID
    prompt = build_system_prompt(names, [("Where is my card?", "card_arrival")])
    assert all(name in prompt for name in names)
    assert "Where is my card? => card_arrival" in prompt
