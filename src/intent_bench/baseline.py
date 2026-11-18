"""Baseline: TF-IDF features + logistic regression, plus a dummy sanity check."""

import json
from itertools import product
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from intent_bench.data import REVISION, SEED, load_banking77, load_split
from intent_bench.evaluate import (
    MAX_AUTO_ERROR_RATE,
    choose_threshold,
    classification_metrics,
    coverage_error_curve,
    measure_latency,
    worst_classes,
)

MODEL_DIR = Path("artifacts/baseline")
MODEL_VERSION = "tfidf-logreg-v1"

# Small grid, chosen on validation only (the test set stays untouched).
GRID = {
    "C": [1.0, 10.0, 30.0, 100.0, 300.0],
    "ngram_range": [(1, 1), (1, 2)],
}


def build_pipeline(C: float, ngram_range: tuple[int, int]) -> Pipeline:
    """TF-IDF word features feeding a multinomial logistic regression."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, sublinear_tf=True)),
            ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED)),
        ]
    )


def main() -> None:
    train, _, label_names = load_banking77()
    train_idx, val_idx = load_split()
    X_tr, y_tr = train["text"].iloc[train_idx], train["label"].iloc[train_idx]
    X_val, y_val = train["text"].iloc[val_idx], train["label"].iloc[val_idx]

    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    dummy_metrics = classification_metrics(y_val, dummy.predict(X_val))
    print(f"dummy: {dummy_metrics}")

    results = []
    for C, ngram_range in product(GRID["C"], GRID["ngram_range"]):
        pipe = build_pipeline(C, ngram_range).fit(X_tr, y_tr)
        metrics = classification_metrics(y_val, pipe.predict(X_val))
        results.append({"C": C, "ngram_range": list(ngram_range), **metrics})
        print(f"C={C:<5} ngram={ngram_range}: {metrics}")

    best = max(results, key=lambda r: r["macro_f1"])
    print(f"best: {best}")

    # Refit the best configuration on the same training part; the validation part
    # stays held out so we can pick the review threshold on it later.
    model = build_pipeline(best["C"], tuple(best["ngram_range"])).fit(X_tr, y_tr)

    # Review threshold: the score is the highest predicted probability.
    proba = model.predict_proba(X_val)
    y_pred, scores = proba.argmax(axis=1), proba.max(axis=1)
    curve = coverage_error_curve(y_val, y_pred, scores)
    review = choose_threshold(curve, MAX_AUTO_ERROR_RATE)
    print(f"review threshold (max auto error {MAX_AUTO_ERROR_RATE:.0%}): {review}")

    worst = worst_classes(y_val, y_pred, label_names)
    print("worst classes on validation:")
    for row in worst:
        print(f"  {row['intent']:<40} f1={row['f1']:.2f}  -> {row['most_confused_with']}")

    latency = measure_latency(lambda t: model.predict_proba([t]), X_val.tolist())
    print(f"latency (CPU, 1 message at a time): {latency}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_revision": REVISION,
        "seed": SEED,
        "label_names": label_names,
        "best_params": {"C": best["C"], "ngram_range": best["ngram_range"]},
        "val_metrics": {k: best[k] for k in ("macro_f1", "accuracy")},
        "dummy_val_metrics": dummy_metrics,
        "review": {"max_auto_error_rate": MAX_AUTO_ERROR_RATE, **review},
        "latency": latency,
        "worst_classes": worst,
        "coverage_error_curve": curve,
        "grid_results": results,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
