"""Score every model once on the held-out test set, using thresholds chosen on validation.

Run ``uv run python -m intent_bench.final_eval`` to write ``reports/``.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from intent_bench import baseline, llm, transformer
from intent_bench.data import REVISION, load_banking77, load_split
from intent_bench.evaluate import (
    bootstrap_macro_f1,
    classification_metrics,
    measure_latency,
    worst_classes,
)

REPORT_DIR = Path("reports")
LLM_MODES = ["zero", "few"]


def review_on_test(y_true, y_pred, scores, threshold: float) -> dict[str, float]:
    """Apply the validation threshold to test: share routed automatically and its error rate."""
    auto = np.asarray(scores) >= threshold
    correct = np.asarray(y_true) == np.asarray(y_pred)
    return {
        "threshold": threshold,
        "coverage": float(auto.mean()),
        "error_rate": float(1 - correct[auto].mean()) if auto.any() else 0.0,
    }


def run_local(X_test, X_lat) -> dict[str, dict]:
    """Baseline and DistilBERT: predictions, scores and CPU latency (one message at a time)."""
    results = {}

    meta = json.loads((baseline.MODEL_DIR / "metadata.json").read_text())
    model = joblib.load(baseline.MODEL_DIR / "model.joblib")
    proba = model.predict_proba(X_test)
    results["tfidf_logreg"] = {
        "meta": meta,
        "y_pred": proba.argmax(axis=1),
        "scores": proba.max(axis=1),
        "latency": measure_latency(lambda t: model.predict_proba([t]), X_lat),
        "cost_per_1000_usd": 0.0,
    }

    meta = json.loads((transformer.MODEL_DIR / "metadata.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(transformer.MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(transformer.MODEL_DIR)
    device = transformer.pick_device()
    model.to(device)
    proba = transformer.predict_proba(model, tokenizer, X_test, device)
    cpu = torch.device("cpu")
    model.to(cpu)
    results["distilbert"] = {
        "meta": meta,
        "y_pred": proba.argmax(axis=1),
        "scores": proba.max(axis=1),
        "latency": measure_latency(
            lambda t: transformer.predict_proba(model, tokenizer, [t], cpu), X_lat
        ),
        "cost_per_1000_usd": 0.0,
    }
    return results


def run_llm(X_test, X_tr, y_tr, label_names) -> dict[str, dict]:
    """LLM zero-shot and few-shot, with the same number of samples used on validation."""
    results = {}
    for mode in LLM_MODES:
        meta = json.loads((llm.OUT_DIR / mode / "metadata.json").read_text())
        res = llm.predict(X_test, mode, meta["samples"], X_tr, y_tr, label_names)
        results[f"llm_{mode}_shot"] = {
            "meta": meta,
            "y_pred": res["y_pred"],
            "scores": res["scores"],
            "latency": {
                "latency_median_ms": float(np.median(res["latencies"])),
                "latency_p95_ms": float(np.percentile(res["latencies"], 95)),
            },
            "cost_per_1000_usd": float(1000 * np.mean(res["costs"])),
            "answers": res["answers"],
            "served_models": res["served_models"],
            "run_dates": res["run_dates"],
        }
    return results


def to_markdown(rows: list[dict]) -> str:
    """Results table for the README."""
    lines = [
        "| Model | Macro-F1 (95% CI) | Accuracy | Auto-routed | Error when auto-routed "
        "| Cost / 1k msgs | Latency median / p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["macro_f1_ci95"]
        rv = r["review_test"]
        lines.append(
            f"| {r['model']} | {r['macro_f1']:.3f} ({lo:.3f}–{hi:.3f}) | {r['accuracy']:.3f} "
            f"| {rv['coverage']:.1%} | {rv['error_rate']:.1%}"
            f"{'' if r['review_meets_target_on_val'] else ' *'} "
            f"| ${r['cost_per_1000_usd']:.3f} "
            f"| {r['latency']['latency_median_ms']:.2f} / {r['latency']['latency_p95_ms']:.2f} ms |"
        )
    lines.append("")
    lines.append(
        "\\* No threshold reached the 5% error target on validation; the strictest one is used."
    )
    return "\n".join(lines)


def main() -> None:
    train, test, label_names = load_banking77()
    train_idx, _ = load_split()
    X_tr = train["text"].iloc[train_idx].tolist()
    y_tr = train["label"].iloc[train_idx].tolist()
    X_test, y_test = test["text"].tolist(), test["label"].to_numpy()
    X_lat = X_test[:1000]  # latency sample, enough for stable median and p95

    results = {**run_local(X_test, X_lat), **run_llm(X_test, X_tr, y_tr, label_names)}

    ci = bootstrap_macro_f1(y_test, {name: r["y_pred"] for name, r in results.items()})
    rows = []
    for name, r in results.items():
        val_review = r["meta"]["review"]
        rows.append(
            {
                "model": name,
                "model_version": r["meta"]["model_version"],
                **classification_metrics(y_test, r["y_pred"]),
                "macro_f1_ci95": [float(x) for x in np.percentile(ci[name], [2.5, 97.5])],
                "val_macro_f1": r["meta"]["val_metrics"]["macro_f1"],
                "n_invalid": int((r["y_pred"] < 0).sum()),
                "review_val": val_review,
                "review_meets_target_on_val": val_review.get("meets_target", True),
                "review_test": review_on_test(
                    y_test, r["y_pred"], r["scores"], val_review["threshold"]
                ),
                "cost_per_1000_usd": r["cost_per_1000_usd"],
                "latency": r["latency"],
                "worst_classes": worst_classes(y_test, r["y_pred"], label_names),
                **({"served_models": r["served_models"], "run_dates": r["run_dates"]}
                   if "served_models" in r else {}),
            }
        )

    # Paired bootstrap: how often each model beats the best one on the same resampled rows.
    best = max(rows, key=lambda r: r["macro_f1"])["model"]
    for row in rows:
        diff = ci[row["model"]] - ci[best]
        row["diff_vs_best_ci95"] = [float(x) for x in np.percentile(diff, [2.5, 97.5])]

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "results.json").write_text(
        json.dumps({"dataset_revision": REVISION, "n_test": len(y_test), "best": best,
                    "models": rows}, indent=2)
    )
    (REPORT_DIR / "results.md").write_text(to_markdown(rows) + "\n")

    preds = pd.DataFrame({"text": X_test, "true": [label_names[y] for y in y_test]})
    for name, r in results.items():
        preds[f"{name}_pred"] = [label_names[p] if p >= 0 else "INVALID" for p in r["y_pred"]]
        preds[f"{name}_score"] = np.round(r["scores"], 4)
    preds.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    print(to_markdown(rows))
    for row in rows:
        lo, hi = row["diff_vs_best_ci95"]
        print(f"{row['model']:<16} val={row['val_macro_f1']:.3f} test={row['macro_f1']:.3f} "
              f"diff vs {best}: [{lo:+.3f}, {hi:+.3f}] invalid={row['n_invalid']}")
    print(f"saved -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
