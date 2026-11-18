"""Fine-tune DistilBERT on BANKING77 and pick the best learning rate and epoch on validation.

Run ``uv run python -m intent_bench.transformer`` to write ``artifacts/distilbert/``.
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

from intent_bench.data import REVISION, SEED, load_banking77, load_split
from intent_bench.evaluate import (
    MAX_AUTO_ERROR_RATE,
    choose_threshold,
    classification_metrics,
    coverage_error_curve,
    measure_latency,
    worst_classes,
)

BASE_MODEL = "distilbert-base-uncased"
MODEL_DIR = Path("artifacts/distilbert")
MODEL_VERSION = "distilbert-ft-v1"

MAX_LENGTH = 64  # p99 is 43 words (~56 tokens), so almost nothing is cut
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATES = [3e-5, 5e-5]
WARMUP_SHARE = 0.1


def pick_device() -> torch.device:
    """Apple GPU (MPS) when available, otherwise CPU. (I use a macbook)"""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def predict_proba(model, tokenizer, texts: list[str], device: torch.device) -> np.ndarray:
    """Class probabilities for a list of texts, computed in batches."""
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), 128):
            batch = tokenizer(
                texts[start : start + 128],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            chunks.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
    return np.concatenate(chunks)


def train_one(lr, tokenizer, X_tr, y_tr, X_val, y_val, n_labels, device):
    """Fine-tune once with a given learning rate; return the best epoch's weights and history."""
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=n_labels)
    model.to(device)

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts), truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
        )
        enc["labels"] = torch.tensor(labels)
        return enc

    loader = DataLoader(
        list(zip(X_tr, y_tr)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = EPOCHS * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_SHARE * total_steps), total_steps
    )

    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start, losses = time.perf_counter(), []
        for batch in loader:
            batch = batch.to(device)
            loss = model(**batch).loss  # cross-entropy over the 77 intents
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        y_pred = predict_proba(model, tokenizer, X_val, device).argmax(axis=1)
        metrics = classification_metrics(y_val, y_pred)
        row = {
            "lr": lr,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "seconds": time.perf_counter() - start,
            **metrics,
        }
        history.append(row)
        print(
            f"lr={lr:.0e} epoch={epoch} loss={row['train_loss']:.3f} "
            f"val_f1={metrics['macro_f1']:.4f} ({row['seconds']:.0f}s)"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    return best_state, history


def main() -> None:
    train, _, label_names = load_banking77()
    train_idx, val_idx = load_split()
    X_tr = train["text"].iloc[train_idx].tolist()
    y_tr = train["label"].iloc[train_idx].tolist()
    X_val = train["text"].iloc[val_idx].tolist()
    y_val = train["label"].iloc[val_idx].tolist()

    device = pick_device()
    print(f"device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    history, best_state, best = [], None, None
    for lr in LEARNING_RATES:
        state, runs = train_one(lr, tokenizer, X_tr, y_tr, X_val, y_val, len(label_names), device)
        history.extend(runs)
        top = max(runs, key=lambda r: r["macro_f1"])
        if best is None or top["macro_f1"] > best["macro_f1"]:
            best, best_state = top, state
    print(f"best: {best}")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={name: i for i, name in enumerate(label_names)},
    )
    model.load_state_dict(best_state)
    model.to(device)

    proba = predict_proba(model, tokenizer, X_val, device)
    y_pred, scores = proba.argmax(axis=1), proba.max(axis=1)
    curve = coverage_error_curve(y_val, y_pred, scores)
    review = choose_threshold(curve, MAX_AUTO_ERROR_RATE)
    print(f"review threshold (max auto error {MAX_AUTO_ERROR_RATE:.0%}): {review}")

    worst = worst_classes(y_val, y_pred, label_names)
    print("worst classes on validation:")
    for row in worst:
        print(f"  {row['intent']:<40} f1={row['f1']:.2f}  -> {row['most_confused_with']}")

    # Latency is measured on CPU, one message at a time, like the API would serve it.
    cpu = torch.device("cpu")
    model.to(cpu)
    latency = measure_latency(lambda t: predict_proba(model, tokenizer, [t], cpu), X_val)
    print(f"latency (CPU, 1 message at a time): {latency}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    metadata = {
        "model_version": MODEL_VERSION,
        "base_model": BASE_MODEL,
        "dataset_revision": REVISION,
        "seed": SEED,
        "label_names": label_names,
        "train_device": str(device),
        "params": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rates": LEARNING_RATES,
            "warmup_share": WARMUP_SHARE,
        },
        "best": {"lr": best["lr"], "epoch": best["epoch"]},
        "val_metrics": {k: best[k] for k in ("macro_f1", "accuracy")},
        "review": {"max_auto_error_rate": MAX_AUTO_ERROR_RATE, **review},
        "latency": latency,
        "worst_classes": worst,
        "history": history,
        "coverage_error_curve": curve,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
