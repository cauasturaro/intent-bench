"""Classify messages with the DeepSeek API (zero-shot or few-shot).

Run ``uv run python -m intent_bench.llm --mode zero|few``. Needs ``DEEPSEEK_API_KEY`` in ``.env``.
"""

import argparse
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from intent_bench.data import REVISION, SEED, load_banking77, load_split
from intent_bench.evaluate import (
    MAX_AUTO_ERROR_RATE,
    choose_threshold,
    classification_metrics,
    coverage_error_curve,
    worst_classes,
)

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"
# USD per 1M tokens, peak (standard) rates; off-peak is half. Checked 2026-11-17 at
# https://api-docs.deepseek.com/quick_start/pricing
PRICE_PER_M = {"input_cache_hit": 0.006, "input_cache_miss": 0.30, "output": 1.20}

OUT_DIR = Path("artifacts/llm")
CACHE_PATH = OUT_DIR / "cache.jsonl"
FEW_SHOT_PER_INTENT = 3
MAX_WORKERS = 16
INVALID = -1  # label id for responses that are not one of the 77 intents
SAMPLE_TEMPERATURE = 1.0  # for the extra samples that measure how consistent the answer is


def build_system_prompt(label_names: list[str], examples: list[tuple[str, str]]) -> str:
    """Instructions, the allowed labels and optional examples. This prefix is identical for
    every message, so the API's prompt cache bills it at the cheap cache-hit rate."""
    lines = [
        "You classify customer-support messages sent to a bank into exactly one intent.",
        "Answer with the intent name only, copied exactly from the list below "
        "(same spelling, case and punctuation). No explanation, no quotes.",
        "",
        "Intents:",
        *label_names,
    ]
    if examples:
        lines += ["", "Examples (message => intent):"]
        lines += [f"{text} => {label}" for text, label in examples]
    return "\n".join(lines)


def pick_examples(texts: list[str], labels: list[int], label_names: list[str]) -> list[tuple[str, str]]:
    """FEW_SHOT_PER_INTENT training messages per intent, chosen with a fixed seed."""
    rng = random.Random(SEED)
    by_label: dict[int, list[str]] = {}
    for text, label in zip(texts, labels):
        by_label.setdefault(label, []).append(text)
    examples = []
    for label in sorted(by_label):
        for text in rng.sample(by_label[label], FEW_SHOT_PER_INTENT):
            examples.append((text, label_names[label]))
    rng.shuffle(examples)  # avoid grouping by intent, which could bias the model
    return examples


def parse_label(answer: str, label_names: list[str]) -> int:
    """Exact match after trimming whitespace; anything else counts as an invalid answer."""
    try:
        return label_names.index(answer.strip())
    except ValueError:
        return INVALID


def cost_usd(usage: dict) -> float:
    """Price of one call from its token counts."""
    return (
        usage["prompt_cache_hit_tokens"] * PRICE_PER_M["input_cache_hit"]
        + usage["prompt_cache_miss_tokens"] * PRICE_PER_M["input_cache_miss"]
        + usage["completion_tokens"] * PRICE_PER_M["output"]
    ) / 1_000_000


class Cache:
    """Append-only JSONL cache of API responses, keyed by model, prompt and message."""

    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                record = json.loads(line)
                self.data[record["key"]] = record

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, record: dict) -> None:
        with self.lock:
            self.data[record["key"]] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")


def classify(
    client: OpenAI, cache: Cache, system_prompt: str, text: str, sample: int | None = None
) -> dict:
    """One API call (or a cache hit). ``sample=None`` is the prediction, at temperature 0;
    ``sample=i`` is the i-th extra answer at SAMPLE_TEMPERATURE, used only for the score.

    Logprobs are not used as a score: at temperature 0 the API returns 0 (probability 1)
    for every chosen token."""
    temperature = 0 if sample is None else SAMPLE_TEMPERATURE
    key_text = f"{MODEL}\n{system_prompt}\n{text}"
    if sample is not None:
        key_text += f"\nT={temperature} sample={sample}"
    key = hashlib.sha256(key_text.encode()).hexdigest()
    if (cached := cache.get(key)) is not None:
        return {**cached, "from_cache": True}

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        temperature=temperature,
        max_tokens=20,
        extra_body={"thinking": {"type": "disabled"}},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    choice = response.choices[0]
    usage = response.usage.model_dump()
    record = {
        "key": key,
        "model": response.model,
        "answer": choice.message.content or "",
        "usage": {
            "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
            "prompt_cache_miss_tokens": usage.get(
                "prompt_cache_miss_tokens", usage["prompt_tokens"]
            ),
            "completion_tokens": usage["completion_tokens"],
        },
        "latency_ms": latency_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    cache.put(record)
    return {**record, "from_cache": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["zero", "few"], required=True)
    parser.add_argument("--limit", type=int, help="only the first N validation messages (smoke test)")
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="extra answers per message at temperature 1; score = share that agree",
    )
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL, max_retries=5)

    train, _, label_names = load_banking77()
    train_idx, val_idx = load_split()
    X_tr = train["text"].iloc[train_idx].tolist()
    y_tr = train["label"].iloc[train_idx].tolist()
    X_val = train["text"].iloc[val_idx].tolist()[: args.limit]
    y_val = train["label"].iloc[val_idx].tolist()[: args.limit]

    examples = pick_examples(X_tr, y_tr, label_names) if args.mode == "few" else []
    system_prompt = build_system_prompt(label_names, examples)
    cache = Cache()

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        records = list(pool.map(lambda t: classify(client, cache, system_prompt, t), X_val))

    y_pred = np.array([parse_label(r["answer"], label_names) for r in records])

    # Self-consistency score: share of extra samples that agree with the temperature-0 answer.
    # Without samples every score is 1, so no message can be sent to review.
    samples = [[] for _ in X_val]
    if args.samples:
        jobs = [(i, k) for i in range(len(X_val)) for k in range(args.samples)]
        with ThreadPoolExecutor(MAX_WORKERS) as pool:
            done = pool.map(lambda j: classify(client, cache, system_prompt, X_val[j[0]], j[1]), jobs)
            for (i, _), record in zip(jobs, done):
                samples[i].append(record)
    scores = np.array(
        [
            np.mean([parse_label(r["answer"], label_names) == p for r in group]) if group else 1.0
            for p, group in zip(y_pred, samples)
        ]
    )
    n_invalid = int((y_pred == INVALID).sum())

    metrics = classification_metrics(y_val, y_pred)
    curve = coverage_error_curve(y_val, y_pred, scores)
    review = choose_threshold(curve, MAX_AUTO_ERROR_RATE)
    worst = worst_classes(y_val, y_pred, label_names)

    calls = [[r, *group] for r, group in zip(records, samples)]  # all calls per message
    costs = [sum(cost_usd(r["usage"]) for r in group) for group in calls]
    # Calls for one message can run in parallel, so its latency is the slowest of them.
    latencies = [max(r["latency_ms"] for r in group) for group in calls]
    tokens = {
        k: int(sum(r["usage"][k] for group in calls for r in group)) for k in records[0]["usage"]
    }

    print(f"mode={args.mode} n={len(records)} invalid={n_invalid} {metrics}")
    print(f"review threshold (max auto error {MAX_AUTO_ERROR_RATE:.0%}): {review}")
    print(f"cost per 1,000 messages: ${1000 * np.mean(costs):.4f} (peak rates) tokens={tokens}")
    print(f"latency: median={np.median(latencies):.0f} ms p95={np.percentile(latencies, 95):.0f} ms")
    for row in worst:
        print(f"  {row['intent']:<40} f1={row['f1']:.2f}  -> {row['most_confused_with']}")

    out = OUT_DIR / args.mode
    out.mkdir(parents=True, exist_ok=True)
    invalid_answers = sorted({r["answer"] for r, p in zip(records, y_pred) if p == INVALID})
    metadata = {
        "model_version": f"{MODEL}-{args.mode}-shot-v1",
        "score": f"self-consistency over {args.samples} samples at T={SAMPLE_TEMPERATURE}"
        if args.samples
        else "none (always 1)",
        "model": MODEL,
        "served_models": sorted({r["model"] for r in records}),
        "run_dates": sorted({r["created_at"][:10] for group in calls for r in group}),
        "dataset_revision": REVISION,
        "seed": SEED,
        "n_messages": len(records),
        "few_shot_per_intent": FEW_SHOT_PER_INTENT if examples else 0,
        "system_prompt": system_prompt,
        "price_per_m_tokens_peak": PRICE_PER_M,
        "val_metrics": metrics,
        "n_invalid": n_invalid,
        "invalid_answers": invalid_answers,
        "review": {"max_auto_error_rate": MAX_AUTO_ERROR_RATE, **review},
        "cost_per_1000_usd": float(1000 * np.mean(costs)),
        "tokens": tokens,
        "latency": {
            "latency_median_ms": float(np.median(latencies)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
        },
        "worst_classes": worst,
        "coverage_error_curve": curve,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
