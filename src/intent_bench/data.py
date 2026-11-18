"""Load BANKING77 and build a reproducible, leakage-free train/validation split.

Run ``uv run python -m intent_bench.data`` to write ``artifacts/split.json``.
"""

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold

DATASET_ID = "PolyAI/banking77"
REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"
SEED = 42
SPLIT_PATH = Path("artifacts/split.json")


def normalize(text: str) -> str:
    """Lowercase and collapse all whitespace, used to detect duplicate messages."""
    return " ".join(text.lower().split())


def load_banking77() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (train, test, label_names) with a 'norm' column added."""
    ds = load_dataset(DATASET_ID, revision=REVISION)
    label_names = ds["train"].features["label"].names

    train = ds["train"].to_pandas()
    test = ds["test"].to_pandas()
    for df in (train, test):
        df["norm"] = df["text"].map(normalize)

    return train, test, label_names


def make_split(train: pd.DataFrame, seed: int = SEED) -> tuple[list[int], list[int]]:
    """Return (train_idx, val_idx): ~20% stratified validation, duplicates kept together."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    # Each of the 5 folds holds out ~20% of the groups; we only need the first one
    train_idx, val_idx = next(
        sgkf.split(train["text"], y=train["label"], groups=train["norm"])
    )
    return train_idx.tolist(), val_idx.tolist()


def save_split(
    train_idx: list[int], val_idx: list[int], seed: int = SEED, path: Path = SPLIT_PATH
) -> None:
    """Save split indices with the metadata needed to reproduce them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "seed": seed,
        "n_splits": 5,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
    path.write_text(json.dumps(payload))


def load_split(path: Path = SPLIT_PATH) -> tuple[list[int], list[int]]:
    """Read saved split indices, refusing a split made from another dataset revision."""
    payload = json.loads(path.read_text())
    if payload["revision"] != REVISION:
        raise ValueError(f"Split was built on revision {payload['revision']}, expected {REVISION}")
    return payload["train_idx"], payload["val_idx"]


if __name__ == "__main__":
    train, test, label_names = load_banking77()
    train_idx, val_idx = make_split(train, seed=SEED)
    save_split(train_idx, val_idx, seed=SEED)
    print(f"train={len(train_idx)} val={len(val_idx)} -> {SPLIT_PATH}")
