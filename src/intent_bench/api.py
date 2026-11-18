"""HTTP API: classify one message and say whether it needs human review.

Run ``uv run uvicorn intent_bench.api:app`` and POST to ``/predict``.
"""

import json
from functools import cache
from typing import Literal

import joblib
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from intent_bench import baseline, llm, transformer
from intent_bench.data import load_banking77, load_split

ModelName = Literal["tfidf_logreg", "distilbert", "llm_few_shot"]

app = FastAPI(title="intent-bench", description="Banking intent classifier with human-review routing.")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000, examples=["When will my new card arrive?"])
    model: ModelName = "distilbert"


class PredictResponse(BaseModel):
    intent: str | None  # None when the LLM answered with something outside the 77 intents
    score: float
    needs_review: bool
    model_version: str


def read_metadata(path) -> dict:
    """Model version, label names and the review threshold chosen on validation."""
    return json.loads((path / "metadata.json").read_text())


@cache
def load_tfidf_logreg():
    """Load the baseline once; returns a text -> (label id, score) function and its metadata."""
    model = joblib.load(baseline.MODEL_DIR / "model.joblib")

    def predict(text: str) -> tuple[int, float]:
        proba = model.predict_proba([text])[0]
        return int(proba.argmax()), float(proba.max())

    return predict, read_metadata(baseline.MODEL_DIR)


@cache
def load_distilbert():
    """Load the fine-tuned DistilBERT once, on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(transformer.MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(transformer.MODEL_DIR)
    cpu = torch.device("cpu")

    def predict(text: str) -> tuple[int, float]:
        proba = transformer.predict_proba(model, tokenizer, [text], cpu)[0]
        return int(proba.argmax()), float(proba.max())

    return predict, read_metadata(transformer.MODEL_DIR)


@cache
def load_llm_few_shot():
    """Few-shot DeepSeek with the same examples and number of samples used in evaluation."""
    meta = read_metadata(llm.OUT_DIR / "few")
    train, _, label_names = load_banking77()
    train_idx, _ = load_split()
    X_tr = train["text"].iloc[train_idx].tolist()
    y_tr = train["label"].iloc[train_idx].tolist()

    def predict(text: str) -> tuple[int, float]:
        res = llm.predict([text], "few", meta["samples"], X_tr, y_tr, label_names)
        return int(res["y_pred"][0]), float(res["scores"][0])

    return predict, {**meta, "label_names": label_names}


LOADERS = {
    "tfidf_logreg": load_tfidf_logreg,
    "distilbert": load_distilbert,
    "llm_few_shot": load_llm_few_shot,
}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the intent; send to review when the score is below the model's threshold."""
    predict_fn, meta = LOADERS[request.model]()
    label, score = predict_fn(request.text)
    valid = label >= 0
    return PredictResponse(
        intent=meta["label_names"][label] if valid else None,
        score=round(score, 4),
        needs_review=not valid or score < meta["review"]["threshold"],
        model_version=meta["model_version"],
    )
