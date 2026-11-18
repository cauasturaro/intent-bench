"""API checks for the local models (the LLM is skipped: it needs network and an API key)."""

import pytest
from fastapi.testclient import TestClient

from intent_bench import baseline, transformer
from intent_bench.api import app

client = TestClient(app)


@pytest.mark.parametrize(
    ("model", "model_dir"),
    [("tfidf_logreg", baseline.MODEL_DIR), ("distilbert", transformer.MODEL_DIR)],
)
def test_predict_local_models(model, model_dir):
    if not (model_dir / "metadata.json").exists():
        pytest.skip(f"{model} is not trained yet")
    response = client.post("/predict", json={"text": "When will my new card arrive?", "model": model})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "card_arrival"
    assert 0 <= body["score"] <= 1
    assert isinstance(body["needs_review"], bool)
    assert body["model_version"]


def test_rejects_empty_text_and_unknown_model():
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "hi", "model": "gpt"}).status_code == 422


def test_health():
    assert client.get("/health").json() == {"status": "ok"}
