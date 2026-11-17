# intent-bench

## What this is

A classifier that reads a customer-support message from a bank (in English), predicts the customer's **intent** (e.g. `card_arrival` for "When will my card arrive?"), and decides whether the message can be **routed automatically** or should go to **human review**.

It classifies messages only: it does not answer customers or perform banking operations.

The project tries to answer one question: **when is it worth training a small model instead of calling an LLM?** To answer it, three approaches are compared on the same data:

| # | Approach | Training |
|---|----------|----------|
| 1 | TF-IDF + Logistic Regression | Trained on CPU (baseline) |
| 2 | DistilBERT | Fine-tuned |
| 3 | LLM via API (DeepSeek) | None (zero-shot and few-shot prompts) |

They are compared on **quality** (macro-F1 is the main metric), **cost per 1,000 messages**, and **latency**. No winner is assumed in advance.

The models are trained and evaluated on [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) (PolyAI, CC BY 4.0): about 13k English banking messages labeled with 77 intents. Dataset revision used: _to be recorded_.

## Stack

| Area | Tools |
|------|-------|
| Language and environment | Python 3.12, [uv](https://docs.astral.sh/uv/) (dependencies pinned in `uv.lock`) |
| Data | Hugging Face `datasets`, `pandas` |
| Baseline model | `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`) |
| Transformer model *(planned)* | PyTorch, Hugging Face `transformers` (`distilbert-base-uncased`) |
| LLM classifier *(planned)* | [DeepSeek API](https://api-docs.deepseek.com/) (OpenAI-compatible), called through the `openai` Python SDK; exact model ID and run dates will be recorded |
| Serving *(planned)* | FastAPI (`POST /predict`) |
| Testing and exploration | `pytest`, Jupyter |

## How to run

### 1. Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv` on macOS). uv installs Python 3.12 automatically if needed.

```bash
git clone https://github.com/cauasturaro/intent-bench.git
cd intent-bench
uv sync              # creates .venv with the exact pinned versions
```

### 2. Run the tests

```bash
uv run pytest
```

### 3. Explore the data

```bash
uv run jupyter lab   # notebooks live in notebooks/
```

### 4. Train, evaluate, and serve *(planned)*

The commands to prepare the data split, train the three models, run the evaluation, and start the API will be added here as each stage is implemented.

## Repository layout

```
src/intent_bench/   reusable code (data, baseline, transformer, llm, evaluate, api)
notebooks/          exploration
tests/              tests
reports/            evaluation report and plots
notes/              study notes per stage
artifacts/          trained models and data splits (not tracked by Git)
```

## Limitations

Results on BANKING77 say nothing about performance in other languages or other customer-support domains. The model score is not a guarantee of correctness.

## License and attribution

Dataset: BANKING77 by PolyAI, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
