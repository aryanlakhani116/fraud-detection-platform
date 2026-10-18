# Fraud Detection Platform

**A production-structured fraud detection system combining classical ML and deep learning, served via FastAPI with a live demo frontend.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-brightgreen.svg)](https://xgboost.ai/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

---

## Table of Contents

- [Problem](#problem)
- [Architecture](#architecture)
- [Data](#data)
- [Design Decisions](#design-decisions)
- [Project Structure](#project-structure)
- [Setup & Running](#setup--running)
- [API Reference](#api-reference)
- [Testing](#testing)
- [What I'd Improve](#what-id-improve)
- [License](#license)

---

## Problem

Fraud detection in production is never one model reading one signal — real systems reason over multiple, independent lines of evidence before flagging a transaction. This project builds the first, fully-working piece of such a system: a **transaction risk scoring API** that evaluates incoming transactions using **two independently-trained models** — a gradient-boosted tree ensemble (XGBoost) and a feedforward neural network built from scratch in PyTorch — trained on identical, honestly-split, real-world data.

The goal was not to chase a leaderboard score. It was to build something that behaves the way a real fraud system has to: trained without leaking the future into the past, evaluated with metrics that mean something on a 96.5%-imbalanced dataset, and served through an API architecture that separates ML logic from HTTP concerns cleanly enough to swap, test, or extend either side independently.

**Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle) — ~590,000 real, anonymized transactions, ~3.5% labeled fraud.

---

## Architecture

### System overview

```mermaid
flowchart LR
    subgraph Data["Data Layer"]
        A[("Raw CSV\ndata/raw/")] --> B["load_data.py"]
        B --> C["preprocess.py\nTabularPreprocessor"]
    end

    subgraph Training["Training (run occasionally)"]
        C --> D["train_classical_ml.py\nXGBoost"]
        C --> E["train_ann.py\nPyTorch ANN"]
        D --> F[("models/\nxgb_model.joblib")]
        E --> G[("models/\nann_model.pt")]
        C --> H[("data/processed/\npreprocessor.joblib")]
    end

    subgraph Inference["Inference (runs per-request)"]
        F --> I["tabular_predictor.py\nTabularFraudPredictor"]
        G --> I
        H --> I
    end

    subgraph API["FastAPI Backend"]
        I -->|loaded once via lifespan| J["app.state.predictor"]
        K["POST /predict/transaction"] --> L["fraud_service.py"]
        L --> J
    end

    subgraph UI["Frontend"]
        M["index.html + app.js"] -->|fetch POST| K
    end

    style Data fill:#1a1d27,stroke:#5b8cff,color:#e8e9ed
    style Training fill:#1a1d27,stroke:#f5b942,color:#e8e9ed
    style Inference fill:#1a1d27,stroke:#3ecf8e,color:#e8e9ed
    style API fill:#1a1d27,stroke:#ff5c5c,color:#e8e9ed
    style UI fill:#1a1d27,stroke:#9aa0ad,color:#e8e9ed
```

### Request lifecycle

```mermaid
sequenceDiagram
    participant U as Browser (frontend)
    participant F as FastAPI route
    participant S as fraud_service
    participant P as TabularFraudPredictor
    participant M as XGBoost / ANN

    U->>F: POST /predict/transaction/?model=xgboost
    Note over F: Pydantic validates TransactionRequest<br/>(gt=0 on amount, required fields)
    F->>S: predict_transaction(payload, predictor, model_choice)
    S->>P: predictor.predict_xgboost(transaction_dict)
    P->>P: preprocessor.transform(df)<br/>(same fit learned at training time)
    P->>M: model.predict_proba(X)
    M-->>P: probability
    P-->>S: {fraud_probability, risk_bucket, model_version}
    S-->>F: TransactionResponse
    F-->>U: 200 OK + JSON
```

**Key property:** the model and preprocessor are loaded **once**, at server startup (via FastAPI's `lifespan`), and reused across every request — not reloaded per-call. See [`app/main.py`](app/main.py).

---

## Data

| | |
|---|---|
| Source | IEEE-CIS Fraud Detection (Kaggle) |
| Rows | ~590,540 |
| Columns (raw) | ~394 |
| Fraud rate | ~3.5% |
| Columns dropped (>90% missing) | see `TabularPreprocessor.cols_to_drop_` after fitting |
| Split method | **Time-based**, not random (see [Design Decisions](#design-decisions)) |

### Class imbalance, visualized

```mermaid
pie showData
    title Transaction Label Distribution (~590K rows)
    "Not Fraud (96.5%)" : 569877
    "Fraud (3.5%)" : 20663
```

A classifier that always predicts "not fraud" scores **96.5% accuracy** while catching zero fraud — this is the reason accuracy is never used as the primary metric anywhere in this project 

---

## Design Decisions

Each decision below follows the same shape: **what was chosen → why → what it cost.**

### 1. Time-based train/test split, not random

The dataset carries a real time column, `TransactionDT`. A random split would let the model train on transactions that occurred chronologically *after* some of its own test transactions — leaking information about the future into training in a way that cannot happen in real deployment, where a model only ever sees the past.

```mermaid
flowchart LR
    subgraph Wrong["❌ Random Split"]
        direction LR
        A1["T1"] --- A2["T2"] --- A3["T3"] --- A4["T4"] --- A5["T5"]
        A1:::train
        A3:::train
        A5:::train
        A2:::test
        A4:::test
    end
    subgraph Right["✅ Time-Based Split"]
        direction LR
        B1["T1"] --- B2["T2"] --- B3["T3"] --- B4["T4"] --- B5["T5"]
        B1:::train
        B2:::train
        B3:::train
        B4:::test
        B5:::test
    end
    classDef train fill:#3ecf8e,color:#0f1117
    classDef test fill:#ff5c5c,color:#0f1117
```

**Cost:** lost the ability to force identical train/test fraud rates via stratification — chronological order decides the split, not balance.

### 2. One shared `TabularPreprocessor` class for both models

XGBoost and the ANN train on **identically** cleaned, encoded, and scaled data — fit once on training data only, applied via `.transform()` to everything downstream (test data and live predictions alike). This makes the PR-AUC comparison in [Results](#results) an honest, apples-to-apples comparison rather than two separately-prepared pipelines that happen to be compared.

### 3. `fit`/`transform` discipline enforced structurally, not just by convention

Every learned statistic (column drop list, medians, one-hot columns, scaler mean/std) is computed **only** inside `fit()`, from whatever data it's given, and stored on `self`. `transform()` never recomputes anything — it only applies what was already learned. This is tested explicitly (`tests/test_preprocess.py::test_transform_does_not_use_test_data_statistics`), not just assumed.

### 4. PR-AUC over accuracy as the primary metric

Covered in [Results](#results) — worth repeating here because it drove every evaluation decision in the project, not just the final number reported.

### 5. Class-weighted loss over SMOTE for the ANN

Tried both. `BCEWithLogitsLoss(pos_weight=...)` performed comparably to SMOTE-based oversampling, without the risk SMOTE carries of generating unrealistic synthetic fraud examples by interpolating between real ones in feature space.

### 6. `src/` never imports from `app/`

```mermaid
flowchart TD
    app["app/ (FastAPI, HTTP)"] -->|imports| src["src/ (pure ML logic)"]
    src -.->|never imports| app
    style src fill:#1a1d27,stroke:#3ecf8e,color:#e8e9ed
    style app fill:#1a1d27,stroke:#5b8cff,color:#e8e9ed
```

All ML logic is framework-agnostic — every `src/` module is callable, testable, and reusable from a plain script, a different web framework, or a notebook, with zero dependency on FastAPI ever being installed.

### 7. PyTorch model saved as a state dict + architecture metadata, not a single object

Unlike `joblib.dump(xgb_model, ...)`, which serializes the complete fitted object, PyTorch's `state_dict()` only captures learned weights — not the architecture that produced them. `train_ann.py` saves `{model_state_dict, input_dim, hidden_sizes, dropout}` together; `tabular_predictor.py` rebuilds an empty `FraudANN` with those exact dimensions before loading the weights into it. Saving only the weights without the shape metadata would make the checkpoint unusable.

---

## Project Structure

```
fraud-detection-platform/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                     # original Kaggle CSV — never modified by code
│   └── processed/                # generated artifacts (preprocessor.joblib)
│
├── models/
│   └── tabular/                  # trained model files (xgb_model.joblib, ann_model.pt)
│
├── src/                           # pure ML logic — zero FastAPI imports
│   ├── config.py                   # single source of truth for paths & settings
│   ├── data/
│   │   ├── load_data.py              # raw CSV → DataFrame
│   │   └── preprocess.py              # TabularPreprocessor (fit/transform), time_based_split
│   ├── training/
│   │   ├── train_classical_ml.py       # trains & saves XGBoost
│   │   ├── ann_model.py                 # FraudANN architecture (nn.Module)
│   │   └── train_ann.py                  # training loop, saves ANN checkpoint
│   └── inference/
│       └── tabular_predictor.py           # loads both models once, serves predictions
│
├── app/                             # HTTP layer — thin, delegates to src/
│   ├── main.py                        # FastAPI app, lifespan model loading
│   ├── schemas/transaction.py          # Pydantic request/response models
│   ├── routes/transaction.py            # POST /predict/transaction/
│   └── services/fraud_service.py         # bridges routes → src/inference
│
├── frontend/                        # plain HTML/CSS/JS — talks to the API over fetch()
│   ├── index.html
│   ├── css/styles.css
│   └── js/
│       ├── api.js                      # only file that calls fetch()
│       └── app.js                       # only file that touches the DOM
│
└── tests/                           # mirrors src/ and app/ structure
    ├── test_preprocess.py
    ├── test_predictor.py
    └── test_api.py
```

**The organizing principle:** files are grouped by **role** (data / training / inference / serving / UI), not by feature. Every file does exactly one job; the name tells you which.

---

## Setup & Running

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd fraud-detection-platform

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

Download `train_transaction.csv` from the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection/data) and place it at:

```
data/raw/train_transaction.csv
```

```bash
# 4. Train both models (saves to models/tabular/ and data/processed/)
python -m src.training.train_classical_ml
python -m src.training.train_ann

# 5. Run the API
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API docs, or open `frontend/index.html` directly in a browser for the live demo (backend must be running).

```bash
# 6. Run the test suite
pytest tests/ -v
```

---

## Results

### 1. **XG-Boost Results**
![XGBoost Result](image.png)

### 2. **ANN Results**
![alt text](image-1.png)
![alt text](image-2.png)


---

## API Reference

### `POST /predict/transaction/`

**Query parameters**

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"xgboost"` | `"xgboost"` or `"ann"` |

**Request body**

```json
{
  "TransactionAmt": 125.50,
  "ProductCD": "W",
  "card4": "visa",
  "card6": "credit",
  "P_emaildomain": "gmail.com"
}
```

**Response `200 OK`**

```json
{
  "fraud_probability": 0.0234,
  "risk_bucket": "low",
  "model_version": "xgboost_v1"
}
```

**Response `422 Unprocessable Entity`** — returned automatically by Pydantic validation (e.g., `TransactionAmt <= 0`) before the route body ever executes.

### `GET /health`

Returns `{"status": "ok"}` — liveness check.

---

## Testing

```mermaid
flowchart TD
    A["tests/test_preprocess.py"] -->|5 tests| B["TabularPreprocessor\nfit/transform correctness,\nno-leakage, save/load"]
    C["tests/test_predictor.py"] -->|fake XGBoost model| D["TabularFraudPredictor\nprediction formatting"]
    E["tests/test_api.py"] -->|TestClient + fake predictor| F["FastAPI routing,\nPydantic validation"]
```

Tests use **fake/stand-in models** (not the real trained XGBoost/ANN) so the suite runs in milliseconds and passes on a fresh clone, before any training has happened. Real model correctness is validated by the PR-AUC reported in [Results](#results), not by the test suite — the test suite validates *pipeline* correctness (no leakage, correct shapes, correct HTTP behavior), which is a different and equally important kind of correctness.

```bash
pytest tests/ -v
```

---

