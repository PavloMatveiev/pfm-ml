# PFM-ML — Transaction Category Classifier (FastAPI + scikit-learn + Docker)

A compact ML microservice that classifies bank transactions into categories.
It synthesizes training data, trains a TF-IDF → Logistic Regression pipeline,
saves a fitted model, and exposes a REST API via FastAPI.

**Stack:** Python 3.11, scikit-learn, pandas, FastAPI, Uvicorn, Docker/Compose  
**Features:** 
- Text: TF-IDF on words for `combined_text` (merchant + description)
- Text: TF-IDF on character n-grams for `merchant_text` (robust to typos/brand variants)
- Numeric: `amount`, `hour`, `day_of_week`, `is_weekend` (scaled)

---

## Quick Start (Docker)

> The image **builds the model during image build** (`RUN python train.py`), so the API is ready the moment the container starts.

```bash
docker compose build --no-cache
docker compose up -d
```

Check health:

```bash
curl -s http://localhost:8000/healthz | jq
```

Predict (bash):

```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"merchant":"McCafe","description":"Lunch","amount":8.99,"iso_datetime":"2025-08-21T12:00:00","topk":3}' | jq
```

Predict (PowerShell):

```powershell
Invoke-RestMethod -Uri 'http://localhost:8000/healthz'

$body = @{merchant="McCafe";description="Lunch";amount=8.99;iso_datetime="2025-08-21T12:00:00";topk=3} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://localhost:8000/predict' -Method POST -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6
```

Stop:

```bash
docker compose down
```

---

## Local Development (no Docker)

```bash
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
source .venv/bin/activate

pip install -r requirements.txt

# Train and save model.pkl
python train.py

# Run API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

CLI prediction without API:
```bash
python predict.py -m "McCafe" -d "Lunch" -a 8.99 -t "2025-08-21T12:00:00"
```

---

## Project Structure

```
.
├─ app/
│  └─ main.py                # FastAPI app: /healthz and /predict
├─ ml/
│  ├─ data.py                # synthetic transaction generation
│  ├─ features.py            # feature building (time, text)
│  ├─ model.py               # build_pipeline(): ColumnTransformer + LogisticRegression
│  └─ io.py                  # saving artifacts
├─ utils/
│  ├─ amount.py              # amount generation per category
│  └─ date_and_time.py       # realistic dates/hours per category
├─ settings.py               # single source of truth (categories, vocab, ranges, hyperparams)
├─ train.py                  # trains and saves model.pkl
├─ predict.py                # CLI prediction
├─ requirements.txt
├─ Dockerfile
└─ docker-compose.yml
```

---

## How It Works (high level)

1. `train.py` generates synthetic transactions and builds features:
   - `combined_text` (merchant + description) → **word TF-IDF** (1–2-grams)
   - `merchant_text` → **character TF-IDF** (3–5-grams, `char_wb`) — robust to misspellings
   - numeric: `amount`, `hour`, `day_of_week`, `is_weekend` → `StandardScaler(with_mean=False)`
2. `ml/model.py::build_pipeline()` assembles:
   - `ColumnTransformer` with the three branches above
   - `LogisticRegression(multi_class="multinomial", solver="saga", class_weight="balanced")`
3. Artifacts are saved as `model.pkl` containing `{"pipeline": <fitted_pipeline>, "labels": ...}`.
4. `app/main.py` loads the pipeline on startup and serves predictions at `/predict`.

---

## API

### `GET /healthz`
Health/readiness probe.

**Response example**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/model.pkl"
}
```

### `POST /predict`
Runs inference and returns top-k classes (or a single label if `predict_proba` is unavailable).

**Request**
```json
{
  "merchant": "McCafe",
  "description": "Lunch",
  "amount": 8.99,
  "iso_datetime": "2025-08-21T12:00:00",
  "topk": 3
}
```

**Response (example)**
```json
{
  "input": {
    "merchant": "McCafe",
    "description": "Lunch",
    "amount": 8.99,
    "iso_datetime": "2025-08-21T12:00:00",
    "topk": 3
  },
  "top1": {"category": "Dining & Coffee", "probability": 0.71},
  "topk": [
    {"category": "Dining & Coffee", "probability": 0.71},
    {"category": "Groceries", "probability": 0.16},
    {"category": "Other", "probability": 0.08}
  ]
}
```

**Validation**
- `merchant`, `description`: non-empty strings
- `amount`: float
- `iso_datetime`: ISO-8601 datetime string; fallback is used on parse failure
- `topk`: integer in [1, 20]

---

## Configuration

Most configuration lives in `settings.py`:
- `categories`, `vocab` (merchant/desc lexicon), `amounts` and defaults
- time synthesis (`base_date`, per-category hour choices, day offsets)
- model hyperparameters (TF-IDF ranges, LogisticRegression params)
- `seed` for reproducibility

Env variables (used by API / Docker):
- `MODEL_PATH` (default `/app/model.pkl`)
- `PORT` (default `8000`)

---

## Retraining

Whenever you change `settings.py` / generation / features / model:
```bash
docker compose build --no-cache
docker compose up -d
```
The new image will re-run `python train.py` at build time and bake a fresh `model.pkl`.

---

## Why TF-IDF + Logistic Regression?

- Excellent baseline for short texts with fast inference
- Character n-grams capture brand variants and typos
- Multinomial logistic regression yields calibrated probabilities
- `class_weight="balanced"` mitigates class imbalance in synthetic data

---

## License

MIT.
