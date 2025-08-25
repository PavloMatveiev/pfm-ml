import os
import pickle
from typing import Annotated, List, Optional, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictFloat, StringConstraints
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

DEFAULT_ISO_DATETIME = "2025-08-24T09:00:00"
DEFAULT_TOPK = 3
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

app = FastAPI(title="PFM-ML Inference API", version="1.0.0")

# -------------------- Schemas --------------------

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TopK = Annotated[int, Field(ge=1, le=20)]

class PredictRequest(BaseModel):
    """Input payload for /predict."""
    merchant: NonEmptyStr = Field(..., description="Merchant name")
    description: NonEmptyStr = Field(..., description="Transaction description")
    amount: StrictFloat = Field(..., description="Transaction amount")
    iso_datetime: str = Field(DEFAULT_ISO_DATETIME, description="ISO datetime")
    topk: TopK = Field(DEFAULT_TOPK, description="How many top classes to return")

class TopKItem(BaseModel):
    """One item of the top-k ranked output."""
    category: str
    probability: float

class PredictResponse(BaseModel):
    """Response for /predict (probabilities or fallback label)."""
    input: PredictRequest
    top1: Optional[TopKItem] = None
    topk: Optional[List[TopKItem]] = None
    prediction: Optional[str] = None  # fallback if model has no predict_proba

class Healthz(BaseModel):
    """Health/readiness response."""
    status: Literal["ok"]
    model_loaded: bool
    model_path: str

# -------------------- Model loading --------------------

_pipeline: Optional[Pipeline] = None
_class_names: Optional[List[str]] = None

def _add_time_features(iso_datetime: str) -> dict:
    """Turn an ISO datetime into simple time features."""
    ts = pd.to_datetime(iso_datetime, errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(DEFAULT_ISO_DATETIME)
    day_of_week = int(ts.dayofweek)
    return {
        "hour": int(ts.hour),
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
    }

def _build_feature_row(merchant: str, description: str, amount: float, iso_datetime: str) -> pd.DataFrame:
    """Construct a single-row DataFrame matching the training schema."""
    t = _add_time_features(iso_datetime)
    return pd.DataFrame([{
        "combined_text": f"{merchant} {description}".lower(),
        "merchant_text": merchant.lower(),
        "amount": float(amount),
        "hour": t["hour"],
        "day_of_week": t["day_of_week"],
        "is_weekend": t["is_weekend"],
    }])

@app.on_event("startup")
def load_model() -> None:
    """Load sklearn pipeline from MODEL_PATH and cache class names."""
    global _pipeline, _class_names
    with open(DEFAULT_MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    _pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload

    # Try to get classes_ from final classifier
    _class_names = None
    if hasattr(_pipeline, "named_steps") and "classifier" in _pipeline.named_steps:
        clf = _pipeline.named_steps["classifier"]
        if hasattr(clf, "classes_"):
            _class_names = list(clf.classes_)
    elif hasattr(_pipeline, "classes_"):
        _class_names = list(_pipeline.classes_) 

# -------------------- Routes --------------------

@app.get("/healthz", response_model=Healthz)
def healthz() -> Healthz:
    """Liveness/readiness probe."""
    return Healthz(status="ok", model_loaded=bool(_pipeline), model_path=DEFAULT_MODEL_PATH)

@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict(req: PredictRequest) -> PredictResponse:
    """Run inference and return top-k probabilities (or a single label if no proba)."""
    if _pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    X = _build_feature_row(req.merchant, req.description, req.amount, req.iso_datetime)

    try:
        if hasattr(_pipeline, "predict_proba"):
            proba = _pipeline.predict_proba(X)[0]
            order = proba.argsort()[::-1]
            k = min(req.topk, len(proba))
            items = [
                TopKItem(
                    category=str(_class_names[i]) if _class_names else str(i),
                    probability=float(proba[i]),
                )
                for i in order[:k]
            ]
            return PredictResponse(input=req, top1=items[0], topk=items)

        # Fallback: plain predict
        label = _pipeline.predict(X)[0]
        return PredictResponse(input=req, prediction=str(label))
    except NotFittedError:
        # Happens only if a raw (unfitted) pipeline was loaded.
        raise HTTPException(
            status_code=503,
            detail="Model is not fitted. Rebuild the image to retrain the model (docker compose build --no-cache).",
        )
