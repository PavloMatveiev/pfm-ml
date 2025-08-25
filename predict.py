import argparse
import json
import pickle
from typing import List, Optional

import pandas as pd

MODEL_PATH = "model.pkl"
DEFAULT_ISO_DATETIME = "2025-08-24T09:00:00"
TOP_K = 3


# ---------- Feature helpers ----------

def add_time_features(iso_datetime: str) -> dict:
    """Extract simple time-based features from an ISO 8601 datetime string.

    Args:
        iso_datetime: Datetime in ISO 8601 format (e.g., "2025-08-21T12:00:00").

    Returns:
        dict: A dictionary with:
            - "hour" (int): 0–23.
            - "day_of_week" (int): Monday=0 … Sunday=6.
            - "is_weekend" (int): 1 if Saturday/Sunday else 0.

    Notes:
        If parsing fails, the function falls back to DEFAULT_ISO_DATETIME.
    """
    ts = pd.to_datetime(iso_datetime, errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(DEFAULT_ISO_DATETIME)
    day_of_week = int(ts.dayofweek)
    return {
        "hour": int(ts.hour),
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
    }


def build_feature_row(merchant: str, description: str, amount: float, iso_datetime: str) -> pd.DataFrame:
    """Construct a single-row DataFrame matching the training feature schema.

    Args:
        merchant: Merchant name as plain text.
        description: Transaction description as plain text.
        amount: Transaction amount (float).
        iso_datetime: Datetime in ISO 8601 format.

    Returns:
        pandas.DataFrame: One-row frame with columns
            ["combined_text", "merchant_text", "amount", "hour", "day_of_week", "is_weekend"].
    """
    t = add_time_features(iso_datetime)
    return pd.DataFrame([{
        "combined_text": f"{merchant} {description}".lower(),
        "merchant_text": merchant.lower(),
        "amount": float(amount),
        "hour": t["hour"],
        "day_of_week": t["day_of_week"],
        "is_weekend": t["is_weekend"],
    }])


# ---------- Model I/O ----------

def load_pipeline(model_path: str = MODEL_PATH):
    """Load the sklearn pipeline saved by train.py.

    Args:
        model_path: Path to a pickle created by train.py. It can be either:
            - a dict with key "pipeline" pointing to the Pipeline, or
            - the Pipeline object itself.

    Returns:
        The deserialized sklearn Pipeline (or object stored in the pickle).

    Raises:
        FileNotFoundError: If the file is missing.
        pickle.UnpicklingError: If the pickle is corrupted or incompatible.
        KeyError: If a dict payload is present but lacks "pipeline".
    """
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    return payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload


def get_class_names(pipeline) -> Optional[List[str]]:
    """Retrieve class names from the final classifier inside the pipeline.

    Args:
        pipeline: A fitted sklearn Pipeline with a final step named "classifier".

    Returns:
        A list of class labels (strings) if available; otherwise None.
    """
    if hasattr(pipeline, "named_steps") and "classifier" in pipeline.named_steps:
        clf = pipeline.named_steps["classifier"]
        if hasattr(clf, "classes_"):
            return list(clf.classes_)
    return None


# ---------- CLI / Inference ----------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prediction.

    Returns:
        argparse.Namespace: Parsed arguments with fields:
            merchant, description, amount, iso_datetime, topk, model_path.
    """
    parser = argparse.ArgumentParser(description="Predict transaction category.")
    parser.add_argument("-m", "--merchant", default="Tesco", help="Merchant name")
    parser.add_argument("-d", "--description", default="groceries", help="Transaction description")
    parser.add_argument("-a", "--amount", type=float, default=43.0, help="Transaction amount")
    parser.add_argument("-t", "--time", dest="iso_datetime", default=DEFAULT_ISO_DATETIME, help="ISO datetime")
    parser.add_argument("-k", "--topk", type=int, default=TOP_K, help="How many top classes to show")
    parser.add_argument("-p", "--model-path", default=MODEL_PATH, help="Path to model.pkl")
    return parser.parse_args()


def main() -> None:
    """Entry point: load model, build features from CLI args, predict, print JSON.

    Workflow:
        1) Parse CLI args.
        2) Load the saved sklearn Pipeline.
        3) Build a one-row feature DataFrame.
        4) Predict probabilities (or labels) with the pipeline.
        5) Print a JSON result to stdout.
    """
    args = parse_args()

    pipeline = load_pipeline(args.model_path)
    class_names = get_class_names(pipeline)

    features_row = build_feature_row(
        merchant=args.merchant,
        description=args.description,
        amount=args.amount,
        iso_datetime=args.iso_datetime,
    )

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(features_row)[0]
        order = proba.argsort()[::-1]
        k = min(args.topk, len(proba))
        topk = [
            {"category": str(class_names[i]) if class_names else str(i), "probability": float(proba[i])}
            for i in order[:k]
        ]
        result = {
            "input": {
                "merchant": args.merchant,
                "description": args.description,
                "amount": float(args.amount),
                "date": args.iso_datetime,
            },
            "top1": topk[0],
            "topk": topk,
        }
    else:
        pred = pipeline.predict(features_row)[0]
        result = {
            "input": {
                "merchant": args.merchant,
                "description": args.description,
                "amount": float(args.amount),
                "date": args.iso_datetime,
            },
            "prediction": str(pred),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
