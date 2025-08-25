from __future__ import annotations
import pandas as pd


def extract_time_features(datetime_series: pd.Series) -> pd.DataFrame:
    """Extract hour / day_of_week / is_weekend from ISO datetime strings."""
    dt = pd.to_datetime(datetime_series)
    return pd.DataFrame(
        {
            "hour": dt.dt.hour,
            "day_of_week": dt.dt.dayofweek,
            "is_weekend": (dt.dt.dayofweek >= 5).astype(int),
        }
    )


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add text views expected by the pipeline."""
    out = df.copy()
    out["combined_text"] = (out["merchant"].fillna("") + " " + out["description"].fillna("")).str.lower()
    out["merchant_text"] = out["merchant"].fillna("").str.lower()
    return out


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all required features to the raw transactions DataFrame."""
    time_df = extract_time_features(df["date"])
    return pd.concat([df, time_df], axis=1)


def get_feature_matrix_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) in the column order expected by the ColumnTransformer."""
    features_df = df[["combined_text", "merchant_text", "amount", "hour", "day_of_week", "is_weekend"]]
    labels = df["label"]
    return features_df, labels
