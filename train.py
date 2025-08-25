"""Train a transaction classifier using synthetic data and central settings."""

from __future__ import annotations
import random
from typing import Dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from settings import SETTINGS
from ml.data import synthesize_dataset
from ml.features import add_text_features, build_feature_dataframe, get_feature_matrix_and_labels
from ml.model import build_pipeline
from ml.io import save_artifacts


def main() -> None:
    # Reproducibility
    random.seed(SETTINGS.seed)

    # 1) Synthesize data
    per_category_overrides: Dict[str, int] = {"Other": 50}
    raw_df = synthesize_dataset(
        default_samples_per_category=60,
        per_category_overrides=per_category_overrides,
        seed=SETTINGS.seed,
    )

    # 2) Features
    df_with_time = build_feature_dataframe(raw_df)
    df_with_text = add_text_features(df_with_time)
    X, y = get_feature_matrix_and_labels(df_with_text)

    # 3) Split (safe stratification)
    label_counts = y.value_counts()
    can_stratify = (
        label_counts.min() >= 2
        and (label_counts * SETTINGS.model.test_size).ge(1).all()
        and (label_counts * (1 - SETTINGS.model.test_size)).ge(1).all()
    )
    stratify_arg = y if can_stratify else None
    if not can_stratify:
        print("[warn] Stratification disabled: some classes are too small for the chosen split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=SETTINGS.model.test_size,
        random_state=SETTINGS.model.random_state,
        stratify=stratify_arg,
    )

    # 4) Build pipeline & train
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 5) Report
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, labels=SETTINGS.categories, zero_division=0))

    # 6) Save artifacts
    save_artifacts(pipeline, SETTINGS.categories)
    print("Saved: model.pkl")


if __name__ == "__main__":
    main()
