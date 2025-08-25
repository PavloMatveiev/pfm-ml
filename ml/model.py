"""Model factory: build a sklearn Pipeline compatible with our features."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from settings import SETTINGS


def build_pipeline() -> Pipeline:
    """Create the text+numeric pipeline used for training and inference.

    Columns expected (in this order):
        - combined_text (str): merchant + description, lowercased
        - merchant_text (str): merchant only, lowercased (char n-grams)
        - amount (float)
        - hour (int)
        - day_of_week (int)
        - is_weekend (int)
    """
    # Text vectorizers
    word_tfidf = TfidfVectorizer(
        min_df=SETTINGS.model.word_tfidf_min_df,
        ngram_range=SETTINGS.model.word_tfidf_ngram,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        min_df=SETTINGS.model.char_tfidf_min_df,
        ngram_range=SETTINGS.model.char_tfidf_ngram,
    )

    # Numeric scaler (sparse-friendly with with_mean=False)
    numeric_scaler = StandardScaler(with_mean=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf_words", word_tfidf, "combined_text"),
            ("tfidf_merchant_char", char_tfidf, "merchant_text"),
            ("numeric", numeric_scaler, ["amount", "hour", "day_of_week", "is_weekend"]),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        solver=SETTINGS.model.solver,
        multi_class="multinomial",
        class_weight=SETTINGS.model.class_weight,
        max_iter=SETTINGS.model.max_iter,
        random_state=SETTINGS.model.random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
