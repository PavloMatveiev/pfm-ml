from __future__ import annotations

import random
from typing import Dict, List, Mapping

import pandas as pd

from settings import SETTINGS, VocabEntry
from utils.date_and_time import rand_time
from utils.amount import rand_amount


def synthesize_rows_for_category(category: str, rows_count: int, vocab: Mapping[str, VocabEntry]) -> List[Dict]:
    """Create `rows_count` synthetic rows for the given category using provided vocab."""
    rows: List[Dict] = []
    merchants = vocab[category]["merchants"]
    descriptions = vocab[category]["desc"]
    for _ in range(rows_count):
        merchant = random.choice(merchants)
        description = random.choice(descriptions)
        rows.append(
            {
                "date": rand_time(category),
                "merchant": merchant,
                "description": description,
                "amount": rand_amount(category),
                "label": category,
            }
        )
    return rows


def synthesize_dataset(
    default_samples_per_category: int = 60,
    per_category_overrides: Dict[str, int] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Build a full synthetic dataset across all categories."""
    if seed is not None:
        random.seed(seed)

    categories = SETTINGS.categories
    vocab = SETTINGS.vocab
    overrides = per_category_overrides or {}

    rows: List[Dict] = []
    for category in categories:
        n = overrides.get(category, default_samples_per_category)
        rows.extend(synthesize_rows_for_category(category, n, vocab))

    # A few extra "unknown-ish" examples to check generalization
    rows.extend(
        [
            {"date": rand_time("Income"),        "merchant": "HSBC",      "description": "PAYROLL BACS CREDIT",  "amount": -1350.0, "label": "Income"},
            {"date": rand_time("Entertainment"), "merchant": "STREAMIO",  "description": "monthly subscription", "amount": 8.99,    "label": "Entertainment"},
            {"date": rand_time("Transport"),     "merchant": "QuickRide", "description": "late night ride",      "amount": 11.2,    "label": "Transport"},
            {"date": rand_time("Other"),         "merchant": "HSBC",      "description": "card payment",         "amount": 24.99,   "label": "Other"},
        ]
    )

    return pd.DataFrame(rows)
