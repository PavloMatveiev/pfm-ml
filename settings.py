from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence, TypedDict, Literal


# ----------------------------
# Amount specs
# ----------------------------

@dataclass(frozen=True)
class AmountSpec:
    """Inclusive range and sign used to sample a monetary amount.

    Attributes:
        low:  Inclusive lower bound of the absolute amount.
        high: Inclusive upper bound of the absolute amount.
        sign: Direction of the amount: 1 for income, -1 for expense.
    """
    low: float
    high: float
    sign: Literal[-1, 1] = -1

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("AmountSpec.low must be <= AmountSpec.high")


# ----------------------------
# Time/date synthesis config
# ----------------------------

@dataclass(frozen=True)
class TimeSettings:
    """Parameters that control generation of synthetic datetimes."""
    base_date: datetime                       # start point for timestamps
    hour_choices: Mapping[str, Sequence[int]] # per-category fixed hour lists
    default_hour_range: tuple[int, int]       # fallback inclusive hour range
    day_offset_range: tuple[int, int]         # inclusive day offset window


# ----------------------------
# Model/training hyperparameters
# ----------------------------

@dataclass(frozen=True)
class ModelSettings:
    """Classifier and featurizer hyperparameters."""
    test_size: float = 0.2
    random_state: int = 42
    word_tfidf_min_df: int = 2
    word_tfidf_ngram: tuple[int, int] = (1, 2)
    char_tfidf_min_df: int = 2
    char_tfidf_ngram: tuple[int, int] = (3, 5)
    solver: str = "saga"
    max_iter: int = 2000
    class_weight: str = "balanced"


# ----------------------------
# Vocabulary types
# ----------------------------

class VocabEntry(TypedDict):
    merchants: list[str]
    desc: list[str]


# ----------------------------
# Top-level settings container
# ----------------------------

@dataclass(frozen=True)
class Settings:
    """Single source of truth for all knobs affecting synthesis and training."""
    categories: list[str]
    vocab: Mapping[str, VocabEntry]
    amounts: Mapping[str, AmountSpec]
    default_amount: AmountSpec
    default_ndigits: int
    time: TimeSettings
    model: ModelSettings
    seed: int = 42


SETTINGS = Settings(
    categories=[
        "Groceries", "Transport", "Dining & Coffee", "Entertainment",
        "Bills & Utilities", "Health & Fitness", "Shopping", "Income", "Other",
    ],
    vocab={
        "Groceries":         {"merchants": ["Tesco", "Sainsbury's", "ALDI", "LIDL", "ASDA", "Co-op", "Morrisons"], "desc": ["groceries", "weekly shop", "food basket", "fresh produce"]},
        "Transport":         {"merchants": ["Uber", "Bolt", "ScotRail", "TFL", "Shell", "BP", "Stagecoach"],       "desc": ["ride home", "bus ticket", "train to work", "petrol", "diesel"]},
        "Dining & Coffee":   {"merchants": ["Starbucks", "Costa", "Caffè Nero", "KFC", "McDonalds", "Dominos"],    "desc": ["morning latte", "burger meal", "pizza deal", "americano", "lunch"]},
        "Entertainment":     {"merchants": ["Netflix", "Spotify", "Steam", "Cineworld", "Disney+"],                "desc": ["subscription", "movie ticket", "monthly sub", "premium plan"]},
        "Bills & Utilities": {"merchants": ["Vodafone", "O2", "EE", "BT", "British Gas", "Octopus Energy"],        "desc": ["mobile bill", "broadband", "energy bill", "council tax"]},
        "Health & Fitness":  {"merchants": ["Boots", "NHS", "PureGym", "The Gym Group", "Holland & Barrett"],      "desc": ["pharmacy", "gym membership", "vitamins", "healthcare"]},
        "Shopping":          {"merchants": ["Amazon", "eBay", "Argos", "Currys", "Primark", "IKEA"],               "desc": ["online order", "charger", "home goods", "t-shirt", "accessories"]},
        "Income":            {"merchants": ["Payroll", "ACME LTD", "Company Ltd", "Employer Ltd", "HSBC"],         "desc": ["monthly salary", "PAYROLL BACS CREDIT", "wage", "payslip"]},
        "Other":             {"merchants": ["Local Market", "HSBC", "Barclays", "NatWest", "Halifax", "Monzo"],    "desc": ["card payment", "transfer", "fee", "charge", "misc purchase"]},
    },
    amounts={
        "Groceries":         AmountSpec(8, 120, -1),
        "Transport":         AmountSpec(3, 70,  -1),
        "Dining & Coffee":   AmountSpec(2, 30,  -1),
        "Entertainment":     AmountSpec(4, 20,  -1),
        "Bills & Utilities": AmountSpec(20, 500, -1),
        "Health & Fitness":  AmountSpec(3, 120, -1),
        "Shopping":          AmountSpec(5, 500, -1),
        "Income":            AmountSpec(800, 2500, +1),
    },
    default_amount=AmountSpec(1, 80, -1),
    default_ndigits=2,
    time=TimeSettings(
        base_date=datetime(2025, 8, 15, 0, 0, 0),
        hour_choices={
            "Transport": [7, 8, 9, 22, 23, 0, 1],
            "Dining & Coffee": [8, 12, 13, 18, 19],
            "Entertainment": [19, 20, 21, 22],
        },
        default_hour_range=(8, 21),
        day_offset_range=(0, 13),
    ),
    model=ModelSettings(),
    seed=42,
)
