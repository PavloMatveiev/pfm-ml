import random
from typing import Literal
from settings import SETTINGS, AmountSpec


def choose_random_amount(
    lower_bound: float,
    upper_bound: float,
    decimal_places: int | None = None,
    *,
    sign: Literal[-1, 1] = 1,
) -> float:
    """Sample a random amount in [lower_bound, upper_bound], apply sign, round.

    Args:
        lower_bound: Inclusive lower bound for the absolute amount.
        upper_bound: Inclusive upper bound for the absolute amount.
        decimal_places: Number of decimal places to round to. If None, uses
            SETTINGS.default_ndigits.
        sign: Direction multiplier: 1 for income, -1 for expense.

    Returns:
        A rounded float with the requested sign and within the provided bounds.

    Raises:
        ValueError: If `decimal_places` is negative.
    """
    if decimal_places is None:
        decimal_places = SETTINGS.default_ndigits
    if decimal_places < 0:
        raise ValueError("decimal_places must be >= 0")

    # Normalize bounds so lower_bound <= upper_bound
    lo, hi = (upper_bound, lower_bound) if lower_bound > upper_bound else (lower_bound, upper_bound)
    sampled_value = random.uniform(lo, hi)
    return round(sign * sampled_value, decimal_places)


def get_amount_spec(category: str) -> AmountSpec:
    """Return the AmountSpec for a category, or the default spec if unknown."""
    return SETTINGS.amounts.get(category, SETTINGS.default_amount)


def rand_amount(category: str, decimal_places: int | None = None) -> float:
    """Convenience wrapper: generate a signed, rounded amount for a category.

    Args:
        category: Category name (e.g., "Groceries", "Income").
        decimal_places: Number of decimal places (None → SETTINGS.default_ndigits).

    Returns:
        A float sampled according to category AmountSpec and rounding settings.
    """
    spec = get_amount_spec(category)
    return choose_random_amount(
        lower_bound=spec.low,
        upper_bound=spec.high,
        decimal_places=decimal_places,
        sign=spec.sign,
    )
