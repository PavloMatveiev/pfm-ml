import random
from typing import overload, Sequence
from datetime import datetime, timedelta
from settings import SETTINGS


@overload
def choose_random_hour(hours: Sequence[int]) -> int: ...
@overload
def choose_random_hour(start_inclusive: int, end_inclusive: int) -> int: ...

def choose_random_hour(hours_or_start: Sequence[int] | int, end_inclusive: int | None = None) -> int:
    """Pick a random hour either from a sequence or an inclusive [start, end] range.

    Args:
        hours_or_start: A sequence of candidate hours (0..23) OR the start value of a range.
        end_inclusive: If provided, the inclusive end value of the range; if None, the first
            argument is treated as a sequence.

    Returns:
        Integer hour in the range [0, 23].

    Raises:
        ValueError: If the sequence is empty or the sampled hour is outside 0..23.
    """
    if end_inclusive is None:
        candidates = list(hours_or_start)
        if not candidates:
            raise ValueError("hours sequence must not be empty")
        hour = random.choice(candidates)
    else:
        start = int(hours_or_start)
        end = int(end_inclusive)
        if start > end:
            start, end = end, start
        hour = random.randint(start, end)

    if not (0 <= hour <= 23):
        raise ValueError("hour must be within 0..23")
    return hour


def choose_random_day(start_day_offset_inclusive: int, end_day_offset_inclusive: int) -> int:
    """Return a random day offset within an inclusive range [start, end]."""
    start, end = start_day_offset_inclusive, end_day_offset_inclusive
    if start > end:
        start, end = end, start
    return random.randint(start, end)


def use_date_offset(start_datetime: datetime, days_offset: int, hours_offset: int) -> datetime:
    """Return a datetime offset from `start_datetime` by given days and hours.

    Normalization is automatic (e.g., +26 hours becomes +1 day + 2 hours).

    Args:
        start_datetime: Base datetime to offset (naive or timezone-aware).
        days_offset: Number of days to add (can be negative).
        hours_offset: Number of hours to add (can be negative).

    Returns:
        New `datetime` instance with the applied offset.
    """
    return start_datetime + timedelta(days=days_offset, hours=hours_offset)


def rand_time(category: str) -> str:
    """Generate an ISO 8601 timestamp (seconds precision) for a given category.

    The hour is chosen from a per-category list (if present), otherwise from a
    default inclusive range. The day is chosen as a random offset from the base
    date. The final timestamp is `SETTINGS.time.base_date + offsets`.

    Args:
        category: Category name used to pick a realistic hour distribution.

    Returns:
        ISO 8601 datetime string, e.g., "2025-08-16T21:00:00".
    """
    time_settings = SETTINGS.time

    # Choose hour: per-category fixed choices or fallback range
    if category in time_settings.hour_choices:
        selected_hour = choose_random_hour(time_settings.hour_choices[category])
    else:
        start_hour, end_hour = time_settings.default_hour_range
        selected_hour = choose_random_hour(start_hour, end_hour)

    # Choose day offset (spreads events across several days)
    selected_day_offset = choose_random_day(*time_settings.day_offset_range)

    # Build final datetime and return ISO string
    timestamp_datetime = use_date_offset(time_settings.base_date, selected_day_offset, selected_hour)
    return timestamp_datetime.isoformat(timespec="seconds")
