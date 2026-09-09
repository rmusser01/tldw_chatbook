"""Time-aware DeepSeek pricing for the source-reader experiment only."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import Any

from tldw_chatbook.LLM_Calls.pricing_catalog import PricingCatalog

_EFFECTIVE_DATE = "2026-08-16"
_VERIFIED_ON = "2026-09-08"
_SOURCE_URL = (
    "https://api-docs.deepseek.com/quick_start/pricing/"
    "?article_id=article_1779470751466_8"
)
_NEWS_URL = "https://api-docs.deepseek.com/news/news260813/"
_PEAK_WINDOWS = ((time(1), time(4)), (time(6), time(10)))
_OFF_PEAK_RATES = {
    "deepseek-v4-flash": (0.22, 0.66, 0.007),
    "deepseek-v4-pro": (0.66, 1.98, 0.022),
}


def _catalog(multiplier: float) -> PricingCatalog:
    models = {
        f"deepseek:{model}": {
            "input_per_mtok": input_rate * multiplier,
            "output_per_mtok": output_rate * multiplier,
            "cache_read_per_mtok": cache_rate * multiplier,
            "cache_write_per_mtok": None,
            "as_of": _EFFECTIVE_DATE,
        }
        for model, (input_rate, output_rate, cache_rate) in _OFF_PEAK_RATES.items()
    }
    return PricingCatalog(config={"models": models})


def peak_catalog() -> PricingCatalog:
    """Return conservative peak rates for source-reader budget preflight."""
    return _catalog(2.0)


def _is_peak(moment: datetime) -> bool:
    utc_time = moment.astimezone(UTC).time().replace(tzinfo=None)
    return any(start <= utc_time < end for start, end in _PEAK_WINDOWS)


def _crosses_rate_boundary(start: datetime, end: datetime) -> bool:
    current_day = start.date()
    final_day = end.date()
    while current_day <= final_day:
        for boundary_time in (time(1), time(4), time(6), time(10)):
            boundary = datetime.combine(current_day, boundary_time, tzinfo=UTC)
            if start < boundary < end:
                return True
        current_day += timedelta(days=1)
    return False


def pricing_for_interval(
    start: datetime, end: datetime
) -> tuple[PricingCatalog, dict[str, Any]]:
    """Select rates that cover the whole execution interval.

    Mixed peak/off-peak intervals use peak prices as an upper bound and are
    explicitly marked ambiguous so experiment reports cannot claim exact savings.

    Args:
        start: Inclusive execution start with timezone information.
        end: Exclusive execution end with timezone information.

    Returns:
        The applicable pricing catalog and provenance/status metadata.

    Raises:
        ValueError: If either datetime is naive or the interval is not ordered.
    """
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("start and end must be aware datetimes")
    if end.tzinfo is None or end.utcoffset() is None:
        raise ValueError("start and end must be aware datetimes")

    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    if start_utc >= end_utc:
        raise ValueError("start and end must form an ordered interval")

    mixed = _crosses_rate_boundary(start_utc, end_utc)
    tier = "mixed" if mixed else ("peak" if _is_peak(start_utc) else "off_peak")
    catalog = peak_catalog() if tier in {"peak", "mixed"} else _catalog(1.0)
    metadata: dict[str, Any] = {
        "tier": tier,
        "exact_pricing": not mixed,
        "cost_status": "bounded" if mixed else "exact",
        "effective_date": _EFFECTIVE_DATE,
        "verified_on": _VERIFIED_ON,
        "source_url": _SOURCE_URL,
        "news_url": _NEWS_URL,
        "peak_intervals_utc": ["[01:00,04:00)", "[06:00,10:00)"],
    }
    return catalog, metadata
