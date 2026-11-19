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

_ACTIVATION = datetime(2026, 9, 10, 4, tzinfo=UTC)
_CURRENT_RATES = {
    "deepseek-flash": (0.15, 0.6, 0.003),
    "deepseek-v4-flash": (0.15, 0.6, 0.003),
    "deepseek-v4-pro": (0.66, 1.98, 0.022),
}


def _catalog(multiplier: float, rates=None, as_of=_EFFECTIVE_DATE) -> PricingCatalog:
    models = {
        f"deepseek:{model}": {
            "input_per_mtok": input_rate * multiplier,
            "output_per_mtok": output_rate * multiplier,
            "cache_read_per_mtok": cache_rate * multiplier,
            "cache_write_per_mtok": None,
            "as_of": as_of,
        }
        for model, (input_rate, output_rate, cache_rate) in (
            rates or _OFF_PEAK_RATES
        ).items()
    }
    return PricingCatalog(config={"models": models})


def peak_catalog() -> PricingCatalog:
    """Return conservative peak rates for source-reader budget preflight."""
    return _catalog(2.0, _CURRENT_RATES, "2026-09-10")


def _is_peak(moment: datetime) -> bool:
    utc_time = moment.astimezone(UTC).time().replace(tzinfo=None)
    return any(start <= utc_time < end for start, end in _PEAK_WINDOWS)


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

    boundaries = {start_utc, end_utc}
    day = start_utc.date()
    while day <= end_utc.date():
        for clock in (time(0), time(1), time(4), time(6), time(10)):
            boundary = datetime.combine(day, clock, tzinfo=UTC)
            if start_utc < boundary < end_utc:
                boundaries.add(boundary)
        day += timedelta(days=1)
    if start_utc < _ACTIVATION < end_utc:
        boundaries.add(_ACTIVATION)
    segments = []
    for moment in sorted(boundaries)[:-1]:
        current = moment >= _ACTIVATION
        peak = _is_peak(moment) and (not current or moment.weekday() < 5)
        segments.append((current, peak))
    mixed = len(set(segments)) > 1
    tier = "mixed" if mixed else ("peak" if segments[0][1] else "off_peak")
    uncertain = any(current and peak for current, peak in segments)
    rates = {}
    for current, peak in segments:
        era_rates = (
            _CURRENT_RATES
            if current
            else {
                **_OFF_PEAK_RATES,
                "deepseek-flash": _OFF_PEAK_RATES["deepseek-v4-flash"],
            }
        )
        for model, values in era_rates.items():
            scaled = tuple(value * (2 if peak else 1) for value in values)
            rates[model] = tuple(
                max(a, b) for a, b in zip(rates.get(model, scaled), scaled)
            )
    current_only = all(current for current, _ in segments)
    exact = not mixed and not uncertain
    catalog = _catalog(1.0, rates, "2026-09-10" if current_only else _EFFECTIVE_DATE)
    metadata: dict[str, Any] = {
        "tier": tier,
        "exact_pricing": exact,
        "cost_status": "exact" if exact else "bounded",
        "effective_date": _EFFECTIVE_DATE,
        "verified_on": _VERIFIED_ON,
        "source_url": _SOURCE_URL,
        "news_url": _NEWS_URL,
        "peak_intervals_utc": ["[01:00,04:00)", "[06:00,10:00)"],
    }
    metadata.update(
        estimate_status="historical_unverified_current",
        calendar="UTC fixture weekdays; Chinese holiday status unknown",
        holiday_status="unknown",
    )
    if current_only:
        metadata.update(
            effective_date="2026-09-10",
            effective_at=_ACTIVATION.isoformat(),
            verified_on="2026-09-29",
            news_url="https://api-docs.deepseek.com/news/news260910/",
            peak_schedule="Monday-Friday excluding Chinese public holidays",
        )
    elif any(current for current, _ in segments):
        metadata["eras"] = [
            "2026-08-16/verified-2026-09-08",
            "2026-09-10/verified-2026-09-29",
        ]
    return catalog, metadata
