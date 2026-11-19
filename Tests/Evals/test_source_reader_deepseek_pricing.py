"""Experiment-only DeepSeek time-of-day pricing tests."""

from datetime import UTC, datetime, timedelta

import pytest

from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Evals.source_reader.deepseek_pricing import (
    peak_catalog,
    pricing_for_interval,
)


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 8, hour, minute, tzinfo=UTC)


@pytest.mark.parametrize(
    ("start", "end", "expected_tier", "expected_input"),
    [
        (_at(0), _at(1), "off_peak", 0.22),
        (_at(1), _at(4), "peak", 0.44),
        (_at(4), _at(6), "off_peak", 0.22),
        (_at(6), _at(10), "peak", 0.44),
        (_at(10), _at(11), "off_peak", 0.22),
    ],
)
def test_exact_boundary_intervals_use_their_single_rate(
    start: datetime,
    end: datetime,
    expected_tier: str,
    expected_input: float,
) -> None:
    catalog, metadata = pricing_for_interval(start, end)

    pricing = catalog.get_pricing("deepseek", "deepseek-v4-flash")
    assert pricing is not None
    assert pricing.input_per_mtok == expected_input
    assert metadata["tier"] == expected_tier
    assert metadata["exact_pricing"] is True


def test_crossing_boundary_uses_peak_bound_and_marks_actual_tier_ambiguous() -> None:
    catalog, metadata = pricing_for_interval(_at(3, 59), _at(4, 1))

    pricing = catalog.get_pricing("deepseek", "deepseek-v4-pro")
    assert pricing is not None
    assert pricing.input_per_mtok == 1.32
    assert metadata["tier"] == "mixed"
    assert metadata["exact_pricing"] is False


def test_interval_spanning_midnight_detects_the_next_days_peak_boundary() -> None:
    start = datetime(2026, 9, 8, 23, 0, tzinfo=UTC)
    end = datetime(2026, 9, 9, 2, 0, tzinfo=UTC)

    catalog, metadata = pricing_for_interval(start, end)

    pricing = catalog.get_pricing("deepseek", "deepseek-v4-flash")
    assert pricing is not None and pricing.output_per_mtok == 1.32
    assert metadata["tier"] == "mixed"
    assert metadata["exact_pricing"] is False


def test_peak_catalog_prices_cached_tokens_at_the_official_peak_rate() -> None:
    usage = ProviderUsage(
        uncached_input=1_000_000,
        cache_read=1_000_000,
        output=1_000_000,
        provider="deepseek",
        model="deepseek-v4-pro",
    )

    cost = peak_catalog().cost_for_usage(usage)

    assert cost is not None
    assert cost.input_cost == 1.32
    assert cost.cache_read_cost == 0.044
    assert cost.output_cost == 3.96
    assert cost.total == 5.324
    assert cost.as_of == "2026-09-10"


def test_catalog_metadata_records_pricing_provenance() -> None:
    _, metadata = pricing_for_interval(_at(10), _at(11))

    assert metadata["effective_date"] == "2026-08-16"
    assert metadata["verified_on"] == "2026-09-08"
    assert metadata["source_url"].startswith(
        "https://api-docs.deepseek.com/quick_start/pricing/"
    )


def test_current_flash_uses_v41_rates_and_current_provenance() -> None:
    start = datetime(2026, 9, 29, 10, 0, tzinfo=UTC)
    catalog, metadata = pricing_for_interval(start, start + timedelta(minutes=1))

    current = catalog.get_pricing("deepseek", "deepseek-flash")
    redirected_legacy = catalog.get_pricing("deepseek", "deepseek-v4-flash")
    assert current is not None
    assert redirected_legacy is not None
    assert (
        current.input_per_mtok,
        current.output_per_mtok,
        current.cache_read_per_mtok,
    ) == (0.15, 0.6, 0.003)
    assert redirected_legacy == current
    assert metadata["effective_at"] == "2026-09-10T04:00:00+00:00"
    assert metadata["verified_on"] == "2026-09-29"
    assert metadata["news_url"].endswith("/news/news260910/")
    assert metadata["peak_schedule"] == (
        "Monday-Friday excluding Chinese public holidays"
    )


def test_current_weekend_peak_clock_is_exactly_off_peak() -> None:
    start = datetime(2026, 9, 26, 1, 0, tzinfo=UTC)  # Saturday
    catalog, metadata = pricing_for_interval(start, start + timedelta(hours=1))

    pricing = catalog.get_pricing("deepseek", "deepseek-flash")
    assert pricing is not None and pricing.input_per_mtok == 0.15
    assert metadata["tier"] == "off_peak"
    assert metadata["exact_pricing"] is True


def test_current_weekday_peak_clock_is_holiday_uncertain_upper_bound() -> None:
    start = datetime(2026, 9, 29, 1, 0, tzinfo=UTC)  # Tuesday
    catalog, metadata = pricing_for_interval(start, start + timedelta(hours=1))

    pricing = catalog.get_pricing("deepseek", "deepseek-flash")
    assert pricing is not None and pricing.input_per_mtok == 0.3
    assert metadata["tier"] == "peak"
    assert metadata["exact_pricing"] is False
    assert metadata["cost_status"] == "bounded"
    assert metadata["holiday_status"] == "unknown"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (datetime(2026, 9, 8, 1), _at(2)),  # noqa: DTZ001 - deliberately naive
        (_at(2), datetime(2026, 9, 8, 3)),  # noqa: DTZ001 - deliberately naive
        (_at(2), _at(2)),
        (_at(2), _at(2) - timedelta(microseconds=1)),
    ],
)
def test_interval_requires_ordered_aware_datetimes(
    start: datetime, end: datetime
) -> None:
    with pytest.raises(ValueError, match="aware.*start.*end|start.*end.*aware|ordered"):
        pricing_for_interval(start, end)
