"""Content-free approval timing records; no transport capture is implied.

A summary qualifies supplied boundary measurements only. Transport observers must
establish actual presentation and bounded clock error before supplying durations.
"""

from __future__ import annotations

import math
import secrets
import statistics
from collections.abc import Mapping, Sequence

STAGES = frozenset(
    {
        "request_complete",
        "local_checks_complete",
        "input_arrived",
        "handler_committing",
        "compositor_paint",
        "native_paint",
        "browser_paint",
        "resolver_entered",
        "host_outcome",
        "worker_released",
        "dispatch",
        "backend_started",
        "grant_applied",
        "result",
        "next_model_output",
    }
)
CLOCKS = frozenset(
    {"python_perf_counter", "native_perf_counter", "browser_performance"}
)
MAX_RECORDS = 4096
_SAMPLE_FIELDS = frozenset(
    {"transport", "clock_qualified", "first_use", "feedback_ms", "actionable_ms"}
)


class ApprovalTimingRecorder:
    """Keep bounded local markers with internally minted correlation tokens."""

    def __init__(self) -> None:
        self._correlations: set[str] = set()
        self._records: list[dict[str, object]] = []

    def new_correlation(self) -> str:
        """Mint a local token, refusing to exceed the bounded sample store."""
        if len(self._correlations) >= MAX_RECORDS:
            raise ValueError("correlation_capacity_exceeded")
        correlation = secrets.token_hex(16)
        self._correlations.add(correlation)
        return correlation

    def record(
        self, stage: str, *, correlation: str, clock: str, timestamp_ns: int
    ) -> None:
        """Record one allowlisted marker without arbitrary event content."""
        if (
            stage not in STAGES
            or clock not in CLOCKS
            or correlation not in self._correlations
        ):
            raise ValueError("unsupported_timing_marker")
        if type(timestamp_ns) is not int or timestamp_ns < 0:  # noqa: E721 - reject bool
            raise ValueError("invalid_timestamp")
        if len(self._records) >= MAX_RECORDS:
            raise ValueError("record_capacity_exceeded")
        self._records.append(
            dict(
                stage=stage,
                correlation=correlation,
                clock=clock,
                timestamp_ns=timestamp_ns,
            )
        )

    def snapshot(self) -> list[dict[str, object]]:
        """Return a detached copy of local timing markers."""
        return [dict(record) for record in self._records]


def summarize_approval_timings(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize warm samples using nearest rank; retain missing boundaries.

    Args:
        samples: Content-free observed durations, with first use marked separately.
    Returns:
        Distribution facts; qualification never asserts a latency-budget pass.
    Raises:
        ValueError: A sample contains unsupported content or invalid measurements.
    """
    for sample in samples:
        if set(sample) != _SAMPLE_FIELDS:
            raise ValueError("unsupported_sample_content")
        if sample["transport"] not in {"native", "browser", "compositor"}:
            raise ValueError("unsupported_transport")
        if not isinstance(sample["clock_qualified"], bool) or not isinstance(
            sample["first_use"], bool
        ):
            raise ValueError("invalid_sample_flags")
        for key in ("feedback_ms", "actionable_ms"):
            value = sample[key]
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
            ):
                raise ValueError("invalid_duration")
    warm = [sample for sample in samples if not sample["first_use"]]
    first_use = [sample for sample in samples if sample["first_use"]]
    missing = {
        key: sum(sample[key] is None for sample in warm)
        for key in ("feedback_ms", "actionable_ms")
    }
    report: dict[str, object] = {
        "sample_count": len(warm),
        "first_use_sample_count": len(first_use),
        "first_use_samples": [dict(sample) for sample in first_use],
        "feedback_budget_ms": 100.0,
        "actionable_budget_ms": 200.0,
        "missing_boundaries": missing,
        "qualified": len(warm) >= 40
        and not any(missing.values())
        and all(sample["clock_qualified"] for sample in warm)
        and len({sample["transport"] for sample in warm}) == 1
        and all(sample["transport"] in {"native", "browser"} for sample in warm),
    }
    for key in ("feedback_ms", "actionable_ms"):
        values = sorted(
            float(sample[key]) for sample in warm if sample[key] is not None
        )
        prefix = key.removesuffix("_ms")
        report[f"{prefix}_median_ms"] = statistics.median(values) if values else None
        report[f"{prefix}_max_ms"] = max(values) if values else None
        # Missing paint must not disappear from the percentile denominator.
        report[f"{prefix}_p95_ms"] = (
            values[math.ceil(0.95 * len(warm)) - 1]
            if values and not missing[key]
            else None
        )
    return report
