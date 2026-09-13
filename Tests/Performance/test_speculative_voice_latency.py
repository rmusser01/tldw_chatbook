"""Deterministic latency and large-history gates for speculative voice."""

from __future__ import annotations

from pathlib import Path

import pytest

from Packaging.speculative_voice_history_gate import (
    BASELINE_LOCATOR_ROWS,
    LARGE_LOCATOR_ROWS,
    completed_pair_locator_inventory,
    _locator_distribution,
    run_completed_pair_history_gate,
)
from Packaging.speculative_voice_latency_gate import (
    LATENCY_TRIALS,
    measure_speculative_voice_latency,
)


@pytest.mark.asyncio
async def test_warm_speculative_voice_latency_distribution_meets_gate() -> None:
    report = await measure_speculative_voice_latency()

    assert report["trial_count"] == LATENCY_TRIALS == 40
    assert report["clock_boundaries"] == {
        "eos": "post_aec_admitted_speech_end",
        "dispatch": "attempt_dispatch_handoff",
        "audible_stop": "transport_abort_fence",
        "first_assistant_audio": "transport_playback_started",
    }
    assert report["eos_to_dispatch"]["p95_ms"] <= 850
    assert report["barge_to_audible_stop"]["p95_ms"] <= 150
    assert report["added_eos_to_replacement_dispatch"]["p95_ms"] <= 850
    assert report["eos_to_first_assistant_audio"]["median_ms"] <= 1_500
    assert report["eos_to_first_assistant_audio"]["p95_ms"] <= 2_500
    assert report["passed"] is True


def test_completed_pair_locator_inventory_is_complete_and_stable() -> None:
    inventory = completed_pair_locator_inventory()

    assert inventory["count"] == 19
    assert {"table": "canvas_revisions", "column": "origin_message_id"} in inventory[
        "locators"
    ]
    distribution = _locator_distribution()
    assert sum(row["baseline_rows"] for row in distribution) == BASELINE_LOCATOR_ROWS
    assert sum(row["large_rows"] for row in distribution) == LARGE_LOCATOR_ROWS
    assert len(inventory["sha256"]) == 64
    assert len(inventory["locators"]) == inventory["count"]


def test_completed_pair_history_gate_meets_large_history_thresholds(
    tmp_path: Path,
) -> None:
    report = run_completed_pair_history_gate(tmp_path)

    assert report["row_counts"] == {
        "baseline": BASELINE_LOCATOR_ROWS,
        "large": LARGE_LOCATOR_ROWS,
    }
    assert report["warmup_count"] == 10
    assert report["trial_count"] == 40
    assert report["measurement_order"] == "ABBA"
    assert report["scan_count"] == 0
    assert report["sqlite"]["pragmas_match"] is True
    assert report["inventory"]["count"] == 19
    assert all(
        locator["large_rows"] == locator["baseline_rows"] * 100
        for locator in report["distribution"]
    )
    assert report["fresh_commit"]["passed"] is True
    assert report["uncertain_retry"]["passed"] is True
    assert report["passed"] is True
