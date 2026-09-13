"""Bounded lifecycle checks for speculative-voice soak qualification."""

from __future__ import annotations

import json

import pytest

from Packaging.qualify_speculative_voice import VoiceQualificationError
from Packaging.qualify_speculative_voice import assemble_soak_report
from Packaging.speculative_voice_soak import (
    run_cancellation_soak,
    run_duplex_soak,
    run_native_process_probe,
)


_DIGEST = "d" * 64
_PASSING_NATIVE_PROBE = {
    "started": True,
    "exit_code": 0,
    "forced_termination": False,
    "reaped": True,
    "passed": True,
}


@pytest.mark.asyncio
async def test_short_duplex_soak_drains_rings_fences_callbacks_and_handles() -> None:
    result = await run_duplex_soak(
        duration_seconds=0.03,
        cycle_interval_seconds=0.001,
        route_change_every=4,
        native_probe=lambda: dict(_PASSING_NATIVE_PROBE),
    )

    assert result["passed"] is True
    assert result["iterations"] >= 1
    assert result["audio_buffers"]["final"] == {
        "capture_frames": 0,
        "render_frames": 0,
        "render_reference_frames": 0,
        "control_events": 0,
    }
    assert result["audio_buffers"]["overflow_count"] == 0
    assert result["post_fence_callbacks_accepted"] == 0
    assert result["device_handles"]["leaked"] == 0
    assert result["tasks"]["final"] <= result["tasks"]["baseline"]


@pytest.mark.asyncio
async def test_short_cancellation_soak_bounds_tasks_orphans_and_callbacks() -> None:
    result = await run_cancellation_soak(
        duration_seconds=0.03,
        cycle_interval_seconds=0.001,
        native_probe=lambda: dict(_PASSING_NATIVE_PROBE),
    )

    assert result["passed"] is True
    assert result["cancellations"] >= 2
    assert result["orphans"] == {"maximum": 2, "final": 0, "limit": 2}
    assert result["post_fence_callbacks_accepted"] == 0
    assert result["tasks"]["final"] <= result["tasks"]["baseline"]


def test_native_extension_process_fixture_exits_and_is_reaped() -> None:
    pytest.importorskip("tldw_voice_aec")

    result = run_native_process_probe(timeout_seconds=5.0)

    assert result == _PASSING_NATIVE_PROBE


def test_soak_report_binds_source_and_propagates_failure() -> None:
    failed = dict(_PASSING_NATIVE_PROBE, passed=False, reaped=False)

    report = assemble_soak_report(
        scenario="duplex-soak",
        source_tree_digest=_DIGEST,
        interpreter={"implementation": "cpython"},
        soak={"native_process": failed, "passed": False},
        git_revision="a" * 40,
    )

    assert report["scenario"] == "duplex-soak"
    assert report["source_tree_digest"] == _DIGEST
    assert report["passed"] is False


def test_soak_report_rejects_unknown_scenario_and_content() -> None:
    with pytest.raises(VoiceQualificationError, match="scenario"):
        assemble_soak_report(
            scenario="unknown",
            source_tree_digest=_DIGEST,
            interpreter={},
            soak={"passed": True},
        )

    with pytest.raises(VoiceQualificationError, match="content-bearing"):
        assemble_soak_report(
            scenario="cancellation-soak",
            source_tree_digest=_DIGEST,
            interpreter={},
            soak={"transcript": "VOICE-POISON", "passed": True},
        )


@pytest.mark.asyncio
async def test_soak_duration_and_report_are_content_free() -> None:
    with pytest.raises(ValueError, match="duration"):
        await run_duplex_soak(duration_seconds=0.0)
    with pytest.raises(ValueError, match="duration"):
        await run_cancellation_soak(duration_seconds=-1.0)

    result = await run_cancellation_soak(
        duration_seconds=0.01,
        cycle_interval_seconds=0.001,
        native_probe=lambda: dict(_PASSING_NATIVE_PROBE),
    )
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in (
        "transcript",
        "response_text",
        "pcm",
        "device_name",
        "credential",
        "request_body",
        "capture_body",
    ):
        assert forbidden not in serialized
