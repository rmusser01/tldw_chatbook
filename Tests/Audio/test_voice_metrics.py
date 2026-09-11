"""Content-free metrics contracts for speculative Console voice."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from tldw_chatbook.Audio.voice_metrics import (
    VoiceAecState,
    VoiceBackendMode,
    VoiceLatencyKind,
    VoiceMetrics,
    VoiceProviderResultClass,
    VoiceUsageCounters,
    VoiceUsageOutcome,
)


def test_metrics_retain_required_content_free_observations() -> None:
    metrics = VoiceMetrics()

    metrics.observe_backend_mode(VoiceBackendMode.NATIVE_LIVE)
    metrics.observe_backend_mode(VoiceBackendMode.ROLLING_WINDOW)
    metrics.observe_latency(VoiceLatencyKind.EOS_TO_DISPATCH, 612.5)
    metrics.observe_latency(VoiceLatencyKind.BARGE_TO_AUDIBLE_STOP, 81)
    metrics.observe_latency(VoiceLatencyKind.REPLACEMENT_DISPATCH, 704)
    metrics.observe_latency(VoiceLatencyKind.FIRST_AUDIO, 1_103)
    metrics.observe_aec(VoiceAecState.WARMING)
    metrics.observe_aec(VoiceAecState.HEALTHY, erle_db=23.5)
    metrics.increment_underruns()
    metrics.increment_restarts(2)
    metrics.increment_conservative_entries()
    metrics.add_duplicated_stt_audio_ms(480)
    metrics.record_provider_result(VoiceProviderResultClass.CANCELLED)
    metrics.record_usage(
        VoiceUsageOutcome.WINNING,
        VoiceUsageCounters(
            calls=1,
            stt_audio_ms=1_000,
            llm_input_tokens=20,
            llm_output_tokens=10,
            tts_characters=40,
            tts_audio_ms=900,
        ),
    )
    metrics.record_usage(
        VoiceUsageOutcome.DISCARDED,
        VoiceUsageCounters(calls=2, llm_output_tokens=12, tts_characters=18),
    )

    snapshot = metrics.snapshot()
    assert snapshot.backend_modes == (
        VoiceBackendMode.NATIVE_LIVE,
        VoiceBackendMode.ROLLING_WINDOW,
    )
    assert snapshot.latencies_ms[VoiceLatencyKind.EOS_TO_DISPATCH] == (612.5,)
    assert snapshot.latencies_ms[VoiceLatencyKind.BARGE_TO_AUDIBLE_STOP] == (81.0,)
    assert snapshot.latencies_ms[VoiceLatencyKind.REPLACEMENT_DISPATCH] == (704.0,)
    assert snapshot.latencies_ms[VoiceLatencyKind.FIRST_AUDIO] == (1_103.0,)
    assert snapshot.aec_states == (VoiceAecState.WARMING, VoiceAecState.HEALTHY)
    assert snapshot.erle_db == (23.5,)
    assert snapshot.underruns == 1
    assert snapshot.restarts == 2
    assert snapshot.conservative_entries == 1
    assert snapshot.duplicated_stt_audio_ms == 480
    assert snapshot.provider_results == (VoiceProviderResultClass.CANCELLED,)
    assert snapshot.winning_usage.calls == 1
    assert snapshot.discarded_usage.calls == 2


@pytest.mark.parametrize(
    ("call", "private_field"),
    [
        (
            lambda metrics: metrics.observe_latency(
                VoiceLatencyKind.FIRST_AUDIO,
                1,
                transcript_text="private",
            ),
            "transcript_text",
        ),
        (
            lambda metrics: metrics.observe_aec(
                VoiceAecState.HEALTHY,
                response_text="private",
            ),
            "response_text",
        ),
        (
            lambda metrics: metrics.record_usage(
                VoiceUsageOutcome.WINNING,
                VoiceUsageCounters(),
                pcm=b"private",
            ),
            "pcm",
        ),
    ],
)
def test_metrics_reject_content_fields(call, private_field: str) -> None:
    with pytest.raises(TypeError, match=private_field):
        call(VoiceMetrics())


def test_snapshot_schema_contains_no_content_bearing_fields() -> None:
    payload = asdict(VoiceMetrics().snapshot())

    assert "transcript" not in repr(payload).lower()
    assert "response_text" not in payload
    assert "pcm" not in payload
    assert "audio_bytes" not in payload


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True])
def test_metrics_reject_invalid_durations(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        VoiceMetrics().observe_latency(VoiceLatencyKind.FIRST_AUDIO, value)
