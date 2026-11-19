"""Deterministic PCM, gate and owner/freshness contracts without audio hardware."""

import struct
import subprocess
import sys
from dataclasses import replace

import pytest

from tldw_chatbook.Audio.speech_presentation import (
    PcmEnvelope,
    SpeechEnvelopeFilter,
    SpeechPresentation,
    SpeechPresentationOwner,
    SpeechPresentationSnapshot,
)


def test_pcm_sample_split_matches_one_chunk_with_known_energy():
    pcm = struct.pack("<960h", *([8192] * 960))
    whole = PcmEnvelope(sample_rate=24000, channels=1)
    split = PcmEnvelope(sample_rate=24000, channels=1)
    expected = whole.push(pcm) + whole.finish()
    actual = (
        split.push(pcm[:1])
        + split.push(pcm[1:673])
        + split.push(pcm[673:])
        + split.finish()
    )
    assert actual == expected
    assert [(w.start_frame, w.frame_count) for w in actual] == [(0, 480), (480, 480)]
    assert all(w.dbfs == pytest.approx(-12.041199826559248) for w in actual)


def test_stereo_antiphase_preserves_energy_instead_of_averaging_to_zero():
    envelope = PcmEnvelope(sample_rate=48000, channels=2)
    windows = envelope.push(struct.pack("<1920h", *([8192, -8192] * 960)))
    assert len(windows) == 1
    assert windows[0].frame_count == 960
    assert windows[0].dbfs == pytest.approx(-12.041199826559248)


def test_fractional_window_rate_does_not_accumulate_source_position_drift():
    pcm = struct.pack("<11025h", *([16384] * 11025))
    envelope = PcmEnvelope(sample_rate=11025, channels=1)
    windows = []
    for start in range(0, len(pcm), 17):
        windows.extend(envelope.push(pcm[start : start + 17]))
    windows.extend(envelope.finish())
    assert len(windows) == 50
    assert sum(w.frame_count for w in windows) == 11025
    assert windows[-1].start_frame + windows[-1].frame_count == 11025
    assert all(w.dbfs == pytest.approx(-6.020599913279624) for w in windows)


def test_zero_and_partial_final_windows_have_correct_positions_and_finish_is_idempotent():
    envelope = PcmEnvelope(sample_rate=24000, channels=1)
    assert envelope.push(b"\x00" * (480 + 12) * 2)[0].dbfs == float("-inf")
    final = envelope.finish()
    assert len(final) == 1
    assert (final[0].start_frame, final[0].frame_count, final[0].dbfs) == (
        480,
        12,
        float("-inf"),
    )
    assert envelope.finish() == ()
    with pytest.raises(ValueError):
        envelope.push(b"\x00\x00")


@pytest.mark.parametrize(
    "rate,channels",
    [
        (0, 1),
        (-1, 1),
        (True, 1),
        ("24000", 1),
        (1, 1),
        (24000, 0),
        (24000, -1),
        (24000, True),
    ],
)
def test_invalid_pcm_format_is_rejected(rate, channels):
    with pytest.raises(ValueError):
        PcmEnvelope(sample_rate=rate, channels=channels)


@pytest.mark.parametrize("channels,tail", [(1, b"x"), (2, b"xx"), (2, b"xxx")])
def test_incomplete_final_pcm_frame_is_rejected(channels, tail):
    envelope = PcmEnvelope(sample_rate=24000, channels=channels)
    assert envelope.push(tail) == ()
    with pytest.raises(ValueError):
        envelope.finish()


def test_rejected_pcm_input_does_not_advance_source_position():
    envelope = PcmEnvelope(sample_rate=24000, channels=1)
    with pytest.raises(ValueError):
        envelope.push("not PCM bytes")
    assert envelope.push(b"\x00" * 960)[0].start_frame == 0


def test_nonzero_pause_bypasses_release_and_reopens_finitely():
    f = SpeechEnvelopeFilter()
    for _ in range(10):
        assert f.update(-12.0) == "open"
    assert f.update(-60.0) == "open"
    assert f.update(-60.0) == "rest"
    assert f.update(-12.0) == "open"
    assert f.update(float("-inf")) == "rest"
    assert f.update(-12.0) == "open"


def test_gate_thresholds_and_actual_window_duration():
    f = SpeechEnvelopeFilter()
    assert f.update(-42.01) == "rest"
    assert f.update(-42.0) == "small"
    assert f.update(-12.0, duration_ms=1000) == "open"
    assert f.update(-45.0) == "open"
    assert f.update(-45.0) == "open"
    assert f.update(-60.0, duration_ms=25) != "rest"
    assert f.update(-60.0, duration_ms=15) == "rest"
    assert f.update(-42.01) == "rest"
    assert f.update(-42.0) == "small"


def test_attack_and_release_cross_the_expected_hysteresis_boundaries():
    attack = SpeechEnvelopeFilter()
    assert attack.update(-30.0) == "small"
    assert attack.update(-12.0) == "open"  # 40 ms attack: -22.9176 after 20 ms
    release = SpeechEnvelopeFilter()
    assert release.update(-12.0) == "open"
    assert [release.update(-40.0) for _ in range(4)] == [
        "open",
        "open",
        "open",
        "small",
    ]
    release.reset()
    assert release.update(-24.0) == "open"
    assert all(release.update(-26.0) == "open" for _ in range(40))
    assert [release.update(-28.0) for _ in range(40)][-1] == "small"


@pytest.mark.parametrize("level", [float("nan"), float("inf"), float("-inf")])
def test_invalid_or_zero_level_resets_without_poisoning_reopen(level):
    f = SpeechEnvelopeFilter()
    assert f.update(-12.0) == "open"
    assert f.update(level) == "rest"
    assert f.update(-12.0) == "open"


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_window_duration_is_rejected(duration):
    with pytest.raises(ValueError):
        SpeechEnvelopeFilter().update(-12.0, duration_ms=duration)


@pytest.fixture
def presentation_case():
    clock = [0.0]
    owner = SpeechPresentationOwner("token", object(), "session", 1, 7, "message", None)
    channel = SpeechPresentation(owner, clock=lambda: clock[0])
    snapshot = SpeechPresentationSnapshot(owner, 1, "playing", 0.0, -12.0, "open", True)
    return channel, owner, snapshot, clock


def test_snapshot_is_latest_only_and_owner_generation_fenced(presentation_case):
    channel, owner, snapshot, clock = presentation_case
    assert channel.read() is None
    channel.publish(snapshot)
    assert channel.read() == snapshot
    newer = replace(snapshot, generation=2, mouth="small", level_dbfs=-30.0)
    channel.publish(newer)
    channel.publish(snapshot)
    channel.publish(replace(newer, owner=replace(owner, character_id=8), mouth="open"))
    assert channel.read() == newer
    clock[0] = 0.02
    newest = replace(newer, observed_at=0.02, mouth="open")
    channel.publish(newest)
    channel.publish(newer)
    assert channel.read() == newest


def test_freshness_expires_without_changing_confirmed_transport(presentation_case):
    channel, _, snapshot, clock = presentation_case
    channel.publish(snapshot)
    clock[0] = 0.1
    assert channel.read().mouth == "open"
    clock[0] = 0.100001
    current = channel.read()
    assert current.state == "playing" and current.mouth == "rest"
    assert current.level_dbfs is None and not current.timing_valid


def test_pause_and_terminal_close_immediately_and_late_progress_cannot_reopen(
    presentation_case,
):
    channel, _, snapshot, clock = presentation_case
    channel.publish(snapshot)
    clock[0] = 0.01
    channel.set_transport("paused")
    assert channel.read().mouth == "rest" and channel.read().state == "paused"
    channel.publish(replace(snapshot, observed_at=0.02))
    assert channel.read().state == "paused"
    clock[0] = 0.03
    channel.set_transport("playing")
    assert channel.read().mouth == "rest"
    channel.publish(replace(snapshot, observed_at=0.03))
    assert channel.read().mouth == "open"
    channel.set_transport("stopped")
    channel.publish(replace(snapshot, generation=99, observed_at=0.03))
    channel.set_transport("playing")
    assert channel.read().state == "stopped" and channel.read().mouth == "rest"


def test_unknown_timing_and_future_or_invalid_observation_are_closed(presentation_case):
    channel, _, snapshot, _clock = presentation_case
    channel.publish(replace(snapshot, timing_valid=False))
    assert channel.read().mouth == "rest" and channel.read().level_dbfs is None
    channel.publish(replace(snapshot, observed_at=1.0))
    assert channel.read().mouth == "rest"


def test_observer_clock_failure_does_not_escape_into_audio(presentation_case):
    _channel, owner, snapshot, _clock = presentation_case

    def broken_clock():
        raise RuntimeError("observer clock failed")

    failed = SpeechPresentation(owner, clock=broken_clock)
    failed.publish(snapshot)
    assert failed.read().mouth == "rest"
    failed.set_transport("failed")
    assert failed.read().state == "failed"


def test_issued_owner_cannot_be_retargeted(presentation_case):
    channel, owner, _snapshot, _clock = presentation_case
    with pytest.raises(AttributeError):
        channel.owner = replace(owner, character_id=8)
    assert channel.owner is owner


def test_missing_level_never_opens_a_qualified_silent_mouth(presentation_case):
    channel, _owner, snapshot, _clock = presentation_case
    channel.publish(replace(snapshot, level_dbfs=None))
    current = channel.read()
    assert current.mouth == "rest" and current.level_dbfs is None
    assert current.timing_valid  # Qualified silence, not loss of device timing.


def test_late_preparing_transport_cannot_regress_playing(presentation_case):
    channel, _owner, snapshot, clock = presentation_case
    channel.publish(snapshot)
    clock[0] = 0.02
    channel.set_transport("preparing")
    assert channel.read().state == "playing"
    channel.publish(replace(snapshot, state="preparing", observed_at=0.02))
    assert channel.read().state == "playing"


def test_digital_zero_snapshot_keeps_qualified_timing_without_infinity(
    presentation_case,
):
    channel, _owner, snapshot, _clock = presentation_case
    channel.publish(replace(snapshot, level_dbfs=float("-inf")))
    current = channel.read()
    assert current.mouth == "rest" and current.level_dbfs is None
    assert current.timing_valid


def test_fresh_import_has_no_device_ui_or_tts_side_effects():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tldw_chatbook.Audio.speech_presentation; assert not any(name == 'textual' or name == 'sounddevice' or name.startswith('tldw_chatbook.TTS') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_gated_nonzero_silence_clears_meter_level_without_losing_timing(
    presentation_case,
):
    channel, _owner, snapshot, _clock = presentation_case
    channel.publish(replace(snapshot, mouth="rest", level_dbfs=-60.0))
    current = channel.read()
    assert current.mouth == "rest" and current.level_dbfs is None
    assert current.timing_valid


def test_two_fractional_duration_raw_windows_close_without_rounding_delay():
    f = SpeechEnvelopeFilter()
    assert f.update(-12.0) == "open"
    assert f.update(-60.0, duration_ms=19.999) == "open"
    assert f.update(-60.0, duration_ms=19.999) == "rest"
