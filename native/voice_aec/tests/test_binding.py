"""Contract tests for the native WebRTC AEC3 Python binding."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import math
import random
import struct
from collections import deque

import pytest


FRAME = bytes(480 * 2)
METRIC_KEYS = {
    "erle_db",
    "delay_ms",
    "delay_estimate_available",
    "delay_estimate_refined",
    "delay_age_blocks",
    "clock_drift",
}


def _processor():
    module = importlib.import_module("tldw_voice_aec")
    return module.AecProcessor(sample_rate=48_000, channels=1)


def test_processor_accepts_one_ten_ms_48khz_mono_frame() -> None:
    processor = _processor()

    processor.analyze_render(FRAME, delay_ms=20)
    cleaned = processor.process_capture(FRAME, delay_ms=20)

    assert isinstance(cleaned, bytes)
    assert len(cleaned) == len(FRAME)
    assert METRIC_KEYS <= processor.metrics().keys()


@pytest.mark.parametrize("sample_rate", [8_000, 16_000, 44_100, 96_000])
def test_processor_rejects_unsupported_sample_rates(sample_rate: int) -> None:
    module = importlib.import_module("tldw_voice_aec")

    with pytest.raises(ValueError, match="sample_rate"):
        module.AecProcessor(sample_rate=sample_rate, channels=1)


@pytest.mark.parametrize("channels", [0, 2, 6])
def test_processor_rejects_non_mono_channels(channels: int) -> None:
    module = importlib.import_module("tldw_voice_aec")

    with pytest.raises(ValueError, match="channels"):
        module.AecProcessor(sample_rate=48_000, channels=channels)


@pytest.mark.parametrize("size", [0, 2, 958, 962, 1_920])
@pytest.mark.parametrize("method_name", ["analyze_render", "process_capture"])
def test_processor_rejects_wrong_frame_size(size: int, method_name: str) -> None:
    processor = _processor()

    with pytest.raises(ValueError, match="960"):
        getattr(processor, method_name)(bytes(size), delay_ms=20)


@pytest.mark.parametrize("value", [bytearray(FRAME), memoryview(FRAME), "not pcm"])
@pytest.mark.parametrize("method_name", ["analyze_render", "process_capture"])
def test_processor_requires_bytes(value: object, method_name: str) -> None:
    processor = _processor()

    with pytest.raises(TypeError, match="bytes"):
        getattr(processor, method_name)(value, delay_ms=20)


@pytest.mark.parametrize("delay_ms", [-1, 1_001])
@pytest.mark.parametrize("method_name", ["analyze_render", "process_capture"])
def test_processor_rejects_invalid_delay(delay_ms: int, method_name: str) -> None:
    processor = _processor()

    with pytest.raises(ValueError, match="delay_ms"):
        getattr(processor, method_name)(FRAME, delay_ms=delay_ms)


def test_reset_allows_processing_to_resume() -> None:
    processor = _processor()
    processor.analyze_render(FRAME, delay_ms=20)
    processor.process_capture(FRAME, delay_ms=20)

    assert processor.reset() is None
    processor.analyze_render(FRAME, delay_ms=0)
    assert len(processor.process_capture(FRAME, delay_ms=0)) == len(FRAME)


def test_metrics_are_finite_floats() -> None:
    processor = _processor()
    processor.analyze_render(FRAME, delay_ms=20)
    processor.process_capture(FRAME, delay_ms=20)

    metrics = processor.metrics()

    assert set(metrics) == METRIC_KEYS
    assert all(isinstance(value, float) for value in metrics.values())
    assert all(
        value == value and abs(value) != float("inf") for value in metrics.values()
    )


def test_binding_does_not_expose_webrtc_types() -> None:
    module = importlib.import_module("tldw_voice_aec")

    assert module.__all__ == [
        "AecProcessor",
        "DUPLEX_ABI_VERSION",
        "NativeDuplexBridge",
    ]
    assert [name for name in dir(module) if not name.startswith("_")] == [
        "AecProcessor",
        "DUPLEX_ABI_VERSION",
        "NativeDuplexBridge",
    ]


def test_processor_constructor_is_keyword_only() -> None:
    module = importlib.import_module("tldw_voice_aec")

    with pytest.raises(TypeError):
        module.AecProcessor(48_000, 1)


def test_binding_is_a_native_extension() -> None:
    spec = importlib.util.find_spec("tldw_voice_aec._native")

    assert spec is not None
    assert spec.origin is not None
    assert any(
        spec.origin.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ), spec.origin


def test_repeated_echo_reference_changes_non_silent_capture() -> None:
    processor = _processor()
    samples = [
        round(10_000 * math.sin(2 * math.pi * 440 * index / 48_000))
        for index in range(480)
    ]
    echo = struct.pack("<480h", *samples)
    cleaned = echo

    for _ in range(80):
        processor.analyze_render(echo, delay_ms=0)
        cleaned = processor.process_capture(echo, delay_ms=0)

    assert cleaned != echo


def test_delayed_echo_exposes_refined_fresh_per_instance_delay_evidence() -> None:
    processor = _processor()
    untouched_processor = _processor()
    rng = random.Random(0xAEC3)
    delayed_render = deque([FRAME] * 4)
    observed: list[dict[str, float]] = []
    capture_energy = 0
    cleaned_energy = 0
    attenuation_frames = 0

    # Pinned AEC3 defaults converge to about 6 dB on this deterministic corpus.
    # Five dB leaves headroom for platform floating-point differences while still
    # proving that the exported ERLE is reachable evidence, not a sentinel value.
    minimum_reachable_erle_db = 5.0
    minimum_meaningful_delay_ms = 20.0

    for _ in range(1_200):
        render = struct.pack(
            "<480h", *(rng.randint(-12_000, 12_000) for _ in range(480))
        )
        capture = delayed_render.popleft()
        delayed_render.append(render)
        processor.analyze_render(render, delay_ms=40)
        cleaned = processor.process_capture(capture, delay_ms=40)
        metrics = processor.metrics()
        observed.append(metrics)
        if (
            metrics["delay_estimate_available"] == 1.0
            and metrics["delay_estimate_refined"] == 1.0
            and metrics["delay_age_blocks"] <= 25.0
            and metrics["clock_drift"] == 0.0
            and metrics["delay_ms"] >= minimum_meaningful_delay_ms
            and metrics["erle_db"] >= minimum_reachable_erle_db
        ):
            capture_samples = struct.unpack("<480h", capture)
            cleaned_samples = struct.unpack("<480h", cleaned)
            capture_energy += sum(sample * sample for sample in capture_samples)
            cleaned_energy += sum(sample * sample for sample in cleaned_samples)
            attenuation_frames += 1
            if attenuation_frames == 100:
                break

    assert observed[-1]["delay_estimate_available"] == 1.0
    assert observed[-1]["delay_estimate_refined"] == 1.0
    assert 0.0 <= observed[-1]["delay_age_blocks"] <= 25.0
    assert observed[-1]["clock_drift"] == 0.0
    assert observed[-1]["delay_ms"] >= minimum_meaningful_delay_ms
    assert observed[-1]["erle_db"] >= minimum_reachable_erle_db
    assert attenuation_frames == 100
    assert cleaned_energy < capture_energy * 0.5

    assert untouched_processor.metrics() == {
        "erle_db": 0.0,
        "delay_ms": 0.0,
        "delay_estimate_available": 0.0,
        "delay_estimate_refined": 0.0,
        "delay_age_blocks": 0.0,
        "clock_drift": 0.0,
    }
