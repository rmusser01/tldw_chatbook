"""Deterministic synthetic corpus helpers for native voice AEC qualification."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Protocol


_FRAME_SAMPLES = 480
_FRAME_BYTES = _FRAME_SAMPLES * 2
_FRAMES_PER_SECOND = 100
_REQUIRED_CASE_KINDS = {
    "stationary_echo",
    "nonlinear_echo",
    "delay_step",
    "clock_drift",
    "double_talk",
    "render_under_overrun",
    "device_reset",
    "bluetooth_latency",
}
_THRESHOLD_KEYS = {
    "median_erle_db_min",
    "p10_erle_db_min",
    "false_barge_events_max",
    "false_barge_render_minutes_min",
    "double_talk_recall_min",
}


class VoiceAecCorpusError(ValueError):
    """Raised when the checked-in corpus contract is malformed or altered."""


class _AecProcessor(Protocol):
    def analyze_render(self, pcm16: bytes, *, delay_ms: int) -> None: ...

    def process_capture(self, pcm16: bytes, *, delay_ms: int) -> bytes: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class VoiceAecCorpusFrame:
    """One deterministic render/capture frame with content-free labels."""

    render_pcm16: bytes
    capture_pcm16: bytes
    delay_ms: int
    near_end_active: bool
    reset_before: bool = False
    render_discontinuity: bool = False


def voice_aec_case_recipe_sha256(case: Mapping[str, object]) -> str:
    """Hash one canonical synthetic recipe, excluding its declared digest."""

    recipe = dict(case)
    recipe.pop("recipe_sha256", None)
    encoded = json.dumps(
        recipe,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_voice_aec_corpus(path: Path) -> dict[str, Any]:
    """Load and strictly validate a content-safe synthetic corpus manifest."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VoiceAecCorpusError("voice AEC corpus manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise VoiceAecCorpusError("voice AEC corpus manifest must be an object")
    if set(manifest) != {
        "schema_version",
        "audio_format",
        "source",
        "thresholds",
        "cases",
    }:
        raise VoiceAecCorpusError("voice AEC corpus manifest fields are invalid")
    if manifest["schema_version"] != 1:
        raise VoiceAecCorpusError("unsupported voice AEC corpus schema")
    if manifest["audio_format"] != {
        "sample_rate_hz": 48_000,
        "channels": 1,
        "sample_format": "pcm_s16le",
        "frame_duration_ms": 10,
    }:
        raise VoiceAecCorpusError("voice AEC corpus audio format is invalid")
    if manifest["source"] != {
        "kind": "deterministic_synthesis",
        "contains_captured_audio": False,
        "contains_user_audio": False,
        "license_spdx": "CC0-1.0",
        "generator": "Packaging/voice_aec_corpus.py",
    }:
        raise VoiceAecCorpusError("voice AEC corpus source declaration is invalid")
    thresholds = manifest["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != _THRESHOLD_KEYS:
        raise VoiceAecCorpusError("voice AEC corpus thresholds are invalid")
    if thresholds != {
        "median_erle_db_min": 20.0,
        "p10_erle_db_min": 10.0,
        "false_barge_events_max": 1,
        "false_barge_render_minutes_min": 30.0,
        "double_talk_recall_min": 0.95,
    }:
        raise VoiceAecCorpusError("voice AEC corpus thresholds changed")

    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise VoiceAecCorpusError("voice AEC corpus cases must be a nonempty list")
    if not all(isinstance(case, dict) for case in cases):
        raise VoiceAecCorpusError("voice AEC corpus cases must be objects")
    if {case.get("kind") for case in cases} != _REQUIRED_CASE_KINDS:
        raise VoiceAecCorpusError("voice AEC corpus case coverage is incomplete")
    identifiers = [case.get("id") for case in cases]
    if any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise VoiceAecCorpusError("voice AEC corpus case IDs must be nonempty")
    if len(set(identifiers)) != len(identifiers):
        raise VoiceAecCorpusError("voice AEC corpus case IDs must be unique")
    for case in cases:
        _validate_case(case)
        if case["recipe_sha256"] != voice_aec_case_recipe_sha256(case):
            raise VoiceAecCorpusError(
                f"voice AEC recipe SHA-256 mismatch for {case['id']}"
            )
    false_barge_seconds = sum(
        float(case["duration_seconds"])
        for case in cases
        if case["measure_false_barge"] is True
    )
    if false_barge_seconds < 1_800.0:
        raise VoiceAecCorpusError("false-barge corpus must render at least 30 minutes")
    return manifest


def iter_voice_aec_case_frames(
    case: Mapping[str, object],
    *,
    limit: int | None = None,
) -> Iterator[VoiceAecCorpusFrame]:
    """Yield a deterministic 48 kHz mono corpus recipe without storing audio."""

    _validate_case(case)
    duration_frames = round(float(case["duration_seconds"]) * _FRAMES_PER_SECOND)
    if limit is not None:
        if limit < 0:
            raise ValueError("frame limit must be non-negative")
        duration_frames = min(duration_frames, limit)

    seed = int(case["seed"])
    render_state = seed
    near_state = seed ^ 0xA5A5A5A5
    render_history: list[tuple[int, ...]] = []
    maximum_delay_frames = 100
    for frame_index in range(duration_frames):
        render_samples, render_state = _synthetic_signal_frame(
            render_state,
            amplitude=int(case["render_amplitude"]),
            frame_index=frame_index,
        )
        render_history.append(render_samples)
        if len(render_history) > maximum_delay_frames + 1:
            del render_history[0]

        delay_ms = _case_delay_ms(case, frame_index)
        delay_frames = max(0, round(delay_ms / 10))
        if delay_frames >= len(render_history):
            delayed_render = (0,) * _FRAME_SAMPLES
        else:
            delayed_render = render_history[-delay_frames - 1]

        near_end_active = _near_end_active(case, frame_index)
        if near_end_active:
            near_end, near_state = _synthetic_signal_frame(
                near_state,
                amplitude=int(case["near_end_amplitude"]),
                frame_index=frame_index + 37,
                cadenced=False,
            )
        else:
            near_end = (0,) * _FRAME_SAMPLES

        capture_samples = tuple(
            _clip_pcm16(_echo_sample(sample, case) + near_end_sample)
            for sample, near_end_sample in zip(delayed_render, near_end)
        )
        reset_frame = int(case["reset_frame"])
        discontinuity_period = int(case["discontinuity_period_frames"])
        yield VoiceAecCorpusFrame(
            render_pcm16=_encode_pcm16(render_samples),
            capture_pcm16=_encode_pcm16(capture_samples),
            delay_ms=delay_ms,
            near_end_active=near_end_active,
            reset_before=reset_frame >= 0 and frame_index == reset_frame,
            render_discontinuity=(
                discontinuity_period > 0
                and frame_index > 0
                and frame_index % discontinuity_period == 0
            ),
        )


def evaluate_voice_aec_corpus(
    manifest: Mapping[str, Any],
    *,
    processor_factory: Callable[[], _AecProcessor],
    frame_limit_per_case: int | None = None,
) -> dict[str, Any]:
    """Run the native-shape AEC contract and return content-free evidence."""

    thresholds = manifest["thresholds"]
    erle_values: list[float] = []
    false_barge_events = 0
    false_barge_frames = 0
    double_talk_expected = 0
    double_talk_detected = 0
    case_results: list[dict[str, object]] = []

    for case in manifest["cases"]:
        processor = processor_factory()
        case_erle: list[float] = []
        case_false_events = 0
        case_double_expected = 0
        case_double_detected = 0
        prior_false_speech = False
        case_error: str | None = None
        warmup_frames = int(case["warmup_seconds"] * _FRAMES_PER_SECOND)
        frames_seen = 0
        try:
            for frame_index, frame in enumerate(
                iter_voice_aec_case_frames(case, limit=frame_limit_per_case)
            ):
                frames_seen += 1
                if frame.reset_before or frame.render_discontinuity:
                    processor.reset()
                if not frame.render_discontinuity:
                    processor.analyze_render(
                        frame.render_pcm16, delay_ms=frame.delay_ms
                    )
                cleaned = processor.process_capture(
                    frame.capture_pcm16,
                    delay_ms=frame.delay_ms,
                )
                if not isinstance(cleaned, bytes) or len(cleaned) != _FRAME_BYTES:
                    raise VoiceAecCorpusError("AEC returned an invalid PCM frame")
                if frame_index < warmup_frames:
                    continue

                speech = _energy_vad(cleaned)
                if case["measure_false_barge"] is True and not frame.near_end_active:
                    false_barge_frames += 1
                    if speech and not prior_false_speech:
                        false_barge_events += 1
                        case_false_events += 1
                    prior_false_speech = speech
                if case["measure_double_talk"] is True and frame.near_end_active:
                    double_talk_expected += 1
                    case_double_expected += 1
                    if speech:
                        double_talk_detected += 1
                        case_double_detected += 1
                if case["measure_erle"] is True and not frame.near_end_active:
                    input_energy = _pcm_energy(frame.capture_pcm16)
                    if input_energy > _FRAME_SAMPLES * 250 * 250:
                        erle = 10.0 * math.log10(
                            (input_energy + 1.0) / (_pcm_energy(cleaned) + 1.0)
                        )
                        erle_values.append(erle)
                        case_erle.append(erle)
        except Exception as exc:
            case_error = type(exc).__name__

        case_passed = case_error is None and frames_seen > 0
        if case["measure_erle"] is True:
            case_passed = case_passed and bool(case_erle)
        if case["measure_double_talk"] is True:
            case_passed = case_passed and case_double_expected > 0
        case_results.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "passed": case_passed,
                "error_class": case_error,
                "frames": frames_seen,
                "median_erle_db": _median(case_erle),
                "false_barge_events": case_false_events,
                "double_talk_recall": _ratio(
                    case_double_detected,
                    case_double_expected,
                ),
            }
        )

    median_erle = _median(erle_values)
    p10_erle = _percentile(erle_values, 0.10)
    false_barge_minutes = false_barge_frames / _FRAMES_PER_SECOND / 60.0
    double_talk_recall = _ratio(double_talk_detected, double_talk_expected)
    aggregate_passed = (
        median_erle >= thresholds["median_erle_db_min"]
        and p10_erle >= thresholds["p10_erle_db_min"]
        and false_barge_events <= thresholds["false_barge_events_max"]
        and false_barge_minutes >= thresholds["false_barge_render_minutes_min"]
        and double_talk_recall >= thresholds["double_talk_recall_min"]
    )
    unqualified = [
        str(result["id"]) for result in case_results if result["passed"] is not True
    ]
    if not aggregate_passed and not unqualified:
        unqualified = ["aggregate-thresholds"]
    passed = aggregate_passed and not unqualified
    return {
        "schema_version": 1,
        "thresholds": dict(thresholds),
        "median_erle_db": median_erle,
        "p10_erle_db": p10_erle,
        "false_barge_events": false_barge_events,
        "false_barge_render_minutes": false_barge_minutes,
        "double_talk_recall": double_talk_recall,
        "case_results": case_results,
        "unqualified_case_ids": unqualified,
        "effective_mode": "full-duplex" if passed else "half-duplex",
        "passed": passed,
    }


def _validate_case(case: Mapping[str, object]) -> None:
    required = {
        "id",
        "kind",
        "duration_seconds",
        "seed",
        "render_amplitude",
        "echo_gain_milli",
        "nonlinear_milli",
        "delay_ms",
        "delay_after_ms",
        "delay_step_frame",
        "drift_interval_frames",
        "near_end_amplitude",
        "near_end_on_frames",
        "near_end_off_frames",
        "warmup_seconds",
        "reset_frame",
        "discontinuity_period_frames",
        "measure_erle",
        "measure_false_barge",
        "measure_double_talk",
        "recipe_sha256",
    }
    if set(case) != required:
        raise VoiceAecCorpusError("voice AEC case fields are invalid")
    if case["kind"] not in _REQUIRED_CASE_KINDS:
        raise VoiceAecCorpusError("voice AEC case kind is invalid")
    integer_fields = required - {
        "id",
        "kind",
        "duration_seconds",
        "warmup_seconds",
        "measure_erle",
        "measure_false_barge",
        "measure_double_talk",
        "recipe_sha256",
    }
    if any(type(case[name]) is not int for name in integer_fields):
        raise VoiceAecCorpusError("voice AEC integer recipe field is invalid")
    if (
        not isinstance(case["duration_seconds"], (int, float))
        or float(case["duration_seconds"]) <= 0
    ):
        raise VoiceAecCorpusError("voice AEC duration must be positive")
    if (
        not isinstance(case["warmup_seconds"], (int, float))
        or float(case["warmup_seconds"]) < 0
    ):
        raise VoiceAecCorpusError("voice AEC warmup must be non-negative")
    if float(case["warmup_seconds"]) >= float(case["duration_seconds"]):
        raise VoiceAecCorpusError("voice AEC warmup must be shorter than the case")
    if (
        not 0 <= int(case["delay_ms"]) <= 1_000
        or not 0 <= int(case["delay_after_ms"]) <= 1_000
    ):
        raise VoiceAecCorpusError("voice AEC delay is out of range")
    if not 0 <= int(case["echo_gain_milli"]) <= 2_000:
        raise VoiceAecCorpusError("voice AEC echo gain is out of range")
    if not 0 <= int(case["nonlinear_milli"]) <= 1_000:
        raise VoiceAecCorpusError("voice AEC nonlinearity is out of range")
    if any(
        type(case[name]) is not bool
        for name in ("measure_erle", "measure_false_barge", "measure_double_talk")
    ):
        raise VoiceAecCorpusError("voice AEC measurement flags must be booleans")
    digest = case["recipe_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise VoiceAecCorpusError("voice AEC recipe SHA-256 is invalid")


def _synthetic_signal_frame(
    state: int,
    *,
    amplitude: int,
    frame_index: int,
    cadenced: bool = True,
) -> tuple[tuple[int, ...], int]:
    samples: list[int] = []
    prior = 0
    active = not cadenced or frame_index % 120 < 95
    for _ in range(_FRAME_SAMPLES):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
        white = ((state >> 16) & 0xFFFF) - 32_768
        prior = (3 * prior + white) // 4
        samples.append((prior * amplitude // 32_768) if active else 0)
    return tuple(samples), state


def _case_delay_ms(case: Mapping[str, object], frame_index: int) -> int:
    delay = int(case["delay_ms"])
    step_frame = int(case["delay_step_frame"])
    if step_frame >= 0 and frame_index >= step_frame:
        delay = int(case["delay_after_ms"])
    drift_interval = int(case["drift_interval_frames"])
    if drift_interval > 0:
        delay += frame_index // drift_interval
    return min(delay, 1_000)


def _near_end_active(case: Mapping[str, object], frame_index: int) -> bool:
    on_frames = int(case["near_end_on_frames"])
    off_frames = int(case["near_end_off_frames"])
    if on_frames <= 0:
        return False
    cycle = on_frames + max(0, off_frames)
    return cycle > 0 and frame_index % cycle < on_frames


def _echo_sample(sample: int, case: Mapping[str, object]) -> int:
    echo = sample * int(case["echo_gain_milli"]) // 1_000
    nonlinear = int(case["nonlinear_milli"])
    if nonlinear:
        squared = abs(sample) * abs(sample) // 32_768
        echo += (squared if sample >= 0 else -squared) * nonlinear // 1_000
    return echo


def _clip_pcm16(value: int) -> int:
    return max(-32_768, min(32_767, value))


def _encode_pcm16(samples: tuple[int, ...]) -> bytes:
    encoded = array("h", samples)
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def _pcm_energy(pcm16: bytes) -> float:
    samples = array("h")
    samples.frombytes(pcm16)
    if sys.byteorder != "little":
        samples.byteswap()
    return float(sum(sample * sample for sample in samples))


def _energy_vad(pcm16: bytes) -> bool:
    samples = array("h")
    samples.frombytes(pcm16)
    if sys.byteorder != "little":
        samples.byteswap()
    # Match VoicePreprocessor's dependency-free fallback VAD exactly so the
    # corpus qualifies the same admission boundary used at runtime.
    return sum(abs(sample) for sample in samples) >= len(samples) * 250


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return float(ordered[index])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
