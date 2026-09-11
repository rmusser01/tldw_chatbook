"""Deterministic, content-safe qualification corpus for speculative voice AEC."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from Packaging.voice_aec_corpus import (
    VoiceAecCorpusError,
    evaluate_voice_aec_corpus,
    iter_voice_aec_case_frames,
    load_voice_aec_corpus,
    voice_aec_case_recipe_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "Tests" / "Audio" / "fixtures" / "voice_aec" / "manifest.json"
REQUIRED_CASE_KINDS = {
    "stationary_echo",
    "nonlinear_echo",
    "delay_step",
    "clock_drift",
    "double_talk",
    "render_under_overrun",
    "device_reset",
    "bluetooth_latency",
}
REQUIRED_THRESHOLDS = {
    "median_erle_db_min": 20.0,
    "p10_erle_db_min": 10.0,
    "false_barge_events_max": 1,
    "false_barge_render_minutes_min": 30.0,
    "double_talk_recall_min": 0.95,
}


def test_manifest_is_licensed_synthetic_and_covers_required_faults() -> None:
    manifest = load_voice_aec_corpus(MANIFEST_PATH)

    assert manifest["schema_version"] == 1
    assert manifest["audio_format"] == {
        "sample_rate_hz": 48_000,
        "channels": 1,
        "sample_format": "pcm_s16le",
        "frame_duration_ms": 10,
    }
    assert manifest["source"] == {
        "kind": "deterministic_synthesis",
        "contains_captured_audio": False,
        "contains_user_audio": False,
        "license_spdx": "CC0-1.0",
        "generator": "Packaging/voice_aec_corpus.py",
    }
    assert manifest["thresholds"] == REQUIRED_THRESHOLDS

    cases = manifest["cases"]
    assert {case["kind"] for case in cases} == REQUIRED_CASE_KINDS
    assert len({case["id"] for case in cases}) == len(cases)
    assert (
        sum(
            float(case["duration_seconds"])
            for case in cases
            if case["kind"] == "stationary_echo"
        )
        >= 1_800.0
    )
    assert all(
        case["recipe_sha256"] == voice_aec_case_recipe_sha256(case) for case in cases
    )
    assert not any(
        forbidden in case
        for case in cases
        for forbidden in ("transcript", "response", "captured_audio", "file_path")
    )


def test_manifest_rejects_recipe_tampering(tmp_path: Path) -> None:
    manifest = MANIFEST_PATH.read_text(encoding="utf-8")
    tampered = tmp_path / "manifest.json"
    tampered.write_text(
        manifest.replace('"echo_gain_milli": 550', '"echo_gain_milli": 551', 1),
        encoding="utf-8",
    )

    with pytest.raises(VoiceAecCorpusError, match="recipe SHA-256"):
        load_voice_aec_corpus(tampered)


def test_synthetic_frames_are_deterministic_and_have_no_external_payload() -> None:
    manifest = load_voice_aec_corpus(MANIFEST_PATH)
    case = next(case for case in manifest["cases"] if case["kind"] == "double_talk")

    first = list(iter_voice_aec_case_frames(case, limit=8))
    second = list(iter_voice_aec_case_frames(case, limit=8))

    assert first == second
    assert len(first) == 8
    assert all(len(frame.render_pcm16) == 960 for frame in first)
    assert all(len(frame.capture_pcm16) == 960 for frame in first)
    assert any(frame.near_end_active for frame in first)


class _PassthroughAec:
    def analyze_render(self, _pcm16: bytes, *, delay_ms: int) -> None:
        assert 0 <= delay_ms <= 1_000

    def process_capture(self, pcm16: bytes, *, delay_ms: int) -> bytes:
        assert 0 <= delay_ms <= 1_000
        return pcm16

    def reset(self) -> None:
        return None


def test_any_unqualified_reference_case_fails_the_overall_gate() -> None:
    manifest = load_voice_aec_corpus(MANIFEST_PATH)

    report = evaluate_voice_aec_corpus(
        manifest,
        processor_factory=lambda: _PassthroughAec(),
        frame_limit_per_case=800,
    )

    double_talk = next(
        result for result in report["case_results"] if result["kind"] == "double_talk"
    )
    assert double_talk["double_talk_recall"] >= 0.95
    assert report["passed"] is False
    assert report["unqualified_case_ids"]
    assert report["effective_mode"] == "half-duplex"


@pytest.mark.skipif(
    importlib.util.find_spec("tldw_voice_aec") is None,
    reason="native AEC companion is not installed in this optional test environment",
)
def test_installed_native_companion_meets_reference_corpus_thresholds() -> None:
    import tldw_voice_aec

    manifest = load_voice_aec_corpus(MANIFEST_PATH)
    report = evaluate_voice_aec_corpus(
        manifest,
        processor_factory=lambda: tldw_voice_aec.AecProcessor(
            sample_rate=48_000,
            channels=1,
        ),
    )

    assert report["median_erle_db"] >= 20.0
    assert report["p10_erle_db"] >= 10.0
    assert report["false_barge_events"] <= 1
    assert report["false_barge_render_minutes"] >= 30.0
    assert report["double_talk_recall"] >= 0.95
    assert report["unqualified_case_ids"] == []
    assert report["effective_mode"] == "full-duplex"
    assert report["passed"] is True
