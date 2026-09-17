"""Reproduce misleading YouTube dependency warnings on a portable STT install."""

from __future__ import annotations

import pytest

from tldw_chatbook.Library import ingest_capabilities as capabilities
from tldw_chatbook.Library.ingest_preflight import analyze_path
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    build_library_ingest_state,
)

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_youtube_preflight_never_advertises_retired_mlx_packages(monkeypatch, platform):
    monkeypatch.setattr("sys.platform", platform)
    monkeypatch.setattr(capabilities, "_is_installed", lambda feature: False)
    result = analyze_path(URL, probe_url=False)
    assert result.type_groups == {"audio_video": [URL]}
    assert not result.errors
    assert not any("mlx" in str(warning).lower() for warning in result.warnings)
    assert any(warning["feature"] == "transcribe_cpp" for warning in result.warnings)


@pytest.mark.parametrize(
    "provider, feature",
    [
        ("default", "faster_whisper"),
        ("faster-whisper", "faster_whisper"),
        ("parakeet-onnx", "parakeet_onnx"),
        ("transcribe-cpp", "transcribe_cpp"),
    ],
)
def test_installed_selected_stt_needs_no_other_backend_or_warning_consent(
    monkeypatch, provider, feature
):
    installed = {"audio_processing", "video_processing", "yt_dlp", feature}
    monkeypatch.setattr(capabilities, "_is_installed", lambda name: name in installed)
    preflight = analyze_path(URL, probe_url=False)
    form = LibraryIngestFormState(
        path=URL,
        preflight=preflight,
        type_options={"audio_video": {"transcription_provider": provider}},
    )
    state = build_library_ingest_state((), form=form, transcribe_cpp_configured=True)
    assert state.warning_lines == []
    assert state.warning_commands == ()
    assert state.forecast.consent_affected == 0


def test_changing_provider_reprojects_captured_warnings_and_preserves_missing_tools(
    monkeypatch,
):
    monkeypatch.setattr(
        capabilities,
        "_is_installed",
        lambda name: name in {"audio_processing", "video_processing", "faster_whisper"},
    )
    preflight = analyze_path(URL, probe_url=False)
    form = LibraryIngestFormState(path=URL, preflight=preflight)
    first = build_library_ingest_state((), form=form)
    assert len(first.warning_lines) == 1
    assert "yt-dlp" in first.warning_lines[0]
    form.type_options = {"audio_video": {"transcription_provider": "parakeet-onnx"}}
    second = build_library_ingest_state((), form=form)
    assert len(second.warning_lines) == 2
    assert any("Parakeet ONNX" in line for line in second.warning_lines)
    form.type_options = {"audio_video": {"transcription_provider": "default"}}
    third = build_library_ingest_state((), form=form)
    assert third.warning_lines == first.warning_lines
    assert form.preflight is preflight


@pytest.mark.parametrize("provider", ["retired-provider", None, 12, [], {}])
def test_invalid_saved_provider_keeps_warnings_and_blocks_start(monkeypatch, provider):
    monkeypatch.setattr(
        capabilities, "_is_installed", lambda name: name == "audio_processing"
    )
    monkeypatch.setattr(
        "tldw_chatbook.Library.library_ingest_state._dependency_installed",
        lambda name: True,
    )
    preflight = analyze_path(URL, probe_url=False)
    form = LibraryIngestFormState(
        path=URL,
        preflight=preflight,
        type_options={"audio_video": {"transcription_provider": provider}},
    )
    state = build_library_ingest_state((), form=form)
    assert any("Faster Whisper" in warning for warning in state.warning_lines)
    assert state.forecast.consent_affected > 0
    assert not state.start_enabled
    assert any(
        error[:2] == ("audio_video", "transcription_provider")
        for error in state.option_errors
    )
