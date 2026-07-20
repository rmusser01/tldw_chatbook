"""Tests for the library ingestion capability discovery module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import tldw_chatbook.Library.ingest_capabilities
from tldw_chatbook.Library.ingest_capabilities import (
    OptionField,
    TypeGroupCapabilities,
    _feature_label,
    _install_hint,
    _is_installed,
    _TYPE_GROUPS,
    get_capabilities,
    get_tooling_warnings,
    get_type_group,
)
from tldw_chatbook.Utils.optional_deps import OPTIONAL_FEATURES


@pytest.mark.parametrize(
    ("path", "expected_group"),
    [
        ("/tmp/document.pdf", "pdf"),
        ("/tmp/recording.mp3", "audio_video"),
        ("/tmp/recording.m4a", "audio_video"),
        ("/tmp/movie.mp4", "audio_video"),
        ("/tmp/movie.mkv", "audio_video"),
        ("/tmp/book.epub", "ebook"),
        ("/tmp/book.mobi", "ebook"),
        ("/tmp/notes.txt", "generic"),
        ("/tmp/notes.md", "generic"),
        ("/tmp/spreadsheet.csv", "generic"),
        ("/tmp/page.html", "generic"),
        ("/tmp/document.docx", "generic"),
    ],
)
def test_get_type_group_maps_extensions(path: str, expected_group: str) -> None:
    assert get_type_group(path) == expected_group


def test_get_capabilities_pdf() -> None:
    caps = get_capabilities("pdf")
    assert isinstance(caps, TypeGroupCapabilities)
    assert caps.group == "pdf"
    assert caps.required_features == ("pdf_processing",)
    assert caps.field_names == ("pdf_engine", "ocr")

    engine_field = caps.fields[0]
    assert isinstance(engine_field, OptionField)
    assert engine_field.name == "pdf_engine"
    assert engine_field.type == "select"
    assert engine_field.default == "pymupdf4llm"
    assert "pymupdf" in engine_field.options
    assert "docling" in engine_field.options


def test_get_capabilities_audio_video() -> None:
    caps = get_capabilities("audio_video")
    assert caps.group == "audio_video"
    assert caps.required_features == ("audio_processing",)
    assert "faster_whisper" in caps.optional_features
    assert caps.field_names == (
        "transcription_model",
        "language",
        "timestamps",
        "diarization",
    )

    model_field = next(f for f in caps.fields if f.name == "transcription_model")
    assert model_field.options == ("tiny", "base", "small", "medium", "large")
    assert model_field.default == "base"


def test_get_capabilities_ebook() -> None:
    caps = get_capabilities("ebook")
    assert caps.group == "ebook"
    assert caps.required_features == ("ebook_processing",)
    assert caps.field_names == ("extraction_method", "include_toc")

    converter_field = next(f for f in caps.fields if f.name == "extraction_method")
    assert converter_field.options == ("filtered", "markdown", "basic")


def test_get_capabilities_generic() -> None:
    caps = get_capabilities("generic")
    assert caps.group == "generic"
    assert caps.required_features == ()
    assert caps.optional_features == ()
    assert caps.field_names == ("chunk_size", "encoding")


def test_get_tooling_warnings_returns_missing_features() -> None:
    with patch(
        "tldw_chatbook.Library.ingest_capabilities._is_installed",
        return_value=False,
    ):
        warnings = get_tooling_warnings("pdf")

    assert len(warnings) == 3
    features = {w["feature"] for w in warnings}
    assert features == {"pdf_processing", "pymupdf4llm", "docling"}

    for warning in warnings:
        assert "hint" in warning
        assert "command" in warning
        assert warning["command"].startswith("pip install")


def test_get_tooling_warnings_empty_when_all_installed() -> None:
    with patch(
        "tldw_chatbook.Library.ingest_capabilities._is_installed",
        return_value=True,
    ):
        warnings = get_tooling_warnings("audio_video")

    assert warnings == []


def test_get_tooling_warnings_generic_never_warns() -> None:
    warnings = get_tooling_warnings("generic")
    assert warnings == []


def test_install_hint_audio_processing_uses_audio_extra() -> None:
    hint = _install_hint("audio_processing")
    assert '[audio]' in hint["command"]
    assert "pip install" in hint["command"]


def test_install_hint_resolves_known_extra_for_every_group_feature() -> None:
    for caps in _TYPE_GROUPS.values():
        for feature in caps.required_features + caps.optional_features:
            hint = _install_hint(feature)
            assert hint["command"].startswith("pip install")
            # Extract the extra name from a command like: pip install -e ".[extra]"
            command = hint["command"]
            start = command.find("[")
            assert start != -1, f"No extra bracket in command for {feature}: {command}"
            extra = command[start + 1 : command.find("]", start)]
            assert extra in OPTIONAL_FEATURES, (
                f"Feature {feature} resolved to unknown extra {extra!r}"
            )


def test_get_tooling_warnings_includes_video_processing_for_audio_video() -> None:
    with patch(
        "tldw_chatbook.Library.ingest_capabilities._is_installed",
        return_value=False,
    ):
        warnings = get_tooling_warnings("audio_video")

    features = {w["feature"] for w in warnings}
    assert "video_processing" in features
    assert "audio_processing" in features


def test_get_type_group_fallback_to_generic_for_unsupported_extension() -> None:
    assert get_type_group("/tmp/unknown.xyz") == "generic"
    assert get_type_group("/tmp/archive.tar.gz") == "generic"


def test_diarization_field_depends_on_diarization_feature() -> None:
    caps = get_capabilities("audio_video")
    diarization_field = next(f for f in caps.fields if f.name == "diarization")
    assert diarization_field.default is False
    assert diarization_field.depends_on == "diarization"


def test_feature_labels_are_distinct_within_each_group() -> None:
    for group, caps in _TYPE_GROUPS.items():
        labels = [
            _feature_label(feature, group)
            for feature in caps.required_features + caps.optional_features
        ]
        assert len(labels) == len(set(labels)), (
            f"Duplicate feature labels in group {group}: {labels}"
        )


def test_feature_label_uses_specific_mapping() -> None:
    assert _feature_label("docling", "pdf") == "Docling"
    assert _feature_label("pymupdf4llm", "pdf") == "PyMuPDF4LLM"
    assert _feature_label("yt_dlp", "audio_video") == "yt-dlp"


def test_feature_label_humanizes_unknown_feature() -> None:
    assert _feature_label("unknown_thing", "generic") == "Unknown Thing"


def test_is_installed_uses_dependencies_available(monkeypatch) -> None:
    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.DEPENDENCIES_AVAILABLE,
        "__test_feature__",
        True,
    )
    assert _is_installed("__test_feature__") is True

    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.DEPENDENCIES_AVAILABLE,
        "__test_feature__",
        False,
    )
    assert _is_installed("__test_feature__") is False


def test_is_installed_falls_back_to_find_spec(monkeypatch) -> None:
    # Ensure the feature is not in the cached registry.
    monkeypatch.delitem(
        tldw_chatbook.Library.ingest_capabilities.DEPENDENCIES_AVAILABLE,
        "soundfile",
        raising=False,
    )

    with patch(
        "tldw_chatbook.Library.ingest_capabilities.importlib.util.find_spec",
        return_value=True,
    ):
        assert _is_installed("soundfile") is True

    with patch(
        "tldw_chatbook.Library.ingest_capabilities.importlib.util.find_spec",
        return_value=None,
    ):
        assert _is_installed("soundfile") is False


def test_is_installed_unknown_feature_returns_false() -> None:
    assert _is_installed("__not_a_real_feature_12345__") is False
