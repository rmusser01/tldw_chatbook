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


@pytest.fixture(autouse=True)
def _clear_installed_probe_cache():
    """Keep memoised dependency probes from leaking between tests."""
    tldw_chatbook.Library.ingest_capabilities.reset_installed_probe_cache()
    yield
    tldw_chatbook.Library.ingest_capabilities.reset_installed_probe_cache()


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


@pytest.mark.parametrize(
    ("url", "expected_group"),
    [
        # A video host, the archetypal import, had no extension to go on.
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "audio_video"),
        ("https://youtu.be/dQw4w9WgXcQ", "audio_video"),
        ("https://vimeo.com/123456", "audio_video"),
        # An extension on a URL is still the better answer than "a web page":
        # a PDF or ebook link should be parsed as one, not scraped as HTML.
        ("https://example.com/paper.pdf", "pdf"),
        ("https://example.com/book.epub", "ebook"),
        ("https://example.com/talk.mp3", "audio_video"),
        # Everything else addressable over http is a page to be clipped.
        ("https://example.com/some-post", "web"),
        ("https://en.wikipedia.org/wiki/Fort_Sumter", "web"),
        ("http://example.com", "web"),
    ],
)
def test_get_type_group_classifies_urls_not_just_extensions(
    url: str, expected_group: str
) -> None:
    """A URL must be grouped by what it *is*, not by whether it has a suffix.

    This mapped by ``detect_file_type`` alone, so any extension-less URL fell
    through to ``unsupported`` -- a YouTube link pre-flighted as an unsupported
    *file* while ``classify_ingest_source`` called the same URL ``video`` and the
    pipeline would have ingested it happily (task-702). The canvas's verdict and
    the pipeline's behaviour have to agree, or the screen lies about what will
    happen.
    """
    assert get_type_group(url) == expected_group


def test_the_canvas_and_the_pipeline_agree_on_what_a_url_is() -> None:
    """Pin the two classifiers against each other, not against my expectations.

    ``get_type_group`` (what the canvas shows) and ``classify_ingest_source``
    (what the pipeline does) are separate functions that answered differently
    for the same URL. Comparing them directly is what makes a future divergence
    fail here rather than on a user's screen.
    """
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
        classify_ingest_source,
    )

    # group -> the pipeline classifications it is allowed to correspond to
    compatible = {
        "audio_video": {"audio", "video"},
        "web": {"article"},
        "pdf": {"pdf", "article"},
        "ebook": {"ebook", "article"},
        "generic": {"document", "plaintext", "html", "xml", "article"},
    }
    for url in (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://example.com/some-post",
        "https://example.com/talk.mp3",
        "https://example.com/paper.pdf",
    ):
        group = get_type_group(url)
        assert group != "unsupported", f"{url} is ingestible but grouped unsupported"
        classified = classify_ingest_source(url)
        assert classified in compatible[group], (
            f"{url}: canvas says {group}, pipeline says {classified}"
        )


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
    assert caps.field_names == ("analyze", "chunk", "chunk_size", "chunk_overlap", "encoding")


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


def test_get_type_group_unsupported_extension() -> None:
    assert get_type_group("/tmp/unknown.xyz") == "unsupported"
    assert get_type_group("/tmp/archive.tar.gz") == "unsupported"


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


def test_is_installed_probes_when_registry_says_unchecked(monkeypatch) -> None:
    """A ``False`` in the registry means "not checked yet", not "not installed".

    ``DEPENDENCIES_AVAILABLE`` is pre-seeded with every key set to ``False``
    and only filled in when something resolves it, which under the default
    lazy mode never happened. Treating that placeholder as authoritative made
    every optional feature look missing: users were told to install packages
    they already had, and every dependent advanced option was permanently
    disabled (task-676).
    """
    from types import SimpleNamespace

    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.DEPENDENCIES_AVAILABLE,
        "__probe_feature__",
        False,
    )
    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.OPTIONAL_FEATURES,
        "__probe_feature__",
        SimpleNamespace(package_dependencies=["loguru"]),
    )
    tldw_chatbook.Library.ingest_capabilities.reset_installed_probe_cache()

    assert _is_installed("__probe_feature__") is True


def test_is_installed_still_false_when_package_absent(monkeypatch) -> None:
    """A genuinely missing package stays missing after the probe."""
    from types import SimpleNamespace

    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.DEPENDENCIES_AVAILABLE,
        "__absent_feature__",
        False,
    )
    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.OPTIONAL_FEATURES,
        "__absent_feature__",
        SimpleNamespace(package_dependencies=["definitely_not_installed_xyz"]),
    )
    tldw_chatbook.Library.ingest_capabilities.reset_installed_probe_cache()

    assert _is_installed("__absent_feature__") is False


def test_installed_feature_produces_no_tooling_warning(monkeypatch) -> None:
    """An installed feature must not be advertised as missing tooling."""
    from types import SimpleNamespace

    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.DEPENDENCIES_AVAILABLE,
        "pdf_processing",
        False,
    )
    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.OPTIONAL_FEATURES,
        "pdf_processing",
        SimpleNamespace(package_dependencies=["loguru"]),
    )
    tldw_chatbook.Library.ingest_capabilities.reset_installed_probe_cache()

    warnings = get_tooling_warnings("pdf")

    assert not any(w["feature"] == "pdf_processing" for w in warnings)


def test_repeated_is_installed_calls_probe_once(monkeypatch) -> None:
    """The per-field render path must not re-probe the filesystem each time."""
    from types import SimpleNamespace

    calls: list[str] = []

    def counting_find_spec(name: str):
        calls.append(name)
        return object()

    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.DEPENDENCIES_AVAILABLE,
        "__counted_feature__",
        False,
    )
    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.OPTIONAL_FEATURES,
        "__counted_feature__",
        SimpleNamespace(package_dependencies=["loguru"]),
    )
    monkeypatch.setattr(
        tldw_chatbook.Library.ingest_capabilities.importlib.util,
        "find_spec",
        counting_find_spec,
    )
    tldw_chatbook.Library.ingest_capabilities.reset_installed_probe_cache()

    for _ in range(5):
        assert _is_installed("__counted_feature__") is True

    assert len(calls) == 1, f"probed {len(calls)} times, expected 1"


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

    # The probe result is memoised (it is asked once per dependent field on
    # every option-panel render), so flipping what ``find_spec`` reports
    # mid-test needs the cache dropped to observe the new answer.
    tldw_chatbook.Library.ingest_capabilities.reset_installed_probe_cache()

    with patch(
        "tldw_chatbook.Library.ingest_capabilities.importlib.util.find_spec",
        return_value=None,
    ):
        assert _is_installed("soundfile") is False


def test_is_installed_unknown_feature_returns_false() -> None:
    assert _is_installed("__not_a_real_feature_12345__") is False


def test_no_group_feature_is_hardwired_unavailable(monkeypatch) -> None:
    """Every feature must be *able* to report installed.

    A feature the lookup cannot resolve is stuck at ``False`` forever, which
    is what permanently disabled the advanced options and produced install
    hints for packages that were already present (task-676). With everything
    importable, every feature the type groups use must come back installed --
    any that does not is resolved by no route at all.

    Only ``depends_on`` is checked: ``enabled_when`` names a sibling form
    field, and routing that through the installed-feature lookup is precisely
    the confusion that disabled chunk size and overlap.
    """
    monkeypatch.setattr(
        tldw_chatbook.Library.ingest_capabilities.importlib.util,
        "find_spec",
        lambda name: object(),
    )
    tldw_chatbook.Library.ingest_capabilities.reset_installed_probe_cache()

    stuck = []
    for caps in _TYPE_GROUPS.values():
        features = set(caps.required_features + caps.optional_features)
        features.update(f.depends_on for f in caps.fields if f.depends_on)
        for feature in features:
            if not _is_installed(feature):
                stuck.append(feature)

    assert not stuck, (
        f"features that can never resolve as installed: {sorted(set(stuck))}"
    )


def test_umbrella_feature_packages_match_optional_deps() -> None:
    """The mirrored package lists must stay in step with the real checks.

    ``_FEATURE_REQUIRED_PACKAGES`` duplicates what ``optional_deps``'
    ``check_*_deps`` functions establish, because those do real imports that
    are far too expensive for a render path. This guards the duplication: each
    mirrored flag must at least be a flag ``optional_deps`` actually tracks.
    """
    from tldw_chatbook.Library.ingest_capabilities import _FEATURE_REQUIRED_PACKAGES
    from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

    for feature in _FEATURE_REQUIRED_PACKAGES:
        assert feature in DEPENDENCIES_AVAILABLE, (
            f"{feature!r} is mirrored here but optional_deps does not track it"
        )


def test_enabled_when_never_names_a_feature() -> None:
    """``enabled_when`` must reference a sibling field, not a package.

    The two were conflated: ``chunk_size``/``chunk_overlap`` declared
    ``depends_on="chunk"``, which the canvas resolved through the
    installed-feature lookup. No package is named "chunk", so both inputs
    were disabled permanently -- even with chunking switched on.
    """
    for caps in _TYPE_GROUPS.values():
        names = set(caps.field_names)
        for field in caps.fields:
            if field.enabled_when is None:
                continue
            assert field.enabled_when in names, (
                f"{caps.group}.{field.name} is gated on {field.enabled_when!r}, "
                "which is not a field in the same group"
            )


def test_a_value_gated_field_names_values_its_gate_can_actually_take() -> None:
    """``enabled_when_values`` must be reachable choices of the gating field.

    A value gate that names something the gate can never hold disables the field
    permanently -- the same silent-forever failure that conflating
    ``enabled_when`` with ``depends_on`` produced. Checking against the gate's
    declared ``options`` is what makes a typo fail here.
    """
    for caps in _TYPE_GROUPS.values():
        by_name = {f.name: f for f in caps.fields}
        for field in caps.fields:
            if not field.enabled_when_values:
                continue
            assert field.enabled_when is not None, (
                f"{caps.group}.{field.name} names gate values but no gate field"
            )
            gate = by_name[field.enabled_when]
            assert gate.options, (
                f"{caps.group}.{field.name} value-gates on {gate.name!r}, "
                "which declares no options to choose from"
            )
            unreachable = set(field.enabled_when_values) - set(gate.options)
            assert not unreachable, (
                f"{caps.group}.{field.name} is gated on values {sorted(unreachable)} "
                f"that {gate.name!r} can never hold"
            )
