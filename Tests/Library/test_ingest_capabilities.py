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
        # (task-3303) Word-processor formats get their own group: the generic
        # panel called a .docx a "plain text file" and had no path to
        # ``process_document``'s OCR options.
        ("/tmp/document.docx", "document"),
        ("/tmp/document.doc", "document"),
        ("/tmp/notes.odt", "document"),
        ("/tmp/letter.rtf", "document"),
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
        # A .docx URL groups as a document (the extension wins, like pdf/epub)
        # while the pipeline fetches it as an article -- same compatibility
        # allowance the pdf/ebook rows above already make.
        "document": {"document", "article"},
        "generic": {"plaintext", "html", "xml", "article"},
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
    assert caps.field_names == ("pdf_engine", "ocr", "ocr_language", "ocr_backend")

    engine_field = caps.fields[0]
    assert isinstance(engine_field, OptionField)
    assert engine_field.name == "pdf_engine"
    assert engine_field.type == "select"
    assert engine_field.default == "pymupdf4llm"
    assert "pymupdf" in engine_field.options
    assert "docling" in engine_field.options
    # (task-3303) docext is a valid ``process_pdf`` engine with no UI path.
    assert "docext" in engine_field.options


def test_pdf_ocr_toggle_is_gated_to_ocr_capable_engines() -> None:
    """(task-3303 AC2) Enable-OCR under pymupdf engines was a silent no-op.

    ``process_pdf`` only OCRs under the docling/docext parsers; the checkbox
    must be inert (with the reason in its label hint) under any other engine.
    """
    caps = get_capabilities("pdf")
    ocr_field = next(f for f in caps.fields if f.name == "ocr")
    assert ocr_field.enabled_when == "pdf_engine"
    assert set(ocr_field.enabled_when_values) == {"docling", "docext"}
    assert ocr_field.hint, "the inert state must carry its reason at the control"


def test_pdf_ocr_detail_fields() -> None:
    """(task-3303 AC2) OCR language rides the OCR toggle; backend rides docext."""
    caps = get_capabilities("pdf")
    language_field = next(f for f in caps.fields if f.name == "ocr_language")
    assert language_field.type == "text"
    assert language_field.default == "en"
    assert language_field.enabled_when == "ocr"

    backend_field = next(f for f in caps.fields if f.name == "ocr_backend")
    assert backend_field.type == "select"
    assert backend_field.default == "auto"
    assert "docext" in backend_field.options
    # ``process_pdf`` consults ocr_backend only when the parser is docext.
    assert backend_field.enabled_when == "pdf_engine"
    assert backend_field.enabled_when_values == ("docext",)


def test_get_capabilities_document() -> None:
    """(task-3303 AC1) The document group exposes ``process_document``'s knobs."""
    caps = get_capabilities("document")
    assert caps.group == "document"
    assert caps.field_names == ("processing_method", "ocr", "ocr_language")

    method_field = next(f for f in caps.fields if f.name == "processing_method")
    assert method_field.type == "select"
    assert method_field.default == "auto"
    assert set(method_field.options) == {"auto", "docling", "native"}

    ocr_field = next(f for f in caps.fields if f.name == "ocr")
    assert ocr_field.type == "checkbox"
    assert ocr_field.default is False
    # OCR only works through docling (``process_document`` docstring), which
    # the auto method selects when installed.
    assert ocr_field.enabled_when == "processing_method"
    assert set(ocr_field.enabled_when_values) == {"auto", "docling"}
    assert ocr_field.depends_on == "docling"
    assert ocr_field.hint, "the inert state must carry its reason at the control"

    language_field = next(f for f in caps.fields if f.name == "ocr_language")
    assert language_field.type == "text"
    assert language_field.default == "en"
    assert language_field.enabled_when == "ocr"


def test_ebook_chunk_method_field() -> None:
    """(task-3303 AC3) Chapter chunking is choosable from the ebook panel."""
    caps = get_capabilities("ebook")
    method_field = next(f for f in caps.fields if f.name == "chunk_method")
    assert method_field.type == "select"
    assert method_field.default == "chapters"
    assert set(method_field.options) == {
        "chapters",
        "sentences",
        "words",
        "paragraphs",
    }


def test_audio_video_translation_and_vad_fields() -> None:
    """(task-3303 AC4) Translate-to-English and VAD are real fields."""
    caps = get_capabilities("audio_video")
    translate_field = next(
        f for f in caps.fields if f.name == "translate_to_english"
    )
    assert translate_field.type == "checkbox"
    assert translate_field.default is False
    # Only faster-whisper translates (``resolve_batch_stt_route``); the
    # semantic default routes a translation request to faster-whisper too.
    assert translate_field.enabled_when == "transcription_provider"
    assert set(translate_field.enabled_when_values) == {"default", "faster-whisper"}
    assert translate_field.hint

    vad_field = next(f for f in caps.fields if f.name == "vad_filter")
    assert vad_field.type == "checkbox"
    assert vad_field.default is False


def test_get_capabilities_audio_video() -> None:
    caps = get_capabilities("audio_video")
    assert caps.group == "audio_video"
    assert caps.required_features == ("audio_processing",)
    assert "faster_whisper" in caps.optional_features
    assert caps.field_names == (
        "transcription_provider",
        "transcription_model_dir",
        "transcription_precision",
        "transcription_model",
        "language",
        "translate_to_english",
        "timestamps",
        "diarization",
        "vad_filter",
    )

    provider_field = next(f for f in caps.fields if f.name == "transcription_provider")
    assert provider_field.options == (
        "default",
        "parakeet-onnx",
        "faster-whisper",
        "transcribe-cpp",
    )
    assert provider_field.default == "default"

    model_dir_field = next(
        f for f in caps.fields if f.name == "transcription_model_dir"
    )
    assert model_dir_field.default == ""
    assert model_dir_field.enabled_when == "transcription_provider"
    assert model_dir_field.enabled_when_values == ("parakeet-onnx",)

    precision_field = next(
        f for f in caps.fields if f.name == "transcription_precision"
    )
    assert precision_field.options == ("int8", "f32")
    assert precision_field.default == "int8"
    assert precision_field.enabled_when == "transcription_provider"
    assert precision_field.enabled_when_values == ("parakeet-onnx",)

    model_field = next(f for f in caps.fields if f.name == "transcription_model")
    assert model_field.options == ("tiny", "base", "small", "medium", "large")
    assert model_field.default == "base"
    assert model_field.enabled_when == "transcription_provider"
    assert model_field.enabled_when_values == ("faster-whisper",)


def test_parakeet_onnx_feature_probes_onnx_asr(monkeypatch) -> None:
    probed = []
    monkeypatch.setattr(
        tldw_chatbook.Library.ingest_capabilities.importlib.util,
        "find_spec",
        lambda name: probed.append(name) or object(),
    )

    assert _is_installed("parakeet_onnx") is True
    assert probed == ["onnx_asr"]


def test_canonical_parakeet_extra_probes_onnx_asr(monkeypatch) -> None:
    """The historical extra name now represents the cross-platform ONNX path."""
    probed = []
    monkeypatch.setattr(
        tldw_chatbook.Library.ingest_capabilities.importlib.util,
        "find_spec",
        lambda name: probed.append(name) or object(),
    )

    assert _is_installed("transcription_parakeet") is True
    assert probed == ["onnx_asr"]


def test_parakeet_recovery_uses_onnx_profile_and_preserves_legacy_mlx_alias() -> None:
    """Recovery copy installs ONNX without making the legacy MLX alias lie."""
    assert _install_hint("parakeet_onnx")["command"] == (
        'pip install -e ".[transcription_parakeet]"'
    )
    assert _install_hint("parakeet_mlx")["command"] == (
        'pip install -e ".[mlx_whisper]"'
    )


def test_get_capabilities_ebook() -> None:
    caps = get_capabilities("ebook")
    assert caps.group == "ebook"
    assert caps.required_features == ("ebook_processing",)
    assert caps.field_names == ("extraction_method", "chunk_method", "include_toc")

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


def test_document_docling_warning_names_ocr_not_pdf_ingestion() -> None:
    """(task-3303) Docling resolves through the pdf extra, whose blurb says
    "PDF ingestion" -- a non sequitur beside a folder of Word documents."""
    with patch(
        "tldw_chatbook.Library.ingest_capabilities._is_installed",
        return_value=False,
    ):
        warnings = get_tooling_warnings("document")

    assert [w["feature"] for w in warnings] == ["docling"]
    assert "PDF" not in warnings[0]["hint"]
    assert "OCR" in warnings[0]["hint"]
    assert warnings[0]["command"], "the recovery command must survive the override"


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
    """An installed feature must not be advertised as missing tooling.

    ``pdf_processing`` has a curated ``_FEATURE_REQUIRED_PACKAGES`` entry, so
    that is the branch the probe takes in production -- patching only
    ``OPTIONAL_FEATURES`` would leave the outcome hostage to whether pymupdf
    happens to be installed in the running venv.
    """
    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities.DEPENDENCIES_AVAILABLE,
        "pdf_processing",
        False,
    )
    monkeypatch.setitem(
        tldw_chatbook.Library.ingest_capabilities._FEATURE_REQUIRED_PACKAGES,
        "pdf_processing",
        ("loguru",),
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


# --- task-3304 (MI-07): disabled state carries a reason ----------------------


def _field(group: str, name: str):
    caps = _TYPE_GROUPS[group]
    return caps, {f.name: f for f in caps.fields}[name]


def test_field_disabled_state_reports_curated_reason_for_closed_value_gate() -> None:
    """A value-gated field under a non-enabling gate value is disabled WITH
    its curated reason -- the annotation the canvas renders at the control."""
    from tldw_chatbook.Library.ingest_capabilities import field_disabled_state

    caps, field = _field("audio_video", "transcription_model_dir")
    disabled, reason = field_disabled_state(
        field,
        caps,
        {"transcription_provider": "default"},
        is_installed=lambda _f: True,
    )
    assert disabled is True
    assert reason == "needs the parakeet-onnx provider"


def test_field_disabled_state_is_clear_when_the_gate_opens() -> None:
    from tldw_chatbook.Library.ingest_capabilities import field_disabled_state

    caps, field = _field("audio_video", "transcription_model_dir")
    disabled, reason = field_disabled_state(
        field,
        caps,
        {"transcription_provider": "parakeet-onnx"},
        is_installed=lambda _f: True,
    )
    assert disabled is False
    assert reason == ""


def test_field_disabled_state_truthy_gate_reason() -> None:
    """Checkbox-gated fields (chunk size under Chunk content) state the
    toggle to flip."""
    from tldw_chatbook.Library.ingest_capabilities import field_disabled_state

    caps, field = _field("generic", "chunk_size")
    disabled, reason = field_disabled_state(
        field, caps, {"chunk": False}, is_installed=lambda _f: True
    )
    assert disabled is True
    assert reason == "needs Chunk content on"


def test_field_disabled_state_missing_dependency_reason() -> None:
    """A packaging gate (depends_on) names the missing feature."""
    from tldw_chatbook.Library.ingest_capabilities import field_disabled_state

    caps, field = _field("audio_video", "diarization")
    disabled, reason = field_disabled_state(
        field, caps, {}, is_installed=lambda _f: False
    )
    assert disabled is True
    assert reason == "needs Speaker diarization installed"


def test_field_disabled_state_suppresses_reason_when_hint_names_the_gate() -> None:
    """task-3303 baked static gate hints into some labels ("docling or
    docext engines only") -- those fields must not get a second, dynamic
    annotation on top (the incumbents' no-double-annotation rule)."""
    from tldw_chatbook.Library.ingest_capabilities import field_disabled_state

    caps, field = _field("pdf", "ocr")
    disabled, reason = field_disabled_state(
        field,
        caps,
        {"pdf_engine": "pymupdf4llm"},
        is_installed=lambda _f: True,
    )
    assert disabled is True
    assert reason == ""


def test_every_value_gated_field_carries_a_disabled_reason() -> None:
    """Meta-guard: a select gate is invisible from inside the field, so a
    value-gated field without curated disabled copy would silently ship
    the MI-07 no-reason state again for the next field added."""
    for caps in _TYPE_GROUPS.values():
        for field in caps.fields:
            if not field.enabled_when_values:
                continue
            if field.hint:
                # A static gate hint in the label already carries the why.
                continue
            assert field.disabled_reason, (
                f"{caps.group}.{field.name} is value-gated with no hint and "
                "no disabled_reason -- its inert state would be unexplained"
            )


# --- task-3305 (MI-09): human display labels for every select option -------


def test_every_select_field_labels_every_option_with_human_copy() -> None:
    """Meta-guard: every ``select`` field must carry a display label for
    every one of its options, and no label may be the raw internal token
    echoed back -- ``pymupdf4llm``, ``url_level``, ``recursive_scraping``
    et al. used to render verbatim as user-facing values (task-3305)."""
    select_fields = 0
    for caps in _TYPE_GROUPS.values():
        for field in caps.fields:
            if field.type != "select":
                continue
            select_fields += 1
            labels = dict(field.option_labels)
            assert set(labels) == set(field.options), (
                f"{caps.group}.{field.name}: option_labels must cover exactly "
                f"the declared options (missing: "
                f"{set(field.options) - set(labels)}, stray: "
                f"{set(labels) - set(field.options)})"
            )
            for value, label in labels.items():
                assert label and label.strip(), (
                    f"{caps.group}.{field.name}.{value}: blank display label"
                )
                assert label != value, (
                    f"{caps.group}.{field.name}.{value}: display label is the "
                    "raw internal token"
                )
                assert "," not in label, (
                    f"{caps.group}.{field.name}.{value}: option labels feed "
                    "comma-joined panel titles and must stay comma-free"
                )
    assert select_fields >= 9, "schema sweep looks broken -- too few selects"


def test_select_option_label_resolves_labels_and_falls_back() -> None:
    """The label helper returns the curated display copy for a mapped value
    and echoes an unmapped value unchanged (never crashes on stale
    persisted values)."""
    from tldw_chatbook.Library.ingest_capabilities import select_option_label

    engine = next(
        f for f in get_capabilities("pdf").fields if f.name == "pdf_engine"
    )
    label = select_option_label(engine, "pymupdf4llm")
    assert label != "pymupdf4llm"
    assert "PyMuPDF4LLM" in label

    bare = OptionField(name="x", label="X", type="select", options=("a",))
    assert select_option_label(bare, "a") == "a"
    assert select_option_label(engine, "no-such-engine") == "no-such-engine"


def test_scope_nouns_exist_for_every_group() -> None:
    """(task-3305, MI-16) Every group carries singular/plural scope nouns so
    the panel scope line can say "every PDF document" instead of gluing the
    category label into an unnatural sentence."""
    for caps in _TYPE_GROUPS.values():
        assert caps.noun_singular, f"{caps.group}: missing noun_singular"
        assert caps.noun_plural, f"{caps.group}: missing noun_plural"


def test_get_type_group_xml_is_unsupported_task_3308() -> None:
    """task-3308 (defer ruling, task-3310 notes): ``.xml`` stays unmapped in
    ``detect_file_type``, so pre-flight must classify it unsupported -- the
    honest state while ``XML_Ingestion.py`` remains unwired. If someone
    wires XML through (extension -> group -> parse), this pin goes red on
    purpose: retire it together with the deferral."""
    assert get_type_group("/tmp/feed.xml") == "unsupported"
    assert get_type_group("/tmp/FEED.XML") == "unsupported"
