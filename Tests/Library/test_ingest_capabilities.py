"""Tests for the library ingestion capability discovery module."""

from __future__ import annotations

import contextlib
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
        # ...but only when the pipeline can actually honor the verdict.
        # (task-3307 xhigh review round) An image URL is the exception:
        # ``classify_ingest_source`` has no image branch, so the pipeline
        # scrapes it as an article. Grouping it as ``image`` made the
        # canvas promise "1 image", mount the OCR panel, and raise an
        # OCR-backend warning that forces the two-press consent -- after
        # which every OCR option was discarded and the URL was fetched as
        # HTML. The canvas must say what the pipeline will do.
        ("https://example.com/chart.png", "web"),
        ("https://example.com/scan.tiff", "web"),
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
        # (task-3307 xhigh review round) NO allowance for image: unlike
        # pdf/ebook/document -- whose panels' options are at worst unused
        # -- an image URL grouped as ``image`` makes the canvas mount the
        # OCR panel and raise the OCR-backend warning that forces the
        # two-press consent, for a pipeline path that never calls
        # ``process_image``. The group must be one the pipeline honors.
        "image": {"image"},
        "generic": {"plaintext", "html", "xml", "article"},
    }
    for url in (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://example.com/some-post",
        "https://example.com/talk.mp3",
        "https://example.com/paper.pdf",
        "https://example.com/chart.png",
        "https://example.com/scan.tiff",
    ):
        group = get_type_group(url)
        assert group != "unsupported", f"{url} is ingestible but grouped unsupported"
        classified = classify_ingest_source(url)
        assert classified in compatible[group], (
            f"{url}: canvas says {group}, pipeline says {classified}"
        )


def test_local_image_files_still_group_as_image() -> None:
    """The URL fix must not cost local images their own panel."""
    assert get_type_group("/home/u/receipt.png") == "image"
    assert get_type_group("/home/u/scan.TIFF") == "image"


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
    translate_field = next(f for f in caps.fields if f.name == "translate_to_english")
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
        "start_time",
        "end_time",
        "cookies_file",
        "summarize_recursively",
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
    # (task-3306) The full faster-whisper catalog the transcription service
    # itself declares (``TranscriptionService.list_available_models``,
    # ``transcription_service.py``) -- the old five-size list silently
    # withheld large-v3, the .en variants, the distil family and the
    # community turbo/CrisperWhisper builds the routing layer passes
    # straight through to ``WhisperModel``. Order mirrors the service list.
    assert model_field.options == (
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "distil-large-v2",
        "distil-medium.en",
        "distil-small.en",
        "distil-large-v3",
        "deepdml/faster-distil-whisper-large-v3.5",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
        "nyrahealth/faster_CrisperWhisper",
    )
    assert model_field.default == "base"
    assert model_field.enabled_when == "transcription_provider"
    assert model_field.enabled_when_values == ("faster-whisper",)


def test_audio_video_trim_cookies_recursive_summary_fields() -> None:
    """(task-3306) Time-range trim, cookies file, recursive summary shapes."""
    caps = get_capabilities("audio_video")

    start_field = next(f for f in caps.fields if f.name == "start_time")
    assert start_field.type == "text"
    assert start_field.default == ""
    assert start_field.hint, "the accepted format must be stated at the control"
    assert start_field.placeholder, "an empty Input needs example content"

    end_field = next(f for f in caps.fields if f.name == "end_time")
    assert end_field.type == "text"
    assert end_field.default == ""
    assert end_field.hint
    assert end_field.placeholder

    cookies_field = next(f for f in caps.fields if f.name == "cookies_file")
    assert cookies_field.type == "text"
    assert cookies_field.default == ""
    # A PATH input, never raw cookie text: these option values persist with
    # the job and echo into config.toml, where a credential must not land.
    assert cookies_field.depends_on == "yt_dlp"
    assert "video URL" in cookies_field.hint, (
        "only the video (yt-dlp) download path consumes the cookie file; "
        "the scope limit belongs at the control"
    )
    assert cookies_field.placeholder

    recursive_field = next(f for f in caps.fields if f.name == "summarize_recursively")
    assert recursive_field.type == "checkbox"
    assert recursive_field.default is False
    # The gate lives in the GENERIC group (Analyze after import), which
    # enabled_when cannot reach across groups -- the dependency is stated
    # in the hint instead (the task-3303 convention).
    assert recursive_field.enabled_when is None
    assert "Analyze" in recursive_field.hint


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
    assert caps.field_names == (
        "analyze",
        "overwrite_existing",
        "custom_prompt",
        "system_prompt",
        "generate_embeddings",
        "keep_original_file",
        "chunk",
        "chunk_size",
        "chunk_overlap",
        "encoding",
    )


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
    assert "[audio]" in hint["command"]
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

    engine = next(f for f in get_capabilities("pdf").fields if f.name == "pdf_engine")
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


# --- task-3307: image ingestion (ship ruling, task-3310 notes) --------------


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/photo.png",
        "/tmp/photo.PNG",
        "/tmp/photo.jpg",
        "/tmp/photo.jpeg",
        "/tmp/animation.gif",
        "/tmp/photo.webp",
        "/tmp/scan.bmp",
        "/tmp/scan.tiff",
        "/tmp/scan.tif",
    ],
)
def test_get_type_group_maps_image_extensions_task_3307(path: str) -> None:
    """Raster formats the image processor's PIL loader opens get the new
    ``image`` group -- they used to pre-flight as unsupported while
    ``Image_Processing_Lib.process_image`` sat unreachable."""
    assert get_type_group(path) == "image"


@pytest.mark.parametrize(
    "path",
    [
        # Not PIL-loadable raster content: SVG is a vector document.
        "/tmp/diagram.svg",
        # An icon container is not a content document to import.
        "/tmp/favicon.ico",
        # HEIC/HEIF need pillow_heif, which no install extra provides.
        "/tmp/photo.heic",
        "/tmp/photo.heif",
    ],
)
def test_image_lookalikes_stay_unsupported_task_3307(path: str) -> None:
    """Formats ``process_image``'s own SUPPORTED_IMAGE_FORMATS table lists
    but the PIL loader cannot actually rasterize (svg), or that need the
    absent pillow_heif opener (heic/heif), or that are icon containers
    rather than content (ico), stay honestly unsupported."""
    assert get_type_group(path) == "unsupported"


def test_image_url_is_grouped_as_web_not_image_task_3307() -> None:
    """An image URL groups as ``web``, unlike pdf/ebook/document.

    (xhigh review round) This test previously asserted the opposite --
    "the extension says what the target IS" -- which is true of pdf/ebook
    only because their panels' options merely go unused. For images the
    verdict had teeth: the OCR panel mounted, the missing-OCR-backend
    warning fired and forced the ingest canvas's two-press consent, and
    then the pipeline (``classify_ingest_source`` has no image branch)
    scraped the URL as HTML with every OCR option discarded. The canvas
    must not promise work the pipeline will not do.
    """
    assert get_type_group("https://example.com/diagram.png") == "web"


def test_get_capabilities_image_task_3307() -> None:
    caps = get_capabilities("image")

    assert caps.group == "image"
    assert caps.noun_singular == "image"
    assert caps.noun_plural == "images"
    # Without at least one OCR backend the group cannot produce content at
    # all (the extracted text IS what gets imported), so the backend
    # umbrella is REQUIRED, not optional.
    assert caps.required_features == ("image_ocr",)

    assert caps.field_names == ("ocr", "ocr_language", "ocr_backend")
    fields = {f.name: f for f in caps.fields}

    ocr = fields["ocr"]
    assert ocr.type == "checkbox"
    # Mirrors process_image's own enable_ocr=True default -- and with OCR
    # off there is nothing to import.
    assert ocr.default is True
    assert ocr.hint, "the OCR toggle must say the text IS the content"

    lang = fields["ocr_language"]
    assert lang.enabled_when == "ocr"
    assert lang.default == "en"
    assert lang.disabled_reason

    backend = fields["ocr_backend"]
    assert backend.type == "select"
    assert backend.default == "auto"
    # The OCR manager's registered backends (OCR_Backends), plus auto.
    assert backend.options == (
        "auto",
        "docext",
        "docling",
        "tesseract",
        "easyocr",
        "paddleocr",
    )
    assert backend.enabled_when == "ocr"
    assert backend.disabled_reason


def test_image_ocr_warning_names_text_extraction_task_3307() -> None:
    """The missing-backend warning must say what it costs (text extraction
    from images), with a real install command."""
    with patch(
        "tldw_chatbook.Library.ingest_capabilities._is_installed",
        return_value=False,
    ):
        warnings = get_tooling_warnings("image")

    assert [w["feature"] for w in warnings] == ["image_ocr"]
    assert "text" in warnings[0]["hint"].lower()
    assert warnings[0]["command"].startswith("pip install")


def test_image_ocr_probe_is_any_of_task_3307(monkeypatch) -> None:
    """``image_ocr`` is an ANY-OF umbrella: one importable backend package
    makes the feature installed; none makes it missing. (The all-of
    ``_FEATURE_REQUIRED_PACKAGES`` grammar would demand every backend at
    once, which no real install has.)"""

    def only_easyocr(name: str):
        return object() if name == "easyocr" else None

    monkeypatch.setattr(
        tldw_chatbook.Library.ingest_capabilities.importlib.util,
        "find_spec",
        only_easyocr,
    )
    tldw_chatbook.Library.ingest_capabilities.reset_installed_probe_cache()
    assert _is_installed("image_ocr") is True

    monkeypatch.setattr(
        tldw_chatbook.Library.ingest_capabilities.importlib.util,
        "find_spec",
        lambda name: None,
    )
    tldw_chatbook.Library.ingest_capabilities.reset_installed_probe_cache()
    assert _is_installed("image_ocr") is False


# ---------------------------------------------------------------------------
# (task-3307 xhigh review round) The ``image_ocr`` umbrella must agree with
# OCR_Backends' OWN registration rules. It re-derived availability from
# single import names, which diverged in two places the review measured:
# ``PADDLEOCR_AVAILABLE`` needs BOTH ``paddle`` and ``paddleocr``, and the
# docext backend needs a companion (gradio_client / transformers / openai)
# for whichever mode it runs in. Preflight therefore reported "an OCR
# backend is installed" for environments where ``ocr_manager`` registers
# nothing -- no warning, and then an empty-extraction failure at import.
#
# The guard drives the REAL backend classes with a simulated environment
# rather than restating the rules, so a change in OCR_Backends breaks it.
# ---------------------------------------------------------------------------

_OCR_PROBED_PACKAGES = (
    "docling",
    "pytesseract",
    "easyocr",
    "paddle",
    "paddleocr",
    "docext",
    "gradio_client",
    "transformers",
    "openai",
)


@contextlib.contextmanager
def _ocr_backends_seeing(available: set[str]):
    """Reload ``OCR_Backends`` as if only ``available`` were importable.

    Its availability flags are computed at import time from
    ``importlib.util.find_spec``, so the only way to ask it about a
    hypothetical environment is to re-execute the module against a patched
    resolver. Restored (and reloaded again for real) on exit.
    """
    import importlib
    import importlib.util

    from tldw_chatbook.Local_Ingestion import OCR_Backends as ocr_mod

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name in _OCR_PROBED_PACKAGES:
            return object() if name in available else None
        return real_find_spec(name, package)

    importlib.util.find_spec = fake_find_spec
    try:
        yield importlib.reload(ocr_mod)
    finally:
        importlib.util.find_spec = real_find_spec
        importlib.reload(ocr_mod)


def _ocr_manager_registers_a_backend(available: set[str]) -> bool:
    """Would some OCR backend's PACKAGE requirements be met in this env?

    Asks the real backend classes, with two documented adjustments:

    * ``TesseractOCRBackend.is_available()`` additionally imports
      pytesseract for real and shells out for the ``tesseract`` binary --
      a runtime condition a ``find_spec`` preflight cannot replicate (and
      should not: the umbrella's contract is "the packages exist"). Its
      package rule, ``TESSERACT_AVAILABLE``, is read directly. The residual
      gap -- pytesseract installed, binary absent -- is real and out of
      this umbrella's reach.
    * The docext backend's mode is configuration-driven and the manager
      constructs it with defaults, so every mode is asked: the umbrella
      cannot know which one a user configured.
    """
    with _ocr_backends_seeing(available) as ocr:
        if ocr.TESSERACT_AVAILABLE:
            return True
        if any(
            backend().is_available()
            for backend in (
                ocr.DoclingOCRBackend,
                ocr.EasyOCRBackend,
                ocr.PaddleOCRBackend,
            )
        ):
            return True
        return any(
            ocr.DocextOCRBackend({"mode": mode}).is_available()
            for mode in ("api", "model", "openai")
        )


def _umbrella_says_installed(monkeypatch, available: set[str]) -> bool:
    """The preflight umbrella's verdict for the same simulated env."""
    caps_mod = tldw_chatbook.Library.ingest_capabilities
    monkeypatch.setattr(caps_mod, "_module_present", lambda pkg: pkg in available)
    caps_mod.reset_installed_probe_cache()
    try:
        return caps_mod._probe_installed("image_ocr")
    finally:
        caps_mod.reset_installed_probe_cache()


def _image_ocr_groups() -> tuple[tuple[str, ...], ...]:
    groups = tldw_chatbook.Library.ingest_capabilities._FEATURE_ANY_PACKAGES[
        "image_ocr"
    ]
    assert all(isinstance(group, tuple) for group in groups), (
        "image_ocr must be an ANY-OF over ALL-OF package groups; a flat "
        "tuple of single names cannot express paddle+paddleocr or "
        "docext+companion"
    )
    return groups


@pytest.mark.parametrize("group", _image_ocr_groups())
def test_each_umbrella_group_really_yields_a_backend(monkeypatch, group) -> None:
    available = set(group)
    assert _umbrella_says_installed(monkeypatch, available) is True
    assert _ocr_manager_registers_a_backend(available) is True, (
        f"the umbrella claims {sorted(available)} is enough, but "
        "OCR_Backends registers nothing"
    )


@pytest.mark.parametrize("group", _image_ocr_groups())
def test_dropping_any_package_from_a_group_loses_the_backend(
    monkeypatch, group
) -> None:
    for missing in group:
        available = set(group) - {missing}
        assert _ocr_manager_registers_a_backend(available) is False, (
            f"{sorted(available)} unexpectedly registers a backend; the "
            "umbrella's groups are now over-strict"
        )
        assert _umbrella_says_installed(monkeypatch, available) is False, (
            f"the umbrella claims {sorted(available)} is enough without "
            f"{missing}, but OCR_Backends registers nothing"
        )


def test_the_backend_roster_the_umbrella_mirrors_is_pinned() -> None:
    """A NEW OCR backend would need a new umbrella group; fail here first."""
    from tldw_chatbook.Local_Ingestion.OCR_Backends import OCRBackendType

    assert {member.value for member in OCRBackendType} == {
        "docling",
        "tesseract",
        "easyocr",
        "paddleocr",
        "docext",
    }


# --- task-14820: required vs optional, asked once ---------------------------


def test_classify_missing_features_splits_required_from_optional() -> None:
    """The forecast has to tell "cannot run at all" from "runs, degraded".

    Consumers used to union both tuples (``count_warning_affected_files``
    did), so a missing REQUIRED backend read exactly like a missing
    optional one -- every warned file was "may fail" and none was a
    forecast failure.
    """
    from tldw_chatbook.Library.ingest_capabilities import (
        classify_missing_features,
    )

    required, optional = classify_missing_features(
        "pdf", {"pdf_processing", "docling", "unrelated_feature"}
    )
    assert required == ("pdf_processing",)
    assert optional == ("docling",)

    # A group whose only stake in the missing feature is optional.
    assert classify_missing_features("document", {"docling"}) == (
        (),
        ("docling",),
    )


def test_classify_missing_features_is_quiet_for_unknown_input() -> None:
    """Unsupported/unknown groups have no declared tooling, and an empty
    missing-set never invents one."""
    from tldw_chatbook.Library.ingest_capabilities import (
        UNSUPPORTED_GROUP,
        classify_missing_features,
    )

    assert classify_missing_features(UNSUPPORTED_GROUP, {"pdf_processing"}) == (
        (),
        (),
    )
    assert classify_missing_features("not-a-group", {"pdf_processing"}) == (
        (),
        (),
    )
    assert classify_missing_features("pdf", set()) == ((), ())
    assert classify_missing_features("pdf", {"", "  "}) == ((), ())


def test_classify_missing_features_agrees_with_the_warning_wall() -> None:
    """The classifier and ``get_tooling_warnings`` must name the same
    features for a group -- the forecast reads one, the user reads the
    other."""
    from tldw_chatbook.Library.ingest_capabilities import (
        classify_missing_features,
    )

    warned = {w["feature"] for w in get_tooling_warnings("pdf")}
    required, optional = classify_missing_features("pdf", warned)
    assert set(required) | set(optional) == warned


def test_every_real_tooling_warning_names_its_feature():
    """The canvas splits warnings on the ``feature`` key: feature-bearing
    ones are missing components (folded, counted, install-commanded),
    featureless ones are advisories (shown in place, never counted).

    That split is only sound if production tooling warnings ALWAYS carry
    a feature. Hand-written test fixtures omitted it and so silently
    became "advisories" -- the fake-matching-the-call-site trap. This
    asserts the real producer's shape instead of trusting a fixture.
    """
    from tldw_chatbook.Library.ingest_capabilities import (
        get_tooling_warnings,
        list_type_groups,
    )

    seen = 0
    for group in list_type_groups():
        for warning in get_tooling_warnings(group):
            seen += 1
            assert str(warning.get("feature") or "").strip(), (
                f"{group} emitted a tooling warning with no feature: {warning}"
            )
    assert seen, (
        "no tooling warnings were produced at all -- this venv has every "
        "optional extra installed, so the guard proved nothing"
    )


# --- task-28007 AC#6: the collapsed Import behavior header states its state --


def test_import_behavior_header_summarises_its_analysis_state():
    """AC#6: "Analyze after import" lives inside a fold that is collapsed by
    default, so the panel's whole point was invisible until it was opened.
    The title carries the state for both values."""
    from tldw_chatbook.Library.ingest_capabilities import type_group_state_summary

    generic = get_capabilities("generic")
    assert type_group_state_summary(generic, {}) == "Import behavior · analysis off"
    assert (
        type_group_state_summary(generic, {"analyze": True})
        == "Import behavior · analysis on"
    )
    # Only the group that owns the toggle gains the clause.
    pdf = get_capabilities("pdf")
    assert type_group_state_summary(pdf, {}) == pdf.label


def test_the_collapsed_generic_title_carries_the_analysis_state():
    """The wiring, not just the helper: the title the Collapsible actually
    renders (and the screen's in-place update assigns) leads with the
    state, and never stutters it a second time as a changed-value pair."""
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        build_type_group_title,
    )

    generic = get_capabilities("generic")
    off = build_type_group_title(generic, {}, is_installed=lambda _f: True)
    assert off.startswith("Import behavior · analysis off"), off

    on = build_type_group_title(generic, {"analyze": True}, is_installed=lambda _f: True)
    assert on.startswith("Import behavior · analysis on"), on
    assert "Analyze after import" not in on, on
