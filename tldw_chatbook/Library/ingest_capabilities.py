"""Capability discovery for library ingestion workflows.

This module exposes per-media-type ingestion settings and availability
warnings without importing heavy optional dependencies. It is intended to
back configuration UIs that need to know which backends are installed and
what options they expose.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    FileIngestionError,
    detect_file_type,
    is_http_url as _is_http_url,
)
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE, OPTIONAL_FEATURES


@dataclass(frozen=True)
class OptionField:
    """A single configurable option exposed for an ingestion type.

    Args:
        name: Machine-readable identifier for the field.
        label: Human-readable label shown in the UI.
        type: Form widget type, e.g. ``select``, ``checkbox``, ``text``,
            ``number``.
        default: Default value when the field is first rendered.
        hint: Optional unit/range hint rendered beside the field label
            (task-2223: chunk size/overlap never stated their unit, and
            the valid range surfaced only via a validation error).
        options: Allowed values for ``select`` fields; empty for other types.
        depends_on: Optional dependency feature ID that must be *installed*
            for this field to be editable. ``None`` means the field needs no
            optional tooling.
        enabled_when: Optional sibling field name whose value must be truthy
            for this field to be editable -- a within-form relationship, not a
            packaging one. Kept separate from ``depends_on`` because resolving
            a sibling field name through the installed-feature lookup silently
            disabled the field forever (no package is ever named "chunk").
        enabled_when_values: Optional allowed values for the ``enabled_when``
            sibling. Empty means the plain truthiness of the sibling decides,
            which is the right test for a checkbox but not for a select: every
            non-empty choice is truthy, so a select-gated field would always
            read as editable. Naming the values that actually enable it keeps
            the form from offering an input the run will ignore.
        disabled_reason: (task-3304, MI-07) Curated copy for the field's
            DISABLED state -- rendered at the control while its
            ``enabled_when`` gate is closed (e.g. "needs the parakeet-onnx
            provider"), per the design system's Inert-actions rule: a
            disabled control carries a text annotation for *why*, never
            dimming alone. Empty falls back to a generic derivation from
            the gate metadata (see :func:`field_disabled_state`); fields
            whose static ``hint`` already names the gate (task-3303's
            "docling or docext engines only" style) deliberately leave both
            empty so the label is never double-annotated.
        option_labels: (task-3305, MI-09) ``(value, display label)`` pairs
            for ``select`` fields -- what the user reads while the raw
            token is what persists and travels to the pipeline. Every
            select option must be covered (meta-tested); labels state the
            consequence where one exists ("Auto (faster-whisper)") and
            stay comma-free because the collapsed panel titles comma-join
            them. Resolve through :func:`select_option_label`, never by
            indexing this tuple directly.
        placeholder: Example content for an empty text Input (e.g.
            ``/path/to/parakeet-model``). Empty falls back to the field
            label -- but a placeholder that merely repeats the label is
            stutter, so empty-by-default text fields should set one.
        directory_picker: Whether a text field gets an adjacent directory
            picker action. The text input remains editable and authoritative.
        backends: Ingestion backends for which this field is meaningful.
            Keeping this on the field makes the capability schema the one
            source for both the field's default and its mode visibility.
    """

    name: str
    label: str
    type: str
    default: Any = None
    options: tuple[str, ...] = ()
    depends_on: str | None = None
    enabled_when: str | None = None
    hint: str = ""
    enabled_when_values: tuple[Any, ...] = ()
    disabled_reason: str = ""
    option_labels: tuple[tuple[str, str], ...] = ()
    placeholder: str = ""
    directory_picker: bool = False
    backends: tuple[str, ...] = ("local", "server")


@dataclass(frozen=True)
class TypeGroupCapabilities:
    """Capabilities and options for a logical ingestion group.

    Args:
        group: Group identifier, e.g. ``pdf``, ``audio_video``.
        label: Human-readable label for the group.
        required_features: Feature IDs that must be installed for the group to
            function at all.
        optional_features: Feature IDs that enhance the group but are not
            strictly required.
        fields: Configurable options for this group.
        noun_singular: (task-3305, MI-16) Singular noun phrase for one item
            of this group ("PDF document", "web page") -- the panel scope
            line composes sentences from it ("Applies to every PDF document
            in this import."), which the bare category ``label`` cannot do
            grammatically ("Applies to all Plain text & HTML in this
            import.").
        noun_plural: Plural counterpart of ``noun_singular``.
    """

    group: str
    label: str
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...]
    fields: tuple[OptionField, ...]
    noun_singular: str = ""
    noun_plural: str = ""

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return the machine names of all configured fields."""
        return tuple(f.name for f in self.fields)


def field_available_for_backend(field: OptionField, backend: str) -> bool:
    """Return whether an option can affect the selected ingest backend.

    Args:
        field: Capability declaration to test.
        backend: Effective ingestion backend (``local`` or ``server``).

    Returns:
        True when the field is declared for ``backend``.
    """
    return backend in field.backends


def capabilities_for_backend(
    capabilities: TypeGroupCapabilities, backend: str
) -> TypeGroupCapabilities:
    """Return the capability view that can affect ``backend``.

    Both canvas composition and the collapsed-title receipt must project the
    same backend-visible fields. Keeping the projection at the capability
    boundary prevents a retained value from an unavailable backend leaking
    into an in-place receipt.

    Args:
        capabilities: Complete declared capability group.
        backend: Effective ingestion backend (``local`` or ``server``).

    Returns:
        A capabilities instance containing only fields available to ``backend``.
    """
    return replace(
        capabilities,
        fields=tuple(
            field
            for field in capabilities.fields
            if field_available_for_backend(field, backend)
        ),
    )


def select_option_label(field: OptionField, value: Any) -> str:
    """Display copy for one select option value.

    (task-3305, MI-09) The single resolution seam between a select's
    persisted internal token and what the user reads -- used by the canvas
    select builder AND the collapsed-title summariser so the two can never
    disagree. An unmapped value (a stale persisted token, a test double)
    echoes through unchanged rather than crashing.

    Args:
        field: The select field whose ``option_labels`` to consult.
        value: The internal option value.

    Returns:
        The curated display label, or ``str(value)`` when none is defined.
    """
    for candidate, label in field.option_labels:
        if candidate == value:
            return label
    return str(value)


# PyPI package names used in the UI/planning documents map to the names that
# Python actually imports. This mapping is used only as a fallback when a
# feature is not already tracked in ``DEPENDENCIES_AVAILABLE``.
_PYPI_TO_IMPORT: dict[str, str] = {
    "beautifulsoup4": "bs4",
    "docling": "docling",
    "ebooklib": "ebooklib",
    "faster-whisper": "faster_whisper",
    "html2text": "html2text",
    "lightning-whisper-mlx": "lightning_whisper_mlx",
    "lxml": "lxml",
    "onnx-asr[cpu]": "onnx_asr",
    "parakeet_onnx": "onnx_asr",
    "parakeet-mlx": "parakeet_mlx",
    "pymupdf": "pymupdf",
    "pymupdf4llm": "pymupdf4llm",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "yt-dlp": "yt_dlp",
}

# Map feature IDs that are not themselves pyproject extras to the extra that
# installs them. Used when building recovery hints.
_FEATURE_TO_EXTRA: dict[str, str] = {
    "audio_processing": "audio",
    "beautifulsoup4": "ebook",
    "docling": "pdf",
    "docext": "ocr_docext",
    "ebook_processing": "ebook",
    "ebooklib": "ebook",
    "faster_whisper": "transcription_faster_whisper",
    "html2text": "ebook",
    "lightning_whisper_mlx": "transcription_lightning_whisper",
    "lxml": "ebook",
    "parakeet_onnx": "transcription_parakeet",
    "parakeet_mlx": "mlx_whisper",
    # (task-3307) The OCR-backend umbrella recovers via the one extra that
    # is explicitly OCR-purposed. Docling (via [pdf]) or a bare
    # `pip install pytesseract` work just as well -- the failure detail
    # names the alternatives; an install COMMAND can only name one extra.
    "image_ocr": "ocr_docext",
    "pdf_processing": "pdf",
    "pymupdf": "pdf",
    "pymupdf4llm": "pdf",
    "scipy": "audio",
    "soundfile": "audio",
    "video_processing": "video",
    "yt_dlp": "audio",
}

# Extra names that belong to the media-type groups, used to provide richer
# fallback hints when a bare feature ID is not explicitly mapped.
_GROUP_EXTRAS: dict[str, tuple[str, ...]] = {
    "pdf": ("pdf",),
    "audio_video": ("audio", "video", "media_processing"),
    "ebook": ("ebook",),
    # Document ingestion's one flagged optional feature (docling) installs
    # via the pdf extra; nothing document-specific has an extra of its own.
    "document": (),
    # (task-3307) The OCR extra is the image group's one install route
    # with a pyproject name.
    "image": ("ocr_docext",),
    "generic": (),
}

# Human-readable labels for individual feature IDs. These keep warning lists
# distinct when multiple features resolve to the same install extra.
_FEATURE_LABELS: dict[str, str] = {
    "audio_processing": "Audio processing",
    "beautifulsoup4": "BeautifulSoup",
    # (task-3304) Named for the disabled-state annotation on the diarization
    # checkbox -- without it the label falls through to the audio group's
    # extra ("Audio ingestion and transcription"), which is not the gate.
    "diarization": "Speaker diarization",
    "docling": "Docling",
    "docext": "Docext",
    "ebook_processing": "E-book processing",
    "ebooklib": "ebooklib",
    "faster_whisper": "Faster Whisper",
    "html2text": "html2text",
    # (task-3307) Named so the image group's warning reads "OCR backend
    # isn't installed" rather than the ocr_docext extra's own label.
    "image_ocr": "OCR backend",
    "lightning_whisper_mlx": "Lightning Whisper MLX",
    "lxml": "lxml",
    "parakeet_onnx": "Parakeet ONNX",
    "parakeet_mlx": "Parakeet MLX",
    "pdf_processing": "PDF processing",
    "pymupdf": "PyMuPDF",
    "pymupdf4llm": "PyMuPDF4LLM",
    "scipy": "SciPy",
    "soundfile": "SoundFile",
    "video_processing": "Video processing",
    "yt_dlp": "yt-dlp",
}


#: Packages that must all be importable for an umbrella *feature flag* to be
#: considered installed. These flags are not package names, so ``find_spec``
#: cannot resolve them directly, and the authoritative checks in
#: ``optional_deps`` establish them with a real ``__import__`` of torch,
#: chromadb, transformers and friends -- far too expensive to run from a
#: render path. The entries below mirror those checks without importing
#: anything; keep them in step with ``check_pdf_processing_deps``,
#: ``check_ebook_processing_deps``, ``check_audio_processing_deps`` and
#: ``check_video_processing_deps``.
_FEATURE_REQUIRED_PACKAGES: dict[str, tuple[str, ...]] = {
    "pdf_processing": ("pymupdf",),
    "ebook_processing": ("ebooklib", "defusedxml"),
    "audio_processing": ("soundfile", "scipy"),
    # Video reuses the audio stack; ``check_video_processing_deps`` sets it
    # straight from ``audio_processing``.
    "video_processing": ("soundfile", "scipy"),
}

#: (task-3307) ANY-OF-over-ALL-OF umbrella features: installed when EVERY
#: package in at least ONE group imports. The all-of grammar above cannot
#: express "one OCR backend, whichever": image ingestion needs SOME backend
#: the OCR manager registers (``OCR_Backends._register_backends``), and no
#: real install carries all five. Import names, not PyPI names (tesseract's
#: package is pytesseract). Deliberately NOT the ``ocr_processing`` flag
#: from ``optional_deps``: ``check_ocr_deps`` reports available when bare
#: ``openai``/``gradio_client`` import -- true for docext's API mode, a lie
#: as "an OCR backend exists".
#:
#: (xhigh review round) The groups mirror ``OCR_Backends``' own
#: ``is_available()`` rules exactly, because a flat list of single import
#: names did not: ``PADDLEOCR_AVAILABLE`` requires BOTH ``paddle`` and
#: ``paddleocr``, and the docext backend requires a companion for whichever
#: mode it runs in. Preflight consequently promised an OCR backend in
#: environments where ``ocr_manager`` registers nothing -- so the image
#: group raised no warning and the import then failed with an empty
#: extraction. ``Tests/Library/test_ingest_capabilities.py`` drives the
#: real backend classes against each group (and each group minus one
#: package), so this table cannot drift from OCR_Backends unnoticed.
_FEATURE_ANY_PACKAGES: dict[str, tuple[tuple[str, ...], ...]] = {
    "image_ocr": (
        ("docling",),
        ("pytesseract",),
        ("easyocr",),
        ("paddle", "paddleocr"),
        ("docext", "gradio_client"),
        ("docext", "transformers"),
        ("docext", "openai"),
    ),
}

#: Memoised results of the ``find_spec`` fallback in :func:`_is_installed`.
#: The option panels ask about every dependent field on every render, so the
#: probe must not hit the import system each time.
_INSTALLED_PROBE_CACHE: dict[str, bool] = {}


def reset_installed_probe_cache() -> None:
    """Forget memoised probe results.

    Intended for tests that change what is importable; production code has no
    reason to call it, since installing a package mid-session is not something
    a running app observes.
    """
    _INSTALLED_PROBE_CACHE.clear()


def _module_present(package: str) -> bool:
    """Return whether ``package`` is importable, without importing it."""
    import_name = _PYPI_TO_IMPORT.get(package, package.replace("-", "_"))
    try:
        return importlib.util.find_spec(import_name) is not None
    except Exception:
        return False


def _probe_installed(feature_id: str) -> bool:
    """Probe the import system for ``feature_id``, memoising the answer."""
    cached = _INSTALLED_PROBE_CACHE.get(feature_id)
    if cached is not None:
        return cached

    required = _FEATURE_REQUIRED_PACKAGES.get(feature_id)
    any_of = _FEATURE_ANY_PACKAGES.get(feature_id)
    if required is not None:
        result = all(_module_present(package) for package in required)
    elif any_of is not None:
        # (task-3307) An any-of-over-all-of umbrella: one COMPLETE group of
        # alternatives makes the feature real.
        result = any(
            all(_module_present(package) for package in group) for group in any_of
        )
    else:
        # A feature is installed only when every package it depends on imports.
        info = OPTIONAL_FEATURES.get(feature_id)
        if info is not None:
            result = all(
                _module_present(package) for package in info.package_dependencies
            )
        else:
            # Several feature IDs are simply the import name (``yt_dlp``,
            # ``faster_whisper``, ``parakeet_mlx`` ...). ``_module_present``
            # routes through the PyPI-to-import mapping first, so a PyPI name
            # is still never handed to ``find_spec`` unmapped.
            result = _module_present(feature_id)

    _INSTALLED_PROBE_CACHE[feature_id] = result
    return result


def _is_installed(feature_id: str) -> bool:
    """Return whether ``feature_id`` is available.

    ``DEPENDENCIES_AVAILABLE`` is pre-seeded with every key set to ``False``
    and only filled in once something resolves it, so a ``False`` there means
    "nobody has checked yet" just as often as it means "not installed". Under
    the default lazy mode nothing ever resolved it, so trusting that
    placeholder reported every optional feature as missing: users were told to
    install packages they already had, and every dependent option was
    permanently disabled.

    So only a ``True`` in the registry is authoritative -- that one was
    positively established. Anything else falls through to a memoised
    ``find_spec`` probe using the explicit PyPI-name-to-import-name mapping.
    PyPI names are never passed directly to ``find_spec``.

    Args:
        feature_id: Dependency flag from ``optional_deps`` or a PyPI package
            name known to this module.

    Returns:
        True when the feature appears to be installed.
    """
    if DEPENDENCIES_AVAILABLE.get(feature_id):
        return True
    return _probe_installed(feature_id)


def _install_hint(feature_id: str) -> dict[str, str]:
    """Return user-facing recovery instructions for a missing feature.

    Args:
        feature_id: Dependency flag or package name.

    Returns:
        Mapping with ``hint`` and ``command`` keys. When the feature maps to
        a known optional extra, the command uses the editable/source install
        form.
    """
    extra = _FEATURE_TO_EXTRA.get(feature_id, feature_id)
    info = OPTIONAL_FEATURES.get(extra)

    if info is not None:
        # ``hint`` is the *capability at stake*, not a sentence: the two
        # metadata fields are used inconsistently across extras (for PDF,
        # label="PDF processing"/what="PDF ingestion"; for audio they are
        # effectively swapped), so any template combining both reads wrong for
        # one of them. Callers compose the final line -- see
        # ``build_warning_lines``, which pairs this with the specific missing
        # feature's own label and drops it when the two would just repeat.
        return {
            "hint": info.unavailable_what,
            "command": info.source_install_command,
        }

    return {
        "hint": "",
        "command": f'pip install -e ".[{feature_id}]"',
    }


#: (task-3303) Group-specific overrides for a warning's "needed for" clause.
#: ``_install_hint`` resolves docling through the pdf extra, whose
#: ``unavailable_what`` says "PDF ingestion" -- accurate for the pdf group,
#: a non sequitur beside a folder of Word documents. Keyed by
#: ``(group, feature)``; absent pairs keep the extra's own wording.
_GROUP_FEATURE_HINTS: dict[tuple[str, str], str] = {
    ("document", "docling"): "scanned-document OCR",
    # (task-3307) ``_install_hint`` resolves image_ocr through the
    # ocr_docext extra, whose blurb names docext's own toolkit; what the
    # user is actually missing is the ability to get any text out of an
    # image at all.
    ("image", "image_ocr"): "extracting text from images",
}

#: Sentinel group returned by :func:`get_type_group` for files this app has no
#: handler for. It is deliberately *not* a key of ``_TYPE_GROUPS``: it has no
#: capabilities, options or tooling of its own. Callers group these files so
#: the pre-flight summary can count them separately.
UNSUPPORTED_GROUP = "unsupported"

#: Scrape methods that fetch more than the one page they are given, and so make
#: the page/depth limits meaningful. ``individual`` is the single-page case, and
#: the only one the local article extractor implements.
MULTI_PAGE_SCRAPE_METHODS = frozenset({"sitemap", "url_level", "recursive_scraping"})

_TYPE_GROUPS: dict[str, TypeGroupCapabilities] = {
    "pdf": TypeGroupCapabilities(
        group="pdf",
        label="PDF documents",
        noun_singular="PDF document",
        noun_plural="PDF documents",
        required_features=("pdf_processing",),
        optional_features=("pymupdf4llm", "docling"),
        fields=(
            OptionField(
                name="pdf_engine",
                label="PDF engine",
                type="select",
                default="pymupdf4llm",
                # (task-3303) docext joins: a valid ``process_pdf`` parser
                # (vision-model OCR) that had no UI path.
                options=("pymupdf", "pymupdf4llm", "docling", "docext"),
                # (task-3305, MI-09) What each engine costs/buys, verified
                # against ``PDF_Processing_Lib``: pymupdf extracts plain
                # text; pymupdf4llm emits Markdown; docling is the
                # layout-aware converter (and an OCR route); docext runs a
                # vision-model OCR backend.
                option_labels=(
                    ("pymupdf", "PyMuPDF (plain text)"),
                    ("pymupdf4llm", "PyMuPDF4LLM (Markdown)"),
                    ("docling", "Docling (layout-aware · OCR-capable)"),
                    ("docext", "Docext (vision-model OCR)"),
                ),
                depends_on="pdf_processing",
            ),
            OptionField(
                name="ocr",
                label="Enable OCR",
                type="checkbox",
                default=False,
                depends_on="pdf_processing",
                # (task-3303) Only the docling/docext parsers OCR
                # (``process_pdf``'s own ``ocr_supported`` flag); under the
                # pymupdf engines the checkbox used to be an offerable
                # silent no-op. The value gate makes it inert there, and
                # the hint carries the reason at the control.
                enabled_when="pdf_engine",
                enabled_when_values=("docling", "docext"),
                hint="docling or docext engines only",
            ),
            OptionField(
                name="ocr_language",
                label="OCR language",
                type="text",
                default="en",
                depends_on="pdf_processing",
                enabled_when="ocr",
                hint="e.g. en, de, fr",
                disabled_reason="needs Enable OCR on",
            ),
            OptionField(
                name="ocr_backend",
                label="OCR backend",
                type="select",
                default="auto",
                # ``process_pdf`` consults this only when the parser is
                # docext ("auto" lets it pick); the names are the OCR
                # manager's registered backends (``OCR_Backends``).
                options=(
                    "auto",
                    "docext",
                    "tesseract",
                    "easyocr",
                    "paddleocr",
                    "docling",
                ),
                option_labels=(
                    ("auto", "Auto (let Docext choose)"),
                    ("docext", "Docext (vision model)"),
                    ("tesseract", "Tesseract"),
                    ("easyocr", "EasyOCR"),
                    ("paddleocr", "PaddleOCR"),
                    ("docling", "Docling"),
                ),
                enabled_when="pdf_engine",
                enabled_when_values=("docext",),
                disabled_reason="needs the docext engine",
            ),
        ),
    ),
    # (task-3303) Word-processor formats (.doc/.docx/.odt/.rtf) used to ride
    # the generic panel, which called them "plain text files" and offered no
    # path to ``process_document``'s processing-method/OCR knobs. The generic
    # group remains their base (analyze/chunk/encoding layer under this group
    # in ``_ingest_job_options``); this group only adds what is
    # document-specific.
    "document": TypeGroupCapabilities(
        group="document",
        label="Word/Office documents",
        noun_singular="Word/Office document",
        noun_plural="Word/Office documents",
        # No required features: per-format native parsers (python-docx,
        # odfpy, striprtf) and docling are ALTERNATIVES, and docling alone
        # can stand in for all of them -- a missing-parser failure is
        # reported per job with the missing package named in its details.
        required_features=(),
        # Docling is the one feature worth flagging up front: without it the
        # OCR toggle below is dead (scanned documents cannot be OCR'd).
        optional_features=("docling",),
        fields=(
            OptionField(
                name="processing_method",
                label="Processing method",
                type="select",
                default="auto",
                # ``process_document(processing_method=...)``: auto prefers
                # docling when installed, else the per-format native parser.
                options=("auto", "docling", "native"),
                option_labels=(
                    ("auto", "Auto (Docling when installed)"),
                    ("docling", "Docling (layout-aware · OCR-capable)"),
                    ("native", "Native per-format parser"),
                ),
            ),
            OptionField(
                name="ocr",
                label="Enable OCR",
                type="checkbox",
                default=False,
                # OCR only runs through docling (``process_document``:
                # "only works with 'docling' method"); auto selects docling
                # when it is installed, so both satisfy the gate.
                depends_on="docling",
                enabled_when="processing_method",
                enabled_when_values=("auto", "docling"),
                hint="docling method only",
            ),
            OptionField(
                name="ocr_language",
                label="OCR language",
                type="text",
                default="en",
                depends_on="docling",
                enabled_when="ocr",
                hint="e.g. en, de, fr",
                disabled_reason="needs Enable OCR on",
            ),
        ),
    ),
    "audio_video": TypeGroupCapabilities(
        group="audio_video",
        label="Audio & video",
        noun_singular="audio/video file",
        noun_plural="audio/video files",
        required_features=("audio_processing",),
        optional_features=(
            "faster_whisper",
            "lightning_whisper_mlx",
            "parakeet_onnx",
            "parakeet_mlx",
            "yt_dlp",
            "video_processing",
        ),
        fields=(
            OptionField(
                name="transcription_provider",
                label="Transcription provider",
                type="select",
                default="default",
                options=(
                    "default",
                    "parakeet-onnx",
                    "faster-whisper",
                    "transcribe-cpp",
                ),
                # (task-3305, MI-09) "default" is not a provider: the batch
                # router sends it to faster-whisper on this surface
                # (task-3301 verified the ingest call site never opens the
                # Parakeet promotion gate) -- say so at the control.
                option_labels=(
                    ("default", "Auto (faster-whisper)"),
                    ("parakeet-onnx", "Parakeet (ONNX)"),
                    ("faster-whisper", "Faster Whisper"),
                    ("transcribe-cpp", "transcribe.cpp (GGUF)"),
                ),
                depends_on="audio_processing",
            ),
            OptionField(
                name="transcription_model_dir",
                label="Local Parakeet model folder",
                type="text",
                default="",
                # (task-3305) Example content, not the label repeated: an
                # empty Input otherwise shows label-as-placeholder stutter.
                placeholder="/path/to/parakeet-model",
                directory_picker=True,
                depends_on="parakeet_onnx",
                enabled_when="transcription_provider",
                enabled_when_values=("parakeet-onnx",),
                disabled_reason="needs the parakeet-onnx provider",
            ),
            OptionField(
                name="transcription_precision",
                label="Parakeet precision",
                type="select",
                default="int8",
                options=("int8", "f32"),
                # (task-3305 meta-rule) every select option carries human
                # copy; this field landed on dev mid-arc, so it is labeled
                # here rather than at its introduction.
                option_labels=(
                    ("int8", "INT8 (smaller · faster)"),
                    ("f32", "Float32 (full precision)"),
                ),
                depends_on="parakeet_onnx",
                enabled_when="transcription_provider",
                enabled_when_values=("parakeet-onnx",),
                disabled_reason="needs the parakeet-onnx provider",
            ),
            OptionField(
                name="transcription_model",
                label="Transcription model",
                type="select",
                default="base",
                # (task-3306) The full catalog the transcription service
                # itself declares for faster-whisper
                # (``TranscriptionService.list_available_models``): the
                # batch router passes an explicit faster-whisper model
                # through untouched, and the service hands it straight to
                # ``WhisperModel`` -- so the old five-size list silently
                # withheld large-v3, every English-only ``.en`` variant,
                # the distil family, and the community turbo/CrisperWhisper
                # builds. Order mirrors the service list.
                options=(
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
                ),
                option_labels=(
                    ("tiny", "Tiny (fastest · least accurate)"),
                    ("tiny.en", "Tiny · English-only"),
                    ("base", "Base (fast)"),
                    ("base.en", "Base · English-only"),
                    ("small", "Small (balanced)"),
                    ("small.en", "Small · English-only"),
                    ("medium", "Medium (more accurate · slower)"),
                    ("medium.en", "Medium · English-only"),
                    ("large-v1", "Large v1 (legacy)"),
                    ("large-v2", "Large v2"),
                    ("large-v3", "Large v3 (most accurate · slowest)"),
                    ("large", "Large (latest large build)"),
                    ("distil-large-v2", "Distil Large v2 (distilled · faster)"),
                    ("distil-medium.en", "Distil Medium · English-only"),
                    ("distil-small.en", "Distil Small · English-only"),
                    ("distil-large-v3", "Distil Large v3 (distilled · faster)"),
                    (
                        "deepdml/faster-distil-whisper-large-v3.5",
                        "Distil Large v3.5 (community build)",
                    ),
                    (
                        "deepdml/faster-whisper-large-v3-turbo-ct2",
                        "Large v3 Turbo (community build)",
                    ),
                    (
                        "nyrahealth/faster_CrisperWhisper",
                        "CrisperWhisper (verbatim · community build)",
                    ),
                ),
                depends_on="faster_whisper",
                enabled_when="transcription_provider",
                enabled_when_values=("faster-whisper",),
                disabled_reason="needs the faster-whisper provider",
            ),
            OptionField(
                name="language",
                label="Language",
                type="text",
                default="en",
                depends_on="audio_processing",
            ),
            OptionField(
                # (task-3303) Maps to ``translation_target_language="en"`` in
                # the job-option builder. Only faster-whisper translates
                # (``resolve_batch_stt_route``), and the semantic default
                # routes a translation request there too -- so the toggle is
                # inert (with the reason in its hint) under the providers
                # that would reject it outright.
                name="translate_to_english",
                label="Translate to English",
                type="checkbox",
                default=False,
                depends_on="faster_whisper",
                enabled_when="transcription_provider",
                enabled_when_values=("default", "faster-whisper"),
                hint="via faster-whisper",
            ),
            OptionField(
                name="timestamps",
                label="Include timestamps",
                type="checkbox",
                default=True,
                depends_on="audio_processing",
            ),
            OptionField(
                name="diarization",
                label="Speaker diarization",
                type="checkbox",
                default=False,
                depends_on="diarization",
            ),
            OptionField(
                # (task-3303) ``process_audio_files(vad_use=...)`` -- skips
                # non-speech segments during transcription.
                name="vad_filter",
                label="Voice activity detection (VAD) filter",
                type="checkbox",
                default=False,
                depends_on="audio_processing",
            ),
            OptionField(
                # (task-3306) ``process_audio_files(start_time=...)`` /
                # ``process_videos(start_time=...)``: local files trim via
                # ffmpeg (-ss/-to/-t), YouTube audio via yt-dlp
                # postprocessor args. The value travels verbatim, so the
                # accepted format is gated at the shared validator seam
                # (``validate_ingest_option_value``).
                name="start_time",
                label="Start at",
                type="text",
                default="",
                placeholder="e.g. 0:30 or 90",
                hint="HH:MM:SS or seconds · blank = from start",
                depends_on="audio_processing",
            ),
            OptionField(
                name="end_time",
                label="Stop at",
                type="text",
                default="",
                placeholder="e.g. 10:00",
                hint="HH:MM:SS or seconds · blank = to end",
                depends_on="audio_processing",
            ),
            OptionField(
                # (task-3306) A cookies FILE PATH, never raw cookie text:
                # this options map persists with the job and echoes into
                # ``[library.ingest_options.audio_video]`` in config.toml,
                # where a credential must not land (and the video
                # processor logs its kwargs at debug level). Only the
                # video (yt-dlp) download path consumes the file
                # (``ydl_opts["cookiefile"]``); the audio downloader's
                # cookies parameter is a JSON dict with different
                # semantics, so the job-option builder deliberately does
                # not feed this path to it.
                name="cookies_file",
                label="Cookies file for gated URLs",
                type="text",
                default="",
                placeholder="/path/to/cookies.txt",
                hint="Netscape cookies.txt · video URLs only",
                depends_on="yt_dlp",
            ),
            OptionField(
                # (task-3306) ``process_audio_files(summarize_recursively=
                # ...)``: with chunking on and analysis running, the
                # processor summarizes each chunk and then combines the
                # summaries (map-reduce) instead of one direct call. The
                # Analyze gate lives in the GENERIC group, which
                # ``enabled_when`` cannot reach across groups (the gate
                # lookup and the per-group value maps are both
                # group-scoped), so the dependency is stated in the hint --
                # the task-3303 convention.
                name="summarize_recursively",
                label="Recursive summary (map-reduce)",
                type="checkbox",
                default=False,
                hint="with Analyze after import · needs chunking",
                depends_on="audio_processing",
            ),
        ),
    ),
    "ebook": TypeGroupCapabilities(
        group="ebook",
        label="E-books",
        noun_singular="e-book",
        noun_plural="e-books",
        required_features=("ebook_processing",),
        optional_features=("html2text", "lxml", "beautifulsoup4"),
        fields=(
            OptionField(
                name="extraction_method",
                label="Extraction method",
                type="select",
                default="filtered",
                options=("filtered", "markdown", "basic"),
                # (task-3305, MI-09) Verified against
                # ``Book_Ingestion_Lib``: filtered follows the spine and
                # skips known front matter; markdown converts with TOC and
                # heading structure; basic reads every document item as
                # plain text.
                option_labels=(
                    ("filtered", "Filtered (skips covers & front matter)"),
                    ("markdown", "Markdown (keeps headings & structure)"),
                    ("basic", "Basic (every section · plain text)"),
                ),
                depends_on="ebook_processing",
            ),
            OptionField(
                # (task-3303) "chapters" maps to the chunker's real
                # ``ebook_chapters`` method in the job-option builder; the
                # other names travel verbatim (all four are methods
                # ``Chunk_Lib.Chunker.chunk_text`` dispatches on). Untouched,
                # no method is forced and ``process_ebook`` applies its own
                # chapters default -- so the schema default and the
                # processor default agree.
                name="chunk_method",
                label="Chunking method",
                type="select",
                default="chapters",
                options=("chapters", "sentences", "words", "paragraphs"),
                option_labels=(
                    ("chapters", "By chapter"),
                    ("sentences", "By sentence"),
                    ("words", "By word count"),
                    ("paragraphs", "By paragraph"),
                ),
                depends_on="ebook_processing",
            ),
            OptionField(
                name="include_toc",
                label="Include table of contents",
                type="checkbox",
                default=True,
                depends_on="ebook_processing",
            ),
        ),
    ),
    # (task-3307, ship ruling recorded in task-3310) Raster images: the
    # imported content IS the text OCR extracts (``process_image``), so the
    # whole panel is the OCR story. Visual features are deliberately not
    # offered: ``persist_parsed_media`` forwards no metadata, so the toggle
    # would compute a dict the pipeline drops (rejected-with-note in the
    # task).
    "image": TypeGroupCapabilities(
        group="image",
        label="Images",
        noun_singular="image",
        noun_plural="images",
        # Without at least one OCR backend no image can produce content --
        # the import always fails -- so the any-of umbrella is REQUIRED,
        # not merely nice to have (contrast the document group, whose
        # native parsers stand in for docling).
        required_features=("image_ocr",),
        optional_features=(),
        fields=(
            OptionField(
                name="ocr",
                label="Extract text (OCR)",
                type="checkbox",
                # Mirrors ``process_image``'s own enable_ocr=True -- and
                # with OCR off there is nothing to import: the job fails
                # honestly at the persist seam.
                default=True,
                depends_on="image_ocr",
                hint="the extracted text is what gets imported",
            ),
            OptionField(
                name="ocr_language",
                label="OCR language",
                type="text",
                default="en",
                depends_on="image_ocr",
                enabled_when="ocr",
                hint="e.g. en, de, fr",
                disabled_reason="needs Extract text (OCR) on",
            ),
            OptionField(
                name="ocr_backend",
                label="OCR backend",
                type="select",
                default="auto",
                # The OCR manager's registered backends
                # (``OCR_Backends._register_backends``), ordered by its own
                # default-selection priority; "auto" lets it pick the best
                # installed one.
                options=(
                    "auto",
                    "docext",
                    "docling",
                    "tesseract",
                    "easyocr",
                    "paddleocr",
                ),
                option_labels=(
                    ("auto", "Auto (best installed backend)"),
                    ("docext", "Docext (vision model)"),
                    ("docling", "Docling"),
                    ("tesseract", "Tesseract"),
                    ("easyocr", "EasyOCR"),
                    ("paddleocr", "PaddleOCR"),
                ),
                depends_on="image_ocr",
                enabled_when="ocr",
                disabled_reason="needs Extract text (OCR) on",
            ),
        ),
    ),
    "generic": TypeGroupCapabilities(
        group="generic",
        label="Import behavior",
        noun_singular="imported item",
        noun_plural="imported items",
        required_features=(),
        optional_features=(),
        fields=(
            OptionField(
                name="analyze",
                label="Analyze after import",
                type="checkbox",
                # Off by default: analysis costs an LLM call per document at
                # ingest time, which a user importing a folder has not asked
                # for and may not have a provider configured for.
                default=False,
                depends_on=None,
            ),
            OptionField(
                name="overwrite_existing",
                label="Overwrite existing",
                type="checkbox",
                default=False,
            ),
            OptionField(
                name="custom_prompt",
                label="Custom prompt",
                type="textarea",
                default="",
                placeholder="Optional instructions for analysis",
                enabled_when="analyze",
                disabled_reason="needs Analyze after import on",
            ),
            OptionField(
                name="system_prompt",
                label="System prompt",
                type="textarea",
                default="",
                placeholder="Optional system instructions for analysis",
                enabled_when="analyze",
                disabled_reason="needs Analyze after import on",
            ),
            OptionField(
                name="generate_embeddings",
                label="Generate embeddings",
                type="checkbox",
                # ADR-005 makes ingestion-time indexing the default. Keeping
                # this on preserves retrieval for users who do not open this
                # panel before importing.
                default=True,
            ),
            OptionField(
                name="keep_original_file",
                label="Keep original file",
                type="checkbox",
                default=False,
                backends=("server",),
            ),
            OptionField(
                name="chunk",
                label="Chunk content",
                type="checkbox",
                # On by default: chunking is local and cheap, and without it
                # imported documents are never chunked for retrieval, which
                # quietly undermines search and RAG for anyone who never
                # opens this panel.
                default=True,
                depends_on=None,
            ),
            OptionField(
                name="chunk_size",
                label="Chunk size",
                type="number",
                default=1000,
                enabled_when="chunk",
                # (task-3301) The unit is words, not characters: every
                # chunking method in the shared service sizes by its own
                # unit (words/sentences/paragraphs), and the ingest
                # pipeline chunks text with the service's word method. The
                # old "characters" hint described a pipeline that never
                # chunked at all.
                hint="words · 100–5000",
                disabled_reason="needs Chunk content on",
            ),
            OptionField(
                name="chunk_overlap",
                label="Chunk overlap",
                type="number",
                default=100,
                enabled_when="chunk",
                hint="words · at least 0",
                disabled_reason="needs Chunk content on",
            ),
            OptionField(
                name="encoding",
                label="Encoding",
                # (task-2100) A select of known encodings instead of free
                # text -- typed garbage silently degraded parsing.
                type="select",
                options=("auto", "utf-8", "utf-16", "latin-1", "cp1252"),
                # (task-3305, MI-09) "auto" per ``_decode_ingest_text``:
                # strict UTF-8 first, then chardet detection, then UTF-8
                # with replacement characters.
                option_labels=(
                    ("auto", "Auto-detect (UTF-8 first)"),
                    ("utf-8", "UTF-8"),
                    ("utf-16", "UTF-16"),
                    ("latin-1", "Latin-1 (ISO-8859-1)"),
                    ("cp1252", "Windows-1252 (Western)"),
                ),
                default="auto",
                depends_on=None,
            ),
        ),
    ),
    "web": TypeGroupCapabilities(
        group="web",
        label="Web pages",
        noun_singular="web page",
        noun_plural="web pages",
        # No local packages are required: the article extractor is part of the
        # app, and a server-backed clip runs entirely on the server.
        required_features=(),
        optional_features=(),
        fields=(
            OptionField(
                name="scrape_method",
                label="What to fetch",
                type="select",
                default="individual",
                # The server's ScrapeMethod enum, read from its OpenAPI
                # document. The local extractor only does the single-page case,
                # which is why that is the default for both backends.
                options=("individual", "sitemap", "url_level", "recursive_scraping"),
                # (task-3305, MI-09) The enum tokens are server-facing; the
                # user-facing question is scope.
                option_labels=(
                    ("individual", "This page only"),
                    ("sitemap", "Site map"),
                    ("url_level", "Pages under this URL"),
                    ("recursive_scraping", "Follow links (recursive)"),
                ),
                depends_on=None,
            ),
            OptionField(
                name="max_pages",
                label="Maximum pages",
                type="number",
                default=3,
                # Only meaningful once more than one page is being fetched.
                enabled_when="scrape_method",
                enabled_when_values=tuple(sorted(MULTI_PAGE_SCRAPE_METHODS)),
                # Accurate because "individual" is the ONLY non-multi-page
                # choice -- if that ever changes, reword to name the gate.
                disabled_reason="single-page fetch selected",
            ),
            OptionField(
                name="max_depth",
                label="Maximum depth",
                type="number",
                default=3,
                enabled_when="scrape_method",
                enabled_when_values=tuple(sorted(MULTI_PAGE_SCRAPE_METHODS)),
                disabled_reason="single-page fetch selected",
            ),
        ),
    ),
}


def list_type_groups() -> list[str]:
    """Return all known ingestion type group identifiers.

    Returns:
        List of group ids such as ``pdf``, ``audio_video``, ``ebook``,
        ``generic``.
    """
    return list(_TYPE_GROUPS.keys())


def get_type_group(path_or_url: str) -> str:
    """Map a file path or URL to a capability group.

    A URL is grouped by what it addresses, not merely by whether its path
    happens to carry a file extension. An extension still wins when there is
    one -- a link to a PDF or an ebook should be parsed as that, not scraped as
    HTML -- but an extension-less URL is a video if it points at a video host
    and otherwise a web page.

    This used to consult ``detect_file_type`` alone, so every extension-less URL
    landed in ``unsupported``: a YouTube link, the archetypal import, pre-flighted
    as an unsupported *file* even though the pipeline's own
    ``classify_ingest_source`` called it ``video`` and would have ingested it
    (task-702).

    Args:
        path_or_url: Local path, or an http(s) URL.

    Returns:
        One of ``pdf``, ``document``, ``image``, ``audio_video``, ``ebook``,
        ``web``, ``generic``, or ``unsupported``. Unsupported file types are
        mapped to ``unsupported`` so the pre-flight summary can surface them
        separately.
    """
    try:
        file_type = detect_file_type(path_or_url)
    except FileIngestionError:
        # No usable extension. For a URL that is normal, not a failure.
        if _is_http_url(path_or_url):
            return _url_type_group(path_or_url)
        return UNSUPPORTED_GROUP

    if file_type == "pdf":
        return "pdf"
    if file_type in ("audio", "video"):
        return "audio_video"
    if file_type == "ebook":
        return "ebook"
    if file_type == "document":
        # (task-3303) Word-processor formats get their own group -- they
        # used to fall through to ``generic``, whose panel called them
        # "plain text files" and reached none of ``process_document``'s
        # options. Placed before the URL check for the same reason pdf/
        # ebook are: an extension on a URL still says what the target IS.
        return "document"
    if file_type == "image" and not _is_http_url(path_or_url):
        # (task-3307) Raster FILES get their own group. Unlike pdf/ebook/
        # document above, the image group deliberately does NOT claim URLs:
        # the pipeline routes every URL through ``classify_ingest_source``,
        # which has no image branch, so ``https://example.com/chart.png``
        # is fetched and scraped as an article and ``process_image`` is
        # never called. Claiming it made the canvas report "1 image", mount
        # the OCR panel and raise the OCR-backend warning that forces the
        # two-press consent -- and then discard every OCR option
        # (task-3307 xhigh review round). Downloading image URLs to OCR
        # them is a real feature, but it is one the pipeline would have to
        # grow first; until then the canvas tells the truth.
        return "image"
    if _is_http_url(path_or_url) and file_type in (
        "plaintext",
        "html",
        "xml",
        "image",
    ):
        # A bare ".html"/".htm" URL is a page to fetch, not a local file to
        # read; the extension says how it is written, not where it lives.
        return _url_type_group(path_or_url)
    return "generic"


def _url_type_group(url: str) -> str:
    """Group an http(s) URL, deferring to the pipeline's own classification.

    Asking ``classify_ingest_source`` rather than re-deriving the rules keeps the
    canvas's verdict and the pipeline's behaviour from drifting apart -- they
    disagreed before, and the screen was the one that lied.
    """
    from ..Local_Ingestion.local_file_ingestion import classify_ingest_source

    try:
        classified = classify_ingest_source(url)
    except FileIngestionError:
        return UNSUPPORTED_GROUP
    if classified in ("audio", "video"):
        return "audio_video"
    if classified == "article":
        return "web"
    return "generic"


def generic_option_default(name: str, fallback: Any = None) -> Any:
    """Return the ``generic`` group's declared default for ``name``.

    (task-3301) The capability schema is the single source of ingest option
    defaults. Three consumers used to carry private copies of this lookup
    (``library_ingest_state``, ``server_ingest_request``) while the local
    job-option builder hardcoded its own values -- which is exactly how the
    UI displayed a chunk overlap of 100 while an untouched local submit fell
    back to 50. Every fallback question about a generic option goes through
    here now.

    Args:
        name: The generic option's machine name (e.g. ``chunk_overlap``).
        fallback: Returned when the schema has no field of that name.

    Returns:
        The schema default, or ``fallback`` for an unknown name.
    """
    for field_spec in _TYPE_GROUPS["generic"].fields:
        if field_spec.name == name:
            return field_spec.default
    return fallback


#: The generic group's Analyze-after-import toggle. Its state rides the
#: collapsed panel title (task-28007 AC#6) instead of hiding inside the
#: fold, so the title builder also drops it from the changed-value pairs
#: rather than stuttering it twice on one line.
ANALYSIS_STATE_FIELD = "analyze"


def type_group_state_summary(
    cap: TypeGroupCapabilities, values: dict[str, Any]
) -> str:
    """The group's collapsed-title label, carrying state the fold hides.

    task-28007 AC#6: "Import behavior" is collapsed by default and its
    headline setting -- whether every imported item gets an LLM analysis --
    was invisible until the panel was opened. Only the group that owns the
    toggle gains the clause; every other label is returned unchanged.

    Args:
        cap: The group's capability schema.
        values: Current per-group option values (a missing key falls back
            to the field's schema default, as everywhere else here).

    Returns:
        ``"Import behavior · analysis on"`` / ``"… · analysis off"`` for the
        owning group, else ``cap.label``.
    """
    gate = next((f for f in cap.fields if f.name == ANALYSIS_STATE_FIELD), None)
    if gate is None:
        return cap.label
    on = bool(values.get(ANALYSIS_STATE_FIELD, gate.default))
    return f"{cap.label} · analysis {'on' if on else 'off'}"


def field_disabled_state(
    field: OptionField,
    cap: TypeGroupCapabilities,
    values: dict[str, Any],
    *,
    is_installed: Any = None,
) -> tuple[bool, str]:
    """Whether ``field`` is currently uneditable, and the reason to render.

    (task-3304, MI-07) The single source for the canvas's disabled
    computation AND the reason annotation shown at the control, so the two
    can never disagree. Two independent gates, checked in the canvas's
    established order:

    1. ``depends_on`` -- a packaging gate. The reason names the missing
       feature ("needs Docling installed").
    2. ``enabled_when`` (optionally with ``enabled_when_values``) -- a
       within-form gate. The reason is the field's curated
       ``disabled_reason`` when present; otherwise a generic derivation
       from the gate metadata. Fields whose static ``hint`` already names
       the gate (task-3303's "docling or docext engines only" labels)
       return an EMPTY reason so the label is never double-annotated --
       the disabled state still shows, the why is already in the label.

    Args:
        field: The option field under evaluation.
        cap: The field's owning group schema (supplies gate siblings).
        values: Current per-group option values (missing keys fall back to
            each gate field's schema default).
        is_installed: Feature-availability probe; defaults to this
            module's :func:`_is_installed`. The canvas passes its own
            module-level reference so tests patching
            ``library_ingest_canvas._is_installed`` keep working.

    Returns:
        ``(disabled, reason)``. ``reason`` is ``""`` whenever the field is
        editable, and may also be ``""`` for a disabled field whose label
        already carries the gate (see above).
    """
    probe = _is_installed if is_installed is None else is_installed
    if field.depends_on is not None and not probe(field.depends_on):
        return True, f"needs {_feature_label(field.depends_on, cap.group)} installed"
    if field.enabled_when is None:
        return False, ""
    gate = next((f for f in cap.fields if f.name == field.enabled_when), None)
    gate_value = values.get(
        field.enabled_when, gate.default if gate is not None else False
    )
    if field.enabled_when_values:
        # A select gate: every non-empty choice is truthy, so the field
        # names the choices that actually enable it.
        if gate_value in field.enabled_when_values:
            return False, ""
    elif bool(gate_value):
        return False, ""
    if field.disabled_reason:
        return True, field.disabled_reason
    if field.hint:
        # The static hint already carries the gate at the control
        # (task-3303's convention); a second annotation would stutter.
        return True, ""
    gate_label = gate.label if gate is not None else field.enabled_when
    if field.enabled_when_values:
        wanted = ", ".join(str(value) for value in field.enabled_when_values)
        return True, f"needs {gate_label}: {wanted}"
    return True, f"needs {gate_label} on"


def field_gate_open(group: str, name: str, values: dict[str, Any]) -> bool:
    """Whether ``name``'s within-form ``enabled_when`` gate is open.

    (task-3303 xhigh review round 2, F9) The job-option builder must consult
    the SAME sibling-field gate the form renders with before forwarding a
    gated value: a checkbox ticked while its gate was open goes stale when
    the gate field changes (the form disables the control but keeps the
    value), and forwarding the stale value can change behavior downstream --
    the concrete incident being ``translate_to_english`` checked under
    provider=default, provider then switched to transcribe-cpp, and the
    stale ``True`` becoming ``translation_target_language='en'`` ->
    ``BatchSTTRoutingError`` -> every audio/video job in the batch FAILED at
    dispatch.

    Only the within-form gate is consulted -- deliberately NOT
    ``depends_on`` (the packaging gate): a value for a field whose optional
    package is missing is handled by the pipeline's own failure/ignore
    paths, and suppressing it here would make a job's options depend on
    which machine resolved them.

    Args:
        group: Type group identifier (e.g. ``audio_video``).
        name: The gated field's machine name.
        values: Current option values for the group (the builder's merged
            ``flat_opts``); a missing gate value falls back to the gate
            field's schema default, mirroring :func:`field_disabled_state`.

    Returns:
        True when the field's gate is open (or it has no gate, or the
        group/field is unknown -- unknown never suppresses a value).
    """
    try:
        cap = get_capabilities(group)
    except KeyError:
        return True
    field = next((f for f in cap.fields if f.name == name), None)
    if field is None or field.enabled_when is None:
        return True
    gate = next((f for f in cap.fields if f.name == field.enabled_when), None)
    gate_value = values.get(
        field.enabled_when, gate.default if gate is not None else False
    )
    if field.enabled_when_values:
        return gate_value in field.enabled_when_values
    return bool(gate_value)


def get_capabilities(group: str) -> TypeGroupCapabilities:
    """Return capabilities metadata for a type group.

    Args:
        group: Type group identifier.

    Returns:
        ``TypeGroupCapabilities`` for the requested group.

    Raises:
        KeyError: If ``group`` is not a known type group.
    """
    return _TYPE_GROUPS[group]


def classify_missing_features(
    group: str, missing: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ``missing`` into ``group``'s REQUIRED and OPTIONAL features.

    (task-14820) The forecast has to tell "this group cannot run at all"
    apart from "this group runs, just less well": a file whose group has
    an unmet REQUIRED feature is a certain failure, while an unmet
    OPTIONAL one only degrades it. That required/optional split is
    declared here, on :class:`TypeGroupCapabilities`, and consumers used
    to re-derive it by unioning both tuples (``count_warning_affected_files``
    did exactly that, which is why every warned file read as "may fail"
    and none as "will fail"). One accessor, so a schema change reaches
    every consumer.

    Args:
        group: Type group identifier. :data:`UNSUPPORTED_GROUP` (and any
            unknown group) has no declared tooling, so nothing classifies.
        missing: Feature IDs known to be unavailable -- typically the
            ``feature`` keys of a pre-flight's warnings, so the answer
            matches what the user was actually told rather than a fresh
            probe of this process.

    Returns:
        ``(required_missing, optional_missing)``, each in the group's own
        declared order, containing only members of ``missing``.
    """
    if group == UNSUPPORTED_GROUP or group not in _TYPE_GROUPS:
        return (), ()
    wanted = {str(feature).strip() for feature in missing} - {""}
    if not wanted:
        return (), ()
    capabilities = get_capabilities(group)
    return (
        tuple(f for f in capabilities.required_features if f in wanted),
        tuple(f for f in capabilities.optional_features if f in wanted),
    )


def get_tooling_warnings(group: str) -> list[dict[str, Any]]:
    """Return install warnings for missing tooling in a type group.

    Args:
        group: Type group identifier.

    Returns:
        List of warning dictionaries with ``feature``, ``label``, ``hint``,
        and ``command`` keys. Empty for :data:`UNSUPPORTED_GROUP`, which has
        no tooling to be missing -- installing something would not make those
        files ingestible.

    Raises:
        KeyError: If ``group`` is neither a known type group nor the
            unsupported sentinel.
    """
    if group == UNSUPPORTED_GROUP:
        return []
    capabilities = get_capabilities(group)
    warnings: list[dict[str, Any]] = []

    for feature in capabilities.required_features + capabilities.optional_features:
        if not _is_installed(feature):
            hint = _install_hint(feature)
            warnings.append(
                {
                    "feature": feature,
                    "label": _feature_label(feature, group),
                    "hint": _GROUP_FEATURE_HINTS.get(
                        (group, feature), hint["hint"]
                    ),
                    "command": hint["command"],
                }
            )

    return warnings


def _feature_label(feature: str, group: str) -> str:
    """Return a human-readable label for a feature ID."""
    if feature in _FEATURE_LABELS:
        return _FEATURE_LABELS[feature]

    extra = _FEATURE_TO_EXTRA.get(feature)
    if extra is not None and extra in OPTIONAL_FEATURES:
        return OPTIONAL_FEATURES[extra].label

    for extra in _GROUP_EXTRAS.get(group, ()):
        if extra in OPTIONAL_FEATURES:
            return OPTIONAL_FEATURES[extra].label

    return feature.replace("_", " ").title()
