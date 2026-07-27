"""Capability discovery for library ingestion workflows.

This module exposes per-media-type ingestion settings and availability
warnings without importing heavy optional dependencies. It is intended to
back configuration UIs that need to know which backends are installed and
what options they expose.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
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
    """

    name: str
    label: str
    type: str
    default: Any = None
    options: tuple[str, ...] = ()
    depends_on: str | None = None
    enabled_when: str | None = None
    enabled_when_values: tuple[Any, ...] = ()


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
    """

    group: str
    label: str
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...]
    fields: tuple[OptionField, ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return the machine names of all configured fields."""
        return tuple(f.name for f in self.fields)


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
    "parakeet_onnx": "transcription_parakeet_onnx",
    "parakeet_mlx": "transcription_parakeet",
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
    "generic": (),
}

# Human-readable labels for individual feature IDs. These keep warning lists
# distinct when multiple features resolve to the same install extra.
_FEATURE_LABELS: dict[str, str] = {
    "audio_processing": "Audio processing",
    "beautifulsoup4": "BeautifulSoup",
    "docling": "Docling",
    "docext": "Docext",
    "ebook_processing": "E-book processing",
    "ebooklib": "ebooklib",
    "faster_whisper": "Faster Whisper",
    "html2text": "html2text",
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
    if required is not None:
        result = all(_module_present(package) for package in required)
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
        required_features=("pdf_processing",),
        optional_features=("pymupdf4llm", "docling"),
        fields=(
            OptionField(
                name="pdf_engine",
                label="PDF engine",
                type="select",
                default="pymupdf4llm",
                options=("pymupdf", "pymupdf4llm", "docling"),
                depends_on="pdf_processing",
            ),
            OptionField(
                name="ocr",
                label="Enable OCR",
                type="checkbox",
                default=False,
                depends_on="pdf_processing",
            ),
        ),
    ),
    "audio_video": TypeGroupCapabilities(
        group="audio_video",
        label="Audio & video",
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
                options=("default", "parakeet-onnx", "faster-whisper"),
                depends_on="audio_processing",
            ),
            OptionField(
                name="transcription_model_dir",
                label="Local Parakeet model folder",
                type="text",
                default="",
                depends_on="parakeet_onnx",
                enabled_when="transcription_provider",
                enabled_when_values=("parakeet-onnx",),
            ),
            OptionField(
                name="transcription_model",
                label="Transcription model",
                type="select",
                default="base",
                options=("tiny", "base", "small", "medium", "large"),
                depends_on="faster_whisper",
                enabled_when="transcription_provider",
                enabled_when_values=("faster-whisper",),
            ),
            OptionField(
                name="language",
                label="Language",
                type="text",
                default="en",
                depends_on="audio_processing",
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
        ),
    ),
    "ebook": TypeGroupCapabilities(
        group="ebook",
        label="E-books",
        required_features=("ebook_processing",),
        optional_features=("html2text", "lxml", "beautifulsoup4"),
        fields=(
            OptionField(
                name="extraction_method",
                label="Extraction method",
                type="select",
                default="filtered",
                options=("filtered", "markdown", "basic"),
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
    "generic": TypeGroupCapabilities(
        group="generic",
        label="Plain text / documents / HTML",
        required_features=(),
        optional_features=(),
        fields=(
            OptionField(
                name="analyze",
                label="Analyze after ingest",
                type="checkbox",
                # Off by default: analysis costs an LLM call per document at
                # ingest time, which a user importing a folder has not asked
                # for and may not have a provider configured for.
                default=False,
                depends_on=None,
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
            ),
            OptionField(
                name="chunk_overlap",
                label="Chunk overlap",
                type="number",
                default=100,
                enabled_when="chunk",
            ),
            OptionField(
                name="encoding",
                label="Encoding",
                type="text",
                default="auto",
                depends_on=None,
            ),
        ),
    ),
    "web": TypeGroupCapabilities(
        group="web",
        label="Web pages",
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
            ),
            OptionField(
                name="max_depth",
                label="Maximum depth",
                type="number",
                default=3,
                enabled_when="scrape_method",
                enabled_when_values=tuple(sorted(MULTI_PAGE_SCRAPE_METHODS)),
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
        One of ``pdf``, ``audio_video``, ``ebook``, ``web``, ``generic``, or
        ``unsupported``. Unsupported file types are mapped to ``unsupported``
        so the pre-flight summary can surface them separately.
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
    if _is_http_url(path_or_url) and file_type in ("plaintext", "html", "xml"):
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
                    "hint": hint["hint"],
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
