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
        depends_on: Optional dependency feature ID that must be installed for
            this field to be editable. ``None`` means the field is always
            available.
    """

    name: str
    label: str
    type: str
    default: Any = None
    options: tuple[str, ...] = ()
    depends_on: str | None = None


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
    "parakeet_mlx": "Parakeet MLX",
    "pdf_processing": "PDF processing",
    "pymupdf": "PyMuPDF",
    "pymupdf4llm": "PyMuPDF4LLM",
    "scipy": "SciPy",
    "soundfile": "SoundFile",
    "video_processing": "Video processing",
    "yt_dlp": "yt-dlp",
}


def _is_installed(feature_id: str) -> bool:
    """Return whether ``feature_id`` is available.

    Checks the cached ``DEPENDENCIES_AVAILABLE`` registry first, then falls
    back to a cheap ``importlib.util.find_spec`` probe using the explicit
    PyPI-name-to-import-name mapping. PyPI names are never passed directly to
    ``find_spec``.

    Args:
        feature_id: Dependency flag from ``optional_deps`` or a PyPI package
            name known to this module.

    Returns:
        True when the feature appears to be installed.
    """
    if feature_id in DEPENDENCIES_AVAILABLE:
        return bool(DEPENDENCIES_AVAILABLE[feature_id])

    import_name = _PYPI_TO_IMPORT.get(feature_id)
    if import_name is None:
        return False

    try:
        return importlib.util.find_spec(import_name) is not None
    except Exception:
        return False


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
        return {
            "hint": f"{info.label} is unavailable: {info.unavailable_what}.",
            "command": info.source_install_command,
        }

    return {
        "hint": f"Optional dependency '{feature_id}' is not installed.",
        "command": f'pip install -e ".[{feature_id}]"',
    }


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
                name="extract_images",
                label="Extract images",
                type="checkbox",
                default=False,
                depends_on="pdf_processing",
            ),
            OptionField(
                name="enable_ocr",
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
            "parakeet_mlx",
            "yt_dlp",
            "video_processing",
        ),
        fields=(
            OptionField(
                name="transcription_backend",
                label="Transcription backend",
                type="select",
                default="faster_whisper",
                options=(
                    "faster_whisper",
                    "lightning_whisper_mlx",
                    "parakeet_mlx",
                ),
                depends_on="audio_processing",
            ),
            OptionField(
                name="transcription_model",
                label="Transcription model",
                type="select",
                default="base",
                options=("tiny", "base", "small", "medium", "large"),
                depends_on="faster_whisper",
            ),
            OptionField(
                name="extract_audio",
                label="Extract audio from video",
                type="checkbox",
                default=False,
                depends_on="video_processing",
            ),
            OptionField(
                name="language",
                label="Language",
                type="text",
                default="en",
                depends_on="audio_processing",
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
                name="html_converter",
                label="HTML converter",
                type="select",
                default="ebooklib",
                options=("ebooklib", "html2text", "beautifulsoup4"),
                depends_on="ebook_processing",
            ),
            OptionField(
                name="extract_toc",
                label="Extract table of contents",
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
                name="chunk_size",
                label="Chunk size",
                type="number",
                default=1000,
                depends_on=None,
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
}


def get_type_group(path_or_url: str) -> str:
    """Map a file path or URL to a capability group.

    Args:
        path_or_url: Local path or URL-like string ending in a filename.

    Returns:
        One of ``pdf``, ``audio_video``, ``ebook``, or ``generic``. Unsupported
        file types are mapped to ``generic`` rather than raising.
    """
    try:
        file_type = detect_file_type(path_or_url)
    except FileIngestionError:
        return "generic"

    if file_type == "pdf":
        return "pdf"
    if file_type in ("audio", "video"):
        return "audio_video"
    if file_type == "ebook":
        return "ebook"
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
        and ``command`` keys.
    """
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
