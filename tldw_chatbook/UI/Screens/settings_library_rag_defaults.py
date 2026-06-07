"""Library and RAG guided defaults for the Settings hub."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from tldw_chatbook.RAG_Search.simplified.config import RAGConfig

from .settings_config_models import SettingsValidationResult


SEARCH_MODES = frozenset({"plain", "semantic", "hybrid"})
CITATION_STYLES = frozenset({"inline", "footnote", "none"})
DEFAULT_SEARCH_MODE = "semantic"
DEFAULT_CITATION_STYLE = "inline"
MIN_RAG_RESULT_COUNT = 1
MAX_RAG_RESULT_COUNT = 100
MIN_RAG_BALANCE = 0.0
MAX_RAG_BALANCE = 1.0
MIN_RAG_SNIPPET_CHARS = 50
MAX_RAG_SNIPPET_CHARS = 10000
MIN_RAG_CONTEXT_CHARS = 1000
MAX_RAG_CONTEXT_CHARS = 1000000


@dataclass(frozen=True)
class SettingsLibraryRagDefaults:
    """Editable Library/RAG defaults exposed in Settings."""

    default_search_mode: str = DEFAULT_SEARCH_MODE
    default_top_k: int = 10
    fts_top_k: int = 10
    vector_top_k: int = 10
    hybrid_alpha: float = 0.5
    score_threshold: float = 0.0
    include_citations: bool = True
    citation_style: str = DEFAULT_CITATION_STYLE
    snippet_max_chars: int = 240
    max_context_size: int = 16000


def _coerce_bool(value: Any, default: bool) -> bool:
    """Coerce common Settings field values to a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    """Coerce a Settings field to int, falling back on parse failures."""
def _coerce_int(value: Any, default: int) -> int:
    """Coerce a Settings field to int, falling back on parse failures."""
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default
    if not parsed.is_integer():
        return default
    return int(parsed)


def _coerce_float(value: Any, default: float) -> float:
    """Coerce a Settings field to float, falling back on parse failures."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _strict_int(value: Any) -> int | None:
    """Return an int only when the value is an unambiguous integer."""
    if isinstance(value, bool):
        return None
    try:
        f = float(str(value).strip())
        if f.is_integer():
            return int(f)
    except (TypeError, ValueError):
        pass
    return None
    if not parsed.is_integer():
        return None
    return int(parsed)


def _strict_float(value: Any) -> float | None:
    """Return a float only when the value is parseable."""
    if isinstance(value, bool):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _rag_section(app_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the nested AppRAGSearchConfig.rag mapping when present."""
    section = app_config.get("AppRAGSearchConfig", {})
    if not isinstance(section, Mapping):
        return {}
    rag = section.get("rag", {})
    return rag if isinstance(rag, Mapping) else {}


def _nested_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a child mapping or an empty mapping when the value is absent."""
    child = parent.get(key, {})
    return child if isinstance(child, Mapping) else {}


def normalise_library_rag_search_mode(value: Any) -> str:
    """Return a safe Library/RAG search mode for widgets.

    Args:
        value: Raw config or draft value.

    Returns:
        A supported search mode, falling back to the semantic default.
    """
    text = str(value).strip()
    return text if text in SEARCH_MODES else DEFAULT_SEARCH_MODE


def normalise_library_rag_citation_style(value: Any) -> str:
    """Return a safe Library/RAG citation style for widgets.

    Args:
        value: Raw config or draft value.

    Returns:
        A supported citation style, falling back to the inline default.
    """
    text = str(value).strip()
    return text if text in CITATION_STYLES else DEFAULT_CITATION_STYLE


def load_library_rag_defaults(app_config: Mapping[str, Any]) -> SettingsLibraryRagDefaults:
    """Load Settings-owned Library/RAG defaults from app configuration.

    Args:
        app_config: Application configuration mapping to read from.

    Returns:
        Coerced Library/RAG defaults with safe fallbacks for malformed numeric
        values.
    """
    fallback = RAGConfig()
    rag = _rag_section(app_config)
    search = _nested_mapping(rag, "search")
    retriever = _nested_mapping(rag, "retriever")

    return SettingsLibraryRagDefaults(
        default_search_mode=str(
            search.get("default_search_mode", fallback.search.default_search_mode)
        ),
        default_top_k=_coerce_int(
            search.get("default_top_k", fallback.search.default_top_k),
            fallback.search.default_top_k,
        ),
        fts_top_k=_coerce_int(
            retriever.get("fts_top_k", fallback.search.fts_top_k),
            fallback.search.fts_top_k,
        ),
        vector_top_k=_coerce_int(
            retriever.get("vector_top_k", fallback.search.vector_top_k),
            fallback.search.vector_top_k,
        ),
        hybrid_alpha=_coerce_float(
            retriever.get("hybrid_alpha", fallback.search.hybrid_alpha),
            fallback.search.hybrid_alpha,
        ),
        score_threshold=_coerce_float(
            search.get("score_threshold", fallback.search.score_threshold),
            fallback.search.score_threshold,
        ),
        include_citations=_coerce_bool(
            search.get("include_citations", fallback.search.include_citations),
            fallback.search.include_citations,
        ),
        citation_style=str(
            search.get("citation_style", fallback.search.citation_style)
        ),
        snippet_max_chars=_coerce_int(
            search.get("snippet_max_chars", fallback.search.snippet_max_chars),
            fallback.search.snippet_max_chars,
        ),
        max_context_size=_coerce_int(
            search.get("max_context_size", fallback.search.max_context_size),
            fallback.search.max_context_size,
        ),
    )


def validate_library_rag_defaults(
    values: SettingsLibraryRagDefaults,
) -> SettingsValidationResult:
    """Validate editable Library/RAG defaults before persistence.

    Args:
        values: Library/RAG defaults to validate.

    Returns:
        Validation state and user-facing recovery copy.
    """
    if values.default_search_mode not in SEARCH_MODES:
        return SettingsValidationResult(
            False,
            "Search mode must be plain, semantic, or hybrid.",
        )
    default_top_k = _strict_int(values.default_top_k)
    if (
        default_top_k is None
        or not MIN_RAG_RESULT_COUNT <= default_top_k <= MAX_RAG_RESULT_COUNT
    ):
        return SettingsValidationResult(
            False,
            "Default results must be between "
            f"{MIN_RAG_RESULT_COUNT} and {MAX_RAG_RESULT_COUNT}.",
        )
    fts_top_k = _strict_int(values.fts_top_k)
    if fts_top_k is None or not (
        MIN_RAG_RESULT_COUNT <= fts_top_k <= MAX_RAG_RESULT_COUNT
    ):
        return SettingsValidationResult(
            False,
            "Keyword results must be between "
            f"{MIN_RAG_RESULT_COUNT} and {MAX_RAG_RESULT_COUNT}.",
        )
    vector_top_k = _strict_int(values.vector_top_k)
    if (
        vector_top_k is None
        or not MIN_RAG_RESULT_COUNT <= vector_top_k <= MAX_RAG_RESULT_COUNT
    ):
        return SettingsValidationResult(
            False,
            "Vector results must be between "
            f"{MIN_RAG_RESULT_COUNT} and {MAX_RAG_RESULT_COUNT}.",
        )
    hybrid_alpha = _strict_float(values.hybrid_alpha)
    if hybrid_alpha is None or not MIN_RAG_BALANCE <= hybrid_alpha <= MAX_RAG_BALANCE:
        return SettingsValidationResult(
            False,
            f"Hybrid balance must be between {MIN_RAG_BALANCE:.1f} and {MAX_RAG_BALANCE:.1f}.",
        )
    score_threshold = _strict_float(values.score_threshold)
    if score_threshold is None or not MIN_RAG_BALANCE <= score_threshold <= MAX_RAG_BALANCE:
        return SettingsValidationResult(
            False,
            f"Score threshold must be between {MIN_RAG_BALANCE:.1f} and {MAX_RAG_BALANCE:.1f}.",
        )
    if values.citation_style not in CITATION_STYLES:
        return SettingsValidationResult(
            False,
            "Citation style must be inline, footnote, or none.",
        )
    snippet_max_chars = _strict_int(values.snippet_max_chars)
    if (
        snippet_max_chars is None
        or not MIN_RAG_SNIPPET_CHARS <= snippet_max_chars <= MAX_RAG_SNIPPET_CHARS
    ):
        return SettingsValidationResult(
            False,
            "Snippet characters must be between "
            f"{MIN_RAG_SNIPPET_CHARS} and {MAX_RAG_SNIPPET_CHARS}.",
        )
    max_context_size = _strict_int(values.max_context_size)
    if (
        max_context_size is None
        or not MIN_RAG_CONTEXT_CHARS <= max_context_size <= MAX_RAG_CONTEXT_CHARS
    ):
        return SettingsValidationResult(
            False,
            "Context budget must be between "
            f"{MIN_RAG_CONTEXT_CHARS} and {MAX_RAG_CONTEXT_CHARS} characters.",
        )
    return SettingsValidationResult(True, "Library/RAG defaults are valid.")


def build_library_rag_save_sections(
    app_config: Mapping[str, Any],
    values: SettingsLibraryRagDefaults,
) -> dict[str, dict[str, Any]]:
    """Build a deep-merged config payload for Settings persistence.

    Args:
        app_config: Existing application configuration used as the merge base.
        values: Validated Library/RAG defaults to persist.

    Returns:
        Config sections suitable for ``SettingsConfigAdapter.save_sections``.
    """
    existing_section = app_config.get("AppRAGSearchConfig", {})
    if not isinstance(existing_section, Mapping):
        existing_section = {}

    section = deepcopy(dict(existing_section))
    rag = section.get("rag", {})
    if not isinstance(rag, Mapping):
        rag = {}
    rag = deepcopy(dict(rag))

    search = deepcopy(dict(_nested_mapping(rag, "search")))
    retriever = deepcopy(dict(_nested_mapping(rag, "retriever")))

    value_map = asdict(values)
    search.update(
        {
            "default_search_mode": value_map["default_search_mode"],
            "default_top_k": value_map["default_top_k"],
            "score_threshold": value_map["score_threshold"],
            "include_citations": value_map["include_citations"],
            "citation_style": value_map["citation_style"],
            "snippet_max_chars": value_map["snippet_max_chars"],
            "max_context_size": value_map["max_context_size"],
        }
    )
    retriever.update(
        {
            "fts_top_k": value_map["fts_top_k"],
            "vector_top_k": value_map["vector_top_k"],
            "hybrid_alpha": value_map["hybrid_alpha"],
        }
    )

    rag["search"] = search
    rag["retriever"] = retriever
    section["rag"] = rag
    return {"AppRAGSearchConfig": section}
