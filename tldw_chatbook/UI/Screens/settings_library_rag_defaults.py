"""Library and RAG guided defaults for the Settings hub."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .settings_config_models import SettingsValidationResult


SEARCH_MODES = frozenset({"plain", "semantic", "hybrid"})
CITATION_STYLES = frozenset({"inline", "footnote", "none"})
CHUNKING_METHODS = frozenset({"words", "sentences", "paragraphs"})
DISTANCE_METRICS = frozenset({"cosine", "l2", "ip"})
DEFAULT_SEARCH_MODE = "semantic"
DEFAULT_CITATION_STYLE = "inline"
DEFAULT_CHUNKING_METHOD = "words"
DEFAULT_DISTANCE_METRIC = "cosine"
#: ``RerankingConfig().model_provider`` (RAG_Search/reranker.py) -- the
#: provider a bare reranker config bills its per-candidate calls to.
#: Hardcoded rather than imported for the same reason ``reranker_top_k``'s
#: default is (reranker.py drags Chat_Functions/Internal_Prompts in behind
#: it); kept honest by
#: ``test_reranker_provider_default_constant_matches_rerankingconfig_default``.
DEFAULT_RERANKER_PROVIDER = "openai"
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
    hybrid_alpha: float = 0.7
    score_threshold: float = 0.0
    include_citations: bool = True
    citation_style: str = DEFAULT_CITATION_STYLE
    snippet_max_chars: int = 240
    max_context_size: int = 16000
    # Embedding (task-3/SP3: extended profile editor)
    embedding_model: str = "mxbai-embed-large-v1"
    embedding_device: str = "auto"
    embedding_batch_size: int = 16
    embedding_max_length: int = 512
    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 100
    chunking_method: str = DEFAULT_CHUNKING_METHOD
    # Vector store
    distance_metric: str = DEFAULT_DISTANCE_METRIC
    # Reranking -- `enable_reranking` controls the PRESENCE of the active
    # profile's `reranking_config` (see settings_rag_profile_adapter.py); a
    # blank `reranker_model`/`reranker_provider` means "use the reranker's
    # own default" (TASK-3502 AC#1: the provider the per-candidate calls are
    # BILLED to used to be unreachable from Settings entirely -- enabling
    # reranking created a bare `RerankingConfig()` and whatever provider it
    # defaulted to was the one that got charged).
    enable_reranking: bool = False
    reranker_provider: str = ""
    reranker_model: str = ""
    # 20 == RerankingConfig().top_k_to_rerank (RAG_Search/reranker.py) --
    # NOT SearchConfig.reranker_top_k (5), a functionally-dead field the RAG
    # engine never reads for reranking (rag_factory.py decides reranking
    # purely from `profile.reranking_config is not None`). Hardcoded rather
    # than imported to avoid pulling reranker.py's heavier import chain
    # (Chat_Functions, Internal_Prompts) into this otherwise-light module;
    # kept honest by test_reranker_top_k_default_matches_reranking_config.
    reranker_top_k: int = 20
    # task-1337 (ADR-030): global [console] toggle -- when True, Console
    # agents get the 18 direct Library tools; when False, they fall back to
    # the bounded Library RAG tool. Global app config, NOT profile-scoped and
    # never serialized into per-conversation session settings; the adapter
    # overlays the live value on load (see load_direct_library_tools).
    direct_library_tools: bool = True


#: Bool string forms recognised by config coercion; anything outside this set
#: is treated as malformed and falls back to the safe default (True), unlike
#: ``coerce_bool_setting`` which maps ANY unrecognised string to False.
_RECOGNIZED_BOOL_STRINGS = frozenset(
    {"true", "1", "t", "y", "yes", "false", "0", "f", "n", "no"}
)


def load_direct_library_tools(app_config: Mapping[str, Any] | None = None) -> bool:
    """Read the global ``[console].direct_library_tools`` toggle.

    Args:
        app_config: Config mapping to read; when ``None``, the live CLI
            config is loaded fresh (so a Settings save applies to the next
            Console run without a restart).

    Returns:
        The configured boolean; ``True`` when the section/key is missing or
        the value is malformed (direct tools are the default retrieval mode).
    """
    if app_config is None:
        # Lazy import: keeps this module free of the config import chain for
        # the adapter's headless callers.
        from tldw_chatbook.config import get_cli_setting

        raw = get_cli_setting("console", "direct_library_tools", True)
    else:
        console = app_config.get("console")
        raw = (
            console.get("direct_library_tools", True)
            if isinstance(console, Mapping)
            else True
        )
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in _RECOGNIZED_BOOL_STRINGS:
            # Route through the existing coercion (whitespace-normalized --
            # coerce_bool_setting itself does not strip).
            from tldw_chatbook.config import coerce_bool_setting

            return coerce_bool_setting(normalized, True)
    return True


#: Approved visible copy for the Console agent retrieval subsection (spec
#: section 8). Rendered below the toggle as plain text -- never a tooltip.
CONSOLE_DIRECT_LIBRARY_TOOLS_COPY = (
    "On: Console agents may automatically list, count, read, and lexically "
    "search your local Library.\n"
    "Off: Direct list, count, view, and lexical search tools are unavailable. "
    "Console agents use Library RAG as the default retrieval method. RAG "
    "currently covers Notes, Media, and Conversations and requires an "
    "available, populated index.\n"
    "Privacy: Retrieved titles, metadata, content, and RAG excerpts are "
    "included in model requests. If you use a cloud model, this Library data "
    "leaves your device and is handled by that provider. Use a local model if "
    "the data must remain on-device.\n"
    "Scope: This setting affects Console agents only. MCP Library access is "
    "controlled separately."
)


def build_library_rag_save_sections(
    app_config: Mapping[str, Any],
    values: SettingsLibraryRagDefaults,
) -> dict[str, dict[str, Any]]:
    """Build config sections persisted alongside a Library/RAG profile save.

    The RAG search/chunking fields themselves live in the active RAG profile
    (written by the profile adapter); this returns the deep-merged ``console``
    section carrying the global retrieval-mode toggle plus a verbatim copy of
    ``AppRAGSearchConfig`` so the two-section write stays atomic and unrelated
    keys in either section survive.

    Args:
        app_config: Existing application configuration mapping.
        values: Validated Library/RAG defaults being saved.

    Returns:
        ``{"console": ..., "AppRAGSearchConfig": ...}`` suitable for
        ``SettingsConfigAdapter.save_sections``.
    """
    console = app_config.get("console")
    merged_console = dict(deepcopy(console)) if isinstance(console, Mapping) else {}
    merged_console["direct_library_tools"] = bool(values.direct_library_tools)
    rag_section = app_config.get("AppRAGSearchConfig")
    return {
        "console": merged_console,
        "AppRAGSearchConfig": (
            dict(deepcopy(rag_section)) if isinstance(rag_section, Mapping) else {}
        ),
    }


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


def _strict_float(value: Any) -> float | None:
    """Return a float only when the value is parseable."""
    if isinstance(value, bool):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


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


def normalise_library_rag_chunking_method(value: Any) -> str:
    """Return a safe Library/RAG chunking method for widgets.

    Args:
        value: Raw config or draft value.

    Returns:
        A supported chunking method, falling back to the words default.
    """
    text = str(value).strip()
    return text if text in CHUNKING_METHODS else DEFAULT_CHUNKING_METHOD


def normalise_library_rag_distance_metric(value: Any) -> str:
    """Return a safe Library/RAG vector-store distance metric for widgets.

    Args:
        value: Raw config or draft value.

    Returns:
        A supported distance metric, falling back to the cosine default.
    """
    text = str(value).strip()
    return text if text in DISTANCE_METRICS else DEFAULT_DISTANCE_METRIC


def library_rag_reranker_providers() -> tuple[str, ...]:
    """Return the chat provider names this build registers.

    Enumerated from ``Chat_Functions.API_CALL_HANDLERS`` -- the exact
    dispatch table ``chat_api_call`` looks the reranker's
    ``model_provider`` up in (``RAG_Search/reranker.py``'s
    ``_call_llm_impl``) -- rather than hand-listed, so a newly registered
    chat provider cannot silently be missing here, and a name that would
    dispatch nowhere cannot be offered. Imported lazily: this module is
    deliberately light (see ``load_direct_library_tools``), and
    ``Chat_Functions`` is not.

    This enumerates what ``chat_api_call`` can ROUTE, which is not the same
    as what the reranker can currently CALL: its credential lookup
    (``_call_llm_impl``) covers far fewer providers than the table has
    rows, so most picks fail at run time -- disclosed on the results screen
    ("Reranking was skipped (No API key found for provider: X)") rather
    than silently. TASK-17065 owns closing that gap, and until it does the
    honest reading of this list is "offered", not "guaranteed callable".

    Returns:
        Every registered chat provider name, ``DEFAULT_RERANKER_PROVIDER``
        first and the rest sorted, so the default reads as the head of the
        list rather than an arbitrary row inside it.
    """
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    rest = sorted(
        name for name in API_CALL_HANDLERS if name != DEFAULT_RERANKER_PROVIDER
    )
    return (DEFAULT_RERANKER_PROVIDER, *rest)


def library_rag_reranker_provider_options() -> list[tuple[str, str]]:
    """Return ``(label, value)`` pairs for the reranker-provider Select.

    The default provider's row is labelled explicitly -- "openai (default)"
    -- and carries that provider's own NAME as its value, not a blank
    sentinel (spec AC#1: "the default made visible rather than implicit").
    Blank stays a legal *field* value meaning "leave the reranker's own
    default alone" (``apply_defaults_to_profile``, the rule
    ``reranker_model`` follows), but the Select never emits it: a user
    switching a profile back from anthropic to openai must actually write
    openai, and staging blank there would silently leave anthropic in place.

    Returns:
        One option per dispatchable provider, default row first.
    """
    default, *rest = library_rag_reranker_providers()
    return [(f"{default} (default)", default)] + [(name, name) for name in rest]


def normalise_library_rag_reranker_provider(value: Any) -> str:
    """Return a safe reranker provider value for the Select widget.

    Args:
        value: Raw profile or draft value.

    Returns:
        The provider name when it is one this build can dispatch, else
        ``DEFAULT_RERANKER_PROVIDER`` -- what a blank field actually
        resolves to at run time (``RerankingConfig``'s own default), so for
        a blank value the control shows the provider that would really be
        billed. A name this build does not register (a hand-edited profile
        file, a provider from a newer build) resolves there too rather than
        raising ``InvalidSelectValueError`` out of ``compose()`` -- and in
        THAT branch the control is showing the default while the profile
        still carries the unrecognised name, which the picker cannot
        currently repair (TASK-17065 owns display + repair for it).
    """
    text = str(value).strip()
    if not text:
        return DEFAULT_RERANKER_PROVIDER
    return (
        text if text in library_rag_reranker_providers() else DEFAULT_RERANKER_PROVIDER
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
    if (
        score_threshold is None
        or not MIN_RAG_BALANCE <= score_threshold <= MAX_RAG_BALANCE
    ):
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
    if not str(values.embedding_model).strip():
        return SettingsValidationResult(
            False, "Embedding model must not be empty."
        )
    embedding_max_length = _strict_int(values.embedding_max_length)
    if embedding_max_length is None or embedding_max_length <= 0:
        return SettingsValidationResult(
            False, "Embedding max length must be positive."
        )
    if values.chunking_method not in CHUNKING_METHODS:
        return SettingsValidationResult(
            False, "Chunking method must be words, sentences, or paragraphs."
        )
    # chunk_size, chunk_overlap, distance_metric, and embedding_batch_size are
    # all already validated by RAGConfig.validate() -- routing through the
    # adapter's hard_config_errors() (rather than re-implementing the same
    # rules here) keeps this function from drifting out of sync with it.
    # Reranker top-k >= 1 (when reranking is enabled) is also folded in
    # there, since RAGConfig itself has no concept of reranking. Imported
    # locally: settings_rag_profile_adapter imports SettingsLibraryRagDefaults
    # from this module at its own top level, so a module-level import here
    # would be circular.
    from .settings_rag_profile_adapter import hard_config_errors

    errors = hard_config_errors(values)
    if errors:
        return SettingsValidationResult(False, errors[0])
    return SettingsValidationResult(True, "Library/RAG defaults are valid.")
