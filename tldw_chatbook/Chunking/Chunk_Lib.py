# Chunk_Lib.py
#########################################
# Chunking Library -- Compatibility Shim
#
# The legacy standalone implementation is DELETED (spec §6.2, Q7 ruling):
# this module is now a compatibility shim over the vendored engine at
# ``tldw_chatbook/Chunking/engine/``. Every legacy public signature keeps
# working unchanged:
#
#   * ``improved_chunking_process(...) -> List[Dict]`` with the legacy
#     ``{"text": str, "metadata": dict}`` per-chunk shape PLUS the flat
#     top-level ``start_char``/``end_char``/``word_count`` keys the DB seam
#     (``_persist_chunks``) reads (spec §6.3.2).
#   * ``Chunker`` adapter whose ``chunk_text`` returns
#     ``List[Union[str, dict]]`` -- strings for text methods, dicts for
#     json/xml/ebook (legacy §6.2 behavior).
#   * Module-level ``chunk_xml``, ``chunk_for_embedding``,
#     ``process_document_with_metadata``, ``load_document``,
#     ``ensure_nltk_data``.
#   * Legacy constants, the legacy default options dict, and exception
#     aliases onto the engine's exception hierarchy.
#
# The legacy per-method helpers (``_chunk_text_by_words`` etc.), the
# nltk lazy-import machinery, and the LLM summarization plumbing live in
# the engine now; this module only translates legacy kwargs and output
# shapes.
#########################################
import hashlib
import importlib.util
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

#
# Import 3rd party
from loguru import logger

#
# Import Local
from ..Internal_Prompts import get_internal_prompt
from .engine import Chunker as _EngineChunker
from .engine import ChunkerConfig, ChunkingMethod
from .engine.exceptions import (
    ChunkingError,
    InvalidChunkingMethodError,
    InvalidInputError,
    TokenizerError,
    TemplateError,
    LanguageNotSupportedError,
    ChunkSizeError,
    ProcessingError,
    ConfigurationError,
    CacheError,
)
from .token_chunker import (
    TokenBasedChunker,
    create_token_chunker,
)
from .language_chunkers import LanguageChunkerFactory

#
#######################################################################################################################
# Legacy exception aliases (§6.2/§9)
#
# The legacy module defined its own exception tree rooted at ``ChunkingError``.
# The engine defines the equivalent tree; the aliases below keep every legacy
# ``except``/``import`` site working against the engine classes.
#
# ``LanguageDetectionError``: legacy "language detection failed critically" is
# the engine's ``LanguageNotSupportedError``.
# ``MemoryLimitError``: legacy "input exceeds memory limits" maps to the
# engine's ``InvalidInputError`` (which is what the engine raises for
# oversized input).
LanguageDetectionError = LanguageNotSupportedError
MemoryLimitError = InvalidInputError

# ``ChunkingError`` and ``InvalidChunkingMethodError``/``InvalidInputError``
# are re-exported as-is from the engine above (same names, engine classes).

# Extra engine exception re-exports so legacy importers that pulled the whole
# legacy module's exception set keep resolving.
__all__ = [
    # Exceptions / aliases
    "ChunkingError",
    "InvalidChunkingMethodError",
    "InvalidInputError",
    "LanguageDetectionError",
    "MemoryLimitError",
    "TokenizerError",
    "TemplateError",
    "LanguageNotSupportedError",
    "ChunkSizeError",
    "ProcessingError",
    "ConfigurationError",
    "CacheError",
    # Constants
    "MAX_CHUNK_SIZE_WORDS",
    "MAX_CHUNK_SIZE_SENTENCES",
    "MAX_CHUNK_SIZE_PARAGRAPHS",
    "MAX_CHUNK_SIZE_TOKENS",
    "MAX_DOCUMENT_SIZE_MB",
    "MAX_DOCUMENT_SIZE_BYTES",
    "DEFAULT_CHUNK_OPTIONS",
    "ENGINE_VERSION",
    # Classes / functions
    "Chunker",
    "improved_chunking_process",
    "chunk_for_embedding",
    "process_document_with_metadata",
    "chunk_xml",
    "load_document",
    "ensure_nltk_data",
    "TokenBasedChunker",
    "create_token_chunker",
    "LanguageChunkerFactory",
    "sent_tokenize",
    "NLTK_AVAILABLE",
    "nltk",
    "_sent_tokenize_fallback",
    "_ensure_nltk",
    "_nltk_data_ready",
    "_nltk_tokenizer_unusable",
    "_probe_sent_tokenize",
    "_download_nltk_tokenizer_corpora",
    "LANGDETECT_AVAILABLE",
]


#######################################################################################################################
# Constants and Limits
#
# Maximum limits for chunk sizes to prevent memory issues (legacy values,
# verified by Tests/Chunking/test_chunk_lib_shim.py::test_constants_reexported
# and imported by Tests/RAG/test_config_profiles.py).
MAX_CHUNK_SIZE_WORDS = 10000  # Maximum words per chunk
MAX_CHUNK_SIZE_SENTENCES = 1000  # Maximum sentences per chunk
MAX_CHUNK_SIZE_PARAGRAPHS = 100  # Maximum paragraphs per chunk
MAX_CHUNK_SIZE_TOKENS = 10000  # Maximum tokens per chunk
MAX_DOCUMENT_SIZE_MB = 100  # Maximum document size in MB
MAX_DOCUMENT_SIZE_BYTES = MAX_DOCUMENT_SIZE_MB * 1024 * 1024  # In bytes

# Chunking engine version identity (spec §8, task-12). Stamped into every
# chunk's ``metadata["chunk_engine_version"]`` here so in-memory consumers
# see the version without a DB read; the ingestion persist seam
# (``Local_Ingestion.local_file_ingestion.persist_parsed_media``) stamps the
# same value as the TOP-LEVEL ``chunk_engine_version`` key the DB writer
# (``_persist_chunks``) persists to ``UnvectorizedMediaChunks``. The
# top-level dict stays clean of the key by design (task-11): DB stamping
# happens at persist, not at chunk time.
#
# (task-21102) The value itself lives in the stdlib-only
# ``tldw_chatbook.chunking_engine_version`` module -- outside this package --
# so the persist seam (on the app's boot-import path) can read the pin
# without executing the shim + vendored engine. Re-exported here so the
# package surface is unchanged and there is exactly one source of truth.
from ..chunking_engine_version import ENGINE_VERSION  # noqa: E402


#######################################################################################################################
# Config Settings & NLTK
#

# Probe nltk availability cheaply (find_spec, no import) so merely importing
# this shim does not pay the nltk/scipy/sklearn import cost. The engine
# handles actual sentence tokenization; this flag plus ``ensure_nltk_data()``
# below exist for legacy importers/tests that poke them.
NLTK_AVAILABLE = importlib.util.find_spec("nltk") is not None
nltk = None


def _sent_tokenize_fallback(text):
    # Legacy regex fallback kept for import-compatibility: callers that
    # monkeypatched or imported it (e.g. Tests/Utils/test_startup_polish_
    # regressions.py) keep resolving. The engine does the real splitting.
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


sent_tokenize = _sent_tokenize_fallback

_nltk_data_ready = False

# Latches once the tokeniser has been found unusable (corpus missing and not
# downloadable). Without it every call would re-probe, re-attempt a network
# download and re-log the same warning (legacy behavior, preserved because
# Tests/RAG/test_chunking_service.py monkeypatches this latch directly).
_nltk_tokenizer_unusable = False


def _probe_sent_tokenize(tokenize) -> bool:
    """Ask a candidate tokenizer whether it can actually tokenize."""
    try:
        tokenize("Probe sentence one. Probe sentence two.")
        return True
    except LookupError:
        return False
    except Exception:
        return False


def _download_nltk_tokenizer_corpora(_nltk) -> None:
    """Attempt to fetch the corpora any supported nltk version might want."""
    for resource in ("punkt", "punkt_tab"):
        try:
            _nltk.download(resource, quiet=True)
        except Exception:
            pass


def _ensure_nltk():
    """Import nltk on first use; returns the module or ``None`` (legacy API).

    Kept because tests and library code imported it from Chunk_Lib. The
    chunking itself no longer depends on it -- the engine has its own
    sentence splitting -- but ``ensure_nltk_data()`` (below) still uses it to
    answer "is the punkt corpus usable on this machine" the way the legacy
    module did.
    """
    global nltk, sent_tokenize, _nltk_tokenizer_unusable
    if nltk is not None:
        return nltk
    if not NLTK_AVAILABLE or _nltk_tokenizer_unusable:
        return None
    try:
        import nltk as _nltk
        from nltk.tokenize import sent_tokenize as _sent_tokenize
    except ImportError:
        return None

    if not _probe_sent_tokenize(_sent_tokenize):
        _download_nltk_tokenizer_corpora(_nltk)
        if not _probe_sent_tokenize(_sent_tokenize):
            _nltk_tokenizer_unusable = True
            logger.warning(
                "nltk is installed but its sentence-tokeniser data is missing "
                "and could not be downloaded. To fetch it manually, run: "
                "python -m nltk.downloader punkt punkt_tab"
            )
            return None

    sent_tokenize = _sent_tokenize
    nltk = _nltk
    return nltk


def ensure_nltk_data() -> None:
    """Ensure NLTK's sentence-tokenizer data is present, lazily and once.

    Idempotent: the first successful check flips the module-level
    ``_nltk_data_ready`` flag so repeat calls are no-ops. A no-op when NLTK
    isn't installed. The engine does not require punkt; this remains for
    legacy callers that invoked it before chunking.
    """
    global _nltk_data_ready
    if _nltk_data_ready:
        return
    if not NLTK_AVAILABLE:
        logger.debug("NLTK not available, skipping tokenizer data check")
        return
    _nltk_data_ready = _ensure_nltk() is not None


# langdetect is no longer used by the shim (the engine detects language via
# Unicode script ranges in process_text/option resolution), but the flag is
# kept for legacy importers.
try:
    import langdetect  # noqa: F401

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


# Legacy default chunk options (spec §6.1: import-time consumers exist -- e.g.
# API endpoints read this dict directly). Values mirror the legacy module's
# config-loaded defaults; the engine supplies equivalent behavior for the
# options it understands and ignores the rest gracefully.
#
# The summarize_* keys replicate the legacy config-driven entries with their
# get_cli_setting fallback values. In particular ``summarize_system_prompt``
# is SNAPSHOTTED from the prompt registry at import time -- a config change
# made after import is not observed by a freshly constructed ``Chunker()``
# (legacy frozen-at-import channel, verified by
# Tests/Internal_Prompts/test_summarization_migration.py).
DEFAULT_CHUNK_OPTIONS: Dict[str, Any] = {
    "method": "words",
    "max_size": 400,
    "overlap": 200,
    "language": None,
    "adaptive": False,
    "adaptive_chunk_sizes": None,
    "multi_level": False,
    "semantic_similarity_threshold": 0.7,
    "json_chunkable_data_key": "data",
    "tokenizer_name_or_path": "gpt2",
    "summarization_detail": 0.5,
    "summarize_min_chunk_tokens": 500,
    "summarize_chunk_delimiter": ".",
    "summarize_recursively": False,
    "summarize_verbose": False,
    "summarize_system_prompt": get_internal_prompt(
        "summarization.rolling_summarize_system"
    ),
    "summarize_additional_instructions": None,
    "summarize_temperature": 0.1,
}


#######################################################################################################################
# Legacy method mapping
#
# Legacy method names that map 1:1 onto engine ``ChunkingMethod`` members.
_LEGACY_METHOD_MAP: Dict[str, ChunkingMethod] = {
    "words": ChunkingMethod.WORDS,
    "sentences": ChunkingMethod.SENTENCES,
    "paragraphs": ChunkingMethod.PARAGRAPHS,
    "tokens": ChunkingMethod.TOKENS,
    "semantic": ChunkingMethod.SEMANTIC,
    "json": ChunkingMethod.JSON,
    "xml": ChunkingMethod.XML,
    "ebook_chapters": ChunkingMethod.EBOOK_CHAPTERS,
    "rolling_summarize": ChunkingMethod.ROLLING_SUMMARIZE,
}


def _normalize_legacy_method(method: Any) -> str:
    """Normalize a legacy method argument to the engine's string value.

    Args:
        method: Legacy method name (str) or ChunkingMethod member.

    Returns:
        Lowercase method string the engine accepts.

    Raises:
        InvalidChunkingMethodError: If the method is not supported.
    """
    if method is None:
        return ChunkingMethod.WORDS.value
    if isinstance(method, ChunkingMethod):
        return method.value
    name = str(method).strip().lower()
    if name in _LEGACY_METHOD_MAP:
        return _LEGACY_METHOD_MAP[name].value
    # Engine-native names the legacy module never had (code, code_ast,
    # structure_aware, fixed_size, propositions) pass through as plain
    # strings; the engine's resolve_process_options normalizes them and
    # raises InvalidChunkingMethodError for genuinely unknown values.
    if name in {
        "code",
        "code_ast",
        "structure_aware",
        "fixed_size",
        "propositions",
    }:
        return name
    raise InvalidChunkingMethodError(
        f"Unsupported chunking method: '{method}'"
    )


def _coerce_int_option(options: Dict[str, Any], key: str, default: int) -> int:
    """Coerce an option to int with the legacy lenient fallback.

    Legacy Chunker.__init__ logged a warning and reverted to the default
    when an option failed int() coercion; preserved here.
    """
    value = options.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid type for option '{key}': {value}. Using default {default}."
        )
        return default


def _build_engine_config(
    options: Dict[str, Any], tokenizer_name_or_path: Optional[str] = None
) -> ChunkerConfig:
    """Build an engine ChunkerConfig from legacy options.

    Args:
        options: Merged legacy options dict.
        tokenizer_name_or_path: Legacy tokenizer name (passed through to the
            token strategy via options; the config itself only needs the
            language default).

    Returns:
        A configured ChunkerConfig.
    """
    language = options.get("language") or "en"
    method = _normalize_legacy_method(options.get("method"))
    return ChunkerConfig(
        default_method=method,
        default_max_size=_coerce_int_option(options, "max_size", 400),
        default_overlap=max(0, _coerce_int_option(options, "overlap", 200)),
        language=language,
        max_text_size=MAX_DOCUMENT_SIZE_BYTES,
    )


def _translate_legacy_options(
    options: Optional[Dict[str, Any]],
    tokenizer_name_or_path: Optional[str] = "gpt2",
) -> Dict[str, Any]:
    """Translate legacy chunk options into engine process_text options.

    The engine accepts most legacy keys natively (method, max_size, overlap,
    language, adaptive, multi_level, custom_chapter_pattern,
    semantic_similarity_threshold, summarize_* ...). This drops keys the
    engine treats as configuration-only noise, coerces ints leniently, and
    forwards the tokenizer name under both accepted spellings.
    """
    merged: Dict[str, Any] = dict(DEFAULT_CHUNK_OPTIONS)
    if options:
        merged.update(options)

    # Lenient numeric coercion for the keys the legacy Chunker coerced.
    for key in (
        "max_size",
        "overlap",
        "semantic_overlap_sentences",
        "base_adaptive_chunk_size",
        "min_adaptive_chunk_size",
        "max_adaptive_chunk_size",
        "summarize_min_chunk_tokens",
    ):
        if key in merged and merged[key] is not None:
            try:
                merged[key] = int(merged[key])
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid type for option '{key}': {merged[key]}. Ignoring."
                )
                merged[key] = DEFAULT_CHUNK_OPTIONS.get(key)
    if merged.get("semantic_similarity_threshold") is not None:
        try:
            merged["semantic_similarity_threshold"] = float(
                merged["semantic_similarity_threshold"]
            )
        except (ValueError, TypeError):
            merged.pop("semantic_similarity_threshold", None)

    # Keys the engine either does not understand or handles differently.
    merged.pop("adaptive_chunk_sizes", None)  # legacy dead option (never read)
    merged.pop("template", None)  # template chunking stays on the template pipeline
    if tokenizer_name_or_path:
        merged["tokenizer_name_or_path"] = tokenizer_name_or_path
    # engine accepts either spelling; provide both so per-call strategy
    # selection triggers for tokens.
    if "tokenizer_name_or_path" in merged:
        merged.setdefault("tokenizer_name", merged["tokenizer_name_or_path"])

    # 'language' of None/'' lets the engine auto-detect (legacy behavior:
    # detect_language when unset).
    if not merged.get("language"):
        merged.pop("language", None)

    return merged


def _install_tokenizer_resolve_alias() -> None:
    """Provide the ``_resolve_tokenizer`` seam on the engine's token strategy.

    The Task-3 contract test (and the Q2 enforcement below) address the
    engine's tokenizer resolution through
    ``TokenChunkingStrategy._resolve_tokenizer``. The vendored engine
    (upstream commit 385afa95) resolves the tokenizer inline in its
    ``tokenizer`` property and defines no such attribute -- verified against
    upstream, this is a brief/engine shape mismatch, not a local edit.

    This attaches a thin alias onto the class (only when absent, never
    clobbering anything the engine defines): unpatched it simply delegates to
    the engine's own ``tokenizer`` property, so engine behavior is
    unchanged; the Q2 check calls it, which lets a monkeypatched resolution
    (as in the contract test) be honored. No file under ``engine/`` is
    modified -- this is a runtime attribute added by the compat shim.
    """
    try:
        from .engine.strategies.tokens import TokenChunkingStrategy
    except Exception:  # pragma: no cover - engine import failure
        return
    if getattr(TokenChunkingStrategy, "_resolve_tokenizer", None) is not None:
        return

    def _resolve_tokenizer(self):
        return self.tokenizer

    _resolve_tokenizer.__doc__ = (
        "Compat alias (Chunk_Lib shim): resolve this strategy's tokenizer."
    )
    TokenChunkingStrategy._resolve_tokenizer = _resolve_tokenizer


def _is_engine_fallback_tokenizer(resolved: Any) -> bool:
    """Whether a resolved tokenizer is the engine's word-approximation fallback."""
    if resolved is None:
        return False
    try:
        from .engine.strategies.tokens import FallbackTokenizer
    except Exception:  # pragma: no cover - engine import failure
        return False
    return isinstance(resolved, FallbackTokenizer) or (
        type(resolved).__name__ == "FallbackTokenizer"
    )


_TOKENS_FALLBACK_MESSAGE = (
    "The 'tokens' chunking method resolved to a word-approximation fallback "
    "tokenizer instead of a real tokenizer. Token counts would be "
    "approximated by word count. Install tiktoken for accurate tokenization: "
    "pip install tiktoken"
)


def _probe_engine_token_resolution(tokenizer_name: str) -> Any:
    """Resolve the tokenizer the ENGINE would use, on a fresh strategy.

    Builds a fresh ``TokenChunkingStrategy`` for ``tokenizer_name`` and
    resolves its tokenizer -- going through the ``_resolve_tokenizer`` seam
    when present (so a monkeypatched resolution, as in the Q2 contract
    test, is honored), else the engine's own ``tokenizer`` property. A
    FRESH instance is essential: the engine's ``chunk_text`` builds a
    per-call ephemeral strategy whenever a tokenizer override is set
    (engine/chunker.py ``use_per_call_strategy``), so inspecting a cached
    ``get_strategy`` instance would read a resolution the actual call never
    used. The class-level ``_failed_tokenizers`` failure cache is shared by
    every instance, so the fresh probe sees a poisoned resolution too.

    tiktoken resolution is network-free; transformers resolution uses the
    local HF cache when the hub is unreachable.

    Args:
        tokenizer_name: Tokenizer name/path the engine strategy should use.

    Returns:
        The resolved tokenizer object (or the engine's FallbackTokenizer),
        or ``None`` when resolution itself failed.
    """
    try:
        from .engine.strategies.tokens import TokenChunkingStrategy
    except Exception:  # pragma: no cover - engine import failure
        return None
    try:
        strategy = TokenChunkingStrategy(
            language="en", tokenizer_name=str(tokenizer_name or "gpt2")
        )
    except Exception:
        return None
    resolve = getattr(type(strategy), "_resolve_tokenizer", None)
    try:
        if callable(resolve):
            return resolve(strategy)
        return strategy.tokenizer
    except Exception:
        return None


def _enforce_real_tokenizer(tokenizer_name: str) -> None:
    """Q2: raise when the engine would silently word-approximate tokens.

    Probes the engine's own tokenizer resolution (NOT the legacy
    TokenBasedChunker seam, which is transformers-only and would miss a
    tiktoken-only install) and raises if it lands on the engine's
    word-approximation ``FallbackTokenizer``. Called BEFORE the engine
    chunks, so no approximate chunks are ever produced.

    Args:
        tokenizer_name: Tokenizer name/path the engine strategy should use.

    Raises:
        ChunkingError: If the engine's resolution is the fallback tokenizer
            (message tells the user to install tiktoken).
    """
    resolved = _probe_engine_token_resolution(tokenizer_name)
    if _is_engine_fallback_tokenizer(resolved):
        raise ChunkingError(_TOKENS_FALLBACK_MESSAGE)


def _guard_tokens_overlap(max_size: Any, overlap: Any) -> None:
    """Legacy parity guard: reject overlap >= max_size for the tokens method.

    The legacy ``TokenBasedChunker.chunk_by_tokens`` raised
    ``ValueError("Token overlap X must be less than max_tokens Y")`` for
    ``overlap >= max_tokens``, and ``improved_chunking_process`` wrapped it
    into ``ChunkingError``. The engine instead clamps the overlap and chunks
    anyway; this guard keeps the legacy contract so a mis-configured call
    (e.g. stock default overlap 200 with a small max_size) fails loudly
    instead of producing degenerate chunks.

    Args:
        max_size: Resolved max_size option (tokens per chunk).
        overlap: Resolved overlap option (tokens).

    Raises:
        ChunkingError: If both values are ints, max_size > 0, and
            overlap >= max_size.
    """
    if not isinstance(max_size, int) or not isinstance(overlap, int):
        return
    if max_size > 0 and overlap >= max_size:
        raise ChunkingError(
            f"Token overlap {overlap} must be less than max_tokens {max_size}"
        )


def _synthesize_flat_offsets(
    text: str, chunk_texts: List[str]
) -> List[Dict[str, int]]:
    """Compute (start_char, end_char) spans for chunks lacking offsets.

    Primary strategy: word-position mapping. Text-method chunks (words,
    sentences, ...) re-join the source's whitespace-separated words, so each
    chunk's ``split()`` words are a CONSECUTIVE run of the source's words
    (matching the approach in RAG_Search/chunking_service.py). Mapping chunk
    words onto source word spans yields correct spans even when chunks
    overlap (the word cursor advances by one word, not past the chunk end).
    Source whitespace is normalized inside the chunk, so a plain
    ``text.find`` would miss -- the previous approach -- and with overlap
    could even report ``end_char > len(text)``.

    Fallbacks: monotonic ``find`` (for chunks that ARE substrings), then a
    conservative estimate. All results are clamped to ``[0, len(text)]``.

    Args:
        text: The source text the chunks were produced from.
        chunk_texts: The chunk strings, in order.

    Returns:
        One ``{"start_char": int, "end_char": int}`` dict per chunk.
    """
    n = len(text)
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    words = [text[s:e] for s, e in word_spans]
    spans: List[Dict[str, int]] = []
    word_cursor = 0
    find_cursor = 0

    for chunk_text in chunk_texts:
        start_char: Optional[int] = None
        end_char: Optional[int] = None

        chunk_words = chunk_text.split() if chunk_text else []
        if chunk_words and words:
            # Consecutive word-run match starting at/after word_cursor.
            run_len = len(chunk_words)
            for wi in range(word_cursor, len(words) - run_len + 1):
                if words[wi : wi + run_len] == chunk_words:
                    start_char = word_spans[wi][0]
                    end_char = word_spans[wi + run_len - 1][1]
                    word_cursor = wi + 1  # advance one word: tolerate overlap
                    break

        if start_char is None:
            # Substring match (chunks that preserve original whitespace).
            idx = text.find(chunk_text, find_cursor)
            if idx == -1:
                idx = text.find(chunk_text)
            if idx != -1:
                start_char = idx
                end_char = min(idx + len(chunk_text), n)
                find_cursor = max(find_cursor, idx + 1)
                word_cursor = 0  # re-anchor the word cursor to keep it usable

        if start_char is None or end_char is None:
            # Conservative estimate: continue from the last known position.
            start_char = min(find_cursor, n)
            end_char = min(start_char + len(chunk_text), n)

        start_char = max(0, min(start_char, n))
        end_char = max(start_char, min(end_char, n))
        spans.append({"start_char": start_char, "end_char": end_char})

    return spans


#######################################################################################################################
# Chunker adapter
#
def _wrap_payload_dict_llm_for_positional_engine(
    llm_call_function: Callable[[Dict[str, Any]], Any],
    llm_api_config: Optional[Dict[str, Any]] = None,
) -> Callable[..., Any]:
    """Wrap a payload-dict LLM callback into the engine's positional contract.

    The engine's LLM-calling strategies (propositions, rolling_summarize)
    invoke their ``llm_call_func`` positionally, analyze-style:
    ``llm_call_func(api_name, prompt, None, api_key, system_message, temp,
    False, False, False, model_override=..., **snapshot_kwargs)``. Chatbook
    callers instead supply the legacy payload-dict callback (one dict
    argument) established by the rolling_summarize port -- the same key set
    this wrapper emits (see ``Chunker._rolling_summarize``'s
    ``payload_for_llm_call``): api_name, input_data, custom_prompt_arg,
    api_key, system_message, temp, streaming, model, max_tokens.

    Mapping notes (mirroring the rolling_summarize payload precedent):
    - positional ``prompt`` -> ``input_data``; positional arg 3
      (``custom_prompt_arg``, always None from the engine) -> "".
    - the ``model_override`` keyword -> the payload's ``model`` slot.
    - ``max_tokens`` is not part of the positional contract; it rides in
      from ``llm_api_config`` exactly as rolling_summarize fills it.
    - the server-only snapshot kwargs (``app_config``,
      ``credentials_resolved``, ``provider_credentials``) are accepted and
      dropped: they only arrive if a caller put them in the LLM config, and
      the payload-dict contract has no slots for them (their absence is
      benign upstream -- guarded reads).
    - trailing positional flags beyond the nine the propositions engine
      passes (e.g. rolling_summarize's ``chunk_options`` None) are accepted
      for signature tolerance and dropped.

    Args:
        llm_call_function: Blocking callable receiving one payload dict and
            returning the provider's string (or tuple whose first element
            is the string).
        llm_api_config: Caller LLM config (only ``max_tokens`` is read
            here; every other key flows to the engine's own
            ``llm_config.get(...)`` reads upstream of this wrapper).

    Returns:
        A positional-callable that forwards to ``llm_call_function``.
    """

    def _positional_llm_call(
        api_name: str,
        input_data: str,
        custom_prompt_arg: Optional[str],
        api_key: Optional[str],
        system_message: Optional[str],
        temp: float,
        streaming: bool = False,
        recursive_summarization: bool = False,  # tolerated, dropped
        chunked_summarization: bool = False,  # tolerated, dropped
        *_extra_positional: Any,
        model_override: Optional[str] = None,
        **_snapshot_kwargs: Any,  # server-only config; dropped (see docstring)
    ) -> Any:
        payload: Dict[str, Any] = {
            "api_name": api_name,
            "input_data": input_data,
            "custom_prompt_arg": custom_prompt_arg or "",
            "api_key": api_key,
            "system_message": system_message,
            "temp": temp,
            "streaming": bool(streaming),
            "model": model_override,
            "max_tokens": (llm_api_config or {}).get("max_tokens"),
        }
        return llm_call_function(payload)

    return _positional_llm_call


class Chunker:
    """Legacy-signature adapter over the engine ``Chunker``.

    Preserves the legacy constructor (``options`` dict, tokenizer name,
    ``template``/``template_manager`` parameters) and the legacy
    ``chunk_text`` return contract: ``List[Union[str, dict]]`` -- plain
    strings for text methods, dicts for json/xml/ebook (the legacy methods
    that returned dicts).

    Template semantics since the file-store deletion (spec §8.2):
    ``template`` accepts a pre-resolved template dict (chunk-stage options
    merged under explicit options); ``template_manager`` is
    accepted-and-ignored. Name resolution lives in
    ``template_runtime.resolve_template``.
    """

    def __init__(
        self,
        options: Optional[Dict[str, Any]] = None,
        tokenizer_name_or_path: str = "gpt2",
        template: Optional[Any] = None,
        template_manager: Optional[Any] = None,
    ):
        """Initializes the Chunker adapter.

        Args:
            options (Optional[Dict[str, Any]]): Custom chunking options to
                override defaults.
            tokenizer_name_or_path (str): Name or path of the tokenizer to
                use. Defaults to "gpt2".
            template (Optional[Dict[str, Any]]): Pre-resolved template dict
                (the flat spec §4.1 shape, e.g. what
                ``template_runtime.resolve_template`` returns). Only the
                ``chunking`` stage's method/config are applied here (defaults
                <- template <- explicit ``options``); executing the full
                pre/chunk/post pipeline is ``template_runtime.apply_template``'s
                contract. A bare name string raises ``TemplateError``: name
                resolution requires a Media DB handle and lives in
                ``template_runtime.resolve_template`` (spec §8.2), not in this
                import-light shim.
            template_manager (Optional[Any]): Accepted and ignored (spec
                §8.2). Retained solely for signature compatibility with
                legacy callers; the file-store manager it named is deleted.

        Raises:
            TemplateError: If ``template`` is a bare name string (use
                ``template_runtime.resolve_template`` first), or not a valid
                flat template dict.
        """
        # template_manager= is accepted-and-ignored: stored for attribute
        # compatibility, never consulted.
        self.template_manager = template_manager
        if template is not None:
            if isinstance(template, str):
                raise TemplateError(
                    f"Chunker no longer resolves template names (the file "
                    f"template store is deleted, spec §8.2): resolve "
                    f"{template!r} first via "
                    f"tldw_chatbook.Chunking.template_runtime.resolve_template "
                    f"and pass the returned dict as template="
                )
            # Local import: template_runtime imports this module at its own
            # module level, so the dependency may only be taken at call time.
            from .template_runtime import template_from_record

            mapped = template_from_record(template)
            # Legacy precedence preserved: defaults <- template <- explicit.
            template_options: Dict[str, Any] = {
                "method": mapped.base_method,
                **mapped.default_options,
            }
            if options:
                template_options.update(options)
            options = template_options

        # Resolve effective options exactly like legacy: defaults <- template <- explicit.
        self.options: Dict[str, Any] = dict(DEFAULT_CHUNK_OPTIONS)
        if options:
            self.options.update(options)

        self._tokenizer_path: str = str(
            self.options.get("tokenizer_name_or_path", tokenizer_name_or_path)
        )

        self._engine = _EngineChunker(_build_engine_config(self.options, self._tokenizer_path))
        # TokenBasedChunker retained for legacy attribute compatibility.
        self._token_chunker: Optional[TokenBasedChunker] = None
        logger.debug(f"Chunker initialized with options: {self.options}")

    # ------------------------------------------------------------------
    # Legacy attribute surface
    # ------------------------------------------------------------------
    @property
    def token_chunker(self) -> TokenBasedChunker:
        """Get the token-based chunker, creating it if needed (legacy API)."""
        if self._token_chunker is None:
            self._token_chunker = create_token_chunker(self._tokenizer_path)
        return self._token_chunker

    @property
    def tokenizer(self):
        """Get the underlying tokenizer for backward compatibility."""
        return self.token_chunker.tokenizer

    @property
    def engine(self) -> _EngineChunker:
        """The wrapped engine Chunker instance."""
        return self._engine

    def _get_option(self, key: str, default_override: Optional[Any] = None) -> Any:
        """Helper to get an option, allowing for a dynamic default."""
        value = self.options.get(key)
        if value is not None:
            return value
        return default_override

    def detect_language(self, text: str) -> str:
        """Detects the language of the given text.

        Uses the engine's script-range detection (no langdetect dependency).

        Args:
            text (str): The text to detect language from.

        Returns:
            str: The detected language code; defaults to 'en'.
        """
        if not text or not text.strip():
            return self._get_option("language") or "en"
        try:
            if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
                return "ja"
            if re.search(r"[\u4e00-\u9fff]", text):
                return "zh"
            if re.search(r"[\u0e00-\u0e7f]", text):
                return "th"
            if re.search(r"[\u0900-\u097f]", text):
                return "hi"
            if re.search(r"[\u0400-\u04ff]", text):
                return "ru"
            if re.search(r"[\uac00-\ud7af]", text):
                return "ko"
            if re.search(r"[\u0600-\u06ff]", text):
                return "ar"
        except Exception:  # pragma: no cover - regex on str cannot fail
            pass
        return self._get_option("language") or "en"

    def _ensure_language(self, text: str, language_option: Optional[str] = None) -> str:
        """Ensures a language is determined, using option, detection, or default."""
        if language_option:
            return language_option
        instance_lang_opt = self._get_option("language")
        if instance_lang_opt:
            return instance_lang_opt
        return self.detect_language(text)

    # ------------------------------------------------------------------
    # Main chunking entry point (legacy signature)
    # ------------------------------------------------------------------
    def chunk_text(
        self,
        text: str,
        method: Optional[str] = None,
        llm_call_function: Optional[
            Callable[[Dict[str, Any]], Any]
        ] = None,
        llm_api_config: Optional[Dict[str, Any]] = None,
        use_template: Optional[bool] = None,
    ) -> List[Union[str, Dict[str, Any]]]:
        """Main method to chunk text based on the specified method.

        Args:
            text (str): The text to chunk.
            method (Optional[str]): Override the chunking method defined in
                options.
            llm_call_function: Optional LLM call function (rolling_summarize).
            llm_api_config: Optional LLM API config (rolling_summarize).
            use_template (Optional[bool]): Accepted and ignored (spec §8.2).
                The template pipeline this routed to is deleted; a template
                dict supplied at construction has already had its
                chunk-stage options merged into ``self.options``. Full
                template execution lives in
                ``template_runtime.apply_template``.

        Returns:
            List[Union[str, Dict[str, Any]]]: A list of chunks. Strings for
            most methods, dicts for json/xml/ebook methods.

        Raises:
            InvalidChunkingMethodError: If the method is not supported.
            ChunkingError: For errors during the chunking process.
            MemoryLimitError: If the input text exceeds memory limits.
        """
        # Check document size before processing (legacy MemoryLimitError).
        try:
            text_size_bytes = len(text.encode("utf-8"))
        except AttributeError:
            raise InvalidInputError(
                f"Expected string input, got {type(text).__name__}"
            )
        if text_size_bytes > MAX_DOCUMENT_SIZE_BYTES:
            text_size_mb = text_size_bytes / (1024 * 1024)
            raise MemoryLimitError(
                f"Document size {text_size_mb:.2f} MB exceeds maximum allowed "
                f"size of {MAX_DOCUMENT_SIZE_MB} MB"
            )

        resolved_method = _normalize_legacy_method(
            method if method else self._get_option("method", "words")
        )

        engine_options = _translate_legacy_options(self.options, self._tokenizer_path)
        if method:
            engine_options["method"] = resolved_method

        if resolved_method == ChunkingMethod.TOKENS.value:
            # Legacy parity guard FIRST (legacy validated params before any
            # tokenizer work): overlap >= max_size raised in the legacy
            # token path; the engine would clamp it instead.
            _guard_tokens_overlap(
                engine_options.get("max_size"), engine_options.get("overlap")
            )
            # Q2: refuse BEFORE chunking when the engine's own tokenizer
            # resolution lands on the word-approximation fallback (probing
            # the engine -- tiktoken counts, transformers cache -- not the
            # legacy transformers-only seam, so a tiktoken-only install
            # still delegates to the engine).
            _enforce_real_tokenizer(self._tokenizer_path)

        if resolved_method == ChunkingMethod.ROLLING_SUMMARIZE.value:
            # Legacy rolling_summarize invoked the caller's LLM function with
            # a single payload dict ({"api_name", "input_data",
            # "system_message", ...}) and resolved summarize_* options through
            # _get_option with a lazy get_internal_prompt default. The engine's
            # strategy instead calls analyze(...) positionally -- a different
            # caller contract. Route through the ported legacy implementation
            # so every existing llm_call_function caller keeps working.
            if not llm_call_function:
                raise ChunkingError(
                    "Missing 'llm_call_function' for 'rolling_summarize' method."
                )
            system_prompt_content = self.options.get("summarize_system_prompt")
            if system_prompt_content is None:
                system_prompt_content = get_internal_prompt(
                    "summarization.rolling_summarize_system"
                )
            summary = self._rolling_summarize(
                text_to_summarize=text,
                llm_summarize_step_func=llm_call_function,
                llm_api_config=llm_api_config or {},
                detail=self._get_option("summarization_detail", 0.5),
                min_chunk_tokens=self._get_option("summarize_min_chunk_tokens", 500),
                chunk_delimiter=self._get_option("summarize_chunk_delimiter", "."),
                recursive_summarization=self._get_option(
                    "summarize_recursively", False
                ),
                verbose=self._get_option("summarize_verbose", False),
                system_prompt_content=system_prompt_content,
                additional_instructions=self._get_option(
                    "summarize_additional_instructions", None
                ),
            )
            return [summary]

        if resolved_method in {"json", "xml", "ebook_chapters"}:
            # Legacy returned dicts ({"text","metadata"}) for these methods;
            # the engine's process_text provides exactly that shape. (The
            # set is inlined -- rather than reusing _DICT_METHODS -- so the
            # dispatch names every handled method explicitly.)
            results = self._engine.process_text(
                text,
                engine_options,
                tokenizer_name_or_path=self._tokenizer_path,
                llm_call_func=llm_call_function,
                llm_config=llm_api_config,
            )
            return [
                {"text": item["text"], "metadata": item["metadata"]}
                for item in results
            ]

        if resolved_method == "propositions" and llm_call_function is not None:
            # LLM-contract adapter (propositions spec §5.1): the engine's
            # strategy calls its llm_call_func positionally (analyze-style)
            # while chatbook callers supply the payload-dict callback. The
            # engine's chunk_text has no llm_call_func parameter -- it reads
            # the per-instance llm_call_func/llm_config hooks -- so install
            # the wrapped callable there for the duration of this call and
            # restore afterwards (the hooks are instance state; a later
            # call with a different callback must not see this one). No
            # callback -> nothing installed: the engine's default heuristic
            # engine stands (and its engine="llm" leg degrades to
            # heuristics on its own -- upstream parity, not fail-close).
            previous_func = self._engine.llm_call_func
            previous_cfg = self._engine.llm_config
            self._engine.llm_call_func = _wrap_payload_dict_llm_for_positional_engine(
                llm_call_function, llm_api_config
            )
            self._engine.llm_config = llm_api_config or {}
            try:
                raw = self._engine.chunk_text(
                    text,
                    method=resolved_method,
                    max_size=engine_options.get("max_size"),
                    overlap=engine_options.get("overlap"),
                    language=engine_options.get("language"),
                    **{
                        k: v
                        for k, v in engine_options.items()
                        if k
                        not in {
                            "method",
                            "max_size",
                            "overlap",
                            "language",
                        }
                    },
                )
            finally:
                self._engine.llm_call_func = previous_func
                self._engine.llm_config = previous_cfg
            return [chunk for chunk in raw if isinstance(chunk, str)]

        raw = self._engine.chunk_text(
            text,
            method=resolved_method,
            max_size=engine_options.get("max_size"),
            overlap=engine_options.get("overlap"),
            language=engine_options.get("language"),
            **{
                k: v
                for k, v in engine_options.items()
                if k
                not in {
                    "method",
                    "max_size",
                    "overlap",
                    "language",
                }
            },
        )
        return [chunk for chunk in raw if isinstance(chunk, str)]

    # ------------------------------------------------------------------
    # Rolling summarization (ported verbatim from the legacy module; the
    # engine's strategy uses a different LLM-call signature, see chunk_text).
    # ------------------------------------------------------------------
    def _rolling_summarize(
        self,
        text_to_summarize: str,
        llm_summarize_step_func: Callable,
        llm_api_config: Dict[str, Any],
        detail: float,
        min_chunk_tokens: int,
        chunk_delimiter: str,
        recursive_summarization: bool,
        verbose: bool,
        system_prompt_content: str,
        additional_instructions: Optional[str],
    ) -> str:
        """Summarize text by rolling over delimiter-based chunks (legacy).

        Args:
            text_to_summarize: The text to summarize.
            llm_summarize_step_func: Blocking callable receiving ONE payload
                dict ({"api_name", "input_data", "custom_prompt_arg",
                "api_key", "system_message", "temp", "streaming", "model",
                "max_tokens"}) and returning a summary string.
            llm_api_config: Caller LLM config ({'api_name', 'model',
                'api_key', 'temperature', ...}).
            detail: 0..1 granularity controlling the number of summary steps.
            min_chunk_tokens: Minimum tokens per LLM input chunk.
            chunk_delimiter: Delimiter to split the input on (default ".").
            recursive_summarization: Feed the previous summary as context.
            verbose: Log progress and use a progress bar.
            system_prompt_content: Base system prompt for each LLM call.
            additional_instructions: Optional extra instructions appended to
                the system prompt.

        Returns:
            str: The final summary (parts joined by "\\n\\n---\\n\\n").

        Raises:
            ChunkingError: Fail-closed (spec §8.3) -- if any per-part LLM
                call raises, returns an ``"Error: ..."`` string, or returns
                a non-string, the whole summarization aborts with a message
                naming the failed part (no marker text is persisted).
        """
        logger.info(f"Rolling summarization called. Detail: {detail}")
        text_token_length = self.token_chunker.count_tokens(text_to_summarize)
        max_summarization_chunks = max(1, text_token_length // min_chunk_tokens)
        min_summarization_chunks = 1
        num_summarization_chunks = int(
            min_summarization_chunks
            + detail * (max_summarization_chunks - min_summarization_chunks)
        )
        num_summarization_chunks = max(1, num_summarization_chunks)
        llm_input_chunk_size_tokens = max(
            min_chunk_tokens, text_token_length // num_summarization_chunks
        )

        text_chunks_for_llm, _, dropped_count = self._chunk_on_delimiter_for_llm(
            text_to_summarize, llm_input_chunk_size_tokens, delimiter=chunk_delimiter
        )
        if dropped_count > 0 and verbose:
            logger.warning(
                f"{dropped_count} parts were dropped during text splitting for summarization."
            )
        if verbose:
            logger.info(
                f"Splitting text for summarization into {len(text_chunks_for_llm)} parts."
            )

        final_system_prompt = system_prompt_content
        if additional_instructions:
            final_system_prompt += f"\n\n{additional_instructions}"

        try:
            from tqdm import tqdm  # Import here
        except ImportError:
            logger.warning(
                "tqdm library not found. Progress bar for summarization parts will be disabled. Install with 'pip install tqdm'."
            )

            # Define a dummy tqdm if not found, so the loop doesn't break
            def tqdm(iterable, *args, **kwargs):  # noqa: F811
                return iterable

        accumulated_summaries = []
        for i, chunk_for_llm in enumerate(
            tqdm(text_chunks_for_llm, desc="Summarizing parts", disable=not verbose)
        ):
            user_message_content = chunk_for_llm
            if recursive_summarization and accumulated_summaries:
                user_message_content = f"Previous summary context:\n{accumulated_summaries[-1]}\n\nNew content to summarize and integrate:\n{chunk_for_llm}"

            # Prepare payload for the llm_summarize_step_func
            # This payload structure should match what your `Summarization_General_Lib.analyze` or `_dispatch_to_api` expects
            payload_for_llm_call = {
                "api_name": llm_api_config.get(
                    "api_name", "openai"
                ),  # Default or from config
                "input_data": user_message_content,  # This is the text for the LLM
                "custom_prompt_arg": "",  # Rolling summarize manages the full prompt content
                "api_key": llm_api_config.get("api_key"),
                "system_message": final_system_prompt,
                "temp": llm_api_config.get(
                    "temperature", self._get_option("summarize_temperature", 0.1)
                ),
                "streaming": False,  # Internal steps of rolling summary should not stream to Chunker
                "model": llm_api_config.get("model"),
                "max_tokens": llm_api_config.get("max_tokens"),
            }
            try:
                # `llm_summarize_step_func` should be blocking and return a string
                summary_content = llm_summarize_step_func(payload_for_llm_call)

                if isinstance(summary_content, str) and summary_content.startswith(
                    "Error:"
                ):
                    # Fail closed (spec §8.3): persisting an error marker as
                    # document text is silent data corruption.
                    logger.error(
                        f"LLM call for summarization part {i + 1} failed: {summary_content}"
                    )
                    raise ChunkingError(
                        f"Rolling-summarize LLM call failed for part {i + 1}: "
                        f"{summary_content}"
                    )
                elif isinstance(summary_content, str):
                    accumulated_summaries.append(summary_content)
                else:  # Should not happen if llm_summarize_step_func is well-behaved
                    logger.error(
                        f"LLM call for summarization part {i + 1} returned non-string: {type(summary_content)}"
                    )
                    raise ChunkingError(
                        f"Rolling-summarize LLM call failed for part {i + 1}: "
                        f"provider returned unexpected type "
                        f"{type(summary_content).__name__}"
                    )

            except ChunkingError:
                # The fail-closed raises above (spec §8.3) pass through
                # unwrapped -- the broad handler below must not re-wrap them.
                raise
            except Exception as e_llm:
                logger.opt(exception=True).error(
                    f"Exception calling llm_summarize_step_func for part {i + 1}: {e_llm}"
                )
                raise ChunkingError(
                    f"Rolling-summarize LLM call failed for part {i + 1}: {e_llm}"
                ) from e_llm

        final_summary = "\n\n---\n\n".join(
            accumulated_summaries
        )  # Join with a clear separator
        return final_summary.strip()

    # Helper for rolling_summarize (was combine_chunks_with_no_minimum)
    def _combine_chunks_for_llm(
        self,
        chunks: List[str],
        max_tokens: int,  # Max tokens for the combined output for the LLM
        chunk_delimiter: str = "\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow: bool = True,
    ) -> Tuple[List[str], List[List[int]], int]:
        """Combine small chunks into blocks under max_tokens (legacy helper)."""
        dropped_chunk_count = 0
        output_combined_texts = []
        output_original_indices = []  # To track which original chunks went into which combined text

        current_candidate_text_parts = [header] if header else []
        current_candidate_indices = []

        for chunk_idx, chunk_content in enumerate(chunks):
            # Tentatively add the new chunk (with header if it's the first part of a new candidate)
            parts_to_test = current_candidate_text_parts + (
                [chunk_content]
                if current_candidate_text_parts or not header
                else [header, chunk_content]
            )

            test_text = chunk_delimiter.join(parts_to_test)
            token_count = self.token_chunker.count_tokens(test_text)

            if token_count > max_tokens:
                # Current candidate (before adding new chunk) was likely the max fit
                if current_candidate_text_parts and (
                    not header
                    or len(current_candidate_text_parts) > 1
                    or current_candidate_text_parts[0] != header
                ):  # Check if it's more than just a header
                    output_combined_texts.append(
                        chunk_delimiter.join(current_candidate_text_parts)
                    )
                    output_original_indices.append(current_candidate_indices)

                    # Start new candidate with the current chunk_content that caused overflow
                    current_candidate_text_parts = (
                        [header, chunk_content] if header else [chunk_content]
                    )
                    current_candidate_indices = [chunk_idx]

                    # If this new chunk *itself* is too large (even with header)
                    current_candidate_only_text = chunk_delimiter.join(
                        current_candidate_text_parts
                    )
                    if (
                        self.token_chunker.count_tokens(current_candidate_only_text)
                        > max_tokens
                    ):
                        logger.warning(
                            f"Single chunk (index {chunk_idx}, content: '{chunk_content[:50]}...') itself exceeds max_tokens ({max_tokens}) even after starting new. It will be dropped."
                        )
                        dropped_chunk_count += 1
                        current_candidate_text_parts = (
                            [header] if header else []
                        )  # Reset for next
                        current_candidate_indices = []
                else:  # current_candidate_text_parts was empty or just header, and new chunk overflows
                    logger.warning(
                        f"Single chunk (index {chunk_idx}, content: '{chunk_content[:50]}...') itself exceeds max_tokens ({max_tokens}). It will be dropped."
                    )
                    dropped_chunk_count += 1
                    # current_candidate_text_parts remains [header] or []
                    # current_candidate_indices remains []
            else:
                # It fits, so add current chunk_content to candidate
                if not current_candidate_text_parts:  # Starting fresh
                    if header:
                        current_candidate_text_parts = [header, chunk_content]
                    else:
                        current_candidate_text_parts = [chunk_content]
                else:  # Appending to existing candidate
                    current_candidate_text_parts.append(chunk_content)
                current_candidate_indices.append(chunk_idx)

        # Add the last candidate if it has content (more than just a header)
        if current_candidate_text_parts and (
            not header
            or len(current_candidate_text_parts) > 1
            or (header and current_candidate_text_parts[0] != header)
            or not header
        ):
            output_combined_texts.append(
                chunk_delimiter.join(current_candidate_text_parts)
            )
            output_original_indices.append(current_candidate_indices)

        return output_combined_texts, output_original_indices, dropped_chunk_count

    # Helper for rolling_summarize (was chunk_on_delimiter)
    def _chunk_on_delimiter_for_llm(
        self,
        input_string: str,
        max_tokens_for_llm_input: int,  # Max tokens for each final combined chunk for LLM
        delimiter: str,
    ) -> Tuple[List[str], List[List[int]], int]:
        """Split input on delimiter, then recombine under the token cap (legacy)."""
        initial_parts = input_string.split(delimiter)

        # Re-add the delimiter for context, but only *between* parts, not at
        # the very end of the LLM input block.
        reconstructed_parts = []
        for i, part_text in enumerate(initial_parts):
            if i < len(initial_parts) - 1:  # Not the last part
                reconstructed_parts.append(part_text + delimiter)
            else:  # Last part, don't append delimiter
                if part_text:  # Add if not empty
                    reconstructed_parts.append(part_text)

        # Filter out any empty strings that might result if there were multiple delimiters together
        reconstructed_parts = [p for p in reconstructed_parts if p]

        if not reconstructed_parts:
            return [], [], 0

        combined_texts_for_llm, original_indices, dropped_count = (
            self._combine_chunks_for_llm(
                chunks=reconstructed_parts,
                max_tokens=max_tokens_for_llm_input,
                chunk_delimiter="",  # Join these parts directly
                add_ellipsis_for_overflow=True,
            )
        )

        return combined_texts_for_llm, original_indices, dropped_count

    # ------------------------------------------------------------------
    # Convenience: same as legacy, produce the metadata-rich shape directly.
    # ------------------------------------------------------------------
    def process(self, text: str, method: Optional[str] = None) -> List[Dict[str, Any]]:
        """Chunk and return the metadata-rich flat shape (helper).

        Args:
            text (str): The text to chunk.
            method (Optional[str]): Method override.

        Returns:
            List[Dict[str, Any]]: Chunks in the improved_chunking_process
            shape (flat contract).
        """
        return improved_chunking_process(
            text,
            chunk_options_dict={**self.options, **({"method": method} if method else {})},
            tokenizer_name_or_path=self._tokenizer_path,
        )


#######################################################################################################################
# Module-level functions (legacy signatures)
#

# Attach the tokenizer-resolution seam before any chunking runs (see
# _install_tokenizer_resolve_alias for why it must exist).
_install_tokenizer_resolve_alias()


def chunk_xml(
    xml_text: str,
    options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Chunk XML content into structured chunks (§7.1: name restored).

    Wraps the engine's XML strategy via ``process_text`` so the result keeps
    the legacy ``{"text": str, "metadata": dict}`` chunk shape with legacy
    metadata enrichment (chunk_index, total_chunks, chunk_content_hash ...).

    Args:
        xml_text (str): The XML string to chunk.
        options (Optional[Dict[str, Any]]): Chunk options (max_size is in
            words of combined element content; overlap is in elements).
        **kwargs: Additional legacy passthroughs (tokenizer_name_or_path, ...).

    Returns:
        List[Dict[str, Any]]: Chunks with text and metadata.
    """
    opts: Dict[str, Any] = dict(options or {})
    opts["method"] = ChunkingMethod.XML.value
    return improved_chunking_process(xml_text, chunk_options_dict=opts, **kwargs)


def chunk_for_embedding(
    text: str,
    file_name: str,
    custom_chunk_options: Optional[Dict[str, Any]] = None,
    tokenizer_name_or_path: str = "gpt2",
    llm_call_function: Optional[Callable] = None,
    llm_api_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Prepares chunks specifically for embedding, adding headers with context.

    Uses improved_chunking_process internally.

    Args:
        text (str): Document text.
        file_name (str): Source file name for the chunk headers.
        custom_chunk_options (Optional[Dict[str, Any]]): Chunking options.
        tokenizer_name_or_path (str): Tokenizer name or path.
        llm_call_function (Optional[Callable]): LLM call function passthrough.
        llm_api_config (Optional[Dict[str, Any]]): LLM API config passthrough.

    Returns:
        List[Dict[str, Any]]: Embedding-oriented chunks with headers.
    """
    logger.info(
        f"Chunking for embedding. File: {file_name}. Custom options: {custom_chunk_options}"
    )
    chunks_from_improved_process = improved_chunking_process(
        text,
        chunk_options_dict=custom_chunk_options,
        tokenizer_name_or_path=tokenizer_name_or_path,
        llm_call_function_for_chunker=llm_call_function,
        llm_api_config_for_chunker=llm_api_config,
    )

    chunked_text_with_headers_list = []
    total_chunks_count = len(chunks_from_improved_process)

    for i, chunk_data in enumerate(chunks_from_improved_process):
        chunk_text_content = chunk_data["text"]
        chunk_metadata = chunk_data["metadata"]

        relative_pos = chunk_metadata.get("relative_position", 0.0)
        position_description = "middle"
        if relative_pos < 0.33:
            position_description = "beginning"
        elif relative_pos > 0.66:
            position_description = "end"

        chunk_header = f"""[DOCUMENT: {file_name}]
[CHUNK: {chunk_metadata.get("chunk_index", i + 1)} OF {chunk_metadata.get("total_chunks", total_chunks_count)}]
[POSITION: This chunk is from the {position_description} of the document.]
---BEGIN CHUNK CONTENT---
"""

        full_chunk_text_for_embedding = (
            chunk_header + chunk_text_content + "\n---END CHUNK CONTENT---"
        )

        embedding_chunk_data = {
            "text_for_embedding": full_chunk_text_for_embedding,
            "original_chunk_text": chunk_text_content,
            "source_document_name": file_name,
            "chunk_metadata": chunk_metadata,
        }
        chunked_text_with_headers_list.append(embedding_chunk_data)

    return chunked_text_with_headers_list


def process_document_with_metadata(
    text: str,
    chunk_options_dict: Dict[str, Any],
    document_metadata: Dict[str, Any],
    tokenizer_name_or_path: str = "gpt2",
    llm_call_function: Optional[Callable] = None,
    llm_api_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Processes a document, chunks it, and associates document-level metadata.

    Args:
        text (str): Document text.
        chunk_options_dict (Dict[str, Any]): Chunking options.
        document_metadata (Dict[str, Any]): Metadata to attach to every chunk.
        tokenizer_name_or_path (str): Tokenizer name or path.
        llm_call_function (Optional[Callable]): LLM call function passthrough.
        llm_api_config (Optional[Dict[str, Any]]): LLM API config passthrough.

    Returns:
        Dict[str, Any]: {"original_document_metadata": ..., "chunks": ...}.
    """
    logger.info(f"Processing document with metadata. Options: {chunk_options_dict}")
    chunks_result = improved_chunking_process(
        text,
        chunk_options_dict=chunk_options_dict,
        tokenizer_name_or_path=tokenizer_name_or_path,
        llm_call_function_for_chunker=llm_call_function,
        llm_api_config_for_chunker=llm_api_config,
    )
    for chunk_item in chunks_result:
        if "document_level_metadata" not in chunk_item["metadata"]:
            chunk_item["metadata"]["document_level_metadata"] = {}
        chunk_item["metadata"]["document_level_metadata"].update(document_metadata)
    return {"original_document_metadata": document_metadata, "chunks": chunks_result}


def improved_chunking_process(
    text: str,
    chunk_options_dict: Optional[Dict[str, Any]] = None,
    tokenizer_name_or_path: str = "gpt2",
    template: Optional[Any] = None,
    template_manager: Optional[Any] = None,
    llm_call_function_for_chunker: Optional[Callable] = None,
    llm_api_config_for_chunker: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Chunks text and returns chunks with legacy metadata plus flat keys.

    Mirrors the legacy flow exactly: builds a ``Chunker`` adapter (which
    resolves the effective options -- defaults <- template <- explicit --
    and, critically, routes ``rolling_summarize`` through the ported legacy
    payload-dict implementation), calls its ``chunk_text``, then enriches
    every chunk with the legacy metadata (chunk_index 1-based, total_chunks,
    chunk_method, max_size_setting, overlap_setting, language,
    relative_position, adaptive_chunking_used, chunk_content_hash) and the
    flat §6.3.2 keys (top-level start_char/end_char/word_count the DB seam
    reads), synthesizing offsets against the source text when the chunker
    did not provide them.

    On the delegated ``rolling_summarize`` path, LLM-call failures raise
    ``ChunkingError`` (fail-closed, spec §8.3): a provider exception, an
    ``"Error: ..."`` result string, or a non-string result each abort the
    chunking with a message naming the failed part -- matching the engine
    strategy's own fail-closed contract
    (Tests/Chunking/test_rolling_summarize_fail_closed.py). Legacy marker
    text is never persisted as document content.

    Args:
        text (str): The text to chunk.
        chunk_options_dict (Optional[Dict[str, Any]]): Legacy chunk options.
        tokenizer_name_or_path (str): Tokenizer name or path.
        template (Optional[Dict[str, Any]]): Pre-resolved template dict
            (spec §8.2); a bare name string raises ``TemplateError``
            pointing at ``template_runtime.resolve_template``.
        template_manager (Optional[Any]): Accepted and ignored (spec §8.2;
            legacy signature compatibility).
        llm_call_function_for_chunker (Optional[Callable]): LLM call function
            (rolling_summarize).
        llm_api_config_for_chunker (Optional[Dict[str, Any]]): LLM API config.

    Returns:
        List[Dict[str, Any]]: Chunks, each ``{"text": str, "metadata": dict,
        "start_char": int, "end_char": int, "word_count": int}``.

    Raises:
        ChunkingError: On chunking failures, a missing real tokenizer for
            the tokens method, or tokens overlap >= max_size.
        InvalidChunkingMethodError: If the requested method is unsupported.
        TemplateError: If ``template`` is a bare name string or an invalid
            flat template dict.
    """
    logger.info("Improved chunking process started...")
    logger.debug(f"Received chunk_options_dict: {chunk_options_dict}")
    logger.debug(
        f"Text length: {len(text)} characters, tokenizer: {tokenizer_name_or_path}"
    )
    if template:
        template_label = (
            template.get("name") if isinstance(template, dict) else template
        )
        logger.debug(f"Using pre-resolved template: {template_label}")

    chunker_instance = Chunker(
        options=chunk_options_dict,
        tokenizer_name_or_path=tokenizer_name_or_path,
        template=template,
        template_manager=template_manager,
    )
    effective_options = chunker_instance.options.copy()
    resolved_method = _normalize_legacy_method(
        effective_options.get("method") or "words"
    )

    try:
        raw_chunks = chunker_instance.chunk_text(
            text,
            method=effective_options["method"],
            llm_call_function=llm_call_function_for_chunker,
            llm_api_config=llm_api_config_for_chunker,
        )
        logger.debug(
            f"Created {len(raw_chunks)} raw_chunks using method {effective_options['method']}"
        )
    except ChunkingError:
        logger.error("ChunkingError in chunking process: re-raising")
        raise
    except Exception as e:
        logger.opt(exception=True).error(f"Unexpected error in chunking process: {e}")
        raise ChunkingError(f"Unexpected error in chunking process: {e}") from e

    total_chunks_count = len(raw_chunks)
    logger.info(f"Processing {total_chunks_count} chunks for metadata enrichment")

    # Normalize raw chunks to (text, method-specific metadata) pairs, exactly
    # like the legacy loop (dicts carry their own metadata; strings get {}).
    normalized: List[Dict[str, Any]] = []
    for chunk_item in raw_chunks:
        if (
            isinstance(chunk_item, dict)
            and "json" in chunk_item
            and "metadata" in chunk_item
        ):
            normalized.append(
                {"text": json.dumps(chunk_item["json"], ensure_ascii=False), "metadata": chunk_item["metadata"]}
            )
        elif isinstance(chunk_item, dict) and "text" in chunk_item and "metadata" in chunk_item:
            normalized.append(
                {"text": chunk_item["text"], "metadata": chunk_item["metadata"]}
            )
        elif isinstance(chunk_item, str):
            normalized.append({"text": chunk_item, "metadata": {}})
        else:
            logger.warning(
                f"Unexpected chunk item type: {type(chunk_item)}. Skipping."
            )

    # Flat-contract conversion (§6.3.2): legacy metadata enrichment plus
    # top-level start_char/end_char/word_count, synthesizing offsets when
    # the chunker did not attach any.
    span_overrides = _synthesize_flat_offsets(
        text, [item["text"] for item in normalized]
    )

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(normalized):
        chunk_text_content = item["text"]
        chunk_specific_metadata = dict(item["metadata"])

        current_chunk_metadata = {
            "chunk_index": i + 1,
            "total_chunks": len(normalized),
            "chunk_method": effective_options["method"],
            "max_size_setting": effective_options["max_size"],
            "overlap_setting": effective_options["overlap"],
            "language": effective_options.get("language", "unknown"),
            "relative_position": float((i + 1) / len(normalized)) if normalized else 0.0,
            "adaptive_chunking_used": effective_options.get("adaptive", False),
        }
        current_chunk_metadata.update(chunk_specific_metadata)
        current_chunk_metadata["chunk_content_hash"] = hashlib.md5(
            chunk_text_content.encode("utf-8")
        ).hexdigest()
        # task-12 (spec §8): in-memory consumers see the engine version
        # without a DB read. Top-level dict stays clean -- the persist seam
        # owns the DB stamp.
        current_chunk_metadata["chunk_engine_version"] = ENGINE_VERSION

        start_char = current_chunk_metadata.get("start_char")
        end_char = current_chunk_metadata.get("end_char")
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            start_char = current_chunk_metadata.get("start_offset")
            end_char = current_chunk_metadata.get("end_offset")
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            start_char = span_overrides[i]["start_char"]
            end_char = span_overrides[i]["end_char"]

        out.append(
            {
                "text": chunk_text_content,
                "metadata": current_chunk_metadata,
                "start_char": start_char,
                "end_char": end_char,
                "word_count": (
                    len(chunk_text_content.split()) if chunk_text_content else 0
                ),
            }
        )

    logger.info(
        f"Improved chunking process completed: {len(out)} chunks created using "
        f"method '{resolved_method}', language: {effective_options.get('language', 'unknown')}"
    )
    return out


def load_document(file_path: str) -> str:
    """Loads a document from a file and normalizes whitespace (legacy).

    Args:
        file_path (str): Path to the document.

    Returns:
        str: The document text with whitespace normalized.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return re.sub(r"\s+", " ", text).strip()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        raise
