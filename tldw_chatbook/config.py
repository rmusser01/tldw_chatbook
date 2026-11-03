from __future__ import annotations

# ADR-126: fence recovery and enroll before any runtime/config imports.
from tldw_chatbook.Backup_Recovery.storage_admission import admit_startup

admit_startup()

# tldw_cli/config.py
# Description: Configuration management for the tldw_cli application.
#

# Imports
import copy
import difflib
import hashlib
import importlib.resources
import importlib.util
import json
import re
import shutil
import sys
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib
import os
import time
from pathlib import Path
import toml
import portalocker
from typing import (
    Any,
    Collection,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Iterator,
)

#
# Third-Party Imports
from loguru import logger


#
# Local Imports
from tldw_chatbook.Backup_Recovery import profile_paths
from tldw_chatbook.Backup_Recovery import config_participants as _config_participants
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Constants import DEFAULT_SPLASH_DURATION_SECONDS
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase

from tldw_chatbook.provider_registry import (  # noqa: F401 - normalize_provider_config_key is re-exported
    ALL_RECORDS,
    CLOUD_PROVIDER_CONFIG_KEYS,
    _ENGINE_CONFIG_KEY_ALIASES,
    normalize_provider_config_key,
)
from tldw_chatbook.Utils.adaptive_reader_state import (
    ITEMS_MAX_WIDTH,
    ITEMS_MIN_WIDTH,
    ITEMS_TARGET_WIDTH,
    normalize_adaptive_reader_preferences,
)
from tldw_chatbook.Utils.console_background_effects import (
    normalize_console_background_effects,
)
from tldw_chatbook.Utils.log_sanitizer import redact_user_paths
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Utils.startup_errors import private_path_repair_hint
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
    atomic_private_write_text,
    create_private_text,
    lexical_path,
    open_private_binary,
    open_private_text_append_stream,
    secure_private_directory,
    verify_trusted_directory,
)
from tldw_chatbook.Utils.sensitive_config_keys import (
    is_sensitive_config_key,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryMigrationSeed
#
#######################################################################################################################
#
# Functions:

def _contains_literal_backslash_x(value: Any) -> bool:
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if type(current) is str:  # Inspect builtin values without custom coercion.
            if "\\x" in current:
                return True
        elif type(current) in (dict, list, tuple):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            # Only builtin containers enter this branch; custom mappings keep the legacy route.
            pending.extend(current.values() if type(current) is dict else current)
    return False


def dumps_cli_config(config_data: Mapping[str, Any]) -> str:
    """Retain the original codec and defer the exceptional literal encoder.

    Args:
        config_data: Original configuration mapping passed to the writer.

    Returns:
        TOML text using the existing ordinary or literal-backslash-x codec.
    """
    if not _contains_literal_backslash_x(config_data):
        return toml.dumps(config_data)
    from tldw_chatbook.Utils.toml_serialization import dumps_cli_config as special

    return special(config_data)


logger.debug("CRITICAL DEBUG: config.py module is being imported/executed NOW.")
# --- Constants ---
# Client ID used by the Server API itself when writing to sync logs


# --- [console_ssh]: SSH ControlMaster lifecycle for remote bindings -------
# (spec 2026-09-24-ssh-remote-workspace-bindings, "ControlMaster lifecycle —
# explicit, and the executor owns its health"). `Tools/remote_workspace_transport.py`
# consumes `control_persist`, `enable_multiplexing` and `connect_timeout_s`;
# the executor's call cap consumes `max_concurrent_calls`; the per-run
# session worker (spec 2026-09-27-ssh-session-worker-and-bundle-cache)
# consumes `session_worker`, `session_idle_s` and `bundle_cache`.

DEFAULT_CONSOLE_SSH_CONTROL_PERSIST = "10m"
DEFAULT_CONSOLE_SSH_CONNECT_TIMEOUT_S = 3
DEFAULT_CONSOLE_SSH_MAX_CONCURRENT_CALLS = 8
DEFAULT_CONSOLE_SSH_SESSION_IDLE_S = 60


@dataclass(frozen=True, slots=True)
class ConsoleSshSettings:
    """Effective ``[console_ssh]`` settings, coerced to their use types.

    Attributes:
        control_persist: OpenSSH ControlPersist duration, passed VERBATIM
            as the ``ControlPersist=`` value on the ``ssh -MNf`` master
            command (never parsed locally; unusable values fall back to the
            default rather than shipping garbage on an argv).
        enable_multiplexing: False disables the whole ControlMaster
            machinery — per-call direct connections, no sockets, no
            masters. The Windows posture (ssh.exe lacks ControlMaster).
        connect_timeout_s: Ceiling on the master handshake and per-call
            connects, in seconds.
        max_concurrent_calls: Cap on in-flight per-binding ssh calls.
        session_worker: Kill switch for the per-run SSH session worker;
            False is pure one-shot (one ssh exchange per call).
        session_idle_s: Idle close for a session worker, in seconds (both
            the laptop registry and the host-side parent).
        bundle_cache: Whether the host loader may cache the worker bundle
            in ``$XDG_RUNTIME_DIR``.
    """

    control_persist: str = DEFAULT_CONSOLE_SSH_CONTROL_PERSIST
    enable_multiplexing: bool = True
    connect_timeout_s: int = DEFAULT_CONSOLE_SSH_CONNECT_TIMEOUT_S
    max_concurrent_calls: int = DEFAULT_CONSOLE_SSH_MAX_CONCURRENT_CALLS
    session_worker: bool = True
    session_idle_s: int = DEFAULT_CONSOLE_SSH_SESSION_IDLE_S
    bundle_cache: bool = True


def get_console_ssh_settings() -> ConsoleSshSettings:
    """Return the effective ``[console_ssh]`` settings.

    The default returned when the section is absent or unusable is always
    the shipped default (multiplexing ON), matching the fallback posture
    of the sibling section accessors: a malformed key must not silently
    degrade remote workspaces into per-call direct connections.

    Returns:
        The coerced settings; per-key fallbacks on invalid values.
    """

    raw_persist = get_cli_setting(
        "console_ssh", "control_persist", DEFAULT_CONSOLE_SSH_CONTROL_PERSIST
    )
    # ssh_config(5) ControlPersist grammar: yes/no, or a TIME FORMAT --
    # bare seconds or number+unit groups such as "10m" or "1h30m".
    control_persist = (
        raw_persist
        if isinstance(raw_persist, str)
        and re.fullmatch(r"yes|no|\d+|(?:\d+[sSmMhHdDwW])+", raw_persist)
        else DEFAULT_CONSOLE_SSH_CONTROL_PERSIST
    )
    # Strict bool (TOML delivers real booleans): a hand-edited non-boolean
    # falls back to the shipped default rather than coercing through the
    # truthy-string ladder, which turned "maybe" into OFF — a silent
    # multiplexing kill switch no user asked for.
    raw_enabled = get_cli_setting("console_ssh", "enable_multiplexing", True)
    enable_multiplexing = raw_enabled if type(raw_enabled) is bool else True
    raw_session = get_cli_setting("console_ssh", "session_worker", True)
    raw_cache = get_cli_setting("console_ssh", "bundle_cache", True)
    return ConsoleSshSettings(
        session_worker=raw_session if type(raw_session) is bool else True,
        session_idle_s=coerce_int_setting(
            get_cli_setting(
                "console_ssh", "session_idle_s", DEFAULT_CONSOLE_SSH_SESSION_IDLE_S
            ),
            DEFAULT_CONSOLE_SSH_SESSION_IDLE_S,
            minimum=1,
        ),
        bundle_cache=raw_cache if type(raw_cache) is bool else True,
        control_persist=control_persist,
        enable_multiplexing=enable_multiplexing,
        connect_timeout_s=coerce_int_setting(
            get_cli_setting(
                "console_ssh",
                "connect_timeout_s",
                DEFAULT_CONSOLE_SSH_CONNECT_TIMEOUT_S,
            ),
            DEFAULT_CONSOLE_SSH_CONNECT_TIMEOUT_S,
            minimum=1,
        ),
        max_concurrent_calls=coerce_int_setting(
            get_cli_setting(
                "console_ssh",
                "max_concurrent_calls",
                DEFAULT_CONSOLE_SSH_MAX_CONCURRENT_CALLS,
            ),
            DEFAULT_CONSOLE_SSH_MAX_CONCURRENT_CALLS,
            minimum=1,
        ),
    )


SERVER_CLIENT_ID = "SERVER_API_V1"
# Client ID for the CLI application instance for its local databases
from tldw_chatbook.Backup_Recovery.isolated_restore import installation_client_id

CLI_APP_CLIENT_ID = installation_client_id()

# --- Path to the CLI's configuration file ---
DEFAULT_CONFIG_PATH = profile_paths.default_config_path()


@lru_cache(maxsize=16)
def _resolve_effective_config_path(
    override: str | None,
    home: str | None,
    userprofile: str | None,
    cwd: str | None,
) -> Path:
    """Cache lexical selection for one environment and relative-path base."""
    del home, userprofile, cwd  # Key the environment read by expanduser/abspath.
    return profile_paths.lexical_path(override or DEFAULT_CONFIG_PATH)


def _get_effective_config_path() -> Path:
    """Return the lexical active CLI config path."""
    override = os.environ.get("TLDW_CONFIG_PATH")
    cwd = os.getcwd() if override and not os.path.isabs(override) else None
    return _resolve_effective_config_path(
        override,
        os.environ.get("HOME"),
        os.environ.get("USERPROFILE"),
        cwd,
    )


def get_cli_config_path() -> Path:
    """Return the effective config path, including any environment override."""

    return _get_effective_config_path()


def get_user_themes_dir() -> Path:
    """Directory holding the user's saved theme TOML files (active profile)."""

    return get_cli_config_path().parent / "themes"


def _optional_package_available(module_name: str) -> bool:
    """Return whether an optional top-level module is installed without importing it."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (AttributeError, ImportError, ValueError):
        return False


def _default_stt_provider_for_platform() -> str:
    """Return the speech-to-text provider this platform prefers, unconfigured.

    macOS prefers the Apple-Silicon-native engines when installed --
    parakeet-mlx first, then lightning-whisper-mlx -- and every other
    platform (and macOS with neither installed) falls back to
    faster-whisper. This is the single source of truth for that
    preference: `load_settings()` uses it for `STT_settings.default_stt_provider`,
    and `CONFIG_TOML_CONTENT` is interpolated with it so the
    `[transcription] default_provider` line a fresh install ships never
    contradicts what this function computes (task-867 -- the template used
    to hardcode "faster-whisper" unconditionally, so the darwin preference
    could never engage on a normal install).
    """
    if sys.platform == "darwin":
        if _optional_package_available("parakeet_mlx"):
            return "parakeet-mlx"
        if _optional_package_available("lightning_whisper_mlx"):
            return "lightning-whisper-mlx"
    return "faster-whisper"


def application_owned_config_directory(config_path: Path) -> Path | None:
    """Return the app-owned default config parent, never a custom parent."""

    if _config_participants.verified_companion_parent(
        sys.modules[__name__], config_path
    ):
        return None
    if os.environ.get("TLDW_CONFIG_PATH"):
        return None
    default_path = lexical_path(DEFAULT_CONFIG_PATH)
    return default_path.parent if lexical_path(config_path) == default_path else None


def _report_config_path_posture(
    result: PrivatePathResult,
    *,
    target_kind: str = "file",
) -> None:
    if result.status is PrivatePathStatus.UNVERIFIED_PLATFORM:
        logger.warning(
            f"Config {target_kind} permission posture is unverified on this platform."
        )
    elif result.status is PrivatePathStatus.HARDENED_PRIVATE:
        logger.info(
            f"Hardened the effective config {target_kind} to the private posture."
        )


# --- Encryption support ---


def _take_startup_unlock() -> tuple[Optional[str], bool]:
    """The startup unlock's outcome, if it is running (TASK-34100.4 round 2).

    `Backup_Recovery.launcher.startup_unlock` imports this module only after
    the master password was typed; taking it here, before the module-level
    load below, means that load decrypts instead of warning "no password is
    set" right after the right password. Read through ``sys.modules`` so
    config never imports the launcher itself.

    Returns:
        ``(password, pending)`` from ``launcher.take_startup_unlock()``, or
        ``(None, False)`` when no startup unlock is in progress.
    """
    launcher = sys.modules.get("tldw_chatbook.Backup_Recovery.launcher")
    take = getattr(launcher, "take_startup_unlock", None)
    if not callable(take):
        return None, False
    return take()


# Cached password for the session; True while a startup reset or re-key is
# about to finish the unlock (the locked import-time load is then expected).
_ENCRYPTION_PASSWORD, _STARTUP_UNLOCK_PENDING = _take_startup_unlock()
_ENCRYPTION_MODULE = None  # Lazily loaded encryption module
_CONFIG_GENERATION = 0
_CONFIG_PERSISTENCE_ERROR = None

#: Permission mode used the first time an encryption-related rewrite of the
#: config file creates it from scratch (no pre-existing file whose mode can
#: be preserved). The config file can hold plaintext API keys and the
#: encryption password verifier, so a freshly created one gets a
#: user-only-readable mode rather than ``atomic_write_text``'s generic
#: 0o644 default. Every encryption entry point passes this alongside
#: ``preserve_existing_mode=True`` so an already-existing file's mode
#: (e.g. a user-tightened 0o600) is never widened by the rewrite -- see
#: task-851 review finding 2.
CONFIG_SECRETS_FILE_MODE = 0o600

# --- Chunking Settings (Default, can be overridden by TOML) ---
global_default_chunk_language = "en"

# --- Default Fallback Configurations (if not found in TOML) ---
# These will be populated from TOML or use these hardcoded dicts as fallbacks.
DEFAULT_APP_TTS_CONFIG = {
    "OPENAI_API_KEY_fallback": "sk-...",  # Note: API keys should primarily come from [API] section or ENV
    "KOKORO_ONNX_MODEL_PATH_DEFAULT": "path/to/your/downloaded/kokoro-v0_19.onnx",
    "KOKORO_ONNX_VOICES_JSON_DEFAULT": "path/to/your/downloaded/voices.json",
    "KOKORO_DEVICE_DEFAULT": "cpu",  # or "cuda"
    "ELEVENLABS_API_KEY_fallback": "el-...",  # Note: API keys should primarily come from [API] section or ENV
    "local_kokoro_default_onnx": {"KOKORO_DEVICE": "cuda:0"},
    "global_tts_settings": {
        # shared settings
    },
}

DEFAULT_DATABASE_CONFIG = {}  # Example, can be populated if needed

DEFAULT_RAG_SEARCH_CONFIG = {
    # Legacy settings for backwards compatibility
    "fts_top_k": 10,
    "vector_top_k": 10,
    "web_vector_top_k": 10,
    "llm_context_document_limit": 10,
    # New comprehensive RAG settings
    "retriever": {
        "fts_top_k": 10,
        "vector_top_k": 10,
        # Hybrid fusion alpha (RRF blend): 0 = FTS only, 1 = vector only.
        # 0.7 matches the tldw_server default (vector-weighted).
        "hybrid_alpha": 0.7,
        "chunk_size": 512,
        "chunk_overlap": 128,
        "media_collection": "media_embeddings",
        "chat_collection": "chat_embeddings",
        "notes_collection": "notes_embeddings",
        "character_collection": "character_embeddings",
    },
    "search": {
        "default_search_mode": "semantic",
        "default_top_k": 10,
        "score_threshold": 0.0,
        "include_citations": True,
        "citation_style": "inline",
        "snippet_max_chars": 240,
        "max_context_size": 16000,
    },
    "processor": {
        "enable_reranking": True,
        "reranker_model": None,
        "reranker_top_k": 5,
        "deduplication_threshold": 0.85,
        "max_context_length": 4096,
        "combination_method": "weighted",
    },
    "generator": {
        "default_model": None,
        "default_temperature": 0.7,
        "max_tokens": 1024,
        "enable_streaming": True,
        "stream_chunk_size": 10,
    },
    "chroma": {
        "persist_directory": None,
        "collection_prefix": "tldw_rag",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "distance_metric": "cosine",
    },
    "cache": {
        "enable_cache": True,
        "cache_ttl": 3600,
        "max_cache_size": 1000,
        "cache_embedding_results": True,
        "cache_search_results": True,
        "cache_llm_responses": False,
    },
    "service": {
        "profile": "hybrid_basic",  # Default profile: "bm25_only", "vector_only", "hybrid_basic", "hybrid_enhanced", "hybrid_full"
        # Available profiles:
        # - bm25_only: Pure keyword search
        # - vector_only: Pure semantic search
        # - hybrid_basic: Combined search without enhancements
        # - hybrid_enhanced: Hybrid with parent retrieval
        # - hybrid_full: All features enabled
        # - fast_search, high_accuracy, balanced, long_context, technical_docs, research_papers, code_search
        "custom_overrides": {
            # Optional: Override specific settings from the profile
            # "enable_parent_retrieval": True,
            # "enable_reranking": False,
            # "enable_parallel_processing": True,
            # "parent_size_multiplier": 3,
            # "expand_context_on_retrieval": True,
            # "clean_pdf_artifacts": True,
            #
            # There is NO "reranking_strategy" key here. One used to be
            # listed, and it read nothing -- no code anywhere loaded it, so
            # setting it selected exactly nothing (TASK-17600 F3). The
            # strategy lives on a RAG PROFILE, as
            # `reranking_config.strategy` in the profile's saved JSON under
            # <user data dir>/rag_profiles/: `ProfileConfig` serialises the
            # whole `RerankingConfig` and rebuilds it on load, so a saved or
            # cloned profile keeps its strategy across restarts (pinned by
            # Tests/RAG/test_config_profiles.py::
            # test_a_saved_profile_round_trips_its_reranking_strategy). The
            # Settings form edits that profile's provider/model/top-k but
            # not its strategy.
            #
            # Reranking strategies: "pointwise" | "pairwise" | "listwise"
            # (each bills an LLM provider per call) or "cross_encoder" (a
            # local model, no provider/credential/network). "cross_encoder"
            # is implemented and selectable but is NOT recommended: it is
            # the only strategy that has been measured here (TASK-16965)
            # and it came out net harmful on average [CAVEAT: that averaged row EXCLUDES `scoped` and `negative` (`UNAVERAGED_CATEGORIES`), and `scoped` is where this strategy WINS -- over all 53 ground-truthed queries hybrid REVERSES sign (MRR 0.731 -> 0.806, +0.075). TASK-16965 final review F1.] -- large gains where
            # retrieval is weak, losses where it is already good. See
            # Docs/superpowers/qa/2026-08-17-cross-encoder/report.md.
        },
    },
    "memory_management": {
        "max_total_size_mb": 1024.0,
        "max_collection_size_mb": 512.0,
        "max_documents_per_collection": 100000,
        "max_age_days": 90,
        "inactive_collection_days": 30,
        "enable_automatic_cleanup": True,
        "cleanup_interval_hours": 24,
        "cleanup_batch_size": 1000,
        "enable_lru_cache": True,
        "memory_limit_bytes": 2147483648,
        "min_documents_to_keep": 100,
        "cleanup_confirmation_required": False,
    },
}

DEFAULT_MEDIA_INGESTION_CONFIG = {
    "pdf": {
        "chunk_method": "semantic",
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": "",
        # OCR settings
        "enable_ocr": False,  # Default to disabled for performance
        "ocr_language": "en",  # Default OCR language
        "ocr_backend": "docling",  # Default OCR backend
        "ocr_confidence_threshold": 0.8,  # Minimum confidence score
    },
    "ebook": {
        "chunk_method": "ebook_chapters",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": "",
    },
    "document": {
        "chunk_method": "sentences",
        "chunk_size": 1500,
        "chunk_overlap": 100,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": "",
        # OCR settings
        "enable_ocr": False,  # Default to disabled for performance
        "ocr_language": "en",  # Default OCR language
        "ocr_backend": "docling",  # Default OCR backend
        "ocr_confidence_threshold": 0.8,  # Minimum confidence score
    },
    "plaintext": {
        "chunk_method": "paragraphs",
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": "",
    },
    "web_article": {
        "chunk_method": "paragraphs",
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": "",
    },
    "audio": {
        "chunk_method": "sentences",
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": "",
        "transcription_model": "base",
        "transcription_language": "en",
        "vad_filter": False,
        "diarize": False,
    },
    "video": {
        "chunk_method": "sentences",
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": "",
        "transcription_model": "base",
        "transcription_language": "en",
        "vad_filter": False,
        "diarize": False,
        "extract_audio_only": True,
    },
    "image": {
        "chunk_method": "visual_blocks",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": "",
        # OCR settings
        "enable_ocr": True,  # Default to enabled for images
        "ocr_backend": "auto",  # Auto-select best available backend
        "ocr_language": "en",
        "ocr_confidence_threshold": 0.8,
        # Visual processing settings
        "extract_visual_features": True,
        "visual_feature_model": "basic",
        "image_preprocessing": True,
        "max_image_size": 4096,  # Max dimension in pixels
    },
}

# OCR Backend Configurations
DEFAULT_OCR_BACKEND_CONFIG = {
    "docext": {
        "mode": "api",  # "api", "model", or "openai"
        "api_url": "http://localhost:7860",
        "model_name": "nanonets/Nanonets-OCR-s",
        "username": "admin",
        "password": "admin",
        "max_new_tokens": 4096,
        # For OpenAI mode
        "openai_base_url": "http://localhost:8000/v1",
        "openai_api_key": "123",
    },
    "tesseract": {
        "config": "",  # Additional tesseract config options
        "lang": "eng",  # Default language
    },
    "easyocr": {"use_gpu": True, "languages": ["en"]},
    "paddleocr": {"use_gpu": True, "lang": "en"},
}

DEFAULT_DIARIZATION_CONFIG = {
    # Enable diarization by default
    "enabled": False,
    # VAD settings
    "vad_threshold": 0.5,
    "vad_min_speech_duration": 0.25,
    "vad_min_silence_duration": 0.25,
    # Segmentation settings
    "segment_duration": 2.0,
    "segment_overlap": 0.5,
    "min_segment_duration": 1.0,
    "max_segment_duration": 3.0,
    # Embedding model
    "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
    "embedding_device": "auto",  # auto, cuda, cpu
    # Clustering settings
    "clustering_method": "spectral",  # spectral, agglomerative
    "similarity_threshold": 0.85,
    "min_speakers": 1,
    "max_speakers": 10,
    # Post-processing
    "merge_threshold": 0.5,  # seconds between segments to merge
    "min_speaker_duration": 3.0,  # minimum total duration per speaker
}

_BUILT_IN_OPENAI_TTS_MAPPINGS = {
    "models": {
        "tts-1": "openai_official_tts-1",
        "tts-1-hd": "openai_official_tts-1-hd",
        "eleven_monolingual_v1": "elevenlabs_english_v1",
        "kokoro": "local_kokoro_default_onnx",
    },
    "voices": {
        "alloy": "alloy",
        "echo": "echo",
        "fable": "fable",
        "onyx": "onyx",
        "nova": "nova",
        "shimmer": "shimmer",
        "RachelEL": "21m00Tcm4TlvDq8ikWAM",
        "k_bella": "af_bella",
        "k_adam": "am_v0adam",
    },
}


def _validate_openai_tts_mappings(payload: object) -> Dict[str, Dict[str, str]]:
    """Return the bounded mapping schema or raise without payload details."""

    if not isinstance(payload, dict):
        raise ValueError("OpenAI TTS mapping schema is invalid")
    validated: Dict[str, Dict[str, str]] = {}
    for section_name in ("models", "voices"):
        section = payload.get(section_name)
        if not isinstance(section, dict) or any(
            type(key) is not str or type(value) is not str
            for key, value in section.items()
        ):
            raise ValueError("OpenAI TTS mapping schema is invalid")
        validated[section_name] = dict(section)
    return validated


def load_openai_mappings() -> Dict:
    """Load OpenAI TTS mappings from packaged resources.

    Load through importlib.resources so wheels and editable installs agree.
    """
    from importlib import resources as importlib_resources

    package = "tldw_chatbook.Config_Files"
    resource_name = "openai_tts_mappings.json"

    try:
        mapping_path = importlib_resources.files(package).joinpath(resource_name)
        with mapping_path.open("r", encoding="utf-8") as f:
            return _validate_openai_tts_mappings(json.load(f))
    except Exception:
        logger.info("OpenAI TTS mappings unavailable; using built-in defaults")
        return copy.deepcopy(_BUILT_IN_OPENAI_TTS_MAPPINGS)


_openai_mappings = load_openai_mappings()

openai_tts_mappings = copy.deepcopy(_BUILT_IN_OPENAI_TTS_MAPPINGS)
# Update openai_tts_mappings with values from _openai_mappings (JSON file takes precedence)
if _openai_mappings:
    openai_tts_mappings["models"].update(_openai_mappings.get("models", {}))
    openai_tts_mappings["voices"].update(_openai_mappings.get("voices", {}))


def deep_merge_dicts(base: Dict, update: Dict) -> Dict:
    """Recursively merges update_dict into base_dict."""
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_encryption_module():
    """Lazily load and return the encryption module."""
    global _ENCRYPTION_MODULE
    if _ENCRYPTION_MODULE is None:
        from tldw_chatbook.Utils.config_encryption import config_encryption

        _ENCRYPTION_MODULE = config_encryption
    return _ENCRYPTION_MODULE


def set_encryption_password(password: str):
    """Set the encryption password for the current session.

    Also invalidates the settings/CLI config caches: any config loaded before the
    password was available (e.g. the module-level ``APP_CONFIG`` primed at import,
    before the startup unlock prompt) holds ciphertext, so it must be dropped and
    re-decrypted on the next load.
    """
    global _ENCRYPTION_PASSWORD, _SETTINGS_CACHE, _CONFIG_CACHE
    global _STARTUP_UNLOCK_PENDING
    _ENCRYPTION_PASSWORD = password
    _STARTUP_UNLOCK_PENDING = False
    _SETTINGS_CACHE = None
    _CONFIG_CACHE = None
    get_encryption_module().clear_derived_key_cache()
    logger.info("Encryption password set for current session")


def get_encryption_password() -> Optional[str]:
    """Get the encryption password for the current session."""
    return _ENCRYPTION_PASSWORD


def clear_encryption_password():
    """Clear the encryption password from memory."""
    global _ENCRYPTION_PASSWORD, _STARTUP_UNLOCK_PENDING
    _ENCRYPTION_PASSWORD = None
    _STARTUP_UNLOCK_PENDING = False
    get_encryption_module().clear_derived_key_cache()
    logger.info("Encryption password cleared from memory")


class _ConfigDecryptionResult(NamedTuple):
    config: Dict[str, Any]
    succeeded: bool


def _decrypt_config_section_with_status(
    config_data: Dict[str, Any],
    *,
    strict: bool = False,
) -> _ConfigDecryptionResult:
    encryption_config = config_data.get("encryption", {})
    if not encryption_config.get("enabled", False):
        return _ConfigDecryptionResult(config_data, True)

    password = get_encryption_password()
    if not password:
        if _STARTUP_UNLOCK_PENDING:
            # A startup reset or re-key installs its outcome right after this
            # import; a locked load now is expected, not a problem to report
            # under the prompt (TASK-34100.4 review round 2).
            logger.debug("Encrypted config loaded before the startup unlock finished.")
        else:
            logger.warning(
                "Encryption is enabled but no password is set. Cannot decrypt config."
            )
        return _ConfigDecryptionResult(config_data, True)

    try:
        enc_module = get_encryption_module()
        if strict:
            decrypted_config = enc_module.decrypt_config_strict(config_data, password)
        else:
            decrypted_config = enc_module.decrypt_config(config_data, password)
        return _ConfigDecryptionResult(decrypted_config, True)
    except Exception as e:
        logger.error(f"Failed to decrypt config: {e}")
        return _ConfigDecryptionResult(config_data, False)


def decrypt_config_section(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decrypt encrypted values in the config if encryption is enabled.

    Args:
        config_data: The config dictionary potentially containing encrypted values

    Returns:
        Config dictionary with decrypted values
    """
    return _decrypt_config_section_with_status(config_data).config


def encrypt_api_keys_in_config(
    config_data: Dict[str, Any], password: str
) -> Dict[str, Any]:
    """
    Encrypt API keys in the config data.

    Args:
        config_data: The config dictionary
        password: The password to use for encryption

    Returns:
        Config dictionary with encrypted API keys
    """
    enc_module = get_encryption_module()
    encrypted_config = copy.deepcopy(config_data)

    # Set encryption metadata
    encryption_config = encrypted_config.get("encryption", {})
    encryption_config["enabled"] = True
    encryption_config["method"] = "AES-256-GCM-scrypt"
    encryption_config["version"] = 1  # New clean version
    # Create password verifier for authentication
    encryption_config["password_verifier"] = enc_module.create_password_verifier(
        password
    )
    encrypted_config["encryption"] = encryption_config

    # Encrypt all sensitive fields in all sections
    def encrypt_sensitive_fields(d: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in d.items():
            # Skip the encryption section itself
            if key == "encryption":
                result[key] = value
                continue

            if isinstance(value, dict):
                # Recursively encrypt nested dictionaries
                result[key] = encrypt_sensitive_fields(value)
            elif isinstance(value, str) and value.strip():
                # Check if this is a sensitive field, using the same
                # predicate every other encryption/redaction/reporting path
                # in the app uses (see Utils/sensitive_config_keys.py for
                # why this used to disagree with itself).
                if is_sensitive_config_key(key):
                    # Skip if already encrypted or is a placeholder
                    if not (
                        enc_module.is_encrypted(value)
                        or (value.startswith("<") and value.endswith(">"))
                    ):
                        result[key] = enc_module.encrypt_value(value, password)
                    else:
                        result[key] = value
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    return encrypt_sensitive_fields(encrypted_config)


def _get_typed_value(
    data_dict: Dict, key: str, default: Any, target_type: type = str
) -> Any:
    """Helper to get value from dict and cast to type, with logging for type errors."""
    value = data_dict.get(key, default)
    if (
        value is default and default is not None
    ):  # if value is the default, it's already typed
        return value
    if value is None:  # If key is missing and default is None
        return None

    try:
        if target_type is bool:
            if isinstance(value, bool):
                return value
            # For bools from TOML strings (shouldn't happen if TOML is well-formed)
            return str(value).lower() in ["true", "1", "t", "y", "yes"]
        if target_type is Path:
            return Path(value) if value else default
        return target_type(value)
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Config key '{key}' has value '{value}' which could not be converted to {target_type}. Using default: '{default}'. Error: {e}"
        )
        return default


def _get_int_timeout_value(data_dict: Dict, key: str, default: int) -> int:
    """Get an integer timeout value, rejecting booleans, non-positive values, and malformed strings.

    Args:
        data_dict: Configuration dictionary
        key: Configuration key to lookup
        default: Default timeout value in seconds

    Returns:
        Integer timeout value, or default if conversion fails, value is boolean, or value is non-positive.
    """
    value = data_dict.get(key, default)
    if value is default:  # Already the default
        return value
    if isinstance(value, bool):  # Reject booleans (int(True) == 1, which is wrong)
        logger.warning(
            f"Config key '{key}' has boolean value {value} which is not valid for timeout. Using default: {default}."
        )
        return default
    try:
        timeout_int = int(value)
        if timeout_int <= 0:
            logger.warning(
                f"Config key '{key}' has value '{value}' which is non-positive and not valid for timeout. Using default: {default}."
            )
            return default
        return timeout_int
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Config key '{key}' has value '{value}' which could not be converted to int timeout. Using default: '{default}'. Error: {e}"
        )
        return default


DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD = 50
MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD = 1
MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD = 100000
DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES = 32768
MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES = 1
MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES = 1024 * 1024

# TASK-870: the single, user-adjustable cap on how much of an agent tool
# result the Console *displays* -- replaces the scattered hardcoded caps
# that used to govern the live step summary (200), the transcript TOOL
# marker (160), and a resumed/persisted step's summary (200). Distinct from
# `Agents.agent_models.RunBudget.max_tool_result_chars` (default 16,000),
# which caps what the MODEL saw -- that value enters the model's own
# history, not the Console's UI, and stays out of this control's reach.
# Default (160) is the transcript TOOL marker's prior cap, kept as-is: the
# inline marker is the primary, always-visible reading surface (every user
# scrolling the transcript sees it), so a fresh install's TRANSCRIPT reads
# unchanged. The Agent rail's live-step and resumed/persisted step summaries
# were previously 200, not 160 -- unifying to one cap means those two
# secondary, optional-panel surfaces now trim 40 characters more than
# before. That is a real, if minor, behaviour change, not a no-op: chosen
# over raising the default to 200 (which would instead grow the marker and
# make the transcript itself noisier) because the transcript is what most
# users actually read, and "View full log" now exists as the full-fidelity
# escape hatch the rail's tighter preview can safely defer to.
# Maximum (2000) is not arbitrary: `agent_runtime.py` already caps a step's
# OWN recorded `result` field at 2000 characters before any of these three
# display paths ever see it -- raising the display cap past that ceiling
# could not reveal a single additional character, only mislead a user into
# thinking a higher setting shows more. Reading the full, untruncated
# result -- beyond what any display cap can reach -- is what the run log
# ("View full log") is for.
DEFAULT_CONSOLE_TOOL_RESULT_DISPLAY_CHARS = 160
MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS = 20
MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS = 2000

# TASK-18600: the Console agent's run budget, exposed in Settings ▸ Console
# Behavior and resolved per run by `console_agent_bridge.console_run_budget()`.
# `[agents] denial_circuit_breaker_limit` is resolved separately per run;
# default 3 stops after a fully settled denied tail, while 0 disables it.
# These override `Agents.agent_models.RunBudget`'s own dataclass defaults
# (8 steps / 240s / 30 turns / 0 tokens / 300s per tool call), which stay
# deliberately conservative for any non-Console caller.
#
# WHICH LIMIT ACTUALLY STOPS A RUN: the token ceiling, not the turn cap.
# `agent_service._make_call_model` re-sends the whole conversation every
# turn (`bound_history_for_send` is a no-op unless `[agents]
# run_log_evict_enabled` is on -- off by default, deliberately: see
# `run_log_eviction.py`), and `ModelTurn.tokens` counts that whole re-sent
# prompt. Spend is therefore QUADRATIC in turn count -- roughly
# `delta * N^2 / 2` for `delta` tokens added per round -- so at a typical
# 800-token round the 25M ceiling is reached around turn 250, and reaching
# turn 2000 would require a ~12-token round. The turn and step caps below
# are backstops that will essentially never bind; they are sized generously
# so they never become the surprise limiter, and the token ceiling is the
# real, intentional governor. Owner decision, 2026-08-18.
#: Provider turns (STEP_MODEL steps) per user message.
DEFAULT_CONSOLE_AGENT_MAX_MODEL_TURNS = 2000
MIN_CONSOLE_AGENT_MAX_MODEL_TURNS = 1
#: Step backstop. A fence tool round costs 3 steps (STEP_MODEL +
#: STEP_TOOL_CALL + STEP_TOOL_RESULT) and the wrap-up reply costs 1, so N
#: turns need `3*(N-1)+1` steps -- 5998 at N=2000. 25000 clears that with
#: room for native multi-call batches, which cost `1 + 2N` steps per turn.
DEFAULT_CONSOLE_AGENT_MAX_STEPS = 25000
MIN_CONSOLE_AGENT_MAX_STEPS = 1
# Mirrored by agent_models.MAX_RUN_CONTROL_STEPS; pinned by Console budget tests
# without importing Agents here (that would create a config import cycle).
MAX_CONSOLE_AGENT_MAX_STEPS = 199_999
#: Wall-clock ceiling for ONE agent run (one user message), in seconds.
#: 86400 = 24h, so a genuinely long-running operation is not cut off. This
#: is a backstop, not a target: Stop cancels at every step boundary and
#: every 0.5s inside the tool-call wrapper, and a hung provider connection
#: is bounded separately by the generation client's own 300s read timeout
#: (`console_provider_gateway.GENERATION_READ_TIMEOUT_SECONDS`).
DEFAULT_CONSOLE_AGENT_MAX_WALL_SECONDS = 86400.0
MIN_CONSOLE_AGENT_MAX_WALL_SECONDS = 1.0
#: Cumulative prompt+completion spend ceiling for ONE run -- see the
#: quadratic-spend note above for why this is the limit that actually
#: fires. PER RUN, not per conversation: `run_agent_loop`'s `total_tokens`
#: is a per-run local, and BOTH child-budget paths (`clamp_child_budget`
#: for turn-scoped/inline children, `contain_child_budget` for threaded
#: survivor candidates) pass this value to a sub-agent UNCHANGED rather
#: than dividing it, so one message's real worst-case aggregate is
#: ~(1 + max_subagents)x this number -- 3x at the shipped
#: `RunBudget.max_subagents = 2`. Note the wall-clock ceiling below does
#: NOT compose the same way: a threaded child's wall comes from
#: `[agents] child_max_wall_seconds` (default 1800), independent of this
#: run's own, so raising the wall here does not extend a child's.
#: 0 = unlimited, which is genuinely dangerous here: the cycle detector
#: keys on exact `(name, args)` repetition (`agent_runtime._detect_cycle`),
#: so a loop with any varying argument escapes it entirely and this ceiling
#: is the ONLY remaining runaway-spend backstop.
DEFAULT_CONSOLE_AGENT_MAX_TOTAL_TOKENS = 25_000_000
MIN_CONSOLE_AGENT_MAX_TOTAL_TOKENS = 0
#: Wall-clock ceiling for ONE tool call, in seconds. Raised from the engine
#: default (300) because a 24h run budget is useless if a single long
#: crawl, ingest, or build dies at five minutes. 0 = no ceiling: the Console
#: resolver translates it to a finite-but-unfireable deadline
#: (`console_agent_bridge.UNLIMITED_TOOL_CALL_DEADLINE_SECONDS`) rather than
#: passing the engine's literal 0, which would bypass the timeout wrapper --
#: the wrapper is also the only thing polling Stop (every 0.5s) while a
#: tool runs, so a literal 0 would silently disable Stop for the duration
#: of every unlimited tool call. Lowering the configured value below ~186s
#: risks the wrapper reporting "timed out" for an MCP call that later
#: really executes on its abandoned thread -- see
#: `RunBudget.max_tool_call_seconds`.
DEFAULT_CONSOLE_AGENT_MAX_TOOL_CALL_SECONDS = 3600.0
#: TASK-25901: transient model failures (429, 5xx, dropped connection) retried
#: inside the loop before a run gives up. Two rides out a brief blip without
#: keeping a user waiting on a provider that is genuinely down; 0 restores the
#: pre-retry behaviour of ending the run on the first failure. Terminal errors
#: (auth, bad request, config) are never retried at any setting.
DEFAULT_CONSOLE_AGENT_MAX_MODEL_RETRIES = 2
MIN_CONSOLE_AGENT_MAX_MODEL_RETRIES = 0

#: TASK-26001: fraction of any budget dimension at which the running agent is
#: told once to wrap up. Clamped to [0.0, 1.0]; 1.0 effectively disables the
#: warning (exhaustion arrives with it).
DEFAULT_CONSOLE_AGENT_BUDGET_WARNING_FRACTION = 0.8
MIN_CONSOLE_AGENT_BUDGET_WARNING_FRACTION = 0.0

MIN_CONSOLE_AGENT_MAX_TOOL_CALL_SECONDS = 0.0

# Ephemeral side chat (Console selection menu): the default prompt template
# for the "More Details" action. ``{selection}`` is the only placeholder the
# side-chat service substitutes (Task-3 renders it via replace, not format).
DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE = "Give me more details about: {selection}"


def coerce_bool_setting(value: Any, default: bool = True) -> bool:
    """Coerce config/app setting values with the same bool rules as load_settings.

    Args:
        value: Raw setting value to coerce.
        default: Fallback value when coercion cannot produce a boolean.

    Returns:
        Coerced boolean value.
    """
    # TASK-32808.4: _get_typed_value returns None unchanged for a None value
    # (other typed getters rely on that passthrough), which violated this
    # function's `-> bool` annotation and let a feature that ships ON read OFF
    # whenever the key was absent. Return the default here, as
    # coerce_int_setting already does.
    if value is None:
        return default
    return _get_typed_value({"value": value}, "value", default, bool)


def coerce_int_setting(
    value: Any,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Coerce integer config/app setting values with optional bounds.

    Args:
        value: Raw setting value to coerce.
        default: Fallback value when coercion fails or bounds reject the value.
        minimum: Optional inclusive lower bound.
        maximum: Optional inclusive upper bound.

    Returns:
        Coerced integer value, or the default when the value is invalid.
    """
    if isinstance(value, bool):
        return default
    # TASK-18600: two inputs used to escape this function as exceptions
    # rather than as the default, and both are reachable from a hand-edited
    # config.toml -- which means both could abort `load_settings` (app
    # startup), not merely mis-set one key.
    #   * `None`: `_get_typed_value` returns None unchanged for a None
    #     value, and the bounds comparison below then raised
    #     `TypeError: '<' not supported between 'NoneType' and 'int'`.
    #   * a non-finite float (`nan`/`inf`, both writable in TOML): `int()`
    #     raises `OverflowError`, which `_get_typed_value` does not catch
    #     (it catches ValueError/TypeError only).
    # Neither is a value any caller can act on, so both resolve to the
    # default like every other unusable input.
    if value is None:
        return default
    if isinstance(value, float) and (
        value != value or value in (float("inf"), float("-inf"))
    ):
        return default
    try:
        coerced = _get_typed_value({"value": value}, "value", default, int)
    except OverflowError:
        return default
    if coerced is None or isinstance(coerced, bool):
        return default
    if minimum is not None and coerced < minimum:
        return default
    if maximum is not None and coerced > maximum:
        return default
    return coerced


def coerce_float_setting(
    value: Any,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce float config/app setting values with optional bounds.

    The float twin of ``coerce_int_setting``, added for the Console agent
    run budget's two duration settings (wall-clock and per-tool-call
    seconds), which are floats in ``RunBudget`` and must survive a TOML
    value written as either ``86400`` or ``86400.0``.

    Args:
        value: Raw setting value to coerce.
        default: Fallback value when coercion fails or bounds reject the value.
        minimum: Optional inclusive lower bound.
        maximum: Optional inclusive upper bound.

    Returns:
        Coerced float value, or the default when the value is invalid.
        ``bool`` is rejected outright (``True`` is not a duration), and a
        non-finite value (``nan``/``inf``) falls back to the default -- an
        ``inf`` wall budget would make the run's wall-clock check
        unfireable rather than merely generous.
    """
    if isinstance(value, bool):
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if coerced != coerced or coerced in (float("inf"), float("-inf")):
        return default
    if minimum is not None and coerced < minimum:
        return default
    if maximum is not None and coerced > maximum:
        return default
    return coerced


# Global cache for load_settings to avoid redundant file I/O
_SETTINGS_CACHE: Optional[Dict[str, Any]] = None
_SETTINGS_CACHE_SOURCE: Optional[Path] = None
#: PERF-06: the config path's identity (every component), stamped before the
#: cached settings were read; a warm hit requires it unchanged.
_SETTINGS_CACHE_POSTURE: Optional[tuple] = None
_SETTINGS_CACHE_LOCK = None  # Will be initialized when needed
#: Serializes the miss->rebuild->store sequence (task-3503).
#:
#: `_SETTINGS_CACHE_LOCK` guards only the cache *cells*; it is released for
#: the rebuild itself, so every thread arriving during a miss used to run
#: the whole rebuild -- re-reading and re-parsing the TOML, re-merging
#: defaults, re-ensuring directories. Measured at 32 bootstrap loads for 8
#: threads on ONE invalidation.
#:
#: REENTRANT on purpose: the rebuild reaches helpers that read configuration
#: again, so a plain Lock would deadlock the rebuilding thread against
#: itself. RLock keeps same-thread reentry behaving exactly as before while
#: admitting only one *thread* at a time.
#:
#: Lock order is ``_SETTINGS_REBUILD_LOCK`` -> ``_CONFIG_FILE_LOCK`` ->
#: ``_SETTINGS_CACHE_LOCK``. Config writes and runtime snapshots use that same
#: order, while warm settings-cache hits take only the cache lock.
_SETTINGS_REBUILD_LOCK = None  # Will be initialized when needed


def _settings_rebuild_lock():
    """Return the process-wide reentrant settings rebuild lock."""

    global _SETTINGS_REBUILD_LOCK
    if _SETTINGS_REBUILD_LOCK is None:
        import threading

        _SETTINGS_REBUILD_LOCK = threading.RLock()
    return _SETTINGS_REBUILD_LOCK


def resolve_tldw_api_config(app_config) -> Dict:
    """Return the [tldw_api] section from either config shape.

    Raw CLI config (load_cli_config_and_ensure_existence) carries [tldw_api]
    at the top level; the app's normalized config (load_settings) keeps the
    raw CLI config nested under COMPREHENSIVE_CONFIG_RAW. Every reader of the
    server endpoint/token must accept both shapes.
    """
    if not isinstance(app_config, dict) and not hasattr(app_config, "get"):
        return {}
    api_config = app_config.get("tldw_api", {}) or {}
    if not isinstance(api_config, dict) or not api_config:
        raw_config = app_config.get("COMPREHENSIVE_CONFIG_RAW", {}) or {}
        nested = raw_config.get("tldw_api", {}) if hasattr(raw_config, "get") else {}
        api_config = nested if isinstance(nested, dict) else {}
    if not isinstance(api_config, dict):
        return {}
    return dict(api_config)


# The [tldw_api] values CONFIG_TOML_CONTENT ships into a fresh profile's
# config file. NOT credentials: the pair exists so readers can recognize an
# untouched template binding (e.g. the Library ingest canvas suppresses its
# server-mode hint when the binding is still the placeholder). Single
# definition here, beside the template's owner module, so the literals never
# spread through the codebase.
TLDW_API_PLACEHOLDER_BASE_URL = "http://127.0.0.1:8000"
TLDW_API_PLACEHOLDER_AUTH_TOKEN = "default-secret-key-for-single-user"

#: Canonical "this is not a real credential" values for provider API keys --
#: PR-T2 Task 7. Shared by `_normalize_legacy_provider_api_key` below and
#: `Chat/provider_readiness.get_provider_readiness`, which imports these
#: three names FROM this module rather than the reverse.
#:
#: (Review round 2 finding: the first fix round had `config.py` import
#: `is_valid_provider_api_key` FROM `Chat/provider_readiness.py`, which
#: reproduced a real cycle -- `config` -> `Chat/__init__` -> `server_chat_
#: conversation_service` -> `runtime_policy.bootstrap` -> back into
#: `config` -- breaking standalone collection of `Tests/RuntimePolicy/`
#: even though the shipped app and the full-suite alphabetical collect
#: order happened to hide it. `config.py` is the dependency ROOT nearly
#: everything else in this app already imports directly (verified: none of
#: this module's own top-level imports -- the `DB.*`/`Utils.*` modules --
#: reach back into `Chat` or `runtime_policy`), so the shared definition
#: belongs here, with `Chat/provider_readiness.py` importing it -- not the
#: reverse. This keeps the actual property Task 7 exists for -- exactly
#: ONE definition of "valid key", consumed by both the credential bridge
#: and the readiness reader -- while fixing the layering.)
PROVIDER_API_KEY_PLACEHOLDERS = frozenset(
    {
        "",
        "<API_KEY_HERE>",
        "YOUR_KEY",
        "your_key",
        "your-api-key",
    }
)


#: The prefix `Utils/config_encryption.ConfigEncryption` writes on every
#: encrypted config value (pinned equal by
#: Tests/test_config_encryption_lifecycle.py). Kept as a literal here so the
#: key resolver below never imports the encryption engine.
ENCRYPTED_CONFIG_VALUE_PREFIX = "enc:"


def is_encrypted_config_value(value: object) -> bool:
    """Whether ``value`` is still-encrypted ``enc:`` config ciphertext.

    TASK-34100.4 (new-protect-summary-03): ciphertext is never a credential.
    A locked session (encryption on, no password) or a value that did not
    decrypt keeps its ``enc:`` form in the loaded config, so every code path
    that reads a key straight from config must treat it as unusable.
    """
    return isinstance(value, str) and value.strip().startswith(
        ENCRYPTED_CONFIG_VALUE_PREFIX
    )


def without_ciphertext(value: Any, absent: Any = None) -> Any:
    """Return ``value`` unchanged, or ``absent`` when it is ``enc:`` ciphertext.

    For key reads that must keep their own blank/placeholder semantics (the
    summarizers distinguish a configured-blank key from a missing one) but
    must never send ciphertext. Provider handlers use
    `resolve_provider_api_key` instead.
    """
    return absent if is_encrypted_config_value(value) else value


def resolve_provider_api_key(value: object) -> Optional[str]:
    """Return `value` stripped, or `None` if it is not a usable provider API
    key (not a string, blank, one of `PROVIDER_API_KEY_PLACEHOLDERS`, or
    still-encrypted `enc:` ciphertext).

    TASK-34100.4 (new-protect-summary-03): ciphertext reads as ABSENT. A
    locked session (encryption on, no password) or a value that did not
    decrypt keeps its `enc:` form in the loaded config; accepting it made
    readiness report Ready and sent the ciphertext as a bearer token.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped in PROVIDER_API_KEY_PLACEHOLDERS:
        return None
    if is_encrypted_config_value(stripped):
        return None
    return stripped or None


def is_valid_provider_api_key(value: object) -> bool:
    """Return whether `value` is a usable provider API key."""
    return resolve_provider_api_key(value) is not None


def resolve_tldw_api_auth_token(value: object) -> Optional[str]:
    """Return `value` stripped, or None if blank/placeholder/synthesized.

    Reuses `resolve_provider_api_key`'s existing blank/placeholder screening
    rather than duplicating the rule; adds only the one extra rejected
    literal this credential's boot-rewrite is known to produce.

    Args:
        value: Raw `[tldw_api] auth_token` config value of any type.

    Returns:
        `value` stripped, or None if it is blank, a known provider-key
        placeholder, or `TLDW_API_PLACEHOLDER_AUTH_TOKEN` -- the value the
        app's own config load synthesizes into `[tldw_api]` when a
        profile's file omits `auth_token` (task-31417).
    """
    resolved = resolve_provider_api_key(value)
    if resolved is None or resolved == TLDW_API_PLACEHOLDER_AUTH_TOKEN:
        return None
    return resolved


class ProviderSettingsError(ValueError):
    """Raised when a selected provider's config table is not a mapping."""


def provider_settings_for_key(
    api_settings: object,
    provider_key: str,
) -> Mapping[str, object]:
    """Return provider settings without mutating the loaded configuration.

    QwenCloud historically accepts normalized aliases with the exact table
    taking precedence. Moonshot and Z.ai require one exact canonical table and
    reject normalized duplicates. Other providers retain the existing
    first-normalized-match behavior.

    Args:
        api_settings: Loaded ``api_settings`` value.
        provider_key: Canonical provider key to resolve.

    Returns:
        The resolved provider settings, or an empty mapping when none exist.

    Raises:
        ProviderSettingsError: If an authoritative table is malformed or a
            strict provider has ambiguous normalized tables.
    """
    if not isinstance(api_settings, Mapping):
        return {}

    if provider_key in {"moonshot", "zai"}:
        canonical_settings = api_settings.get(provider_key)
        aliases = tuple(
            configured_provider
            for configured_provider in api_settings
            if configured_provider != provider_key
            and normalize_provider_config_key(configured_provider) == provider_key
        )
        if provider_key in api_settings:
            if not isinstance(canonical_settings, Mapping) or aliases:
                raise ProviderSettingsError(
                    f"api_settings.{provider_key} must be one unambiguous configuration table."
                )
            return canonical_settings
        return {}

    if provider_key == "qwencloud":
        if provider_key in api_settings:
            canonical_settings = api_settings[provider_key]
            if not isinstance(canonical_settings, Mapping):
                raise ProviderSettingsError(
                    "api_settings.qwencloud must be a configuration table."
                )
            alias_settings: Mapping[str, object] = {}
            for configured_provider, configured_value in api_settings.items():
                configured_key = str(configured_provider)
                if configured_key == provider_key:
                    continue
                if normalize_provider_config_key(configured_key) != provider_key:
                    continue
                if isinstance(configured_value, Mapping):
                    alias_settings = configured_value
                break
            merged = dict(alias_settings)
            merged.update(canonical_settings)
            return merged

        for configured_provider, configured_value in api_settings.items():
            if normalize_provider_config_key(configured_provider) != provider_key:
                continue
            if not isinstance(configured_value, Mapping):
                raise ProviderSettingsError(
                    "api_settings.qwencloud must be a configuration table."
                )
            return configured_value
        return {}

    for configured_provider, configured_value in api_settings.items():
        if normalize_provider_config_key(configured_provider) != provider_key:
            continue
        if isinstance(configured_value, Mapping):
            return configured_value
        return {}

    return {}


#: Providers whose legacy `[API] <provider>_api_key` TOML key and
#: `<PROVIDER>_API_KEY` environment variable both follow the plain
#: `_normalize_legacy_provider_api_key` convention, keyed by the SAME
#: normalized provider key `Chat/provider_readiness.provider_config_key`
#: would derive (e.g. `"anthropic"`, not `"anthropic_api"`) -- this is what
#: lets the bridge below write into `api_settings.<provider_key>`, the
#: table `get_provider_readiness` actually reads.
#:
#: `mistral` IS included, deliberately, even though the shipped default
#: config's own decorative table is `[api_settings.mistralai]` (matching
#: `[providers]`'s `MistralAI` listing): `LLM_Calls/LLM_API_Calls.py`'s
#: `chat_with_mistral` (~4617-4621) reads `api_settings.mistral` -- NOT
#: `api_settings.mistralai` -- so `"mistral"` (what `provider_config_key
#: ("Mistral")` computes, and what this bridge writes into) IS the live
#: table the spend path already reads. An earlier revision of this comment
#: claimed bridging under `"mistral"` would create "a second, disconnected
#: table" -- that was backwards: `[api_settings.mistralai]` is the
#: disconnected one; `api_settings.mistral` was simply absent from the
#: shipped defaults, not wrong to write into. (Whether `MistralAI`-display
#: users should also read `[api_settings.mistralai]`, and whether that
#: default table should be renamed, is a separate, pre-existing question
#: out of scope for this task.)
_LEGACY_PROVIDER_API_KEY_BRIDGE: tuple[tuple[str, str, str], ...] = (
    ("openai", "openai_api_key", "OPENAI_API_KEY"),
    ("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY"),
    ("cohere", "cohere_api_key", "COHERE_API_KEY"),
    ("groq", "groq_api_key", "GROQ_API_KEY"),
    ("huggingface", "huggingface_api_key", "HUGGINGFACE_API_KEY"),
    ("openrouter", "openrouter_api_key", "OPENROUTER_API_KEY"),
    ("deepseek", "deepseek_api_key", "DEEPSEEK_API_KEY"),
    ("google", "google_api_key", "GOOGLE_API_KEY"),
    ("mistral", "mistral_api_key", "MISTRAL_API_KEY"),
)


def _normalize_legacy_provider_api_key(
    api_settings_section: Dict[str, Any],
    provider_key: str,
    *,
    raw_api_section: Dict[str, Any],
    toml_key: str,
    env_var: str,
    bridged_from_legacy: set[str] | None = None,
) -> Optional[str]:
    """One normalized credential for `provider_key` -- PR-T2 Task 7.

    The harm this closes: a config with only `[API] anthropic_api_key` set
    spent real money through the Library path (`LLM_Calls/LLM_API_Calls.
    py`'s `chat_with_anthropic` reads the legacy `anthropic_api` dict built
    a few lines below this function's call sites, which DOES see that key)
    while Console's own readiness check (`Chat/provider_readiness.get_
    provider_readiness`, reading only `api_settings.<provider>.api_key`)
    showed a blocking "Connect a provider" wall for the identical config.
    Both the modern `api_settings` table and the legacy `<provider>_api`
    dict must be built from the SAME resolved value -- never two
    independent reads of `[API]` -- or the split just moves.

    Which of the two a handler reads varies by provider, and the bridge is
    only as true as the table the handler actually opens: `chat_with_
    anthropic` reads the legacy dict, `chat_with_mistral` reads
    `api_settings.mistral`, and `chat_with_google` reads `api_settings.
    google` -- the last only since PR-T2 review round 3 (finding I4), which
    found it reading `api_settings.google_api`, a table nothing in this app
    has ever produced. Google was therefore the one provider for which
    this docstring's claim was FALSE for a bridged `[API] google_api_key`:
    it could not reach the spend path at all while readiness reported
    ready. Pinned by `Tests/Chat/test_google_native_tools.py::test_google_
    api_key_comes_from_the_api_settings_google_table` (handler half) and
    `Tests/Chat/test_provider_readiness.py::test_legacy_only_google_key_
    lands_in_the_table_chat_with_google_reads` (config half).

    **Scope of the "both readers agree" guarantee: the two CONFIG sources
    only** -- the modern `api_settings.<provider>.api_key` table and the
    legacy `[API] <provider>_api_key`. It does NOT extend to the
    environment variable for every provider, and google is a known open
    case (PR-T2 review round 4, R1; pre-existing, filed separately, NOT
    closed here). With only `GOOGLE_API_KEY` set: readiness reports ready
    (it has its own env fallback), this bridge deliberately does not write
    env-sourced values into `api_settings` (see the paragraph above on
    `api_key_source` and prefill safety -- that behavior is pinned and
    correct), google's legacy top-level dict was removed as dead code
    (nothing read it), and `chat_with_google` has no env fallback of
    its own. A caller that passes no explicit `api_key` -- the Library RAG
    path among them -- is therefore gated OPEN and fails at the wire. Any
    claim that google is fully closed must say "config-sourced" or name
    this gap.

    Precedence: (1) an explicit, non-placeholder `api_settings.<provider_
    key>.api_key` always wins -- a value entered through the Settings
    screen (the modern location) was a deliberate choice; (2) the
    environment variable; (3) the legacy `[API] <toml_key>` value.

    **This precedence deliberately inverts the module docstring's general
    "env vars -> config.toml -> defaults" priority for this one credential
    lookup.** Before this task, `chat_with_openai` ALREADY overlaid a
    canonical `api_settings.openai` value onto the legacy dict for `api_
    key`/`api_base_url` (see its own call site, `LLM_API_Calls.py:561-580`)
    while every OTHER provider's `<provider>_api` dict resolved env-before-
    TOML with no `api_settings` input at all -- two silently different
    rules per provider. Extending "modern config wins" to all 9 bridged
    providers, rather than restoring env-first everywhere, is the choice
    that makes `get_provider_readiness` (which always shows the config
    value first) and the actual spend agree on which key was used -- the
    entire point of this task. A user who explicitly set `api_settings.
    <provider>.api_key` (e.g. via the Settings screen) now spends with that
    value even if a stale environment variable is also set; this is a
    real, user-visible behavior change and is intentional. See PR-T2 Task
    7's test `test_modern_api_settings_key_outranks_the_env_var_for_the_
    spending_path` (`Tests/Chat/test_provider_readiness.py`) and Task 8's
    docs pass, which should call this out explicitly.

    Only the (3) TOML-sourced fallback is ever written back into
    `api_settings_section` -- never an (2) env-sourced one. Writing an
    env-derived secret into `api_settings` would flip its reported
    `api_key_source` from `env:...` to `config:...`, and `provider_
    readiness.chat_api_key_field_state` treats a `config:` source as safe
    to prefill and persist in the inline Chat-Defaults API-key field --
    silently exposing a secret that was never typed into config. `get_
    provider_readiness`'s own environment fallback already reports the
    env-only case as ready without this rewrite, so nothing is lost by
    leaving `api_settings` untouched there.

    A provider table that does not yet exist is created only when there is
    an actual TOML-sourced value to write into it -- never eagerly, since
    several tests pin an untouched config's `api_settings` as exactly `{}`.

    "Non-placeholder" is delegated to THIS module's own `resolve_provider_
    api_key` (see its definition above, near `PROVIDER_API_KEY_
    PLACEHOLDERS`) -- the SAME function `Chat/provider_readiness.get_
    provider_readiness` runs each of ITS candidate sources through (it
    imports the name from here, aliased `_valid_api_key`), applied here to
    all three sources rather than per-branch, so neither reader can see a
    value the other rejects or a differently-trimmed form of it. An earlier
    revision of this function defined its own single-value placeholder
    check (`value != "<API_KEY_HERE>"`), which missed four of the five
    placeholders that function recognizes (`""`, `"YOUR_KEY"`, `"your_
    key"`, `"your-api-key"`). Concretely: `[api_settings.anthropic] api_key
    = "YOUR_KEY"` alongside a REAL `[API] anthropic_api_key` used to have
    the bridge treat `"YOUR_KEY"` as "explicit modern config wins" and
    write it into the legacy `anthropic_api` dict `chat_with_anthropic`
    spends through, while readiness (which DID recognize `"YOUR_KEY"`)
    correctly reported not-ready -- recreating, inside the very function
    meant to end it, the exact kind of two-readers-disagree split this
    task exists to close. A review round then found a second layering bug
    in the first fix for that: routing the import through `Chat/provider_
    readiness.py` (i.e. `config` importing FROM `Chat`) created a real
    cycle back through `Chat/__init__` -> `server_chat_conversation_
    service` -> `runtime_policy.bootstrap` -> `config`, breaking
    standalone collection of `Tests/RuntimePolicy/` (masked in the
    full-suite run only by alphabetical import ordering). The definition
    now lives here instead, with `provider_readiness.py` importing it --
    matching the rest of the codebase's dependency direction (nearly every
    other module already imports `config` directly; `config` importing
    from a submodule of `Chat` was the anomaly).

    Args:
        api_settings_section: The (already TOML-loaded) `api_settings`
            table, mutated in place only when a TOML-sourced fallback
            applies.
        provider_key: Normalized provider key, e.g. `"anthropic"` -- must
            match `Chat/provider_readiness.provider_config_key`'s output
            for this provider.
        raw_api_section: The raw `[API]` TOML section (legacy namespace).
        toml_key: The legacy key within `raw_api_section`, e.g.
            `"anthropic_api_key"`.
        env_var: The conventional environment variable, e.g.
            `"ANTHROPIC_API_KEY"`.
        bridged_from_legacy: When given, `provider_key` is added to it if
            the TOML-sourced fallback filled the table (TASK-33001.15).

    Returns:
        The normalized credential (feeds the legacy `<provider>_api` dict
        built just below each call site), or `None` when no source
        resolves one.
    """
    provider_table = api_settings_section.get(provider_key)
    # EVERY source goes through `resolve_provider_api_key` -- the same
    # function `get_provider_readiness` runs each of its own candidates
    # through -- and what this function returns is that resolved (stripped,
    # non-placeholder) value, never the raw one (PR-T2 review round 3,
    # findings I5 + minor 2). Two ways the earlier per-branch handling
    # split the two readers back apart:
    #   * The env branch was a bare `os.getenv(env_var)` truth test, so
    #     `ANTHROPIC_API_KEY="YOUR_KEY"` landed verbatim in the legacy
    #     `anthropic_api` dict while readiness (which validates every
    #     source) said not-ready -- gated surfaces failed safe, but an
    #     ungated `chat_api_call` caller (Evals, briefings, agent runs)
    #     would send the placeholder as a credential.
    #   * The modern/legacy branches validated the STRIPPED form but
    #     returned the raw one, so `api_key = " sk-xyz "` showed
    #     `sk-xyz` in readiness and spent ` sk-xyz ` (a 401).
    table = provider_table if isinstance(provider_table, dict) else {}
    # TASK-33641.11 (ADR-012): a recorded `credential_source` decides as it
    # does for `provider_readiness.resolve_provider_credential` (not imported:
    # that is a cycle). "none" uses no key; "stored" the table's key or the
    # legacy one copied below, never the variable; "environment" only the
    # variable the table names. No source keeps the order below.
    source = table.get("credential_source")
    source = source.strip().lower() if type(source) is str else None
    if source == "none":
        return None
    if source == "environment":
        named = table.get("api_key_env_var")
        named = named.strip() if isinstance(named, str) else ""
        return resolve_provider_api_key(os.getenv(named)) if named else None

    modern_value = resolve_provider_api_key(table.get("api_key"))
    if modern_value:
        return modern_value  # explicit modern config always wins

    env_value = (
        None if source == "stored" else resolve_provider_api_key(os.getenv(env_var))
    )
    if env_value:
        return env_value

    legacy_value = resolve_provider_api_key(raw_api_section.get(toml_key))
    if legacy_value:
        if not isinstance(provider_table, dict):
            provider_table = {}
            api_settings_section[provider_key] = provider_table
        provider_table["api_key"] = legacy_value
        if bridged_from_legacy is not None:
            bridged_from_legacy.add(provider_key)
        return legacy_value

    return None


def _config_file_posture(config_path: Path) -> tuple:
    """``lstat`` identity of every component from ``/`` down to the config file.

    Each entry is (dev, ino, type, mode, uid), plus nlink for the file itself,
    or ``None`` for a missing component. A replaced file, a new hard link, a
    symlink swap of the file, or a parent renamed and replaced by a symlink to
    itself all change it (Qodo, #2903). ``lstat`` only: nothing is opened or
    followed.
    """
    import stat as _stat

    posture = []
    for component in (*reversed(config_path.parents), config_path):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            posture.append(None)
            continue
        except OSError:
            posture.append(("unreadable",))
            continue
        entry = (
            info.st_dev,
            info.st_ino,
            _stat.S_IFMT(info.st_mode),
            _stat.S_IMODE(info.st_mode),
            info.st_uid,
        )
        posture.append(entry + (info.st_nlink,) if component == config_path else entry)
    return tuple(posture)


def _is_plain_owned_file(posture: tuple) -> bool:
    """Whether a posture's last entry is a regular file this user owns, linked once.

    Only that may stand in for the config file a load just created: the
    guarded member checks reject anything else, so a symlink or other object
    swapped in before the re-stamp keeps the stale stamp and the next read
    meets those checks (Qodo, #2903).

    Args:
        posture: A :func:`_config_file_posture` result.

    Returns:
        True for a plain, owned, singly linked file.
    """
    import stat as _stat

    leaf = posture[-1] if posture else None
    # Windows has no geteuid and reports no owner in st_uid; ownership is
    # only comparable where it exists (Qodo, #2903).
    geteuid = getattr(os, "geteuid", None)
    return (
        leaf is not None
        and len(leaf) == 6
        and leaf[2] == _stat.S_IFREG
        and (geteuid is None or leaf[4] == geteuid())
        and leaf[5] == 1
    )


def _settings_cache_hit(active_config_path: Path) -> dict | None:
    """Return the installed settings for ``active_config_path``, or None.

    One short lock and one ``lstat`` per path component. Shared by the
    unguarded warm paths and the guarded rebuild's re-check. The config
    path's identity must still be the one stamped before the cached settings
    were read: the guarded path rejected a replaced or symlinked config
    member, or a symlinked parent, on every read, and a warm hit must not
    hide one (Qodo, #2903), so any change is a miss.
    """
    global _SETTINGS_CACHE_LOCK

    if _SETTINGS_CACHE_LOCK is None:
        import threading

        _SETTINGS_CACHE_LOCK = threading.Lock()
    with _SETTINGS_CACHE_LOCK:
        cached = _SETTINGS_CACHE
        source = _SETTINGS_CACHE_SOURCE
        posture = _SETTINGS_CACHE_POSTURE
    if cached is None or source != active_config_path:
        return None
    if not _posture_unchanged(active_config_path, cached, posture):
        return None
    return cached


#: Per-thread record of the last UI-loop stretch that re-checked the posture.
_POSTURE_STRETCH = None


def _posture_unchanged(active_config_path: Path, cached: dict, posture: tuple) -> bool:
    """Whether the config path still has ``posture``; one check per loop stretch.

    TASK-33620.15: a warm hit ``lstat``s every path component, and one Console
    send made 114-243 warm hits on the UI loop (53-251 ms in the mounted
    harness; a 294 ms live stall in the 0.2 s sync tick, where each ``lstat``
    gives up the GIL to busy workers). On the thread running an event loop,
    one check now covers the callbacks the loop runs before its next batch,
    which no other code on that thread can interleave with; a rebuild or a
    write installs a new settings object, which is never matched. Every
    caller off a loop -- workers, tests, scripts -- still checks every hit.

    Args:
        active_config_path: The config path the warm hit is for.
        cached: The installed settings object that hit.
        posture: The identity stamped when ``cached`` was read.

    Returns:
        True while the path's identity is the stamped one.
    """
    global _POSTURE_STRETCH
    import asyncio

    loop = asyncio._get_running_loop()
    if loop is None:
        return _config_file_posture(active_config_path) == posture
    if _POSTURE_STRETCH is None:
        _POSTURE_STRETCH = _threading.local()
    stretch = _POSTURE_STRETCH
    seen = getattr(stretch, "seen", None)
    if (
        seen is not None
        and seen[0] is loop
        and seen[1] is cached
        and seen[2] is posture
        and seen[3] == active_config_path
    ):
        return True
    if _config_file_posture(active_config_path) != posture:
        return False
    stretch.seen = (loop, cached, posture, active_config_path)
    loop.call_soon(_end_posture_stretch, stretch)
    return True


def _end_posture_stretch(stretch: Any) -> None:
    """Make the next loop batch re-check the config path's identity."""
    stretch.seen = None


def load_settings(
    force_reload: bool = False,
    *,
    reload_bootstrap: bool | None = None,
) -> Dict:
    """Return the merged application settings, rebuilding at most once.

    PERF-06 (TASK-33265): a warm hit is served here, ahead of the ADR-126
    admission handshake. It is a pure in-memory read and paid ~650
    ``open()`` calls per call (the 4 Hz Console poll, every composer
    keystroke). A miss or forced reload runs the guarded
    :func:`_load_settings_guarded`, exactly as before; this mirrors
    TASK-32804.1's ``_warm_config_cache_hit`` for ``get_cli_setting``.

    Args:
        force_reload: Rebuild even on a cache hit.
        reload_bootstrap: Whether the rebuild also force-reloads the CLI
            bootstrap config from disk. ``None`` (default) follows
            ``force_reload``, preserving the historical behavior. TASK-21124:
            ``_publish_runtime_config_unlocked`` passes ``False`` because it
            has already installed a fresh bootstrap cache under the write
            lock -- re-reading and re-parsing the file it just wrote was one
            of the write path's redundant TOML parses.

    Returns:
        The merged settings mapping.
    """
    if not force_reload:
        # Lock-free by design, like TASK-21124's get_cli_setting fast path: a
        # write holds the config locks through fsyncs and TOML parses, and a
        # loop-side read must not stall behind it. A read overlapping a write
        # may return the pre-write settings; the writer invalidates the cache
        # before it returns, so a read after a completed write sees the new
        # value.
        cached = _settings_cache_hit(_get_effective_config_path())
        if cached is not None:
            return cached
    return _load_settings_guarded(
        force_reload=force_reload,
        reload_bootstrap=reload_bootstrap,
    )


@_config_participants.guarded
def _load_settings_guarded(
    force_reload: bool = False,
    *,
    reload_bootstrap: bool | None = None,
) -> dict:
    """Guarded body of :func:`load_settings`: serialize the cache-miss rebuild.

    Thin wrapper over :func:`_load_settings_uncached` that serializes the
    cache-miss rebuild (task-3503).

    Args:
        force_reload: Rebuild even on a cache hit.
        reload_bootstrap: Whether the rebuild also force-reloads the CLI
            bootstrap config from disk. ``None`` (default) follows
            ``force_reload``, preserving the historical behavior. TASK-21124:
            ``_publish_runtime_config_unlocked`` passes ``False`` because it
            has already installed a fresh bootstrap cache under the write
            lock -- re-reading and re-parsing the file it just wrote was one
            of the write path's redundant TOML parses.

    Returns:
        The merged settings mapping.
    """
    active_config_path = _get_effective_config_path()

    def _cache_hit():
        return _settings_cache_hit(active_config_path)

    if not force_reload:
        cached = _cache_hit()
        if cached is not None:
            return cached

    # Miss: serialize the rebuild. Whoever loses the race re-checks the cache
    # and returns the winner's freshly built settings rather than repeating
    # the entire rebuild.
    with _settings_rebuild_lock():
        if not force_reload:
            cached = _cache_hit()
            if cached is not None:
                logger.debug(
                    "load_settings: returning configuration rebuilt by another thread"
                )
                return cached
        return _load_settings_uncached(
            force_reload=force_reload,
            reload_bootstrap=reload_bootstrap,
        )


@_config_participants.guarded
def _load_settings_uncached(
    force_reload: bool = False,
    *,
    reload_bootstrap: bool | None = None,
) -> Dict:
    """
    Loads all settings from TOML config files, environment variables, or defaults into a dictionary.
    It first loads a base config (e.g., server-local), then attempts to load a user-specific
    CLI config which can override or extend the base settings.

    Args:
        force_reload: If True, bypasses the cache and reloads from disk.
        reload_bootstrap: Whether the CLI bootstrap config is also
            force-reloaded from disk; ``None`` follows ``force_reload``
            (see :func:`load_settings`, TASK-21124).

    Returns:
        Dictionary containing all configuration settings.
    """
    global _SETTINGS_CACHE, _SETTINGS_CACHE_SOURCE, _SETTINGS_CACHE_LOCK
    global _SETTINGS_CACHE_POSTURE
    active_config_path = _get_effective_config_path()
    # Stamped before anything is read: a file swapped mid-read leaves a stale
    # stamp, so the next warm read misses rather than trusting it.
    posture = _config_file_posture(active_config_path)

    # Initialize lock on first use to avoid import issues
    if _SETTINGS_CACHE_LOCK is None:
        import threading

        _SETTINGS_CACHE_LOCK = threading.Lock()

    # Thread-safe cache check
    with _SETTINGS_CACHE_LOCK:
        if (
            _SETTINGS_CACHE is not None
            and _SETTINGS_CACHE_SOURCE == active_config_path
            and _SETTINGS_CACHE_POSTURE == posture
            and not force_reload
        ):
            logger.debug("load_settings: Returning cached configuration (cache hit)")
            return _SETTINGS_CACHE
        _SETTINGS_CACHE = None
        _SETTINGS_CACHE_SOURCE = None

    current_file_path = Path(__file__).resolve()
    # config.py is in project_root/tldw_server_api/app/core/config.py
    ACTUAL_PROJECT_ROOT = current_file_path.parent  # /project_root/
    APP_COMPONENT_ROOT = current_file_path.parent  # /project_root/tldw_server_api/
    logger.info(
        f"Determined ACTUAL_PROJECT_ROOT for general paths: {ACTUAL_PROJECT_ROOT}"
    )
    logger.info(f"Determined APP_COMPONENT_ROOT for config files: {APP_COMPONENT_ROOT}")

    # --- Load Comprehensive Config from TOML ---
    # load_cli_config_and_ensure_existence() already deep-merges
    # DEFAULT_CONFIG_FROM_TOML with the user's CLI config file (creating the
    # file with defaults on first run) and decrypts the result; reuse that
    # single read+parse instead of re-opening the same file here a second
    # time. The historical "primary/server-component config" previously
    # probed at APP_COMPONENT_ROOT/Config_Files/config.toml never exists in
    # the packaged app (no installer/build step writes it, and pyproject.toml
    # only packages *.json/*.md from that directory) so merging it was always
    # a no-op; dropping the probe changes nothing observable.
    bootstrap = _load_cli_config_bootstrap(
        force_reload=(force_reload if reload_bootstrap is None else reload_bootstrap)
    )
    toml_config_data = copy.deepcopy(bootstrap.config)
    # Idempotent no-op when already decrypted (or encryption disabled) --
    # kept so a session password entered *after* the CLI cache above was
    # primed with ciphertext still yields plaintext here. Without this,
    # app.app_config (populated from load_settings) could carry `enc:`
    # ciphertext keys that the Chat send path passes to providers verbatim,
    # failing auth.
    toml_config_data = decrypt_config_section(toml_config_data)
    logger.debug(
        "load_settings: Configuration loaded from disk (cache miss or forced reload)"
    )
    # logger.debug(f"Final toml_config_data after potential merge: {toml_config_data}") # Optional: for verbose debugging

    # --- Extract settings from the (potentially merged) TOML, with fallbacks ---
    # Helper to get values from specific TOML sections within the final toml_config_data
    def get_toml_section(section_name: str, default_val: Optional[Dict] = None) -> Dict:
        return toml_config_data.get(
            section_name, default_val if default_val is not None else {}
        )

    api_section = get_toml_section("API")  # This will now check the merged config
    # If [API] exists in the user's CLI config, it would have merged with/overridden the CLI defaults' [API]
    # Same applies to all other sections retrieved below.

    paths_section = get_toml_section("Paths")
    logging_section_server = get_toml_section("Logging")
    processing_section = get_toml_section("Processing")
    chunking_section = get_toml_section("Chunking")
    embeddings_section = get_toml_section("Embeddings")
    embedding_config_section = get_toml_section(
        "embedding_config"
    )  # Get the [embedding_config] table
    chat_dicts_section = get_toml_section("ChatDictionaries")
    auto_save_section = get_toml_section("AutoSave")
    stt_settings_section = get_toml_section("STTSettings")
    tts_settings_section = get_toml_section("TTSSettings")
    search_engines_section = get_toml_section("SearchEngines")
    search_settings_section = get_toml_section("SearchSettings")
    web_scraper_section = get_toml_section("WebScraper")
    confluence_section = get_toml_section("Confluence")
    library_section = get_toml_section("library")

    final_api_settings = get_toml_section("api_settings")
    final_logging_settings = get_toml_section("logging")
    final_providers_settings = get_toml_section("providers")
    final_general_settings_cli = get_toml_section("general")
    final_database_settings_cli = get_toml_section("database")
    final_model_catalog_settings_cli = get_toml_section("model_catalog")
    final_chat_defaults_cli = copy.deepcopy(get_toml_section("chat_defaults"))
    if not isinstance(final_chat_defaults_cli, dict):
        final_chat_defaults_cli = {}
    final_character_defaults_cli = get_toml_section("character_defaults")
    final_notes_settings_cli = get_toml_section("notes")
    # (task 11, spec §9.1/AC 40) The [chunking] table -- the config tier of
    # the ingest template resolution order (``[chunking] default_template``).
    # Distinct from the legacy CamelCase "Chunking" server section above.
    final_chunking_settings_cli = get_toml_section("chunking")
    # (task-34389, ADR-229) The [sync] table -- the sync-journal retention
    # window the startup maintenance pass reads.
    final_sync_settings_cli = get_toml_section("sync")
    final_image_generation_settings_cli = get_toml_section("image_generation")
    final_video_generation_settings_cli = get_toml_section("video_generation")
    # F-E fix: the first-run wizard's own state (setup_started/setup_completed)
    # lives under [first_run] in the raw TOML and was never projected through
    # here -- every other section the app reads via app_config (chat_defaults,
    # notes, console, ...) is listed below, but first_run was simply absent,
    # so should_offer_wizard()/should_show_resume_toast() (which read
    # app_config, populated from THIS function's return value, not the raw
    # loader) never saw a completed/started run and the wizard re-offered on
    # every launch even after real completion.
    final_first_run_settings_cli = get_toml_section("first_run")
    # Console run hooks (spec 2026-09-11, Task 4): the RAW [hooks] table
    # passes through untouched. Same shape as the first_run projection
    # above -- a section only one consumer reads, and that consumer
    # (Agents/run_hooks.py load_hooks_config) re-validates on every fire,
    # so coercing here would only duplicate that logic. Without this
    # projection the runtime could not reach [hooks] at all: the dict
    # returned below is curated, not the parsed TOML.
    final_hooks_settings_cli = get_toml_section("hooks")
    final_console_settings_cli = copy.deepcopy(get_toml_section("console"))
    if not isinstance(final_console_settings_cli, dict):
        final_console_settings_cli = {}
    final_console_settings_cli["collapse_large_pastes"] = coerce_bool_setting(
        final_console_settings_cli.get("collapse_large_pastes", True),
        True,
    )
    final_console_settings_cli["stack_collapsed_rail_labels"] = coerce_bool_setting(
        final_console_settings_cli.get("stack_collapsed_rail_labels", False),
        False,
    )
    from tldw_chatbook.Utils.reasoning_config import resolve_console_reasoning_config

    reasoning_resolution = resolve_console_reasoning_config(final_console_settings_cli)
    for diagnostic in reasoning_resolution.diagnostics:
        logger.warning("Invalid Console reasoning configuration: {}", diagnostic)
    final_console_settings_cli.update(reasoning_resolution.settings.model_dump())
    _rail_layout_scope = final_console_settings_cli.get("rail_layout_scope")
    final_console_settings_cli["rail_layout_scope"] = (
        _rail_layout_scope.strip().lower()
        if isinstance(_rail_layout_scope, str)
        and _rail_layout_scope.strip().lower() in {"global", "workspace"}
        else "global"
    )
    # task-17652: status-row placement relative to the composer. Validation
    # lives in UI/Console_Modules/status_row.resolve_status_chips_position;
    # normalizing here keeps a typo from ever reaching compose order.
    _status_chips_position = final_console_settings_cli.get("status_chips_position")
    if not (
        isinstance(_status_chips_position, str)
        and _status_chips_position.strip().lower() in ("above", "below")
    ):
        _status_chips_position = "above"
    else:
        _status_chips_position = _status_chips_position.strip().lower()
    final_console_settings_cli["status_chips_position"] = _status_chips_position
    final_console_settings_cli["paste_collapse_threshold"] = coerce_int_setting(
        final_console_settings_cli.get(
            "paste_collapse_threshold",
            DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
        ),
        DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
        minimum=MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
        maximum=MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    )
    # TASK-870: same coercion shape as paste_collapse_threshold above.
    final_console_settings_cli["tool_result_display_chars"] = coerce_int_setting(
        final_console_settings_cli.get(
            "tool_result_display_chars",
            DEFAULT_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
        ),
        DEFAULT_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
        minimum=MIN_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
        maximum=MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS,
    )
    # TASK-18600: the Console agent run budget. Same coercion shape as the
    # two settings above; the two duration keys go through the float twin.
    # Floors only, no ceilings -- these are deliberately user-owned
    # trade-offs (same call as `max_parallel_runs`), and an out-of-range or
    # unparsable value falls back to the shipped default rather than
    # silently clamping to a number the user never chose.
    final_console_settings_cli["agent_max_model_turns"] = coerce_int_setting(
        final_console_settings_cli.get(
            "agent_max_model_turns",
            DEFAULT_CONSOLE_AGENT_MAX_MODEL_TURNS,
        ),
        DEFAULT_CONSOLE_AGENT_MAX_MODEL_TURNS,
        minimum=MIN_CONSOLE_AGENT_MAX_MODEL_TURNS,
    )
    final_console_settings_cli["agent_max_steps"] = coerce_int_setting(
        final_console_settings_cli.get(
            "agent_max_steps",
            DEFAULT_CONSOLE_AGENT_MAX_STEPS,
        ),
        DEFAULT_CONSOLE_AGENT_MAX_STEPS,
        minimum=MIN_CONSOLE_AGENT_MAX_STEPS,
        maximum=MAX_CONSOLE_AGENT_MAX_STEPS,
    )
    final_console_settings_cli["agent_max_wall_seconds"] = coerce_float_setting(
        final_console_settings_cli.get(
            "agent_max_wall_seconds",
            DEFAULT_CONSOLE_AGENT_MAX_WALL_SECONDS,
        ),
        DEFAULT_CONSOLE_AGENT_MAX_WALL_SECONDS,
        minimum=MIN_CONSOLE_AGENT_MAX_WALL_SECONDS,
    )
    final_console_settings_cli["agent_max_total_tokens"] = coerce_int_setting(
        final_console_settings_cli.get(
            "agent_max_total_tokens",
            DEFAULT_CONSOLE_AGENT_MAX_TOTAL_TOKENS,
        ),
        DEFAULT_CONSOLE_AGENT_MAX_TOTAL_TOKENS,
        minimum=MIN_CONSOLE_AGENT_MAX_TOTAL_TOKENS,
    )
    final_console_settings_cli["agent_max_tool_call_seconds"] = coerce_float_setting(
        final_console_settings_cli.get(
            "agent_max_tool_call_seconds",
            DEFAULT_CONSOLE_AGENT_MAX_TOOL_CALL_SECONDS,
        ),
        DEFAULT_CONSOLE_AGENT_MAX_TOOL_CALL_SECONDS,
        minimum=MIN_CONSOLE_AGENT_MAX_TOOL_CALL_SECONDS,
    )
    final_console_settings_cli["local_tools_enabled"] = coerce_bool_setting(
        final_console_settings_cli.get("local_tools_enabled", True),
        True,
    )
    final_console_settings_cli["raw_cli_permitted"] = coerce_bool_setting(
        final_console_settings_cli.get("raw_cli_permitted", False),
        False,
    )
    for key in (
        "project_instructions_startup_max_bytes",
        "project_instructions_nested_max_bytes",
    ):
        final_console_settings_cli[key] = coerce_int_setting(
            final_console_settings_cli.get(
                key, DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES
            ),
            DEFAULT_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
            minimum=MIN_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
            maximum=MAX_CONSOLE_PROJECT_INSTRUCTIONS_MAX_BYTES,
        )
    workspace_root = final_console_settings_cli.get("workspace_root", "")
    if not isinstance(workspace_root, str):
        workspace_root = ""
    final_console_settings_cli["workspace_root"] = workspace_root.strip()
    # Ephemeral side chat (selection menu): strings need presence-validation
    # only -- non-strings fall back, mirroring workspace_root above.
    sidechat_model = final_console_settings_cli.get("sidechat_model", "")
    if not isinstance(sidechat_model, str):
        sidechat_model = ""
    final_console_settings_cli["sidechat_model"] = sidechat_model.strip()
    sidechat_prompt_template = final_console_settings_cli.get(
        "sidechat_prompt_template", DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE
    )
    if not isinstance(sidechat_prompt_template, str):
        sidechat_prompt_template = DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE
    final_console_settings_cli["sidechat_prompt_template"] = (
        sidechat_prompt_template.strip()
    )
    background_effects = final_console_settings_cli.get("background_effects")
    if not isinstance(background_effects, dict):
        background_effects = {}
    final_console_settings_cli["background_effects"] = (
        normalize_console_background_effects(background_effects).to_config()
    )

    # --- MCP Settings ---
    final_mcp_settings_cli = copy.deepcopy(get_toml_section("mcp"))
    if not isinstance(final_mcp_settings_cli, dict):
        final_mcp_settings_cli = {}
    for mcp_switch in ("expose_local_tools", "expose_character_tools"):
        final_mcp_settings_cli[mcp_switch] = coerce_bool_setting(
            final_mcp_settings_cli.get(mcp_switch, False),
            False,
        )

    # --- Application Mode ---
    single_user_mode_str = os.getenv(
        "APP_MODE", _get_typed_value(processing_section, "app_mode", "single")
    ).lower()
    single_user_mode = single_user_mode_str != "multi"

    # --- Single-User Settings ---
    single_user_fixed_id = int(
        os.getenv(
            "SINGLE_USER_FIXED_ID",
            _get_typed_value(processing_section, "single_user_fixed_id", "0", int),
        )
    )
    os.getenv(
        "API_KEY",
        _get_typed_value(
            api_section, "single_user_api_key", "default-secret-key-for-single-user"
        ),
    )

    # --- Paths ---
    api_section_legacy = get_toml_section(
        "API"
    )  # For legacy direct API key access if any
    paths_section_legacy = get_toml_section("Paths")
    get_toml_section("Processing")
    get_toml_section("Chunking")

    # --- User Name ---
    default_users_name_fallback = "default_user"
    users_name_from_toml_general = _get_typed_value(
        final_general_settings_cli, "users_name", default_users_name_fallback, str
    )
    users_name = os.getenv("USERS_NAME", users_name_from_toml_general)

    users_db_configured = (
        os.getenv(
            "USERS_DB_ENABLED",
            _get_typed_value(processing_section, "users_db_enabled", "false", str),
        ).lower()
        == "true"
    )
    log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
    _get_typed_value(logging_section_server, "log_level", log_level_env, str).upper()

    # --- Load specific configurations from TOML or use defaults ---
    app_tts_config = get_toml_section("AppTTSConfig")  # For APP_CONFIG related values
    app_database_config = get_toml_section("AppDatabaseConfig")  # For DATABASE_CONFIG
    app_rag_search_config = get_toml_section(
        "AppRAGSearchConfig"
    )  # For RAG_SEARCH_CONFIG

    # API Keys (Prioritize ENV, then TOML, then None)
    def get_api_key(
        toml_key: str, env_var: str, section: Dict = api_section_legacy
    ) -> Optional[str]:
        return os.getenv(env_var, section.get(toml_key))

    # PR-T2 Task 7: these 9 providers' legacy `[API] <provider>_api_key`
    # value and `api_settings.<provider>.api_key` are resolved together by
    # `_normalize_legacy_provider_api_key` -- ONE normalization -- so
    # `final_api_settings` (below, "api_settings" in config_dict, what
    # `get_provider_readiness` reads) and this same value (below, the
    # `<provider>_api` dicts `chat_with_<provider>` spends through) can
    # never disagree about the same config. `elevenlabs` is untouched: it
    # is a TTS credential, not a Chat provider `get_provider_readiness`
    # resolves.
    _legacy_api_key_providers: set[str] = set()
    _bridged_provider_api_keys = {
        provider_key: _normalize_legacy_provider_api_key(
            final_api_settings,
            provider_key,
            raw_api_section=api_section_legacy,
            toml_key=toml_key,
            env_var=env_var,
            bridged_from_legacy=_legacy_api_key_providers,
        )
        for provider_key, toml_key, env_var in _LEGACY_PROVIDER_API_KEY_BRIDGE
    }
    openai_api_key = _bridged_provider_api_keys["openai"]
    anthropic_api_key = _bridged_provider_api_keys["anthropic"]
    cohere_api_key = _bridged_provider_api_keys["cohere"]
    groq_api_key = _bridged_provider_api_keys["groq"]
    huggingface_api_key = _bridged_provider_api_keys["huggingface"]
    openrouter_api_key = _bridged_provider_api_keys["openrouter"]
    deepseek_api_key = _bridged_provider_api_keys["deepseek"]
    mistral_api_key = _bridged_provider_api_keys["mistral"]
    elevenlabs_api_key = get_api_key("elevenlabs_api_key", "ELEVENLABS_API_KEY")

    # Determine platform-specific default STT provider
    default_stt_provider = _default_stt_provider_for_platform()
    if sys.platform == "darwin":
        logger.debug(
            f"Darwin platform-preferred STT provider resolved to: {default_stt_provider}"
        )

    # TASK-22223: `normalize_adaptive_reader_preferences` comes from the
    # stdlib-only leaf `Utils/adaptive_reader_state.py` (module-top import).
    # This function runs at config-module import (`load_settings()` at module
    # scope), so NOTHING here may import a feature package -- a previous
    # `Library.library_adaptive_reader_state` import claimed to be lazy but
    # executed the whole Library `__init__` service stack on every config
    # import and closed a live cycle through `runtime_policy.bootstrap`.
    # Guarded by `Tests/Packaging/test_config_import_closure.py`; share logic
    # with features through config-safe leaf modules only.
    legacy_media_reader = (
        library_section.get("media_reader", {})
        if isinstance(library_section.get("media_reader"), Mapping)
        else {}
    )
    raw_reader = (
        library_section.get("reader", {})
        if isinstance(library_section.get("reader"), Mapping)
        else {}
    )
    shared_raw = {
        key: os.getenv(
            f"TLDW_LIBRARY_READER_{key.upper()}",
            raw_reader.get(key, legacy_media_reader.get(key)),
        )
        for key in ("library_open", "custom_widths_enabled", "library_width")
    }
    shared_preferences = normalize_adaptive_reader_preferences(shared_raw)
    shared_width = normalize_adaptive_reader_preferences(
        {**shared_raw, "custom_widths_enabled": True}
    ).library_width
    normalized_reader = {
        **copy.deepcopy(raw_reader),
        "library_open": shared_preferences.library_open,
        "custom_widths_enabled": shared_preferences.custom_widths_enabled,
        "library_width": shared_width,
    }
    normalized_destination_readers: dict[str, dict[str, Any]] = {}
    for section_name in (
        "media_reader",
        "collections_reader",
        "conversations_reader",
        "notes_reader",
        "prompts_reader",
        "skills_reader",
        "artifacts_reader",
    ):
        raw_destination = (
            library_section.get(section_name, {})
            if isinstance(library_section.get(section_name), Mapping)
            else {}
        )
        destination_preferences = normalize_adaptive_reader_preferences(
            {
                "custom_widths_enabled": True,
                "items_open": os.getenv(
                    f"TLDW_LIBRARY_{section_name.upper()}_ITEMS_OPEN",
                    raw_destination.get("items_open"),
                ),
                "items_width": os.getenv(
                    f"TLDW_LIBRARY_{section_name.upper()}_ITEMS_WIDTH",
                    raw_destination.get("items_width"),
                ),
            }
        )
        normalized_destination_readers[section_name] = {
            **copy.deepcopy(raw_destination),
            "items_open": destination_preferences.items_open,
            "items_width": destination_preferences.items_width,
        }
        if section_name == "notes_reader":
            files_tree_width = os.getenv(
                "TLDW_LIBRARY_NOTES_READER_FILES_TREE_WIDTH",
                raw_destination.get("files_tree_width"),
            )
            try:
                parsed_files_tree_width = (
                    int(files_tree_width.strip())
                    if isinstance(files_tree_width, str)
                    else files_tree_width
                )
            except ValueError:
                parsed_files_tree_width = ITEMS_TARGET_WIDTH
            if (
                type(parsed_files_tree_width) is not int
                or not ITEMS_MIN_WIDTH <= parsed_files_tree_width <= ITEMS_MAX_WIDTH
            ):
                parsed_files_tree_width = ITEMS_TARGET_WIDTH
            files_tree_preferences = normalize_adaptive_reader_preferences(
                {
                    "custom_widths_enabled": True,
                    "items_open": os.getenv(
                        "TLDW_LIBRARY_NOTES_READER_FILES_TREE_OPEN",
                        raw_destination.get("files_tree_open"),
                    ),
                    "items_width": parsed_files_tree_width,
                }
            )
            normalized_destination_readers[section_name].update(
                {
                    "files_tree_open": files_tree_preferences.items_open,
                    "files_tree_width": files_tree_preferences.items_width,
                }
            )
    config_dict = {
        # General App
        "APP_MODE_STR": single_user_mode_str,
        "SINGLE_USER_MODE": single_user_mode,
        "LOG_LEVEL": final_logging_settings.get("file_log_level", "INFO").upper(),
        "PROJECT_ROOT": ACTUAL_PROJECT_ROOT,
        "API_COMPONENT_ROOT": APP_COMPONENT_ROOT,
        "USERS_NAME": users_name,
        # --- Pass through the full tables ---
        "general": final_general_settings_cli,  # For TUI settings like default_tab
        "logging": final_logging_settings,  # For TUI log settings like log_max_bytes
        "database": final_database_settings_cli,  # For TUI DB paths
        "api_settings": final_api_settings,  # CRUCIAL for local API calls
        # TASK-33001.15: provider tables whose `api_key` above is not in the
        # table on disk -- the loader copied it from legacy `[API]`. Names
        # only, always present; a Settings save that records "stored" for one
        # of them writes the key into the table (provider_setup_persistence).
        "legacy_api_key_providers": tuple(sorted(_legacy_api_key_providers)),
        "providers": final_providers_settings,  # For UI dropdowns
        "model_catalog": final_model_catalog_settings_cli,
        "chat_defaults": final_chat_defaults_cli,
        "character_defaults": final_character_defaults_cli,
        "appearance": copy.deepcopy(toml_config_data.get("appearance", {})),
        "notes": final_notes_settings_cli,  # For notes auto-save settings
        "chunking": final_chunking_settings_cli,  # Template default for ingest (§9.1)
        "sync": final_sync_settings_cli,  # Sync-journal retention (task-34389, ADR-229)
        "console": final_console_settings_cli,  # For Console behavior settings
        "first_run": final_first_run_settings_cli,  # Wizard setup_started/setup_completed flags
        "hooks": final_hooks_settings_cli,  # Raw [hooks] table for Agents/run_hooks.py (Console run hooks)
        "image_generation": final_image_generation_settings_cli,  # For Image_Generation/config.py loader
        "video_generation": final_video_generation_settings_cli,  # For Video_Generation/config.py loader
        "mcp": final_mcp_settings_cli,  # For MCP server settings
        "persona_buddy": copy.deepcopy(toml_config_data.get("persona_buddy", {})),
        **(
            {"buddy_interaction": copy.deepcopy(toml_config_data["buddy_interaction"])}
            if "buddy_interaction" in toml_config_data
            else {}
        ),
        # [skills] disabled_builtins is read from app_config on every skills
        # read (TASK-32954); without this a disabled built-in returned on restart.
        "skills": copy.deepcopy(toml_config_data.get("skills", {})),
        "agents": copy.deepcopy(toml_config_data.get("agents", {})),
        "tools": copy.deepcopy(toml_config_data.get("tools", {})),
        # [analysis_defaults] is the Library's analysis provider/model/call
        # shape: the Media reader's Generate, bulk Analyze and
        # Analyze-after-import all resolve it from app_config through
        # Library/ingest_analysis.py, and the Media viewer's analysis panel
        # seeds its inputs from it. It was never carried over, so analysis
        # reported "no provider configured" for every user (TASK-34000.5,
        # review finding L-02). Tests/test_config_load_settings_table_guard.py
        # fails on the next table a reader needs that this dict drops.
        "analysis_defaults": copy.deepcopy(
            toml_config_data.get("analysis_defaults", {})
        ),
        # Single User
        "SINGLE_USER_FIXED_ID": single_user_fixed_id,
        # Auth
        "SINGLE_USER_API_KEY": get_api_key(
            "single_user_api_key", "API_KEY", section=api_section_legacy
        )
        or "default-secret-key-for-single-user",
        "DATABASE_URL": os.getenv(
            "DATABASE_URL",
            paths_section_legacy.get(
                "database_url",
                f"sqlite:///{ACTUAL_PROJECT_ROOT / 'user_databases' / 'single_user' / 'tldw.db'}",
            ),
        ),
        "USERS_DB_CONFIGURED": users_db_configured,
        # --- Configurations migrated from load_and_log_configs ---
        "anthropic_api": {
            "api_key": anthropic_api_key,
            "model": api_section_legacy.get("anthropic_model", "claude-sonnet-5"),
            "streaming": api_section_legacy.get("anthropic_streaming", False),
            "temperature": api_section_legacy.get("anthropic_temperature", 0.7),
            "top_p": api_section_legacy.get("anthropic_top_p", 0.95),
            "top_k": api_section_legacy.get("anthropic_top_k", 100),
            "max_tokens": api_section_legacy.get("anthropic_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("anthropic_api_timeout", 90),
            "api_retries": api_section_legacy.get(
                "anthropic_api_retry", 3
            ),  # Key name consistency
            "api_retry_delay": api_section_legacy.get("anthropic_api_retry_delay", 5),
        },
        "cohere_api": {
            "api_key": cohere_api_key,
            "model": api_section_legacy.get("cohere_model", "command-a-03-2025"),
            "streaming": api_section_legacy.get("cohere_streaming", False),
            "temperature": api_section_legacy.get("cohere_temperature", 0.7),
            "max_p": api_section_legacy.get(
                "cohere_max_p", 0.95
            ),  # Note: check param name, Cohere might use 'p' or 'top_p'
            "top_k": api_section_legacy.get("cohere_top_k", 100),
            "max_tokens": api_section_legacy.get("cohere_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("cohere_api_timeout", 90),
            "api_retries": api_section_legacy.get("cohere_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("cohere_api_retry_delay", 5),
        },
        "deepseek_api": {
            "api_key": deepseek_api_key,
            "model": api_section_legacy.get("deepseek_model", "deepseek-v4-flash"),
            "streaming": api_section_legacy.get("deepseek_streaming", False),
            "temperature": api_section_legacy.get("deepseek_temperature", 0.7),
            "top_p": api_section_legacy.get("deepseek_top_p", 0.95),
            "min_p": api_section_legacy.get("deepseek_min_p", 0.05),
            "max_tokens": api_section_legacy.get("deepseek_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("deepseek_api_timeout", 90),
            "api_retries": api_section_legacy.get("deepseek_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("deepseek_api_retry_delay", 5),
        },
        "groq_api": {
            "api_key": groq_api_key,
            "model": api_section_legacy.get("groq_model", "llama-3.3-70b-versatile"),
            "streaming": api_section_legacy.get("groq_streaming", False),
            "temperature": api_section_legacy.get("groq_temperature", 0.7),
            "top_p": api_section_legacy.get("groq_top_p", 0.95),
            "max_tokens": api_section_legacy.get("groq_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("groq_api_timeout", 90),
            "api_retries": api_section_legacy.get("groq_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("groq_api_retry_delay", 5),
        },
        "huggingface_api": {
            "api_key": huggingface_api_key,
            "use_router_url_format": api_section_legacy.get(
                "huggingface_use_router_url_format", False
            ),
            "router_base_url": api_section_legacy.get(
                "huggingface_router_base_url",
                "https://router.huggingface.co/hf-inference",
            ),
            "api_base_url": api_section_legacy.get(
                "huggingface_api_base_url", "https://router.huggingface.co/v1"
            ),
            "api_chat_path": api_section_legacy.get(
                "huggingface_api_chat_path", "chat/completions"
            ),
            "model": api_section_legacy.get("huggingface_model", "openai/gpt-oss-120b"),
            "streaming": api_section_legacy.get("huggingface_streaming", False),
            "temperature": api_section_legacy.get("huggingface_temperature", 0.7),
            "top_p": api_section_legacy.get("huggingface_top_p", 0.95),
            "min_p": api_section_legacy.get("huggingface_min_p", 0.05),
            "max_tokens": api_section_legacy.get("huggingface_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("huggingface_api_timeout", 90),
            "api_retries": api_section_legacy.get("huggingface_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("huggingface_api_retry_delay", 5),
        },
        "mistral_api": {
            "api_key": mistral_api_key,
            "model": api_section_legacy.get("mistral_model", "mistral-large-latest"),
            "streaming": api_section_legacy.get("mistral_streaming", False),
            "temperature": api_section_legacy.get("mistral_temperature", 0.7),
            "top_p": api_section_legacy.get("mistral_top_p", 0.95),
            "max_tokens": api_section_legacy.get("mistral_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("mistral_api_timeout", 90),
            "api_retries": api_section_legacy.get("mistral_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("mistral_api_retry_delay", 5),
        },
        "openrouter_api": {
            "api_key": openrouter_api_key,
            "model": api_section_legacy.get(
                "openrouter_model", "microsoft/wizardlm-2-8x22b"
            ),
            "streaming": api_section_legacy.get("openrouter_streaming", False),
            "temperature": api_section_legacy.get("openrouter_temperature", 0.7),
            "top_p": api_section_legacy.get("openrouter_top_p", 0.95),
            "min_p": api_section_legacy.get("openrouter_min_p", 0.05),
            "top_k": api_section_legacy.get("openrouter_top_k", 100),
            "max_tokens": api_section_legacy.get("openrouter_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("openrouter_api_timeout", 90),
            "api_retries": api_section_legacy.get("openrouter_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("openrouter_api_retry_delay", 5),
        },
        "openai_api": {  # OpenAI specific model params, API key is separate
            "api_key": openai_api_key,  # This is now the primary OpenAI API key
            "model": api_section_legacy.get("openai_model", "gpt-5.6-terra"),
            "streaming": api_section_legacy.get("openai_streaming", False),
            "temperature": api_section_legacy.get("openai_temperature", 0.7),
            "top_p": api_section_legacy.get("openai_top_p", 0.95),
            "max_tokens": api_section_legacy.get("openai_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("openai_api_timeout", 90),
            "api_retries": api_section_legacy.get("openai_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("openai_api_retry_delay", 5),
        },
        "elevenlabs_api": {  # Primarily for the API key, other settings in TTS
            "api_key": elevenlabs_api_key,
        },
        # Local APIs from LocalAPI section
        "kobold_api": {
            "api_ip": api_section_legacy.get(
                "kobold_api_IP", "http://127.0.0.1:5000/api/v1/generate"
            ),
            "api_streaming_ip": api_section_legacy.get(
                "kobold_openai_api_IP", "http://127.0.0.1:5001/v1/chat/completions"
            ),
            "api_key": api_section_legacy.get("kobold_api_key", ""),
            "streaming": api_section_legacy.get("kobold_streaming", False),
            "api_timeout": api_section_legacy.get("kobold_api_timeout", 90),
            "api_retries": api_section_legacy.get("kobold_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("kobold_api_retry_delay", 5),
        },
        "llama_cpp_api": {  # Renamed for clarity, assuming llama.cpp server
            "api_ip": api_section_legacy.get(
                "llama_api_IP", "http://127.0.0.1:8080/v1/chat/completions"
            ),
            "api_key": api_section_legacy.get("llama_api_key", ""),
            "streaming": api_section_legacy.get("llama_streaming", False),
            "temperature": api_section_legacy.get("llama_temperature", 0.7),
            "top_p": api_section_legacy.get("llama_top_p", 0.95),
            "min_p": api_section_legacy.get("llama_min_p", 0.05),
            "top_k": api_section_legacy.get("llama_top_k", 100),
            "max_tokens": api_section_legacy.get("llama_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("llama_api_timeout", 90),
            "api_retries": api_section_legacy.get("llama_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("llama_api_retry_delay", 5),
        },
        "ooba_api": {
            "api_ip": api_section_legacy.get(
                "ooba_api_IP", "http://127.0.0.1:5000/v1/chat/completions"
            ),
            "api_key": api_section_legacy.get("ooba_api_key", ""),
            "streaming": api_section_legacy.get("ooba_streaming", False),
            "temperature": api_section_legacy.get("ooba_temperature", 0.7),
            "top_p": api_section_legacy.get("ooba_top_p", 0.95),
            "min_p": api_section_legacy.get("ooba_min_p", 0.05),
            "top_k": api_section_legacy.get("ooba_top_k", 100),
            "max_tokens": api_section_legacy.get("ooba_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("ooba_api_timeout", 90),
            "api_retries": api_section_legacy.get("ooba_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("ooba_api_retry_delay", 5),
        },
        "tabby_api": {
            "api_ip": api_section_legacy.get(
                "tabby_api_IP", "http://127.0.0.1:5000/api/v1/generate"
            ),
            "api_key": api_section_legacy.get("tabby_api_key", None),
            "model": api_section_legacy.get(
                "tabby_model", None
            ),  # Tabby model might be part of URL or configured in Tabby
            "streaming": api_section_legacy.get("tabby_streaming", False),
            "temperature": api_section_legacy.get("tabby_temperature", 0.7),
            "top_p": api_section_legacy.get("tabby_top_p", 0.95),
            "top_k": api_section_legacy.get("tabby_top_k", 100),
            "min_p": api_section_legacy.get("tabby_min_p", 0.05),
            "max_tokens": api_section_legacy.get("tabby_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("tabby_api_timeout", 90),
            "api_retries": api_section_legacy.get("tabby_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("tabby_api_retry_delay", 5),
        },
        "vllm_api": {
            "api_ip": api_section_legacy.get(
                "vllm_api_IP", "http://127.0.0.1:5000/v1/chat/completions"
            ),  # Corrected key
            "api_key": api_section_legacy.get("vllm_api_key", None),
            "model": api_section_legacy.get("vllm_model", None),
            "streaming": api_section_legacy.get("vllm_streaming", False),
            "temperature": api_section_legacy.get("vllm_temperature", 0.7),
            "top_p": api_section_legacy.get("vllm_top_p", 0.95),
            "top_k": api_section_legacy.get("vllm_top_k", 100),
            "min_p": api_section_legacy.get("vllm_min_p", 0.05),
            "max_tokens": api_section_legacy.get("vllm_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("vllm_api_timeout", 90),
            "api_retries": api_section_legacy.get("vllm_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("vllm_api_retry_delay", 5),
        },
        "ollama_api": {
            "api_url": api_section_legacy.get(
                "ollama_api_IP", "http://127.0.0.1:11434/api/generate"
            ),  # ollama_api_url or IP
            "api_key": api_section_legacy.get(
                "ollama_api_key", None
            ),  # Ollama doesn't typically use API keys
            "model": api_section_legacy.get("ollama_model", None),
            "streaming": api_section_legacy.get("ollama_streaming", False),
            "temperature": api_section_legacy.get("ollama_temperature", 0.7),
            "top_p": api_section_legacy.get("ollama_top_p", 0.95),
            "max_tokens": api_section_legacy.get(
                "ollama_max_tokens", 4096
            ),  # Ollama might handle max_tokens differently (num_predict)
            "api_timeout": api_section_legacy.get("ollama_api_timeout", 90),
            "api_retries": api_section_legacy.get("ollama_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("ollama_api_retry_delay", 5),
        },
        "aphrodite_api": {
            "api_ip": api_section_legacy.get(
                "aphrodite_api_IP", "http://127.0.0.1:8080/v1/chat/completions"
            ),
            "api_key": api_section_legacy.get("aphrodite_api_key", ""),
            "model": api_section_legacy.get("aphrodite_model", ""),
            "max_tokens": api_section_legacy.get("aphrodite_max_tokens", 4096),
            "streaming": api_section_legacy.get("aphrodite_streaming", False),
            "api_timeout": api_section_legacy.get(
                "aphrodite_api_timeout", 90
            ),  # Original used llama_api_timeout
            "api_retries": api_section_legacy.get("aphrodite_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("aphrodite_api_retry_delay", 5),
        },
        "custom_openai_api": {
            "api_ip": api_section_legacy.get(
                "custom_openai_api_ip", "http://127.0.0.1:5000/v1/chat/completions"
            ),
            "api_key": api_section_legacy.get("custom_openai_api_key", None),
            "model": api_section_legacy.get("custom_openai_api_model", None),
            "streaming": api_section_legacy.get("custom_openai_api_streaming", False),
            "temperature": api_section_legacy.get("custom_openai_api_temperature", 0.7),
            "top_p": api_section_legacy.get("custom_openai_api_top_p", 0.95),
            "min_p": api_section_legacy.get(
                "custom_openai_api_min_p", 0.05
            ),  # Original used top_k, ensure consistency
            "max_tokens": api_section_legacy.get("custom_openai_api_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("custom_openai_api_timeout", 90),
            "api_retries": api_section_legacy.get("custom_openai_api_retry", 3),
            "api_retry_delay": api_section_legacy.get(
                "custom_openai_api_retry_delay", 5
            ),
        },
        "custom_openai_api_2": {  # Ensure key names are consistent e.g. custom_openai2_api_min_p
            "api_ip": api_section_legacy.get(
                "custom_openai2_api_ip", "http://127.0.0.1:5000/v1/chat/completions"
            ),
            "api_key": api_section_legacy.get("custom_openai2_api_key", None),
            "model": api_section_legacy.get("custom_openai2_api_model", None),
            "streaming": api_section_legacy.get("custom_openai2_api_streaming", False),
            "temperature": api_section_legacy.get(
                "custom_openai2_api_temperature", 0.7
            ),
            "top_p": api_section_legacy.get(
                "custom_openai2_api_top_p", 0.95
            ),  # original had custom_openai_api2_top_p
            "min_p": api_section_legacy.get(
                "custom_openai2_api_min_p", 0.05
            ),  # original had custom_openai_api2_top_k
            "max_tokens": api_section_legacy.get("custom_openai2_api_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("custom_openai2_api_timeout", 90),
            "api_retries": api_section_legacy.get("custom_openai2_api_retry", 3),
            "api_retry_delay": api_section_legacy.get(
                "custom_openai2_api_retry_delay", 5
            ),
        },
        "llm_api_settings": {  # General LLM settings
            "default_api": api_section_legacy.get("default_api", "openai"),
            "local_api_timeout": api_section_legacy.get(
                "local_api_timeout", 90
            ),  # Note: this was also in Local-API Settings before
            "local_api_retries": api_section_legacy.get(
                "local_api_retry", 3
            ),  # Key name consistency
            "local_api_retry_delay": api_section_legacy.get("local_api_retry_delay", 5),
        },
        "output_path": _get_typed_value(paths_section, "output_path", "results", Path),
        "system_preferences": {
            "save_video_transcripts": _get_typed_value(
                paths_section, "save_video_transcripts", True, bool
            ),
        },
        "processing_choice": _get_typed_value(
            processing_section, "processing_choice", "cpu"
        ),
        "chat_dictionaries": {
            "enable_chat_dictionaries": _get_typed_value(
                chat_dicts_section, "enable_chat_dictionaries", False, bool
            ),
            "post_gen_replacement": _get_typed_value(
                chat_dicts_section, "post_gen_replacement", False, bool
            ),
            "post_gen_replacement_dict": _get_typed_value(
                chat_dicts_section, "post_gen_replacement_dict", ""
            ),
            "chat_dict_chat_prompts": _get_typed_value(
                chat_dicts_section, "chat_dictionary_chat_prompts", ""
            ),
            "chat_dict_RAG_prompts": _get_typed_value(
                chat_dicts_section, "chat_dictionary_RAG_prompts", ""
            ),
            "chat_dict_replacement_strategy": _get_typed_value(
                chat_dicts_section,
                "chat_dictionary_replacement_strategy",
                "character_lore_first",
            ),
            "chat_dict_max_tokens": _get_typed_value(
                chat_dicts_section, "chat_dictionary_max_tokens", 1000, int
            ),
            "default_rag_prompt": _get_typed_value(
                chat_dicts_section, "default_rag_prompt", ""
            ),
            "chat_dicts_folder": "",  # Will be set dynamically below
        },
        "chunking_config": {
            # Global defaults
            "chunking_method": _get_typed_value(
                chunking_section, "chunking_method", "words"
            ),
            "chunk_max_size": _get_typed_value(
                chunking_section, "chunk_max_size", 400, int
            ),
            "chunk_overlap": _get_typed_value(
                chunking_section, "chunk_overlap", 200, int
            ),
            "adaptive_chunking": _get_typed_value(
                chunking_section, "adaptive_chunking", False, bool
            ),
            "multi_level": _get_typed_value(
                chunking_section, "chunking_multi_level", False, bool
            ),
            "chunk_language": _get_typed_value(
                chunking_section, "chunk_language", global_default_chunk_language
            ),  # Use global default
            # Per-type overrides (example for article, repeat for others: audio, book, etc.)
            "article_chunking_method": _get_typed_value(
                chunking_section, "article_chunking_method", "words"
            ),
            "article_chunk_max_size": _get_typed_value(
                chunking_section, "article_chunk_max_size", 400, int
            ),
            "article_chunk_overlap": _get_typed_value(
                chunking_section, "article_chunk_overlap", 200, int
            ),
            "article_adaptive_chunking": _get_typed_value(
                chunking_section, "article_adaptive_chunking", False, bool
            ),
            "article_chunking_multi_level": _get_typed_value(
                chunking_section, "article_chunking_multi_level", False, bool
            ),
            "article_language": _get_typed_value(
                chunking_section, "article_language", "en"
            ),
            "audio_chunking_method": _get_typed_value(
                chunking_section, "audio_chunking_method", "words"
            ),
            "audio_chunk_max_size": _get_typed_value(
                chunking_section, "audio_chunk_max_size", 400, int
            ),
            "audio_chunk_overlap": _get_typed_value(
                chunking_section, "audio_chunk_overlap", 200, int
            ),
            "audio_adaptive_chunking": _get_typed_value(
                chunking_section, "audio_adaptive_chunking", False, bool
            ),
            "audio_chunking_multi_level": _get_typed_value(
                chunking_section, "audio_chunking_multi_level", False, bool
            ),
            "audio_language": _get_typed_value(
                chunking_section, "audio_language", "en"
            ),
            "book_chunking_method": _get_typed_value(
                chunking_section, "book_chunking_method", "ebook_chunk_by_chapter"
            ),
            "book_chunk_max_size": _get_typed_value(
                chunking_section, "book_chunk_max_size", 400, int
            ),
            "book_chunk_overlap": _get_typed_value(
                chunking_section, "book_chunk_overlap", 200, int
            ),
            "book_adaptive_chunking": _get_typed_value(
                chunking_section, "book_adaptive_chunking", False, bool
            ),
            "book_chunking_multi_level": _get_typed_value(
                chunking_section, "book_chunking_multi_level", False, bool
            ),
            "book_language": _get_typed_value(chunking_section, "book_language", "en"),
            "document_chunking_method": _get_typed_value(
                chunking_section, "document_chunking_method", "words"
            ),
            "document_chunk_max_size": _get_typed_value(
                chunking_section, "document_chunk_max_size", 400, int
            ),
            "document_chunk_overlap": _get_typed_value(
                chunking_section, "document_chunk_overlap", 200, int
            ),
            "document_adaptive_chunking": _get_typed_value(
                chunking_section, "document_adaptive_chunking", False, bool
            ),
            "document_chunking_multi_level": _get_typed_value(
                chunking_section, "document_chunking_multi_level", False, bool
            ),
            "document_language": _get_typed_value(
                chunking_section, "document_language", "en"
            ),
            "mediawiki_article_chunking_method": _get_typed_value(
                chunking_section, "mediawiki_article_chunking_method", "words"
            ),
            "mediawiki_article_chunk_max_size": _get_typed_value(
                chunking_section, "mediawiki_article_chunk_max_size", 400, int
            ),
            "mediawiki_article_chunk_overlap": _get_typed_value(
                chunking_section, "mediawiki_article_chunk_overlap", 200, int
            ),
            "mediawiki_article_adaptive_chunking": _get_typed_value(
                chunking_section, "mediawiki_article_adaptive_chunking", False, bool
            ),
            "mediawiki_article_chunking_multi_level": _get_typed_value(
                chunking_section, "mediawiki_article_chunking_multi_level", False, bool
            ),
            "mediawiki_article_language": _get_typed_value(
                chunking_section, "mediawiki_article_language", "en"
            ),
            "mediawiki_dump_chunking_method": _get_typed_value(
                chunking_section, "mediawiki_dump_chunking_method", "words"
            ),
            "mediawiki_dump_chunk_max_size": _get_typed_value(
                chunking_section, "mediawiki_dump_chunk_max_size", 400, int
            ),
            "mediawiki_dump_chunk_overlap": _get_typed_value(
                chunking_section, "mediawiki_dump_chunk_overlap", 200, int
            ),
            "mediawiki_dump_adaptive_chunking": _get_typed_value(
                chunking_section, "mediawiki_dump_adaptive_chunking", False, bool
            ),
            "mediawiki_dump_chunking_multi_level": _get_typed_value(
                chunking_section, "mediawiki_dump_chunking_multi_level", False, bool
            ),
            "mediawiki_dump_language": _get_typed_value(
                chunking_section, "mediawiki_dump_language", "en"
            ),
            "obsidian_note_chunking_method": _get_typed_value(
                chunking_section, "obsidian_note_chunking_method", "words"
            ),
            "obsidian_note_chunk_max_size": _get_typed_value(
                chunking_section, "obsidian_note_chunk_max_size", 400, int
            ),
            "obsidian_note_chunk_overlap": _get_typed_value(
                chunking_section, "obsidian_note_chunk_overlap", 200, int
            ),
            "obsidian_note_adaptive_chunking": _get_typed_value(
                chunking_section, "obsidian_note_adaptive_chunking", False, bool
            ),
            "obsidian_note_chunking_multi_level": _get_typed_value(
                chunking_section, "obsidian_note_chunking_multi_level", False, bool
            ),
            "obsidian_note_language": _get_typed_value(
                chunking_section, "obsidian_note_language", "en"
            ),
            "podcast_chunking_method": _get_typed_value(
                chunking_section, "podcast_chunking_method", "sentences"
            ),
            "podcast_chunk_max_size": _get_typed_value(
                chunking_section, "podcast_chunk_max_size", 300, int
            ),
            "podcast_chunk_overlap": _get_typed_value(
                chunking_section, "podcast_chunk_overlap", 30, int
            ),
            "podcast_adaptive_chunking": _get_typed_value(
                chunking_section, "podcast_adaptive_chunking", False, bool
            ),
            "podcast_chunking_multi_level": _get_typed_value(
                chunking_section, "podcast_chunking_multi_level", False, bool
            ),
            "podcast_language": _get_typed_value(
                chunking_section, "podcast_language", "en"
            ),
            "text_chunking_method": _get_typed_value(
                chunking_section, "text_chunking_method", "words"
            ),
            "text_chunk_max_size": _get_typed_value(
                chunking_section, "text_chunk_max_size", 400, int
            ),
            "text_chunk_overlap": _get_typed_value(
                chunking_section, "text_chunk_overlap", 200, int
            ),
            "text_adaptive_chunking": _get_typed_value(
                chunking_section, "text_adaptive_chunking", False, bool
            ),
            "text_chunking_multi_level": _get_typed_value(
                chunking_section, "text_chunking_multi_level", False, bool
            ),
            "text_language": _get_typed_value(chunking_section, "text_language", "en"),
            "video_chunking_method": _get_typed_value(
                chunking_section, "video_chunking_method", "words"
            ),
            "video_chunk_max_size": _get_typed_value(
                chunking_section, "video_chunk_max_size", 400, int
            ),
            "video_chunk_overlap": _get_typed_value(
                chunking_section, "video_chunk_overlap", 200, int
            ),
            "video_adaptive_chunking": _get_typed_value(
                chunking_section, "video_adaptive_chunking", False, bool
            ),
            "video_chunking_multi_level": _get_typed_value(
                chunking_section, "video_chunking_multi_level", False, bool
            ),
            "video_language": _get_typed_value(
                chunking_section, "video_language", "en"
            ),
        },
        "embedding_config": {
            "embedding_provider": _get_typed_value(
                embeddings_section, "embedding_provider", "openai"
            ),
            "embedding_model": _get_typed_value(
                embeddings_section, "embedding_model", "text-embedding-3-large"
            ),
            "onnx_model_path": _get_typed_value(
                embeddings_section,
                "onnx_model_path",
                "./Models/onnx_models/text-embedding-3-small.onnx",
                Path,
            ),
            "model_dir": _get_typed_value(
                embeddings_section, "model_dir", "./Models", Path
            ),
            "embedding_api_url": _get_typed_value(
                embeddings_section,
                "embedding_api_url",
                "http://localhost:8080/v1/embeddings",
            ),
            "embedding_api_key": _get_typed_value(
                embeddings_section, "embedding_api_key", ""
            ),
            "chunk_size": _get_typed_value(
                embeddings_section, "chunk_size", 400, int
            ),  # This was 'chunk_size' in old Embeddings, also in Chunking
            "chunk_overlap": _get_typed_value(
                embeddings_section, "overlap", 200, int
            ),  # This was 'overlap' in old Embeddings
            "models": embedding_config_section.get(
                "models", {}
            ),  # Include the models from the embedding_config section
        },
        "auto_save": {
            "save_character_chats": _get_typed_value(
                auto_save_section, "save_character_chats", False, bool
            ),
            "save_rag_chats": _get_typed_value(
                auto_save_section, "save_rag_chats", False, bool
            ),
        },
        "STT_settings": {  # Corrected key from STT-Settings
            "default_stt_provider": _get_typed_value(
                stt_settings_section, "default_stt_provider", default_stt_provider
            ),
        },
        "tts_settings": {
            "default_tts_provider": _get_typed_value(
                tts_settings_section, "default_tts_provider", "openai"
            ),
            "tts_voice": _get_typed_value(
                tts_settings_section, "default_tts_voice", "shimmer"
            ),  # General default voice
            "local_tts_device": _get_typed_value(
                tts_settings_section, "local_tts_device", "cpu"
            ),
            # OpenAI TTS
            "default_openai_tts_model": _get_typed_value(
                tts_settings_section, "default_openai_tts_model", "tts-1-hd"
            ),
            "default_openai_tts_voice": _get_typed_value(
                tts_settings_section, "default_openai_tts_voice", "shimmer"
            ),
            "default_openai_tts_speed": _get_typed_value(
                tts_settings_section, "default_openai_tts_speed", 1.0, float
            ),
            "default_openai_tts_output_format": _get_typed_value(
                tts_settings_section, "default_openai_tts_output_format", "mp3"
            ),
            "default_openai_tts_streaming": _get_typed_value(
                tts_settings_section, "default_openai_tts_streaming", False, bool
            ),
            # Google TTS
            "default_google_tts_model": _get_typed_value(
                tts_settings_section, "default_google_tts_model", "en"
            ),  # FIXME: Review defaults
            "default_google_tts_voice": _get_typed_value(
                tts_settings_section, "default_google_tts_voice", "en"
            ),  # FIXME: Review defaults
            "default_google_tts_speed": _get_typed_value(
                tts_settings_section, "default_google_tts_speed", 1.0, float
            ),  # FIXME: Review defaults
            # ElevenLabs TTS
            "default_eleven_tts_model": _get_typed_value(
                tts_settings_section,
                "default_eleven_tts_model",
                "eleven_multilingual_v2",
            ),  # FIXME: Placeholder
            "default_eleven_tts_voice": _get_typed_value(
                tts_settings_section, "default_eleven_tts_voice", "Rachel"
            ),  # FIXME: Placeholder
            "default_eleven_tts_language_code": _get_typed_value(
                tts_settings_section, "default_eleven_tts_language_code", "en-US"
            ),  # FIXME
            "default_eleven_tts_voice_stability": _get_typed_value(
                tts_settings_section, "default_eleven_tts_voice_stability", 0.5, float
            ),  # FIXME
            "default_eleven_tts_voice_similiarity_boost": _get_typed_value(
                tts_settings_section,
                "default_eleven_tts_voice_similiarity_boost",
                0.75,
                float,
            ),  # FIXME
            "default_eleven_tts_voice_style": _get_typed_value(
                tts_settings_section, "default_eleven_tts_voice_style", 0.0, float
            ),  # FIXME
            "default_eleven_tts_voice_use_speaker_boost": _get_typed_value(
                tts_settings_section,
                "default_eleven_tts_voice_use_speaker_boost",
                True,
                bool,
            ),  # FIXME
            "default_eleven_tts_output_format": _get_typed_value(
                tts_settings_section,
                "default_eleven_tts_output_format",
                "mp3_44100_192",
            ),
            # AllTalk TTS (from load_and_log_configs, now integrated)
            "alltalk_api_ip": _get_typed_value(
                tts_settings_section,
                "alltalk_api_ip",
                "http://127.0.0.1:7851/v1/audio/speech",
            ),
            "default_alltalk_tts_model": _get_typed_value(
                tts_settings_section, "default_alltalk_tts_model", "alltalk_model"
            ),
            "default_alltalk_tts_voice": _get_typed_value(
                tts_settings_section, "default_alltalk_tts_voice", "alloy"
            ),
            "default_alltalk_tts_speed": _get_typed_value(
                tts_settings_section, "default_alltalk_tts_speed", 1.0, float
            ),
            "default_alltalk_tts_output_format": _get_typed_value(
                tts_settings_section, "default_alltalk_tts_output_format", "mp3"
            ),
            # Kokoro TTS
            "kokoro_model_path": _get_typed_value(
                tts_settings_section,
                "kokoro_model_path",
                "Databases/kokoro_models",
                Path,
            ),
            "default_kokoro_tts_model": _get_typed_value(
                tts_settings_section, "default_kokoro_tts_model", "pht"
            ),
            "default_kokoro_tts_voice": _get_typed_value(
                tts_settings_section, "default_kokoro_tts_voice", "sky"
            ),
            "default_kokoro_tts_speed": _get_typed_value(
                tts_settings_section, "default_kokoro_tts_speed", 1.0, float
            ),
            "default_kokoro_tts_output_format": _get_typed_value(
                tts_settings_section, "default_kokoro_tts_output_format", "wav"
            ),
            # Self-hosted OpenAI API TTS
            "default_openai_api_tts_model": _get_typed_value(
                tts_settings_section, "default_openai_api_tts_model", "tts-1-hd"
            ),
            "default_openai_api_tts_voice": _get_typed_value(
                tts_settings_section, "default_openai_api_tts_voice", "shimmer"
            ),
            "default_openai_api_tts_speed": _get_typed_value(
                tts_settings_section, "default_openai_api_tts_speed", 1.0, float
            ),  # Was '1' string
            "default_openai_api_tts_output_format": _get_typed_value(
                tts_settings_section, "default_openai_api_tts_output_format", "mp3"
            ),  # key was default_openai_tts_api_output_format
            "default_openai_api_tts_streaming": _get_typed_value(
                tts_settings_section, "default_openai_api_tts_streaming", False, bool
            ),
        },
        "search_settings_general": {  # Renamed from 'search_settings' to avoid conflict with SearchEngines section for keys
            "default_search_provider": _get_typed_value(
                search_settings_section, "search_provider_default", "duckduckgo"
            ),
            "search_language_query": _get_typed_value(
                search_settings_section, "search_language_query", "en"
            ),
            "search_language_analysis": _get_typed_value(
                search_settings_section, "search_language_analysis", "en"
            ),
            "search_default_max_queries": _get_typed_value(
                search_settings_section, "search_default_max_queries", 5, int
            ),
            "search_enable_subquery": _get_typed_value(
                search_settings_section, "search_enable_subquery", False, bool
            ),
            "search_enable_subquery_count_max": _get_typed_value(
                search_settings_section, "search_enable_subquery_count_max", 3, int
            ),
            "search_result_rerank": _get_typed_value(
                search_settings_section, "search_result_rerank", False, bool
            ),
            "search_result_max": _get_typed_value(
                search_settings_section, "search_result_max", 10, int
            ),
            "search_result_max_per_query": _get_typed_value(
                search_settings_section, "search_result_max_per_query", 10, int
            ),
            "search_result_blacklist": _get_typed_value(
                search_settings_section, "search_result_blacklist", ""
            ),
            "search_result_display_type": _get_typed_value(
                search_settings_section, "search_result_display_type", "text"
            ),
            "search_result_display_metadata": _get_typed_value(
                search_settings_section, "search_result_display_metadata", True, bool
            ),
            "search_result_save_to_db": _get_typed_value(
                search_settings_section, "search_result_save_to_db", True, bool
            ),
            "search_result_analysis_tone": _get_typed_value(
                search_settings_section, "search_result_analysis_tone", "neutral"
            ),
            "relevance_analysis_llm": _get_typed_value(
                search_settings_section, "relevance_analysis_llm", "openai"
            ),
            "final_answer_llm": _get_typed_value(
                search_settings_section, "final_answer_llm", "openai"
            ),
            "relevance_llm_timeout_s": _get_int_timeout_value(
                search_settings_section, "relevance_llm_timeout_s", 30
            ),
            "relevance_scrape_timeout_s": _get_int_timeout_value(
                search_settings_section, "relevance_scrape_timeout_s", 30
            ),
            # 240 default (task-1356 review ruling). Fix round 1: this key
            # is NOT required to stay under the agent runtime's 300s
            # max_tool_call_seconds (Agents/agent_models.py RunBudget.
            # max_tool_call_seconds) -- the runtime automatically allots the
            # web_deep_search tool its own per-call timeout of this value
            # plus ~50s slack (wait_for grace + thread-join + scheduling
            # jitter) via LocalToolProvider.timeout_for, so a deadline-hit
            # deep search can still return its partial synthesis instead of
            # being killed by the outer tool-call ceiling first, for any
            # value an operator sets here.
            "deep_search_timeout_s": _get_int_timeout_value(
                search_settings_section, "deep_search_timeout_s", 240
            ),
        },
        "search_engine_specific_settings": {  # API Keys for various search engines from 'SearchEngines' TOML table
            "baidu_search_api_key": _get_typed_value(
                search_engines_section, "baidu_search_api_key", ""
            ),
            "bing_country_code": _get_typed_value(
                search_engines_section, "bing_country_code", ""
            ),
            "bing_search_api_url": _get_typed_value(
                search_engines_section, "bing_search_api_url", ""
            ),
            "brave_country_code": _get_typed_value(
                search_engines_section, "brave_country_code", ""
            ),
            "google_search_api_url": _get_typed_value(
                search_engines_section, "google_search_api_url", ""
            ),
            "google_search_engine_id": _get_typed_value(
                search_engines_section, "google_search_engine_id", ""
            ),
            "google_simp_trad_chinese": _get_typed_value(
                search_engines_section, "google_simp_trad_chinese", False, bool
            ),
            "limit_google_search_to_country": _get_typed_value(
                search_engines_section, "limit_google_search_to_country", False, bool
            ),
            "google_search_country": _get_typed_value(
                search_engines_section, "google_search_country", ""
            ),
            "google_search_country_code": _get_typed_value(
                search_engines_section, "google_search_country_code", ""
            ),
            "google_search_filter_setting": _get_typed_value(
                search_engines_section, "google_filter_setting", ""
            ),
            "google_user_geolocation": _get_typed_value(
                search_engines_section, "google_user_geolocation", False, bool
            ),
            "google_ui_language": _get_typed_value(
                search_engines_section, "google_ui_language", ""
            ),
            "google_limit_search_results_to_language": _get_typed_value(
                search_engines_section,
                "google_limit_search_results_to_language",
                False,
                bool,
            ),
            "google_site_search_include": _get_typed_value(
                search_engines_section, "google_site_search_include", ""
            ),
            "google_site_search_exclude": _get_typed_value(
                search_engines_section, "google_site_search_exclude", ""
            ),
            "google_sort_results_by": _get_typed_value(
                search_engines_section, "google_sort_results_by", ""
            ),
            "google_default_search_results": _get_typed_value(
                search_engines_section, "google_default_search_results", 10, int
            ),
            "google_safe_search": _get_typed_value(
                search_engines_section, "google_safe_search", False, bool
            ),
            "google_enable_site_search": _get_typed_value(
                search_engines_section, "google_enable_site_search", False, bool
            ),
            "yandex_search_engine_id": _get_typed_value(
                search_engines_section, "yandex_search_engine_id", ""
            ),
        },
        "search_engines_keys": {  # API Keys for various search engines from 'SearchEngines' TOML table
            "baidu_search_api_key": _get_typed_value(
                search_engines_section, "search_engine_api_key_baidu", ""
            ),
            "bing_search_api_key": _get_typed_value(
                search_engines_section, "bing_search_api_key", ""
            )
            or _get_typed_value(
                search_engines_section, "search_engine_api_key_bing", ""
            ),
            "brave_search_api_key": _get_typed_value(
                search_engines_section, "brave_search_api_key", ""
            ),
            "brave_search_ai_api_key": _get_typed_value(
                search_engines_section, "brave_search_ai_api_key", ""
            ),
            "duckduckgo_search_api_key": _get_typed_value(
                search_engines_section, "duckduckgo_search_api_key", ""
            ),
            "google_search_api_key": _get_typed_value(
                search_engines_section, "google_search_api_key", ""
            ),
            "kagi_search_api_key": _get_typed_value(
                search_engines_section, "kagi_search_api_key", ""
            ),
            "searx_search_api_url": _get_typed_value(
                search_engines_section, "searx_search_api_url", ""
            )
            or _get_typed_value(search_engines_section, "search_engine_searx_api", ""),
            "tavily_search_api_key": _get_typed_value(
                search_engines_section, "tavily_search_api_key", ""
            ),
            "serper_search_api_key": _get_typed_value(
                search_engines_section, "serper_search_api_key", ""
            ),
            "exa_search_api_key": _get_typed_value(
                search_engines_section, "exa_search_api_key", ""
            ),
            "yandex_search_api_key": _get_typed_value(
                search_engines_section, "yandex_search_api_key", ""
            ),
            "yandex_search_folder_id": _get_typed_value(
                search_engines_section, "yandex_search_folder_id", ""
            ),
        },
        # NOTE: the former "prompts_strings" loader was removed once the
        # Internal_Prompts registry took over these web-search prompts. Its
        # only consumer was the unused CONFIG_PROMPT_SITUATE_CHUNK_CONTEXT
        # constant. The [Prompts] TOML keys themselves remain — the registry
        # reads them as its legacy-override tier (via legacy_config_path).
        "web_scraper_settings": {
            "web_scraper_api_key": _get_typed_value(
                web_scraper_section, "web_scraper_api_key", ""
            ),
            "web_scraper_api_url": _get_typed_value(
                web_scraper_section, "web_scraper_api_url", ""
            ),
            # ... (all web scraper settings)
        },
        "confluence": {
            "base_url": _get_typed_value(
                confluence_section, "base_url", os.getenv("CONFLUENCE_BASE_URL", "")
            ),
            "auth_method": _get_typed_value(
                confluence_section,
                "auth_method",
                os.getenv("CONFLUENCE_AUTH_METHOD", "api_token"),
            ),
            "username": _get_typed_value(
                confluence_section, "username", os.getenv("CONFLUENCE_USERNAME", "")
            ),
            "api_token": _get_typed_value(
                confluence_section, "api_token", os.getenv("CONFLUENCE_API_TOKEN", "")
            ),
            "oauth_token": _get_typed_value(
                confluence_section,
                "oauth_token",
                os.getenv("CONFLUENCE_OAUTH_TOKEN", ""),
            ),
            "password": _get_typed_value(
                confluence_section, "password", os.getenv("CONFLUENCE_PASSWORD", "")
            ),
            "browser": _get_typed_value(confluence_section, "browser", "all"),
            "space_keys": _get_typed_value(confluence_section, "space_keys", [], list),
            "max_pages_per_space": _get_typed_value(
                confluence_section, "max_pages_per_space", 100, int
            ),
            "max_crawl_depth": _get_typed_value(
                confluence_section, "max_crawl_depth", 5, int
            ),
            "include_attachments": _get_typed_value(
                confluence_section, "include_attachments", False, bool
            ),
            "follow_links": _get_typed_value(
                confluence_section, "follow_links", False, bool
            ),
            "rate_limit_delay": _get_typed_value(
                confluence_section, "rate_limit_delay", 0.5, float
            ),
        },
        # Configurations from hardcoded dicts (now from TOML or fallback to Python dicts)
        "APP_TTS_CONFIG": {**DEFAULT_APP_TTS_CONFIG, **app_tts_config},
        "APP_DATABASE_CONFIG": {**DEFAULT_DATABASE_CONFIG, **app_database_config},
        "APP_RAG_SEARCH_CONFIG": {**DEFAULT_RAG_SEARCH_CONFIG, **app_rag_search_config},
        "acp": get_toml_section("acp"),
        "permission_summary": get_toml_section("permission_summary"),
        "library": {
            **copy.deepcopy(library_section),
            "ingest_directory_scan_limit": coerce_int_setting(
                library_section.get("ingest_directory_scan_limit", 1000),
                1000,
                minimum=1,
            ),
            # (TASK-19556) Opt-in, OFF by default: see the [library] block in
            # the default TOML below for why a link check is not free.
            "ingest_url_preflight_probe": coerce_bool_setting(
                library_section.get("ingest_url_preflight_probe", False),
                False,
            ),
            "ingest_options": library_section.get("ingest_options", {})
            if isinstance(library_section.get("ingest_options"), dict)
            else {},
            "reader": normalized_reader,
            **normalized_destination_readers,
        },
        "COMPREHENSIVE_CONFIG_RAW": toml_config_data,  # Store the raw TOML data if needed
        "OPENAI_API_KEY": openai_api_key,  # Top-level convenience access
    }

    # Bridge TTSSettings defaults into APP_TTS_CONFIG for runtime use
    # This ensures TTS event handlers can find the user's selected defaults
    if "default_provider" not in config_dict["APP_TTS_CONFIG"]:
        config_dict["APP_TTS_CONFIG"]["default_provider"] = config_dict[
            "tts_settings"
        ].get("default_tts_provider", "openai")
    if "default_voice" not in config_dict["APP_TTS_CONFIG"]:
        config_dict["APP_TTS_CONFIG"]["default_voice"] = config_dict[
            "tts_settings"
        ].get("default_tts_voice", "alloy")
    if "default_model" not in config_dict["APP_TTS_CONFIG"]:
        config_dict["APP_TTS_CONFIG"]["default_model"] = config_dict[
            "tts_settings"
        ].get("default_openai_tts_model", "tts-1")
    if "default_format" not in config_dict["APP_TTS_CONFIG"]:
        config_dict["APP_TTS_CONFIG"]["default_format"] = config_dict[
            "tts_settings"
        ].get("default_openai_tts_output_format", "mp3")
    if "default_speed" not in config_dict["APP_TTS_CONFIG"]:
        config_dict["APP_TTS_CONFIG"]["default_speed"] = config_dict[
            "tts_settings"
        ].get("default_openai_tts_speed", 1.0)

    # Populate the rest of chunking_config (tedious but necessary)
    chunking_types = [
        "audio",
        "book",
        "document",
        "mediawiki_article",
        "mediawiki_dump",
        "obsidian_note",
        "podcast",
        "text",
        "video",
    ]
    for ctype in chunking_types:
        # Use direct defaults from chunking_section or hardcoded fallbacks
        default_method = _get_typed_value(chunking_section, "chunking_method", "words")
        default_max_size = _get_typed_value(
            chunking_section, "chunk_max_size", 400, int
        )
        default_overlap = _get_typed_value(chunking_section, "chunk_overlap", 200, int)
        default_adaptive = _get_typed_value(
            chunking_section, "adaptive_chunking", False, bool
        )
        default_multi_level = _get_typed_value(
            chunking_section, "chunking_multi_level", False, bool
        )
        default_language = _get_typed_value(
            chunking_section, "chunk_language", global_default_chunk_language
        )

        # Only set if not already defined in lines 494-562
        if f"{ctype}_chunking_method" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunking_method"] = (
                _get_typed_value(
                    chunking_section, f"{ctype}_chunking_method", default_method
                )
            )
        if f"{ctype}_chunk_max_size" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunk_max_size"] = (
                _get_typed_value(
                    chunking_section, f"{ctype}_chunk_max_size", default_max_size, int
                )
            )
        if f"{ctype}_chunk_overlap" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunk_overlap"] = _get_typed_value(
                chunking_section, f"{ctype}_chunk_overlap", default_overlap, int
            )
        if f"{ctype}_adaptive_chunking" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_adaptive_chunking"] = (
                _get_typed_value(
                    chunking_section,
                    f"{ctype}_adaptive_chunking",
                    default_adaptive,
                    bool,
                )
            )
        if f"{ctype}_chunking_multi_level" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunking_multi_level"] = (
                _get_typed_value(
                    chunking_section,
                    f"{ctype}_chunking_multi_level",
                    default_multi_level,
                    bool,
                )
            )
        if f"{ctype}_language" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_language"] = _get_typed_value(
                chunking_section, f"{ctype}_language", default_language
            )

    # Set the chat dictionaries folder path dynamically
    from .Utils.paths import get_user_data_dir

    chat_dicts_folder = (
        get_user_data_dir()
        if bootstrap.succeeded
        else profile_paths.user_data_dir(toml_config_data)
    ) / "chat_dicts"
    config_dict["chat_dictionaries"]["chat_dicts_folder"] = str(chat_dicts_folder)

    # Create the chat dictionaries folder if it doesn't exist. task-32901:
    # a umask-derived mkdir lands 0775 on umask-002 machines and the very
    # next boot's admission check refuses it; the private lifecycle creates
    # 0700 and hardens an existing 0775 instance in place.
    try:
        if bootstrap.succeeded:
            with _config_participants.operation(
                sys.modules[__name__],
                route="config_chat_dicts",
                target=chat_dicts_folder,
            ):
                secure_private_directory(
                    chat_dicts_folder, create=True, application_owned=True
                )
        logger.debug(f"Ensured chat dictionaries folder exists: {chat_dicts_folder}")
    except RecoveryRequired:
        raise
    except PrivatePathError as e:
        # task-32901: a pre-existing shared-writable instance (pre-fix
        # umask-002 machines) is refused by the admission pre-check before
        # the hardened create above can repair it; name the exact repair,
        # reason-gated and shell-safe (see private_path_repair_hint).
        repair = private_path_repair_hint(e)
        suffix = f" Repair once with: {repair}" if repair else ""
        logger.error(
            f"Could not create chat dictionaries folder {chat_dicts_folder}: {e}.{suffix}"
        )
    except Exception as e:
        logger.error(
            f"Could not create chat dictionaries folder {chat_dicts_folder}: {e}"
        )

    if bootstrap.succeeded:
        if posture and posture[-1] is None:
            # The file was missing when stamped and this load's bootstrap
            # created it: stamp what now exists, or every fresh profile pays a
            # second rebuild. A file that existed keeps its pre-read stamp, so
            # a mid-read swap still misses (Qodo, #2903).
            created = _config_file_posture(active_config_path)
            if _is_plain_owned_file(created):
                posture = created
        with _SETTINGS_CACHE_LOCK:
            _SETTINGS_CACHE = config_dict
            _SETTINGS_CACHE_SOURCE = active_config_path
            _SETTINGS_CACHE_POSTURE = posture
            logger.debug("load_settings: Configuration cached for future use")

    return config_dict


# --- Define API Models (Combined Cloud & Local) ---
# (Keep your existing API_MODELS_BY_PROVIDER and LOCAL_PROVIDERS dictionaries)
API_MODELS_BY_PROVIDER = {
    "OpenAI": [
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.6-luna",
        "gpt-4.1-2025-04-14",
        "o4-mini-2025-04-16",
        "o3-2025-04-16",
        "o3-mini-2025-01-31",
        "o1-2024-12-17",
        "chatgpt-4o-latest",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
        "gpt-4o-mini-2024-07-18",
    ],
    "Anthropic": [
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-fable-5",
        "claude-haiku-4-5",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-2.1",
        "claude-2.0",
    ],
    "Cohere": [
        "command-a-03-2025",
        "command-r7b-12-2024",
        "command-r-plus-04-2024",
        "command-r-plus",
        "command-r-08-2024",
        "command-r-03-2024",
        "command",
        "command-nightly",
        "command-light",
        "command-light-nightly",
    ],
    "DeepSeek": ["deepseek-v4-flash", "deepseek-v4-pro"],
    "Groq": [
        "gemma2-9b-it",
        "mmeta-llama/Llama-Guard-4-12B",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-70b-8192",
        "llama3-8b-8192",
    ],
    "Google": [
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ],
    "HuggingFace": [
        "openai/gpt-oss-120b",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ],
    "MistralAI": [
        "open-mistral-nemo",
        "mistral-medium-2505",
        "codestral-2501",
        "mistral-saba-2502",
        "mistral-large-2411",
        "ministral-3b-2410",
        "ministral-8b-2410",
        "mistral-moderation-2411",
        "devstral-small-2505",
        "mistral-small-2503",
    ],
    "OpenRouter": [
        "openai/gpt-4o-mini",
        "anthropic/claude-3.7-sonnet",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.5-pro-preview",
        "google/gemini-2.5-flash-preview",
        "deepseek/deepseek-chat-v3-0324:free",
        "deepseek/deepseek-chat-v3-0324",
        "openai/gpt-4.1",
        "anthropic/claude-sonnet-4",
        "deepseek/deepseek-r1:free",
        "anthropic/claude-3.7-sonnet:thinking",
        "google/gemini-flash-1.5-8b",
        "mistralai/mistral-nemo",
        "google/gemini-2.5-flash-preview-05-20",
    ],
}

#######################################################################################################################
# --- CLI User Configuration Section ---
#######################################################################################################################

# --- Configuration File Content (for reference or auto-creation for the CLI) ---
# Packaged data file (Config_Files/default_config.toml); the two placeholders in it
# are resolved just below. Read as bytes so no newline translation can alter it.
CONFIG_TOML_CONTENT = (
    importlib.resources.files("tldw_chatbook.Config_Files")
    .joinpath("default_config.toml")
    .read_bytes()
    .decode("utf-8")
)

# Resolve the `[transcription] default_provider` placeholder to this platform's
# preferred engine before the template is parsed or ever written to disk
# (task-867). `CONFIG_TOML_CONTENT` feeds both `DEFAULT_CONFIG_FROM_TOML`
# below -- the baseline every config load merges the user's file on top of --
# and the literal bytes a fresh install writes to `config.toml`, so this one
# substitution fixes both without any reader needing to special-case an
# absent key.
CONFIG_TOML_CONTENT = CONFIG_TOML_CONTENT.replace(
    "__DEFAULT_TRANSCRIPTION_PROVIDER__", _default_stt_provider_for_platform()
)
# Same single-substitution discipline for the splash duration (PR #2329):
# the placeholder keeps the literal out of the template and every reader --
# the widget, the app fallback, the Settings viewer -- shares the one
# constant from Constants.py.
CONFIG_TOML_CONTENT = CONFIG_TOML_CONTENT.replace(
    "__DEFAULT_SPLASH_DURATION__", str(DEFAULT_SPLASH_DURATION_SECONDS)
)

try:
    DEFAULT_CONFIG_FROM_TOML: Dict[str, Any] = tomllib.loads(CONFIG_TOML_CONTENT)
except tomllib.TOMLDecodeError as e:
    logger.critical(
        f"FATAL: Could not parse internal DEFAULT_CONFIG_TOML_CONTENT: {e}. Application cannot start correctly."
    )
    DEFAULT_CONFIG_FROM_TOML = {}  # Should not happen with valid TOML string


def _merge_user_config_over_shipped_defaults(
    user_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Deep-merge one raw user config over the shipped template.

    Every published view of the config (bootstrap load, post-write install,
    atomic snapshot) is built here, so they agree. TASK-33620.10: the
    template's ``[chat_defaults] streaming = true`` must not shadow a user's
    legacy ``enable_streaming``, which is read only when the user's file has no
    ``streaming``. When the file holds the legacy key alone, the template's
    ``streaming`` is left out of the merged view and readers bridge the legacy
    value as before.

    Args:
        user_config: The user's raw (parsed, migrated) config file.

    Returns:
        A new merged mapping; neither input is modified.
    """
    merged = deep_merge_dicts(DEFAULT_CONFIG_FROM_TOML, dict(user_config))
    user_chat_defaults = user_config.get("chat_defaults")
    if (
        isinstance(user_chat_defaults, Mapping)
        and "enable_streaming" in user_chat_defaults
        and "streaming" not in user_chat_defaults
    ):
        merged_chat_defaults = merged.get("chat_defaults")
        if isinstance(merged_chat_defaults, dict):
            merged_chat_defaults.pop("streaming", None)
    return merged


# --- Primary Configuration Loading Logic for the CLI ---
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_CACHE_SOURCE: Optional[Path] = None
#: TASK-26038: (mtime_ns, size) of the config file at its last successful
#: load, and the last monotonic time we statted it. A THROTTLED inline
#: metadata check on the read path picks up external edits without a
#: filesystem watcher or a polling thread, while keeping TASK-21124's
#: near-zero hot path (one stat per throttle window, not per read).
_CONFIG_FILE_STAMP: Optional[tuple[int, int]] = None
_CONFIG_STAT_CHECKED_MONOTONIC: float = 0.0
#: How often (seconds) the read path may re-stat the file. An external edit
#: is picked up on the next read after this window. Small enough to feel
#: immediate, large enough that a burst of ~400 get_cli_setting calls in one
#: render statts at most once.
_CONFIG_STAT_THROTTLE_SECONDS: float = 1.0
_FIRST_PROFILE_CREATED_THIS_SESSION = False


def first_profile_created_this_session() -> bool:
    """Return whether this process created the active profile config.

    Returns:
        ``True`` when this process created the active profile configuration;
        otherwise ``False``.
    """
    return _FIRST_PROFILE_CREATED_THIS_SESSION


# --- TASK-26039: advisory unknown/deprecated config-key validation ---

#: Dotted section prefixes whose sub-keys are legitimately user-defined, so
#: unknown-key reporting must stay silent under them (AC#6). Kept conservative
#: -- only genuinely dynamic-keyed sections -- so real typos elsewhere still
#: surface. Each entry is a tuple path prefix.
_FREEFORM_CONFIG_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("agents",),  # deliberately EMPTY in the default shape: the
    # authoritative defaults live in
    # Agents/agent_service.py (two-homes drift),
    # so documented overrides here would all flag
    # as unknown (Qodo #13, PR #2301)
    ("console", "reasoning_history_overrides"),
    ("console", "reasoning_native_tool_overrides"),
    ("api_settings",),  # provider configs incl. user-added custom providers
    ("providers",),  # provider display sections + model lists
    ("model_capabilities", "models"),  # arbitrary model names
    ("model_capabilities", "patterns"),  # arbitrary model-name patterns
    ("SearchEngines",),  # per-engine configs
    ("Prompts",),  # user prompt content
    ("prompts",),
)

#: Keys/sections known to be renamed or removed, mapped to their replacement
#: dotted path (AC#3). A user path equal to a key here, or nested under it, is
#: reported as deprecated (naming the replacement) instead of unknown.
_DEPRECATED_CONFIG_KEYS: Dict[str, str] = {
    # Legacy provider keys: `[API] <provider>_api_key` was superseded by the
    # `[api_settings.<provider>] api_key` table (still honored as the lowest
    # -precedence fallback -- see _normalize_legacy_provider_api_key).
    "API": "api_settings",
}


@dataclass(frozen=True)
class ConfigKeyFinding:
    """One advisory config-key finding."""

    path: str  # dotted path, e.g. "general.focus_mdoe"
    kind: str  # "unknown" | "deprecated"
    suggestion: Optional[str] = None  # near-miss key or replacement path


def _is_freeform_config_path(path: tuple[str, ...]) -> bool:
    return any(
        len(path) >= len(prefix) and path[: len(prefix)] == prefix
        for prefix in _FREEFORM_CONFIG_PREFIXES
    )


def _deprecated_config_replacement(path: tuple[str, ...]) -> Optional[str]:
    """Return the replacement dotted path if ``path`` is (under) a deprecated key."""
    for i in range(1, len(path) + 1):
        prefix = ".".join(path[:i])
        replacement = _DEPRECATED_CONFIG_KEYS.get(prefix)
        if replacement is not None:
            tail = path[i:]
            return ".".join((replacement, *tail)) if tail else replacement
    return None


def validate_config_keys(
    user_config: Mapping[str, Any],
    reference: Optional[Mapping[str, Any]] = None,
) -> list[ConfigKeyFinding]:
    """Report keys present in ``user_config`` that no code reads (TASK-26039).

    Advisory only -- the caller never rejects the config. A key absent from the
    ``reference`` shape (the programmatic defaults) is ``unknown``; a near miss
    against a sibling reference key carries that suggestion (AC#2); a key known
    to be renamed is ``deprecated`` with its replacement (AC#3); nested tables
    are covered (AC#5); user-extensible sections are exempt (AC#6).
    """
    if reference is None:
        reference = DEFAULT_CONFIG_FROM_TOML
    findings: list[ConfigKeyFinding] = []

    def _walk(user: Mapping[str, Any], ref: Mapping[str, Any], path: tuple[str, ...]):
        for key, value in user.items():
            here = (*path, key)
            replacement = _deprecated_config_replacement(here)
            if replacement is not None:
                findings.append(
                    ConfigKeyFinding(".".join(here), "deprecated", replacement)
                )
                continue
            if key in ref:
                ref_value = ref[key]
                if isinstance(value, Mapping) and isinstance(ref_value, Mapping):
                    if not _is_freeform_config_path(here):
                        _walk(value, ref_value, here)
                continue
            if _is_freeform_config_path(here):
                continue
            siblings = [k for k in ref.keys() if isinstance(k, str)]
            match = difflib.get_close_matches(key, siblings, n=1, cutoff=0.8)
            suggestion = ".".join((*path, match[0])) if match else None
            findings.append(ConfigKeyFinding(".".join(here), "unknown", suggestion))

    _walk(user_config, reference, ())
    return findings


def format_config_key_report(findings: Sequence[ConfigKeyFinding]) -> str:
    """Render key findings as a short advisory block (empty string when none)."""
    if not findings:
        return ""
    lines: list[str] = []
    for f in findings:
        if f.kind == "deprecated":
            lines.append(f"deprecated key '{f.path}' -> use '{f.suggestion}'")
        elif f.suggestion:
            lines.append(f"unknown key '{f.path}' (did you mean '{f.suggestion}'?)")
        else:
            lines.append(f"unknown key '{f.path}' (no code reads it)")
    return "Advisory: " + "; ".join(lines)


#: TASK-26040: the config file's schema version. Bumped when a numbered
#: migration is added to ``_CONFIG_MIGRATIONS``; a fresh config is created
#: carrying this version. An unversioned (pre-existing) file is treated as
#: the baseline (0) and migrated forward, never rejected.
CONFIG_SCHEMA_VERSION_KEY = "config_schema_version"
_CURRENT_CONFIG_SCHEMA_VERSION = 1

#: Numbered stepwise migrations: ``{target_version: fn(config) -> config}``.
#: Each transforms a config AT (target-1) into (target). Empty today -- this
#: is the first versioned config -- but the runner is exercised by tests and
#: ready for the first key rename, mirroring the DB migration pattern.
_CONFIG_MIGRATIONS: Dict[int, Any] = {}


def migrate_config_forward(
    config: Dict[str, Any],
) -> tuple[Dict[str, Any], bool, Optional[str]]:
    """Run stepwise forward migrations on one config (TASK-26040).

    Returns ``(migrated, changed, conflict)``:
    * ``migrated`` -- the config transformed to the current version (or the
      input unchanged when already current, or when a conflict blocks it).
    * ``changed`` -- whether anything (including a first version stamp) changed.
    * ``conflict`` -- a human-readable reason when the config is from a NEWER
      version than this code understands (AC#5); the config is returned
      untouched rather than mangled.
    """
    version = config.get(CONFIG_SCHEMA_VERSION_KEY, 0)
    try:
        version = int(version)
    except (TypeError, ValueError):
        version = 0  # a junk version stamp is treated as the baseline
    if version > _CURRENT_CONFIG_SCHEMA_VERSION:
        return (
            config,
            False,
            (
                f"config schema version {version} is newer than this "
                f"application supports ({_CURRENT_CONFIG_SCHEMA_VERSION}); "
                "not migrating -- upgrade the application or restore an "
                "older config."
            ),
        )
    if version == _CURRENT_CONFIG_SCHEMA_VERSION:
        return config, False, None
    migrated = copy.deepcopy(config)
    for target in range(version + 1, _CURRENT_CONFIG_SCHEMA_VERSION + 1):
        migration = _CONFIG_MIGRATIONS.get(target)
        if migration is not None:
            migrated = migration(migrated)
    migrated[CONFIG_SCHEMA_VERSION_KEY] = _CURRENT_CONFIG_SCHEMA_VERSION
    return migrated, True, None


#: TASK-26040: set when the loaded config declares a schema version NEWER
#: than this application understands. The config is served untouched (never
#: mangled by a downgrade migration); this signal lets ``app.py`` surface a
#: loud, user-visible warning, mirroring ``ConfigLoadFailure``.
_CONFIG_SCHEMA_CONFLICT: Optional[str] = None


def get_config_schema_conflict() -> Optional[str]:
    """Return the newer-than-supported schema warning, if the last load hit one."""
    return _CONFIG_SCHEMA_CONFLICT


def _has_pending_config_migration(from_version: int) -> bool:
    """Whether any real migration FUNCTION exists between ``from_version`` and now.

    A bare version stamp (no function in range) is NOT worth a full-file
    rewrite -- it would strip the user's hand-written comments -- so the stamp
    rides the next natural save instead. Only an actual content transform
    forces the persist path.
    """
    return any(
        _CONFIG_MIGRATIONS.get(v) is not None
        for v in range(from_version + 1, _CURRENT_CONFIG_SCHEMA_VERSION + 1)
    )


def migrate_config_file_if_needed() -> Optional[Path]:
    """Persist a forward migration of the on-disk config (TASK-26040 AC#3/#4).

    Runs under the write lock. Reads the raw file, and only when an actual
    migration function must run does it back up the original and atomically
    rewrite the migrated result. A failed migration raises before any write,
    leaving the original file untouched. Returns the backup path when a
    migration was written, else ``None`` (no file, already current, bare
    stamp only, or a newer-than-supported version).
    """
    if not _CONFIG_MIGRATIONS:
        return None  # no real migration exists yet -- free no-op
    config_path = _get_effective_config_path()
    with _config_write_lock(config_path):
        current_serialized = _try_read_cli_config_serialized_unlocked(config_path)
        if current_serialized is None:
            return None  # no file yet -- creation stamps the version itself
        raw = tomllib.loads(current_serialized)
        raw_version = raw.get(CONFIG_SCHEMA_VERSION_KEY, 0)
        try:
            raw_version = int(raw_version)
        except (TypeError, ValueError):
            raw_version = 0
        if raw_version >= _CURRENT_CONFIG_SCHEMA_VERSION:
            return None
        if not _has_pending_config_migration(raw_version):
            return None
        migrated, changed, conflict = migrate_config_forward(raw)
        if conflict is not None or not changed:
            return None
        backup_path = _write_serialized_config_artifact_unlocked(
            _advanced_backup_path(config_path),
            current_serialized,
            config_path=config_path,
        )
        persisted = _config_data_for_persistence(migrated)
        raw_written = _write_raw_cli_config_unlocked(config_path, persisted)
        _publish_runtime_config_unlocked(
                raw_config=raw_written, config_path=config_path
            )
        return backup_path


#: (path, mtime_ns, size) of the corrupt file most recently preserved aside,
#: with the aside path. On a persistent parse failure the cache is never
#: populated, so every read re-enters this path; without this dedup a live TUI
#: would copy the same broken file and mint a new .corrupt-<stamp> on every
#: read (lane-7 review Important #2). Only a genuinely changed corrupt file
#: (the user edited it again, still broken) earns a fresh aside.
_LAST_PRESERVED_CORRUPT_KEY: Optional[tuple[str, int, int]] = None
_LAST_PRESERVED_CORRUPT_ASIDE: Optional[Path] = None


def _preserve_corrupt_config_aside(config_path: Path) -> Optional[Path]:
    """Copy an unparseable config file aside so the user's edits survive.

    TASK-26036 AC#2. Best-effort: a failure to copy must never break the
    fallback path (the whole point is resilience), so any error is logged
    and swallowed. Deduplicated by the corrupt file's (mtime_ns, size) so a
    persistently-broken file is preserved once, not on every read. Returns
    the aside path (freshly made or the prior one for an unchanged file).
    """
    global _LAST_PRESERVED_CORRUPT_KEY, _LAST_PRESERVED_CORRUPT_ASIDE
    try:
        source = Path(config_path)
        if not source.exists():
            return None
        stat = source.stat()
        key = (str(source), stat.st_mtime_ns, stat.st_size)
        if (
            key == _LAST_PRESERVED_CORRUPT_KEY
            and _LAST_PRESERVED_CORRUPT_ASIDE is not None
            and _LAST_PRESERVED_CORRUPT_ASIDE.exists()
        ):
            return _LAST_PRESERVED_CORRUPT_ASIDE
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        aside = source.with_name(f"{source.name}.corrupt-{stamp}")
        shutil.copy2(source, aside)
        logger.warning(f"Preserved unparseable config {source} at {aside}")
        _LAST_PRESERVED_CORRUPT_KEY = key
        _LAST_PRESERVED_CORRUPT_ASIDE = aside
        return aside
    except Exception as exc:  # noqa: BLE001 -- resilience path never raises
        logger.warning(f"Could not preserve corrupt config aside: {exc!r}")
        return None


class ConfigLoadFailure(NamedTuple):
    """The most recent config parse failure -- TASK-13157.

    ``_load_cli_config_bootstrap_unlocked`` already logs a TOMLDecodeError
    via loguru, but both of its callers (``load_cli_config_and_ensure_
    existence`` and ``load_settings``) discard the ``_ConfigBootstrapResult.
    succeeded`` flag and hand back bare in-memory defaults with no signal a
    caller can act on. A live-verification incident found this produces a
    silent, user-invisible fallback to the ``default_user`` profile -- the
    loaded config simply has no ``[general] users_name`` because the file
    that would have carried it never parsed. This module-level record is the
    "user-visible path" half of the fix: ``app.py`` reads it once at boot
    (mirroring the existing ``_instance_lock_status``/``_maybe_warn_second_
    instance`` pattern) and surfaces a persistent notification naming the
    file and the parse error, instead of leaving the degradation invisible.
    """

    path: Path
    message: str


#: Set inside the TOMLDecodeError branch below, cleared on the next
#: successful bootstrap. See ``ConfigLoadFailure`` and ``get_config_load_
#: failure()``.
_LAST_CONFIG_LOAD_FAILURE: Optional[ConfigLoadFailure] = None


def get_config_load_failure() -> Optional[ConfigLoadFailure]:
    """Return the most recent CLI config TOML parse failure, if any.

    ``None`` means the last bootstrap attempt (or no attempt yet) did not
    hit a TOML parse error -- it says nothing about OTHER bootstrap failure
    modes (permission/posture errors, decryption failures), which already
    have their own handling and are out of scope for this signal.
    """

    return _LAST_CONFIG_LOAD_FAILURE


class _ConfigBootstrapResult(NamedTuple):
    config: Dict[str, Any]
    succeeded: bool


@_config_participants.guarded
def _load_cli_config_bootstrap_unlocked(
    force_reload: bool = False,
) -> _ConfigBootstrapResult:
    global _CONFIG_CACHE, _CONFIG_CACHE_SOURCE, _LAST_CONFIG_LOAD_FAILURE
    global _FIRST_PROFILE_CREATED_THIS_SESSION
    global _CONFIG_FILE_STAMP, _CONFIG_STAT_CHECKED_MONOTONIC
    global _CONFIG_SCHEMA_CONFLICT
    config_path = _get_effective_config_path()
    if (
        _CONFIG_CACHE is not None
        and _CONFIG_CACHE_SOURCE == config_path
        and not force_reload
    ):
        return _ConfigBootstrapResult(_CONFIG_CACHE, True)

    # TASK-26036: retain the last successfully loaded config for THIS path
    # before clearing the cache, so a parse failure on a force-reload can
    # serve it instead of reverting security-relevant settings to defaults.
    retained_good = (
        copy.deepcopy(_CONFIG_CACHE)
        if _CONFIG_CACHE is not None and _CONFIG_CACHE_SOURCE == config_path
        else None
    )
    _CONFIG_CACHE = None
    _CONFIG_CACHE_SOURCE = None
    # TASK-26040 (lane-7 review Minor): clear any prior schema conflict so a
    # later parse-failure load does not retain a stale "newer version" warning.
    _CONFIG_SCHEMA_CONFLICT = None

    # Start with the programmatic defaults defined in CONFIG_TOML_CONTENT
    loaded_config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    bootstrap_succeeded = False
    application_directory = application_owned_config_directory(config_path)
    if application_directory is not None:
        directory_result = secure_private_directory(
            application_directory,
            create=True,
            application_owned=True,
        )
        _report_config_path_posture(directory_result, target_kind="directory")
    logger.info(f"Attempting to load CLI config from: {config_path}")
    try:
        with open_private_binary(config_path) as opened:
            _report_config_path_posture(opened.result)
            user_config_from_file = tomllib.load(opened.stream)
        # TASK-26040: migrate the RAW file forward before the default merge --
        # an unversioned file must be seen as the baseline (0), not inherit the
        # default's current version and skip its migrations. A newer-than-code
        # version is served untouched with a recorded conflict warning.
        user_config_from_file, _schema_changed, _schema_conflict = (
            migrate_config_forward(user_config_from_file)
        )
        from tldw_chatbook.Utils.reasoning_config import (
            migrate_legacy_reasoning_history,
        )

        migrate_legacy_reasoning_history(user_config_from_file)
        _CONFIG_SCHEMA_CONFLICT = _schema_conflict
        if _schema_conflict is not None:
            logger.warning(_schema_conflict)
        loaded_config = _merge_user_config_over_shipped_defaults(user_config_from_file)
        logger.info(f"Successfully loaded and merged CLI config from {config_path}")
        decryption = _decrypt_config_section_with_status(loaded_config, strict=True)
        if decryption.succeeded:
            loaded_config = decryption.config
            bootstrap_succeeded = True
        else:
            loaded_config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)
    except FileNotFoundError:
        logger.info(
            f"CLI Config file not found at {config_path}. Creating with default values from CONFIG_TOML_CONTENT."
        )
        created = create_private_text(
            config_path,
            CONFIG_TOML_CONTENT,
            application_owned_directory=application_directory,
        )
        _report_config_path_posture(created)
        logger.info(f"Created default CLI config file at {config_path}")
        loaded_config["_first_run"] = True
        _FIRST_PROFILE_CREATED_THIS_SESSION = True
        bootstrap_succeeded = True
    except PrivatePathError as exc:
        if application_directory is not None and exc.result.reason == "missing_parent":
            logger.info(
                f"CLI Config file not found at {config_path}. Creating with default values from CONFIG_TOML_CONTENT."
            )
            created = create_private_text(
                config_path,
                CONFIG_TOML_CONTENT,
                application_owned_directory=application_directory,
            )
            _report_config_path_posture(created)
            logger.info(f"Created default CLI config file at {config_path}")
            loaded_config["_first_run"] = True
            _FIRST_PROFILE_CREATED_THIS_SESSION = True
            bootstrap_succeeded = True
        else:
            raise
    except RecoveryRequired:
        raise
    except tomllib.TOMLDecodeError as e:
        logger.opt(exception=True).error(
            f"Error decoding CLI TOML config file {config_path}: {e}. Using internal defaults + any previous successful load."
        )
        # TASK-13157: this except branch is the exact point a real user's
        # config parse failure previously vanished -- logged here, then
        # thrown away by every caller (`load_cli_config_and_ensure_
        # existence`/`load_settings` both return only `.config`, never
        # `.succeeded`). Recording it lets `app.py` surface a loud,
        # user-visible notification instead of a silent `default_user`
        # fallback (see `ConfigLoadFailure`/`get_config_load_failure`).
        # TASK-26036: preserve the unparseable file aside (never lose the
        # user's edits) and serve the LAST KNOWN GOOD config instead of
        # built-in defaults, so a mid-edit break can't silently revert
        # encryption/database/provider settings.
        aside = _preserve_corrupt_config_aside(config_path)
        in_effect = "built-in defaults"
        if retained_good is not None:
            loaded_config = copy.deepcopy(retained_good)
            in_effect = "the last successfully loaded configuration"
        _LAST_CONFIG_LOAD_FAILURE = ConfigLoadFailure(
            path=config_path,
            message=(
                f"{e} (now serving {in_effect}"
                + (f"; the unreadable file was kept at {aside.name}" if aside else "")
                + ")"
            ),
        )
    except Exception as e:
        logger.opt(exception=True).error(
            f"An unexpected error occurred while loading CLI config {config_path}: {e}. Using internal defaults + any previous successful load."
        )

    if bootstrap_succeeded:
        _CONFIG_CACHE = loaded_config
        _CONFIG_CACHE_SOURCE = config_path
        # TASK-26038: stamp the file we just loaded so a later external edit
        # is detected, and reset the throttle so the next read re-checks.
        _CONFIG_FILE_STAMP = _current_config_file_stamp(config_path)
        _CONFIG_STAT_CHECKED_MONOTONIC = time.monotonic()
        # A later successful load (e.g. the user or the app repaired the
        # file) retires any previously recorded parse failure.
        _LAST_CONFIG_LOAD_FAILURE = None
    # Log the keys of the configuration being returned to verify its structure
    logger.debug(
        f"load_cli_config_and_ensure_existence returning config with top-level keys: {list(loaded_config.keys())}"
    )
    if "api_settings" in loaded_config:
        logger.debug(
            f"  'api_settings' found with keys: {list(loaded_config.get('api_settings', {}).keys())}"
        )
    else:
        logger.warning(
            "  'api_settings' key NOT FOUND in the loaded configuration for load_cli_config_and_ensure_existence."
        )

    return _ConfigBootstrapResult(loaded_config, bootstrap_succeeded)


def _warm_config_cache_hit() -> Optional[Dict[str, Any]]:
    """Return the installed config on a warm hit, or None to fall through.

    TASK-32804.1: `get_cli_setting` has ~400 call sites, many on the Textual
    event loop, and each funnels into `_load_cli_config_bootstrap`, which is
    `@config_participants.guarded` -- so every read, even a warm cache hit,
    paid the ADR-126 admission handshake (`operation(...)`). Measured at
    ~4.8 ms per warm read, tens of ms per second of streaming on the paint
    loop. The handshake protects the file/build lifecycle, not a pure
    in-memory cache read, so a warm hit is resolved HERE, ahead of the
    guarded call, amortising the handshake to actual (re)derivations.

    This repeats the guarded fast path's exact lock-free check (all atomic
    global reads, plus the throttled external-edit stat), and returns None
    on any miss so the guarded bootstrap runs unchanged. The soundness
    reasoning is `_load_cli_config_bootstrap`'s (a brand-new dict per
    publish; a miss re-reads; the edit check only turns a hit into a
    forced re-read, never a false hit).
    """
    config_path = _get_effective_config_path()
    generation_before = _CONFIG_GENERATION
    cached_config = _CONFIG_CACHE
    cached_source = _CONFIG_CACHE_SOURCE
    if (
        cached_config is not None
        and cached_source == config_path
        and _CONFIG_CACHE is cached_config
        and _CONFIG_GENERATION == generation_before
        and not _external_edit_detected(config_path)
    ):
        return cached_config
    return None


def config_read_would_wait() -> bool:
    """Whether a config read on this thread would wait for another thread.

    TASK-35645: warm settings and CLI config caches never wait; a cold one
    waits while another thread holds the config locks (a rebuild or a write),
    so they are probed without blocking, in their usual order. A storage
    pause makes a cold read raise rather than wait, as it always has.
    """
    if (
        _warm_config_cache_hit() is not None
        and _settings_cache_hit(_get_effective_config_path()) is not None
    ):
        return False
    with ExitStack() as acquired:
        for lock in (_settings_rebuild_lock(), _config_file_lock()):
            if not lock.acquire(blocking=False):
                return True
            acquired.callback(lock.release)
    return False


def load_cli_config_and_ensure_existence(
    force_reload: bool = False,
) -> Dict[str, Any]:  # Renamed from load_cli_config
    """Load the merged CLI configuration, creating the file if absent.

    Reads ~/.config/tldw_cli/config.toml, creating it from
    CONFIG_TOML_CONTENT defaults on first run, and uses those programmatic
    defaults as the base for the merge. A warm cache hit is served without
    the ADR-126 admission handshake (TASK-32804.1).

    Args:
        force_reload: Rebuild from disk even on a warm cache hit, bypassing
            the fast path and re-running the guarded bootstrap.

    Returns:
        The installed configuration mapping (programmatic defaults merged
        with the on-disk config).
    """
    # TASK-32804.1: serve a warm read without entering the admission
    # handshake (see `_warm_config_cache_hit`). A miss or a forced reload
    # falls through to the guarded bootstrap exactly as before.
    if not force_reload:
        warm = _warm_config_cache_hit()
        if warm is not None:
            return warm
    return _load_cli_config_bootstrap(force_reload=force_reload).config


def _setting_value_for_log(key: Any, value: Any) -> str:
    """Return a safe representation of a config setting value for logs."""
    if is_sensitive_config_key(key):
        return repr("<redacted>")
    return repr(value)


def _maybe_encrypt_setting_value(
    config_data: Dict[str, Any], key: Any, value: Any
) -> Any:
    encryption_config = config_data.get("encryption", {})
    if not (
        isinstance(encryption_config, dict)
        and encryption_config.get("enabled", False)
        and is_sensitive_config_key(key)
        and isinstance(value, str)
        and value
        and not value.startswith("enc:")
        and not (value.startswith("<") and value.endswith(">"))
    ):
        return value

    password = get_encryption_password()
    if not password:
        return value
    try:
        enc_module = get_encryption_module()
        encrypted_value = enc_module.encrypt_value(value, password)
        logger.info(f"Encrypted {key} in config section")
        return encrypted_value
    except Exception as error:
        logger.error(
            "Failed to encrypt config value (key={}, exception_category={}).",
            key,
            type(error).__name__,
        )
        raise


def _target_config_section(config_data: Dict[str, Any], section: str) -> Dict[str, Any]:
    current_level = config_data
    for part in section.split("."):
        next_level = current_level.setdefault(part, {})
        if not isinstance(next_level, dict):
            raise TypeError(part)
        current_level = next_level
    return current_level


# Eagerly created at import (single-threaded), so every caller shares one
# lock. Lazy init would let the first two concurrent workers each build a
# separate lock and defeat serialization on the first write.
import threading as _threading  # noqa: E402

from tldw_chatbook.Utils.private_paths import (
    _lock_stream_for_callbacks,
    open_private_lock_stream,
)

_LOCK_STREAM_SELECTOR_BINDING = (
    _lock_stream_for_callbacks,
    _lock_stream_for_callbacks.__code__,
    _lock_stream_for_callbacks.__globals__,
    tuple(cell.cell_contents for cell in _lock_stream_for_callbacks.__closure__),
)


_CONFIG_FILE_LOCK = _threading.RLock()


def _config_file_lock():
    """Return the shared config-file read-modify-write lock.

    Returns:
        The process-wide lock serializing config file saves/deletes across
        Textual thread workers so concurrent cycles cannot drop updates.
    """
    return _CONFIG_FILE_LOCK


def _bind_config_interprocess_lock(definition):
    """Capture the original optional selector before caller-marker lookup."""
    @contextmanager
    def _config_interprocess_lock(config_path: Path) -> Iterator[None]:
        """Hold one OS-backed lock across a whole-file config transaction."""

        with _config_participants.operation(sys.modules[__name__], target=config_path):
            lock_path = config_path.with_name(f"{config_path.name}.lock")
            application_directory = application_owned_config_directory(config_path)
            lock_opener = None
            binding = globals().get("_LOCK_STREAM_SELECTOR_BINDING")
            if binding is definition:
                selector, selector_code, selector_globals, selector_closure = definition
                if (
                    globals().get("_lock_stream_for_callbacks") is selector
                    and selector.__code__ is selector_code
                    and selector.__globals__ is selector_globals
                    and selector.__defaults__ is None
                    and selector.__kwdefaults__ is None
                    and selector.__closure__ is not None
                    and len(selector.__closure__) == len(selector_closure)
                    and all(cell.cell_contents is value for cell, value in zip(selector.__closure__, selector_closure, strict=True))
                ):
                    lock_opener = selector(
                        create_private_text, open_private_text_append_stream, open_private_lock_stream, globals()
                    )
            if lock_opener is None:
                try:
                    create_private_text(
                        lock_path,
                        "",
                        application_owned_directory=application_directory,
                    )
                except FileExistsError:
                    pass
                stream = open_private_text_append_stream(
                    lock_path,
                    application_owned_directory=application_directory,
                )
            else:
                stream = lock_opener(
                    lock_path, application_owned_directory=application_directory
                )
            locked = False
            try:
                from tldw_chatbook.Backup_Recovery import raw_participants

                operation = raw_participants.runtime_operation(config_path)
                while True:
                    _config_participants.check_lock_wait(sys.modules[__name__], operation)
                    try:
                        portalocker.lock(
                            stream,
                            portalocker.LockFlags.EXCLUSIVE
                            | portalocker.LockFlags.NON_BLOCKING,
                        )
                        locked = True
                        break
                    except portalocker.exceptions.AlreadyLocked:
                        _threading.Event().wait(0.05)
                _config_participants.check_lock_wait(sys.modules[__name__], operation)
                yield
            finally:
                if locked:
                    try:
                        portalocker.unlock(stream)
                    except Exception as error:
                        logger.error(
                            "Configuration write lock release failed (error_type={}).",
                            type(error).__name__,
                        )
                try:
                    stream.close()
                except Exception as error:
                    logger.error(
                        "Configuration write lock close failed (error_type={}).",
                        type(error).__name__,
                    )

    return _config_interprocess_lock

_config_interprocess_lock = _bind_config_interprocess_lock(_LOCK_STREAM_SELECTOR_BINDING)
del _bind_config_interprocess_lock


# The thread inside each config file's write transaction. The OS lock is taken
# on a fresh descriptor every time, so unlike the RLocks beside it, it is not
# reentrant: the holding thread can never get it a second time.
_CONFIG_WRITE_LOCK_HOLDERS: dict[Path, _threading.Thread] = {}


def _refuse_config_write_lock_reentry(config_path: Path) -> Path:
    """Refuse the calling thread a write transaction it is already inside.

    A caller that opens an operation of its own around the write lock calls
    this first: a refusal raised inside an entered operation is recorded as a
    failed config operation, which keeps the config participant from draining
    until the next write that succeeds.

    Returns:
        The key the write lock records its holder under.

    Raises:
        RecoveryRequired: The calling thread is already inside this file's
            write transaction, in an outer frame or in another asyncio task.
            Waiting would block the holder for good (TASK-33641.6).
    """

    # Keyed as the operation compares its target with the selection.
    holder_key = profile_paths.lexical_path(config_path)
    if _CONFIG_WRITE_LOCK_HOLDERS.get(holder_key) is _threading.current_thread():
        raise RecoveryRequired("raw_operation_provenance_invalid")
    return holder_key


@contextmanager
def _config_write_lock(config_path: Path) -> Iterator[None]:
    """Serialize one config write transaction within and across processes.

    Raises:
        RecoveryRequired: The calling thread is already inside this file's
            write transaction; refused before anything is entered.
    """

    holder_key = _refuse_config_write_lock_reentry(config_path)
    with _config_participants.operation(sys.modules[__name__], target=config_path):
        with (
            _settings_rebuild_lock(),
            _config_file_lock(),
            _config_interprocess_lock(config_path),
        ):
            _CONFIG_WRITE_LOCK_HOLDERS[holder_key] = _threading.current_thread()
            try:
                yield
            finally:
                _CONFIG_WRITE_LOCK_HOLDERS.pop(holder_key, None)


def _current_config_file_stamp(config_path: Path) -> Optional[tuple[int, int]]:
    """(mtime_ns, size) for the config file, or None when absent/unreadable."""
    try:
        st = os.stat(config_path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _external_edit_detected(config_path: Path) -> bool:
    """TASK-26038: THROTTLED check for an external edit since the last load.

    Returns True (invalidating the cache) at most once per external change,
    and stats at most once per ``_CONFIG_STAT_THROTTLE_SECONDS`` so the read
    hot path keeps its near-zero cost. No thread, no watcher.
    """
    global _CONFIG_STAT_CHECKED_MONOTONIC
    if _CONFIG_FILE_STAMP is None:
        return False
    now = time.monotonic()
    if now - _CONFIG_STAT_CHECKED_MONOTONIC < _CONFIG_STAT_THROTTLE_SECONDS:
        return False
    _CONFIG_STAT_CHECKED_MONOTONIC = now
    current = _current_config_file_stamp(config_path)
    return current is not None and current != _CONFIG_FILE_STAMP


@_config_participants.guarded
def _load_cli_config_bootstrap(
    force_reload: bool = False,
) -> _ConfigBootstrapResult:
    """Load or create the config while serializing the file/cache lifecycle.

    TASK-21124 -- LOCK-FREE FAST PATH. `get_cli_setting` has ~398 call
    sites, many on the Textual event loop, and every one funnels through
    here; taking `_config_file_lock()` before the cache check meant one
    config write (which holds that lock through two fsyncs and its TOML
    parses) stalled every loop-side read for the whole write. A warm cache
    hit now returns without touching the lock.

    Why the unlocked reads below are safe (CPython, GIL builds -- the only
    builds this app supports):

    * Each global read (`_CONFIG_GENERATION`, `_CONFIG_CACHE`,
      `_CONFIG_CACHE_SOURCE`) is a single atomic reference load; a reader
      can never see a partially-assigned cell.
    * Every publication installs a BRAND-NEW dict object (built via
      `copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)` + `deep_merge_dicts`, both
      of which construct fresh objects) -- a previously published dict is
      never re-installed. Therefore the `_CONFIG_CACHE is cached_config`
      re-check proves no install happened between the two cache reads, so
      the (cache, source) pair read here belongs to one single install and
      cannot be torn across two different installs.
    * Writers store in the order: cache=None, source=None, <build>,
      cache=new, source=path, all under `_config_file_lock`. In
      `_load_cli_config_bootstrap_unlocked` and
      `_invalidate_config_caches` the pre-clear is explicit; for
      `_install_bootstrap_cache_from_raw` the coupling is IMPLICIT -- it
      does no pre-clear itself, and the invariant holds only because
      every `raw_config` it receives comes from
      `_write_raw_cli_config_unlocked`, whose `_invalidate_config_caches()`
      call performed the cache=None/source=None stores moments earlier
      under the same lock. Combined with the identity re-check, a hit
      therefore returns the config that IS the currently installed cache
      for the caller's path.
    * The `_CONFIG_GENERATION` sandwich (read, ..., re-read) is the
      double-check the task's AC names, but it is NOT what makes the read
      sound -- the identity re-check above carries the soundness on its
      own (review of TASK-21124 proved by mutation that reordering the
      publish-time bump relative to the cache install leaves every
      guarantee intact). The generation term adds conservatism only: when
      a publication lands between the two generation reads, the reader
      declines the hit and re-validates through the locked path instead.
    * A write's invalidate window (cache=None between file replace and
      republish) makes readers MISS and serialize through the lock below,
      which is the pre-existing behavior for every miss.

    The fast path deliberately returns the SAME shared mutable dict the
    locked cache-hit path has always returned (no defensive copy) -- the
    copy semantics of `load_cli_config_and_ensure_existence` are unchanged.
    """

    edit_detected = False
    if not force_reload:
        config_path = _get_effective_config_path()
        generation_before = _CONFIG_GENERATION
        cached_config = _CONFIG_CACHE
        cached_source = _CONFIG_CACHE_SOURCE
        base_hit = (
            cached_config is not None
            and cached_source == config_path
            and _CONFIG_CACHE is cached_config
            and _CONFIG_GENERATION == generation_before
        )
        # TASK-26038: only stat when we would otherwise hit -- a miss already
        # re-reads. The (throttled) check turns a hit into a forced re-read,
        # never a false hit, so the lock-free soundness reasoning is intact.
        if base_hit:
            edit_detected = _external_edit_detected(config_path)
            if not edit_detected:
                return _ConfigBootstrapResult(cached_config, True)

    with _config_file_lock():
        # An external edit forces the locked path past its own cache fast
        # path so the changed file is actually re-read (and re-stamped).
        return _load_cli_config_bootstrap_unlocked(
            force_reload=force_reload or edit_detected
        )


@_config_participants.guarded
def _prepare_config_parent(config_path: Path) -> Path | None:
    """Secure the default config directory or verify a custom parent."""

    application_directory = application_owned_config_directory(config_path)
    if application_directory is not None:
        result = secure_private_directory(
            application_directory,
            create=True,
            application_owned=True,
        )
        _report_config_path_posture(result, target_kind="directory")
    else:
        verify_trusted_directory(
            config_path.parent,
            allow_shared_sticky=False,
        )
    return application_directory


@_config_participants.guarded
def _read_raw_cli_config_unlocked(config_path: Path) -> Dict[str, Any]:
    """Read the on-disk config mapping while the config lock is held."""

    try:
        with open_private_binary(config_path) as opened:
            _report_config_path_posture(opened.result)
            loaded = tomllib.load(opened.stream)
    except FileNotFoundError:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError("The CLI config must contain a top-level table")
    return loaded


def _invalidate_config_caches() -> None:
    """Drop config/settings caches after a successful whole-file write."""

    global _CONFIG_CACHE, _CONFIG_CACHE_SOURCE
    global _SETTINGS_CACHE, _SETTINGS_CACHE_SOURCE

    _CONFIG_CACHE = None
    _CONFIG_CACHE_SOURCE = None
    _SETTINGS_CACHE = None
    _SETTINGS_CACHE_SOURCE = None


class ConfigSerializationError(ValueError):
    """A config rewrite would have committed unparseable TOML -- TASK-13157.

    ``ValueError`` on purpose: ``replace_cli_config_serialized`` already uses
    a plain ``ValueError`` as its "config write rejected" signal (see
    ``_enforce_existing_encryption``), and ``persist_cli_config_for_
    shutdown`` already catches ``ValueError`` in its failure tuple -- making
    this a subclass means every existing call site's exception handling
    covers it without modification.
    """


@_config_participants.guarded
def _write_raw_cli_config_unlocked(
    config_path: Path,
    config_data: Mapping[str, Any],
) -> Dict[str, Any]:
    """Atomically write a private on-disk config while the lock is held.

    Returns:
        The verify parse-back of the exact serialized text committed to
        disk -- byte-for-byte what the next read of the file will produce.
        TASK-21124: callers hand this to
        ``_publish_runtime_config_unlocked(raw_config=...)`` so the publish
        step reuses this parse instead of re-reading and re-parsing the
        file it just wrote (twice: once for the bootstrap cache, once again
        inside ``load_settings(force_reload=True)``). The TASK-13157 guard
        below is therefore not a redundant extra parse anymore -- it IS the
        single serialization-side parse, and publishing its output is
        strictly more faithful than publishing the input mapping (it is
        the post-round-trip view the next boot would see).

    TASK-13157: every config-rewrite pass (settings-screen edits, the
    first-run wizard, and the quit's encryption pass in
    ``persist_cli_config_for_shutdown``) ends up here. The in-memory
    ``config_data`` is always a plain ``dict`` with
    inherently unique keys, so it cannot itself carry a duplicate key -- but
    ``toml.dumps`` (the third-party encoder) and ``tomllib`` (the stdlib
    reader the NEXT boot uses) are two independent implementations with no
    guaranteed round-trip contract between them. Before this fix, nothing
    verified the encoder's own output was still valid TOML before it was
    committed to disk; a bad serialization would sit there until the next
    boot's read failed, at which point the failure was ALSO silent (see
    ``ConfigLoadFailure``). Parsing the freshly serialized text back through
    the exact reader used on the next boot, before the atomic write, makes
    it categorically impossible for this function to leave behind a file its
    own next read cannot parse. That is the full extent of the guarantee:
    this guard proves PARSEABILITY, not fidelity -- it never compares the
    parsed-back value against ``config_data``, so an encoder-level mangling
    that still happens to produce valid TOML sails through unnoticed
    (measured: ``toml.dumps`` silently drops control characters such as
    ``\\x00``/``\\x1b`` from string values, and the mangled output still
    parses back cleanly). The write is NOT idempotent by construction and
    can change the on-disk result relative to what was requested; what is
    guaranteed is narrower -- it can never regress a config this function
    can write at all into a file the next read cannot parse.
    """

    application_directory = _prepare_config_parent(config_path)
    serialized = dumps_cli_config(dict(config_data))
    try:
        parsed_back = tomllib.loads(serialized)
    except tomllib.TOMLDecodeError as exc:
        logger.error(
            "Refusing to write CLI config: serialized TOML failed to parse back "
            "(error_type={})",
            type(exc).__name__,
        )
        raise ConfigSerializationError(
            f"Refusing to write {config_path}: the serialized configuration "
            f"does not parse back as valid TOML ({exc})"
        ) from exc
    _commit_serialized_cli_config_unlocked(
        config_path, serialized, application_directory
    )
    return parsed_back


def _commit_serialized_cli_config_unlocked(
    config_path: Path,
    serialized: str,
    application_directory: Path | None,
) -> None:
    """Atomically commit exact serialized config text while the lock is held.

    Shared by `_write_raw_cli_config_unlocked` and the encryption lifecycle's
    byte-exact rollback (TASK-34100.4), so a restore advances an owned
    recovery binding exactly like an ordinary write does. Not itself
    `_config_participants.guarded` (that allowlist is closed): every caller
    already runs inside `_config_write_lock`'s participant operation.
    """

    from .Backup_Recovery.config_binding import preserve_owned_binding

    with preserve_owned_binding(
        sys.modules[__name__], config_path, serialized
    ) as precondition:
        result = atomic_private_write_text(
            config_path,
            serialized,
            application_owned_directory=application_directory,
            target_precondition=precondition,
        )
    _report_config_path_posture(result)
    _invalidate_config_caches()


def _install_bootstrap_cache_from_raw(
    raw_config: Mapping[str, Any],
    config_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Install the bootstrap cache from an already-parsed raw config mapping.

    TASK-21124: replicates exactly the success tail of
    `_load_cli_config_bootstrap_unlocked` (merge the programmatic defaults,
    decrypt strictly, publish cache + source, retire any recorded parse
    failure) without re-reading or re-parsing the file. Used by
    `_publish_runtime_config_unlocked` with the verify parse-back a write
    just produced. Must be called with `_config_file_lock` held.

    TASK-21235: ``config_path`` is the path the CALLER's read/write used.
    It used to be recomputed here via `_get_effective_config_path()` at
    publish time, so a ``TLDW_CONFIG_PATH`` flip between the caller's read
    and this publish would label path-A data as the path-B cache
    (``_CONFIG_CACHE_SOURCE`` is exactly what the fast path's cache
    identity re-check compares against). Callers that hold a path pass it;
    the fallback recompute covers only callers with no path of their own.

    Returns:
        The installed merged+decrypted config, or ``None`` when strict
        decryption failed -- the caller then falls back to the full locked
        reload, which reproduces the historical failure handling
        (cache left empty, in-memory defaults returned).
    """

    global _CONFIG_CACHE, _CONFIG_CACHE_SOURCE, _LAST_CONFIG_LOAD_FAILURE
    global _CONFIG_FILE_STAMP, _CONFIG_STAT_CHECKED_MONOTONIC

    if config_path is None:
        config_path = _get_effective_config_path()
    merged = _merge_user_config_over_shipped_defaults(raw_config)
    decryption = _decrypt_config_section_with_status(merged, strict=True)
    if not decryption.succeeded:
        return None
    loaded_config = decryption.config
    # Same store order as `_load_cli_config_bootstrap_unlocked` (cache, then
    # source); the fast path's identity re-check makes either order safe --
    # see `_load_cli_config_bootstrap`. NOTE an implicit coupling: this
    # function performs no cache=None/source=None pre-clear of its own; the
    # documented writer store order holds only because every caller's
    # `raw_config` comes from `_write_raw_cli_config_unlocked`, whose
    # `_invalidate_config_caches()` did that pre-clear moments earlier under
    # the same lock. A new caller sourcing `raw_config` elsewhere must
    # preserve that ordering.
    _CONFIG_CACHE = loaded_config
    _CONFIG_CACHE_SOURCE = config_path
    _LAST_CONFIG_LOAD_FAILURE = None
    # TASK-26038 (lane-7 review Important #1): stamp the file WE just wrote so a
    # read after the stat-throttle window does not mistake our own write for an
    # external edit and force a redundant locked re-read. This is the write-path
    # twin of the stamp refresh in `_load_cli_config_bootstrap_unlocked`;
    # refreshing only one twin left the TASK-21124 coalescing broken on the
    # first read after every write.
    _CONFIG_FILE_STAMP = _current_config_file_stamp(config_path)
    _CONFIG_STAT_CHECKED_MONOTONIC = time.monotonic()
    return loaded_config


@_config_participants.guarded
def _publish_runtime_config_unlocked(
    raw_config: Mapping[str, Any] | None = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reload caches and publish one complete in-process config generation.

    Args:
        raw_config: Optional already-parsed raw on-disk mapping (the verify
            parse-back returned by `_write_raw_cli_config_unlocked`). When
            given and installable, the bootstrap cache is rebuilt from it
            without re-reading the file, and the settings rebuild reuses
            that just-primed cache (`reload_bootstrap=False`) -- TASK-21124
            takes one write from four TOML parses to two (the inherent
            read-modify-write read plus the TASK-13157 verify parse).
        config_path: TASK-21235: the path the caller's read/write used, so
            the bootstrap cache is published under the path its data was
            actually read from even if ``TLDW_CONFIG_PATH`` flips mid-write.
            Omitted by callers with no path of their own (the raw-less
            reload path recomputes it under the same lock).

    The `_CONFIG_GENERATION` bump is kept last for consistency, but the
    ordering is not load-bearing: the fast path's soundness rests on its
    cache-identity re-check, and the generation sandwich only adds
    conservatism (see `_load_cli_config_bootstrap` -- the TASK-21124
    review proved a bump-first mutant equivalent). Callers hold the write
    lock.
    """

    global settings, _CONFIG_GENERATION, _CONFIG_PERSISTENCE_ERROR

    loaded: Optional[Dict[str, Any]] = None
    if raw_config is not None:
        loaded = _install_bootstrap_cache_from_raw(
            raw_config, config_path=config_path
        )
    if loaded is None:
        bootstrap = _load_cli_config_bootstrap_unlocked(force_reload=True)
        if not bootstrap.succeeded:
            raise ValueError("Configuration runtime reload failed")
        loaded = bootstrap.config
    settings = load_settings(force_reload=True, reload_bootstrap=False)
    _CONFIG_GENERATION += 1
    _CONFIG_PERSISTENCE_ERROR = None
    return loaded


class RuntimeConfigSnapshot(NamedTuple):
    """A defensive normalized config view paired with its write generation."""

    generation: int
    values: Dict[str, Any]


class AtomicConfigSnapshot(NamedTuple):
    """A locked authoritative config view paired with its published generation."""

    generation: int
    values: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class LiteralSettingsMutation:
    """Exact set/delete operations addressed by literal TOML mapping paths."""

    section_values: Mapping[tuple[str, ...], Mapping[str, object]]
    delete_keys: Mapping[tuple[str, ...], Collection[str]]


@dataclass(frozen=True, slots=True)
class AtomicLiteralMutationSnapshot:
    """Authoritative raw and effective config views held under the write lock."""

    generation: int
    raw_values: Mapping[str, object]
    effective_values: Mapping[str, object]
    config_path: Path | None = None


@dataclass(frozen=True, slots=True)
class LiteralConfigMutationResult:
    """Two-phase literal mutation outcome with an optional fresh settings view."""

    file_replaced: bool
    caches_reloaded: bool
    settings_view: Mapping[str, object] | None
    failure_phase: str | None


@dataclass(frozen=True, slots=True)
class HookConfigSnapshot:
    """Detached saved hook section paired with its actual file identity."""

    config_path: Path
    section_present: bool
    section: object = field(repr=False)
    section_stamp: str
    profile_data_dir: Path | None = field(default=None, repr=False)


def _hooks_config_snapshot(
    config_path: Path, raw: Mapping[str, object]
) -> HookConfigSnapshot:
    present = "hooks" in raw
    section = copy.deepcopy(raw.get("hooks"))
    stamp: dict[str, object] = {"present": present}
    if present:
        stamp["section"] = section
    encoded = toml.dumps(stamp).encode("utf-8")
    return HookConfigSnapshot(
        config_path.resolve(),
        present,
        section,
        hashlib.sha256(encoded).hexdigest(),
        profile_paths.user_data_dir(raw),
    )


@contextmanager
def locked_hooks_config_snapshot() -> Iterator[HookConfigSnapshot]:
    """Hold the config writer lock while reading and consuming hook authority."""
    config_path = validate_path_simple(
        _get_effective_config_path(),
        require_exists=False,
        probe_existing=False,
        reject_shell_metacharacters=False,
    )
    with _config_write_lock(config_path):
        if _get_effective_config_path().resolve() != config_path.resolve():
            raise ValueError("Hook configuration file changed; reload.")
        yield _hooks_config_snapshot(
            config_path, _read_raw_cli_config_unlocked(config_path)
        )


def read_hooks_config_snapshot() -> HookConfigSnapshot:
    """Read current saved hook definitions without consulting runtime caches."""
    with locked_hooks_config_snapshot() as snapshot:
        return snapshot


def replace_hooks_config_snapshot(
    expected: HookConfigSnapshot,
    replacement: Mapping[str, object],
) -> LiteralConfigMutationResult:
    """Replace the current hook section through the canonical guarded writer."""
    owned = copy.deepcopy(dict(replacement))

    def build(snapshot: AtomicLiteralMutationSnapshot) -> LiteralSettingsMutation:
        if snapshot.config_path is None:
            raise ValueError("Hook configuration file unavailable.")
        current = _hooks_config_snapshot(snapshot.config_path, snapshot.raw_values)
        if (
            current.config_path != expected.config_path
            or current.section_stamp != expected.section_stamp
            or current.profile_data_dir != expected.profile_data_dir
        ):
            raise ValueError("Hook configuration changed; reload before saving.")
        if current.section_present and not isinstance(current.section, Mapping):
            raise ValueError("Repair malformed Hooks section in Advanced Config.")
        old = current.section if isinstance(current.section, Mapping) else {}
        return LiteralSettingsMutation(
            section_values={
                ("hooks",): {
                    **{
                        key: value
                        for key, value in old.items()
                        if key not in {"enabled", "hook"}
                    },
                    **owned,
                }
            },
            delete_keys={
                ("hooks",): tuple(
                    key
                    for key in ("enabled", "hook")
                    if key in old and key not in owned
                )
            },
        )

    return apply_literal_settings_transaction_to_cli_config(build)


def _atomic_config_values_from_raw(
    config_data: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge and decrypt the exact raw mapping already read under the write lock."""

    merged = _merge_user_config_over_shipped_defaults(config_data)
    decryption = _decrypt_config_section_with_status(merged, strict=True)
    if not decryption.succeeded:
        raise ValueError("Authoritative configuration could not be decrypted")
    return decryption.config


def get_atomic_config_snapshot() -> AtomicConfigSnapshot:
    """Read the authoritative config through the same lock used by mutations."""

    config_path = _get_effective_config_path()
    with _config_write_lock(config_path):
        raw = _read_raw_cli_config_unlocked(config_path)
        return AtomicConfigSnapshot(
            generation=_CONFIG_GENERATION,
            values=_atomic_config_values_from_raw(raw),
        )


def get_runtime_config_generation() -> int:
    """Read the publication counter without locks, I/O, or copying config.

    This is a baseline, not an acceptance fence. Callers must finish with
    ``run_if_runtime_config_generation_current``; a concurrent publication
    conservatively invalidates the captured baseline.
    """
    return _CONFIG_GENERATION


def get_runtime_config_snapshot(
    *,
    force_reload: bool = False,
) -> RuntimeConfigSnapshot:
    """Return a defensive current runtime config view.

    PERF-06 (TASK-33265): a warm snapshot is served without the ADR-126
    admission handshake (a deep copy of the installed settings under the
    same two in-process locks). A miss or forced reload runs the guarded
    :func:`_get_runtime_config_snapshot_guarded`, exactly as before.

    Args:
        force_reload: Rebuild the settings from disk even when they are cached.

    Returns:
        A snapshot of the current config generation and a deep copy of the
        merged settings, safe for the caller to mutate.
    """

    if not force_reload:
        # Never block here: a held lock means a write is in flight, and the
        # guarded path waits for it pause-aware, so recovery can cancel the
        # wait (Qodo, #2903).
        rebuild, file_lock = _settings_rebuild_lock(), _config_file_lock()
        if rebuild.acquire(blocking=False):
            try:
                if file_lock.acquire(blocking=False):
                    try:
                        # Copy the hit that was checked: a second lookup could
                        # miss after a lock-free clear and rebuild under the
                        # locks, reversing admission-then-locks (Qodo, #2903).
                        cached = _settings_cache_hit(_get_effective_config_path())
                        if cached is not None:
                            return RuntimeConfigSnapshot(
                                generation=_CONFIG_GENERATION,
                                values=copy.deepcopy(cached),
                            )
                    finally:
                        file_lock.release()
            finally:
                rebuild.release()
    return _get_runtime_config_snapshot_guarded(force_reload=force_reload)


@_config_participants.guarded
def _get_runtime_config_snapshot_guarded(
    *,
    force_reload: bool = False,
) -> RuntimeConfigSnapshot:
    """Guarded body of :func:`get_runtime_config_snapshot` (miss or reload)."""

    with _settings_rebuild_lock(), _config_file_lock():
        values = load_settings(force_reload=force_reload)
        return RuntimeConfigSnapshot(
            generation=_CONFIG_GENERATION,
            values=copy.deepcopy(values),
        )


def _published_runtime_config_snapshot() -> RuntimeConfigSnapshot:
    """Read the already-published config generation without filesystem I/O."""

    with _settings_rebuild_lock(), _config_file_lock():
        return RuntimeConfigSnapshot(
            generation=_CONFIG_GENERATION,
            values=copy.deepcopy(settings),
        )


def run_if_runtime_config_generation_current(
    expected_generation: int,
    action: Callable[[], bool],
) -> bool:
    """Linearize a nonblocking action against runtime config publication.

    Lock order is always ``config -> action-owned lock``. The supplied action
    must be process-local and nonblocking; first-chat handoff acknowledgement
    uses this to acquire only the pending-handoff lock. Pending handoff stage,
    claim, and release paths never acquire the config lock, so there is no
    reverse ``handoff -> config`` edge.

    Args:
        expected_generation: Runtime generation the caller validated.
        action: Nonblocking action to execute while publication is fenced.

    Returns:
        ``True`` only when the generation matched and the action succeeded.
    """

    if type(expected_generation) is not int or expected_generation < 0:
        raise ValueError("Expected config generation must be nonnegative")
    if not callable(action):
        raise TypeError("Config generation action must be callable")
    with _config_file_lock():
        if _CONFIG_GENERATION != expected_generation:
            return False
        return action() is True


def _encryption_enabled(config_data: Mapping[str, Any]) -> bool:
    encryption = config_data.get("encryption", {})
    return isinstance(encryption, Mapping) and encryption.get("enabled", False) is True


def _enforce_existing_encryption(
    current_config: Mapping[str, Any],
    replacement: Mapping[str, Any],
) -> None:
    if _encryption_enabled(current_config) and not _encryption_enabled(replacement):
        raise ValueError(
            "Encrypted config replacement must keep encryption enabled; "
            "disable encryption explicitly"
        )


_REVISION_OWNED_CONFIG_SECTIONS = frozenset({"speech_studio"})


def _preserve_revision_owned_sections(
    current: Mapping[str, Any],
    replacement: Mapping[str, Any],
) -> Dict[str, Any]:
    """Keep dedicated revision-owned sections out of whole-config writers."""

    selected = copy.deepcopy(dict(replacement))
    for section in _REVISION_OWNED_CONFIG_SECTIONS:
        if section in current:
            selected[section] = copy.deepcopy(current[section])
        else:
            selected.pop(section, None)
    return selected


@_config_participants.guarded
def _try_read_cli_config_serialized_unlocked(config_path: Path) -> str | None:
    try:
        with open_private_binary(config_path) as opened:
            _report_config_path_posture(opened.result)
            return opened.stream.read().decode("utf-8")
    except FileNotFoundError:
        return None


@_config_participants.guarded
def _read_cli_config_serialized_unlocked(config_path: Path) -> str:
    serialized = _try_read_cli_config_serialized_unlocked(config_path)
    if serialized is None:
        _load_cli_config_bootstrap_unlocked(force_reload=True)
        with open_private_binary(config_path) as opened:
            _report_config_path_posture(opened.result)
            return opened.stream.read().decode("utf-8")
    return serialized


@_config_participants.guarded
def read_cli_config_serialized() -> str:
    """Return the effective config's exact serialized on-disk representation."""

    with _config_file_lock():
        return _read_cli_config_serialized_unlocked(get_cli_config_path())


@dataclass(frozen=True, repr=False)
class ConfigFileSnapshot:
    """Exact in-memory raw file/profile identity; absence differs from empty text."""

    path: Path
    serialized: str | None


class ConfigSnapshotConflictError(ValueError):
    """The raw editor's file/profile baseline no longer matches the current file."""

    def __init__(self) -> None:
        super().__init__("The config file or profile changed. Reload it before saving.")


class ConfigPostCommitError(RuntimeError):
    """Report a committed raw replacement whose snapshot or runtime refresh failed.

    Attributes:
        snapshot: Exact committed file snapshot, or None if its read failed.
        backup_path: Backup created before replacement, if a previous file existed.
    """

    def __init__(
        self, snapshot: ConfigFileSnapshot | None, backup_path: Path | None
    ) -> None:
        """Retain recovery state without exposing config text or the original error.

        Args:
            snapshot: Committed snapshot when its post-write read succeeded.
            backup_path: Existing config's backup, or None when none was created.
        """
        super().__init__("Config saved to disk, but post-save refresh failed.")
        self.snapshot = snapshot
        self.backup_path = backup_path


def read_cli_config_snapshot() -> ConfigFileSnapshot:
    """Read exact serialized config without parsing or creating a missing file."""

    _refuse_config_write_lock_reentry(get_cli_config_path())
    with _config_participants.operation(sys.modules[__name__]):
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            return ConfigFileSnapshot(
                config_path, _try_read_cli_config_serialized_unlocked(config_path)
            )


def _advanced_backup_path(config_path: Path) -> Path:
    return config_path.with_suffix(config_path.suffix + ".bak")


def _write_serialized_config_artifact_unlocked(
    path: Path,
    serialized: str,
    *,
    config_path: Path,
) -> Path:
    if path in (config_path, _advanced_backup_path(config_path)):
        route, target = "config", config_path
    else:
        prefix, suffix = "config_backup_", ".toml"
        if (
            path.parent != config_path.parent
            or not path.name.startswith(prefix)
            or not path.name.endswith(suffix)
        ):
            raise ValueError("invalid_config_snapshot_target")
        route, target = "config_snapshot", path.name[len(prefix) : -len(suffix)]
        if _config_snapshot_path(config_path, target) != path:
            raise ValueError("invalid_config_snapshot_target")
    with _config_participants.operation(
        sys.modules[__name__], route=route, target=target
    ):
        application_directory = _prepare_config_parent(config_path)
        result = atomic_private_write_text(
            path,
            serialized,
            application_owned_directory=application_directory,
        )
        _report_config_path_posture(result, target_kind="snapshot")
        return result.lexical_path


@_config_participants.guarded
def read_cli_config_backup_serialized() -> str:
    """Return the exact serialized advanced-editor backup."""

    with _config_file_lock():
        config_path = get_cli_config_path()
        backup_path = _advanced_backup_path(config_path)
        with open_private_binary(backup_path) as opened:
            _report_config_path_posture(opened.result, target_kind="snapshot")
            return opened.stream.read().decode("utf-8")


def replace_cli_config_serialized(
    serialized: str,
    *,
    create_backup: bool = True,
) -> tuple[Dict[str, Any], Path | None]:
    """Validate and replace raw TOML without downgrading encryption.

    Raises:
        ConfigPostCommitError: Replacement succeeded but post-save refresh failed.
        RecoveryRequired: The calling thread is already inside this file's
            write transaction; refused before the guarded body opens.
    """

    _refuse_config_write_lock_reentry(get_cli_config_path())
    return _replace_cli_config_serialized_guarded(
        serialized, create_backup=create_backup
    )


@_config_participants.guarded
def _replace_cli_config_serialized_guarded(
    serialized: str,
    *,
    create_backup: bool,
) -> tuple[dict[str, Any], Path | None]:
    """Guarded body of :func:`replace_cli_config_serialized`."""

    loaded, backup_path, _ = _replace_cli_config_serialized(
        serialized, create_backup=create_backup
    )
    return loaded, backup_path


def replace_cli_config_snapshot(
    serialized: str,
    expected_snapshot: ConfigFileSnapshot,
    *,
    create_backup: bool = True,
) -> tuple[Dict[str, Any], Path | None, ConfigFileSnapshot]:
    """Replace raw TOML only while its exact file/profile baseline still matches.

    The comparison, backup, replacement and returned on-disk snapshot share one
    write lock. A conflict leaves both config and backup unchanged.

    Raises:
        ConfigSnapshotConflictError: The current file or profile no longer matches.
        ConfigPostCommitError: The file was replaced but its snapshot or runtime
            refresh failed. The exception carries the committed snapshot when
            available; callers must not report this as a failed disk write.
    """

    if not isinstance(expected_snapshot, ConfigFileSnapshot):
        raise TypeError("A config file snapshot is required for guarded replacement")
    return _replace_cli_config_serialized(
        serialized,
        create_backup=create_backup,
        expected_snapshot=expected_snapshot,
    )


def _replace_cli_config_serialized(
    serialized: str,
    *,
    create_backup: bool,
    expected_snapshot: ConfigFileSnapshot | None = None,
) -> tuple[Dict[str, Any], Path | None, ConfigFileSnapshot]:
    """Own raw replacements, optionally guarded by an exact editor baseline."""

    replacement = tomllib.loads(serialized)
    if not isinstance(replacement, dict):
        raise TypeError("The CLI config must contain a top-level table")

    config_path = get_cli_config_path()
    with _config_write_lock(config_path):
        if expected_snapshot is not None and (
            expected_snapshot.path != config_path
            or config_path != get_cli_config_path()
        ):
            raise ConfigSnapshotConflictError()
        current_serialized = _try_read_cli_config_serialized_unlocked(config_path)
        if (
            expected_snapshot is not None
            and expected_snapshot.serialized != current_serialized
        ):
            raise ConfigSnapshotConflictError()
        if current_serialized is None:
            current_config: Dict[str, Any] = {}
        else:
            current_config = tomllib.loads(current_serialized)
        replacement = _preserve_revision_owned_sections(
            current_config,
            replacement,
        )
        _enforce_existing_encryption(current_config, replacement)
        persisted = _config_data_for_persistence(replacement)
        backup_path: Path | None = None
        if create_backup and current_serialized is not None:
            backup_path = _write_serialized_config_artifact_unlocked(
                _advanced_backup_path(config_path),
                current_serialized,
                config_path=config_path,
            )
        raw_written = _write_raw_cli_config_unlocked(config_path, persisted)
        saved_snapshot = None
        try:
            saved_serialized = _try_read_cli_config_serialized_unlocked(config_path)
            if saved_serialized is None:
                raise OSError("The committed config snapshot is unavailable.")
            saved_snapshot = ConfigFileSnapshot(config_path, saved_serialized)
            loaded = _publish_runtime_config_unlocked(
                raw_config=raw_written, config_path=config_path
            )
        except Exception:  # Post-commit errors must retain the successful disk outcome.
            raise ConfigPostCommitError(saved_snapshot, backup_path) from None
        return loaded, backup_path, saved_snapshot


def _upgrade_legacy_ciphertext(
    table: Mapping[str, Any], password: str
) -> tuple[Dict[str, Any], bool]:
    """Re-encrypt every v1 ``enc:`` value in place under ``password``.

    Walks tables as decryption does (it never reads arrays). A value stays
    encrypted whatever its key, so nothing is written back in plaintext.

    Returns:
        The upgraded copy, and whether any value changed.
    """

    enc_module = get_encryption_module()
    upgraded = False
    result: Dict[str, Any] = {}
    for key, value in table.items():
        if isinstance(value, Mapping):
            result[key], nested = _upgrade_legacy_ciphertext(value, password)
            upgraded = upgraded or nested
        elif enc_module.is_legacy(value):
            plaintext = enc_module.decrypt_value(value, password)
            result[key] = enc_module.encrypt_value(plaintext, password)
            upgraded = True
        else:
            result[key] = value
    return result, upgraded


def persist_cli_config_for_shutdown() -> bool:
    """Finish encryption at rest for an unlocked encrypted config at quit.

    Every save has already written the file, and every load merges the
    shipped template under it. Writing that merge back re-inserted template
    keys a save had deleted, reordered the file and stripped its comments
    (TASK-35835). What a quit still carries is encryption work that only an
    unlocked session can do, on the file's own contents:

    * a secret added in plaintext (a hand edit) is encrypted;
    * values and the password check still in the v1 format are re-encrypted
      as v2. A v1 value costs a 2^20 scrypt derivation (about 1 GiB, 3 s) at
      every launch, and short of a password change nothing else upgrades it.

    With neither, or with the session locked, nothing is written.

    Returns:
        False when the file cannot be loaded safely or the write fails.
    """

    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            from .Backup_Recovery.profile_open import unchanged_selected_config

            # Settings writes already persist. A rewrite would invalidate the
            # selected recovery generation's config fingerprint.
            if unchanged_selected_config(config_path):
                return True
            bootstrap = _load_cli_config_bootstrap_unlocked(force_reload=True)
            if not bootstrap.succeeded:
                logger.warning(
                    "Shutdown config persistence skipped because the current "
                    "file could not be loaded safely."
                )
                return False
            serialized = _try_read_cli_config_serialized_unlocked(config_path)
            current = tomllib.loads(serialized) if serialized is not None else {}
            encryption = current.get("encryption", {})
            password = get_encryption_password()
            if not (
                isinstance(encryption, Mapping)
                and encryption.get("enabled", False)
                and password
            ):
                return True
            persisted, upgraded = _upgrade_legacy_ciphertext(current, password)
            if _contains_unencrypted_sensitive_value(persisted):
                persisted = _config_data_for_persistence(persisted)
            elif not upgraded:
                return True
            # The file may have changed since the load above; check the exact
            # document, so a value under another password is never stranded.
            _validate_encrypted_document(persisted, password)
            raw_written = _write_raw_cli_config_unlocked(config_path, persisted)
            _publish_runtime_config_unlocked(
                raw_config=raw_written, config_path=config_path
            )
        return True
    except (OSError, TypeError, ValueError, toml.TomlDecodeError) as exc:
        logger.warning(
            "Shutdown config persistence failed (error_type={}).",
            type(exc).__name__,
        )
        return False


def _contains_unencrypted_sensitive_value(config_data: Mapping[str, Any]) -> bool:
    """Return whether an encrypted config mapping contains plaintext secrets."""

    enc_module = get_encryption_module()
    for key, value in config_data.items():
        if key == "encryption":
            continue
        if isinstance(value, Mapping):
            if _contains_unencrypted_sensitive_value(value):
                return True
            continue
        if not (
            is_sensitive_config_key(key) and isinstance(value, str) and value.strip()
        ):
            continue
        if value.startswith("<") and value.endswith(">"):
            continue
        if not enc_module.is_encrypted(value):
            return True
    return False


def _config_data_for_persistence(
    config_data: Mapping[str, Any],
) -> Dict[str, Any]:
    """Preserve encryption-at-rest when persisting a decrypted config view."""

    selected = copy.deepcopy(dict(config_data))
    encryption = selected.get("encryption", {})
    if not (isinstance(encryption, Mapping) and encryption.get("enabled", False)):
        return selected

    password = get_encryption_password()
    if password:
        return encrypt_api_keys_in_config(selected, password)
    if _contains_unencrypted_sensitive_value(selected):
        raise ValueError(
            "Cannot persist plaintext secrets while config encryption is locked"
        )
    return selected


def replace_cli_config(config_data: Mapping[str, Any]) -> Dict[str, Any]:
    """Atomically replace the effective config and refresh all config caches."""

    config_path = get_cli_config_path()
    with _config_write_lock(config_path):
        current = _read_raw_cli_config_unlocked(config_path)
        replacement = _preserve_revision_owned_sections(current, config_data)
        _enforce_existing_encryption(current, replacement)
        persisted = _config_data_for_persistence(replacement)
        raw_written = _write_raw_cli_config_unlocked(config_path, persisted)
        return _publish_runtime_config_unlocked(
                raw_config=raw_written, config_path=config_path
            )


def _config_snapshot_path(config_path: Path, timestamp: str) -> Path:
    """Select one config-owned snapshot basename before any filesystem effect."""
    if (
        not isinstance(timestamp, str)
        or not timestamp
        or any(char in timestamp for char in ("/", "\\", "\x00"))
    ):
        raise ValueError("invalid_config_snapshot_target")
    return config_path.parent / f"config_backup_{timestamp}.toml"


def export_cli_config_snapshot(
    config_data: Mapping[str, Any] | None = None,
    *,
    timestamp: str | None = None,
) -> Path:
    """Create an owner-only snapshot beside the effective config file."""

    snapshot_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    with (
        _config_participants.operation(
            sys.modules[__name__], route="config_snapshot", target=snapshot_timestamp
        ),
        _config_file_lock(),
    ):
        config_path = get_cli_config_path()
        snapshot_path = _config_snapshot_path(config_path, snapshot_timestamp)
        serialized = _try_read_cli_config_serialized_unlocked(config_path)
        if serialized is None:
            if config_data is None:
                raise FileNotFoundError(config_path)
            serialized = dumps_cli_config(_config_data_for_persistence(config_data))
        result_path = _write_serialized_config_artifact_unlocked(
            snapshot_path,
            serialized,
            config_path=config_path,
        )
    return result_path


@dataclass(frozen=True, slots=True)
class ConfigMutationResult:
    """Outcome of an atomic config file replacement and cache reload."""

    file_replaced: bool
    caches_reloaded: bool
    failure_phase: Literal["before_replace", "cache_reload"] | None
    conflict: bool = False
    conflict_reason: Literal["identity_changed"] | None = None

    @property
    def fully_applied(self) -> bool:
        """Return whether both persistence phases completed."""
        return self.file_replaced and self.caches_reloaded


def _valid_nonnegative_revision(value: object) -> int | None:
    """Return an exact nonnegative revision or ``None`` when malformed."""

    if type(value) is not int or value < 0:
        return None
    return value


def _without_revision(
    values: Mapping[str, Any],
    revision_key: str,
) -> Dict[str, Any]:
    """Return a defensive section copy without its concurrency revision."""

    comparable = copy.deepcopy(dict(values))
    comparable.pop(revision_key, None)
    return comparable


def replace_revisioned_settings_section_to_cli_config(
    section: str,
    values: Mapping[str, Any],
    *,
    expected_revision: int,
    revision_key: str = "revision",
) -> ConfigMutationResult:
    """Atomically replace one top-level revisioned configuration section.

    A missing or structurally malformed section has revision zero, allowing an
    explicit owner to recover only its own section. The comparison, whole
    section replacement, atomic file write, and cache publication all occur
    while the existing process lock and a stable private interprocess lock are
    held.

    Args:
        section: Exact top-level section owned by the caller.
        values: Complete replacement section, including its next revision.
        expected_revision: Revision observed by the caller.
        revision_key: Key carrying the persisted section revision.

    Returns:
        A mutation result with ``conflict=True`` when the observed revision is
        stale. Invalid transitions fail before replacement.
    """

    try:
        if type(section) is not str or not section or "." in section:
            raise ValueError("Revisioned section must be a top-level name")
        if not isinstance(values, Mapping):
            raise TypeError("Revisioned section values must be a mapping")
        if type(revision_key) is not str or not revision_key:
            raise ValueError("Revision key must be a non-empty string")
        validated_expected = _valid_nonnegative_revision(expected_revision)
        if validated_expected is None:
            raise ValueError("Expected revision must be a nonnegative integer")
        replacement = copy.deepcopy(dict(values))
        replacement_revision = _valid_nonnegative_revision(
            replacement.get(revision_key)
        )
        if replacement_revision != validated_expected + 1:
            raise ValueError("Replacement revision must advance by exactly one")
        config_path = _get_effective_config_path()
    except Exception as error:
        logger.error(
            "Revisioned configuration replacement failed "
            "(phase=validation, section={}, error_type={}).",
            section if isinstance(section, str) else "invalid",
            type(error).__name__,
        )
        return ConfigMutationResult(False, False, "before_replace")

    with ExitStack() as locks:
        try:
            locks.enter_context(_config_write_lock(config_path))
        except Exception as error:
            logger.error(
                "Revisioned configuration replacement failed "
                "(phase=lock, section={}, error_type={}).",
                section,
                type(error).__name__,
            )
            return ConfigMutationResult(False, False, "before_replace")
        try:
            config_data = _read_raw_cli_config_unlocked(config_path)
        except Exception as error:
            logger.error(
                "Revisioned configuration replacement failed "
                "(phase=read, section={}, error_type={}).",
                section,
                type(error).__name__,
            )
            return ConfigMutationResult(False, False, "before_replace")

        current = config_data.get(section)
        current_revision = 0
        current_revision_is_valid = False
        if isinstance(current, Mapping):
            parsed_revision = _valid_nonnegative_revision(current.get(revision_key))
            if parsed_revision is not None:
                current_revision = parsed_revision
                current_revision_is_valid = True
        if current_revision != validated_expected:
            logger.info(
                "Revisioned configuration replacement rejected stale writer: "
                "section={}, expected_revision={}, current_revision={}",
                section,
                validated_expected,
                current_revision,
            )
            return ConfigMutationResult(False, False, None, conflict=True)

        if (
            current_revision_is_valid
            and isinstance(current, Mapping)
            and _without_revision(
                current,
                revision_key,
            )
            == _without_revision(replacement, revision_key)
        ):
            return ConfigMutationResult(False, False, None)

        config_data[section] = replacement
        try:
            persisted = _config_data_for_persistence(config_data)
            raw_written = _write_raw_cli_config_unlocked(config_path, persisted)
        except Exception as error:
            logger.error(
                "Revisioned configuration replacement failed "
                "(phase=before_replace, section={}, error_type={}).",
                section,
                type(error).__name__,
            )
            return ConfigMutationResult(False, False, "before_replace")

        try:
            _publish_runtime_config_unlocked(
                raw_config=raw_written, config_path=config_path
            )
        except Exception as error:
            logger.error(
                "Revisioned configuration replacement failed "
                "(phase=cache_reload, section={}, error_type={}).",
                section,
                type(error).__name__,
            )
            return ConfigMutationResult(True, False, "cache_reload")

        return ConfigMutationResult(True, True, None)


def _delete_config_keys(
    config_data: Dict[str, Any],
    delete_keys: Mapping[str, Collection[str]],
) -> bool:
    """Delete exact keys and report whether the config changed."""
    return _delete_literal_config_keys(
        config_data,
        {tuple(section.split(".")): keys for section, keys in delete_keys.items()},
    )


def _literal_config_section(
    config_data: Dict[str, Any],
    path: tuple[str, ...],
    *,
    create: bool,
) -> Dict[str, Any] | None:
    """Resolve one literal mapping path without interpreting punctuation."""

    current_level: Any = config_data
    for part in path:
        if not isinstance(current_level, dict):
            raise TypeError(part)
        if part not in current_level:
            if not create:
                return None
            current_level[part] = {}
        current_level = current_level[part]
    if not isinstance(current_level, dict):
        raise TypeError(".".join(path))
    return current_level


def _delete_literal_config_keys(
    config_data: Dict[str, Any],
    delete_keys: Mapping[tuple[str, ...], Collection[str]],
) -> bool:
    """Delete exact keys beneath literal paths and report whether any existed."""

    changed = False
    missing = object()
    for path, keys in delete_keys.items():
        current_level = _literal_config_section(config_data, path, create=False)
        if current_level is None:
            continue
        for key in keys:
            if current_level.pop(key, missing) is not missing:
                changed = True
    return changed


def _validate_config_mutation_targets(
    section_values: Mapping[str, Mapping[Any, Any]],
    delete_keys: Mapping[str, Collection[str]],
) -> None:
    """Validate input shapes and reject overlapping set/delete targets."""
    if not isinstance(section_values, Mapping) or not isinstance(
        delete_keys,
        Mapping,
    ):
        raise TypeError("Configuration mutations must use mappings")

    set_targets: set[tuple[str, Any]] = set()
    for section, values in section_values.items():
        if not isinstance(section, str) or not section:
            raise TypeError("Configuration sections must be non-empty strings")
        if section.partition(".")[0] in _REVISION_OWNED_CONFIG_SECTIONS:
            raise ValueError(
                "Revision-owned configuration requires its dedicated writer"
            )
        if not isinstance(values, Mapping):
            raise TypeError("Configuration section values must be mappings")
        for key in values:
            set_targets.add((section, key))

    delete_targets: set[tuple[str, str]] = set()
    for section, keys in delete_keys.items():
        if not isinstance(section, str) or not section:
            raise TypeError("Configuration sections must be non-empty strings")
        if section.partition(".")[0] in _REVISION_OWNED_CONFIG_SECTIONS:
            raise ValueError(
                "Revision-owned configuration requires its dedicated writer"
            )
        if isinstance(keys, (str, bytes)) or not isinstance(keys, Collection):
            raise TypeError("Configuration delete keys must be collections")
        for key in keys:
            if not isinstance(key, str) or not key:
                raise TypeError("Configuration delete keys must be non-empty strings")
            delete_targets.add((section, key))

    if set_targets.intersection(delete_targets):
        raise ValueError("Configuration mutation cannot set and delete the same key")


def _validate_literal_config_mutation_targets(
    mutation: LiteralSettingsMutation,
) -> None:
    """Validate literal paths and reject overlapping exact set/delete targets."""

    if not isinstance(mutation, LiteralSettingsMutation):
        raise TypeError("Literal mutation builder returned an invalid result")
    if not isinstance(mutation.section_values, Mapping) or not isinstance(
        mutation.delete_keys,
        Mapping,
    ):
        raise TypeError("Literal configuration mutations must use mappings")

    set_targets: set[tuple[tuple[str, ...], str]] = set()
    for path, values in mutation.section_values.items():
        _validate_literal_config_path(path)
        if not isinstance(values, Mapping):
            raise TypeError("Literal configuration section values must be mappings")
        for key in values:
            if type(key) is not str or not key:
                raise TypeError("Literal configuration keys must be non-empty strings")
            set_targets.add((path, key))

    delete_targets: set[tuple[tuple[str, ...], str]] = set()
    for path, keys in mutation.delete_keys.items():
        _validate_literal_config_path(path)
        if isinstance(keys, (str, bytes)) or not isinstance(keys, Collection):
            raise TypeError("Literal configuration delete keys must be collections")
        for key in keys:
            if type(key) is not str or not key:
                raise TypeError(
                    "Literal configuration delete keys must be non-empty strings"
                )
            delete_targets.add((path, key))

    if set_targets.intersection(delete_targets):
        raise ValueError("Configuration mutation cannot set and delete the same key")


def _detach_literal_settings_mutation(
    mutation: LiteralSettingsMutation,
) -> LiteralSettingsMutation:
    """Copy a builder-owned mutation before validation or application."""

    if not isinstance(mutation, LiteralSettingsMutation):
        raise TypeError("Literal mutation builder returned an invalid result")
    if not isinstance(mutation.section_values, Mapping) or not isinstance(
        mutation.delete_keys,
        Mapping,
    ):
        raise TypeError("Literal configuration mutations must use mappings")

    section_values: dict[tuple[str, ...], dict[str, object]] = {}
    for path, values in mutation.section_values.items():
        if type(path) is not tuple:
            raise TypeError("Literal configuration paths must be tuples")
        if not isinstance(values, Mapping):
            raise TypeError("Literal configuration section values must be mappings")
        owned_path = tuple(part for part in path)
        section_values[owned_path] = copy.deepcopy(dict(values))

    delete_keys: dict[tuple[str, ...], tuple[str, ...]] = {}
    for path, keys in mutation.delete_keys.items():
        if type(path) is not tuple:
            raise TypeError("Literal configuration paths must be tuples")
        if isinstance(keys, (str, bytes)) or not isinstance(keys, Collection):
            raise TypeError("Literal configuration delete keys must be collections")
        owned_path = tuple(part for part in path)
        delete_keys[owned_path] = tuple(keys)

    return LiteralSettingsMutation(
        section_values=section_values,
        delete_keys=delete_keys,
    )


def _validate_literal_config_path(path: object) -> None:
    """Validate one exact TOML mapping path."""

    if (
        type(path) is not tuple
        or not path
        or any(type(part) is not str or not part for part in path)
    ):
        raise TypeError("Literal configuration paths must be non-empty string tuples")
    if path[0] in _REVISION_OWNED_CONFIG_SECTIONS:
        raise ValueError("Revision-owned configuration requires its dedicated writer")


def _literal_mutation_log_shape(
    mutation: LiteralSettingsMutation,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return value-free path/key shapes suitable for diagnostic logging."""

    sets = {
        repr(path): list(values.keys())
        for path, values in mutation.section_values.items()
    }
    deletes = {repr(path): list(keys) for path, keys in mutation.delete_keys.items()}
    return sets, deletes


def _current_settings_view() -> Mapping[str, object]:
    """Return a detached copy of the freshly published normalized settings."""

    return copy.deepcopy(settings)


def _apply_literal_mutation_unlocked(
    config_data: Dict[str, Any],
    mutation: LiteralSettingsMutation,
) -> bool:
    """Apply one validated literal mutation to the authoritative raw mapping."""

    deleted_any = _delete_literal_config_keys(config_data, mutation.delete_keys)
    set_any = False
    for path, values in mutation.section_values.items():
        if not values:
            continue
        current_level = _literal_config_section(config_data, path, create=True)
        assert current_level is not None
        for key, value in values.items():
            current_level[key] = _maybe_encrypt_setting_value(config_data, key, value)
            set_any = True
    return set_any or deleted_any


def _apply_literal_settings_transaction_locked(
    mutation_builder: Callable[
        [AtomicLiteralMutationSnapshot], LiteralSettingsMutation
    ],
    *,
    mutation_precondition: Callable[[], bool] | None,
    locked_snapshot_precondition: Callable[[AtomicConfigSnapshot], bool] | None = None,
    publish_noop: bool,
    validate_literal_targets: bool,
    before_replace: Callable[[], None] | None = None,
    after_replace: Callable[[], None] | None = None,
) -> LiteralConfigMutationResult:
    """Run one authoritative read/build/replace/publication transaction."""
    try:
        if not callable(mutation_builder):
            raise TypeError("Literal configuration mutation builder must be callable")
        if mutation_precondition is not None and not callable(mutation_precondition):
            raise TypeError("Configuration mutation precondition must be callable")
        if locked_snapshot_precondition is not None and not callable(
            locked_snapshot_precondition
        ):
            raise TypeError("Locked configuration precondition must be callable")
        if before_replace is not None and not callable(before_replace):
            raise TypeError("Before-replace callback must be callable")
        if after_replace is not None and not callable(after_replace):
            raise TypeError("After-replace callback must be callable")
        config_path = _get_effective_config_path()
    except Exception as error:
        logger.error(
            "Configuration mutation failed "
            "(phase=resolve_path, config_path=unresolved, error_type={}).",
            type(error).__name__,
        )
        return LiteralConfigMutationResult(False, False, None, "before_replace")

    with ExitStack() as locks:
        try:
            locks.enter_context(_config_write_lock(config_path))
        except Exception as error:
            logger.error(
                "Configuration mutation failed "
                "(phase=lock, config_path={}, error_type={}).",
                config_path,
                type(error).__name__,
            )
            return LiteralConfigMutationResult(False, False, None, "before_replace")
        try:
            config_data = _read_raw_cli_config_unlocked(config_path)
        except Exception as error:
            logger.error(
                "Configuration mutation failed "
                "(phase=read, config_path={}, error_type={}).",
                config_path,
                type(error).__name__,
            )
            return LiteralConfigMutationResult(False, False, None, "before_replace")

        if mutation_precondition is not None:
            try:
                if mutation_precondition() is not True:
                    return LiteralConfigMutationResult(
                        False, False, None, "identity_changed"
                    )
            except Exception as error:
                logger.error(
                    "Configuration mutation failed "
                    "(phase=precondition, error_type={}).",
                    type(error).__name__,
                )
                return LiteralConfigMutationResult(False, False, None, "before_replace")

        try:
            effective_values = _atomic_config_values_from_raw(config_data)
            if locked_snapshot_precondition is not None:
                legacy_snapshot = AtomicConfigSnapshot(
                    generation=_CONFIG_GENERATION,
                    values=copy.deepcopy(effective_values),
                )
                if locked_snapshot_precondition(legacy_snapshot) is not True:
                    return LiteralConfigMutationResult(
                        False, False, None, "identity_changed"
                    )
            snapshot = AtomicLiteralMutationSnapshot(
                generation=_CONFIG_GENERATION,
                raw_values=copy.deepcopy(config_data),
                effective_values=copy.deepcopy(effective_values),
                config_path=config_path,
            )
            mutation = _detach_literal_settings_mutation(mutation_builder(snapshot))
            if validate_literal_targets:
                _validate_literal_config_mutation_targets(mutation)
            logged_sets, logged_deletes = _literal_mutation_log_shape(mutation)
            logger.info(
                "Attempting to apply literal settings mutation: sets={}, deletes={}",
                logged_sets,
                logged_deletes,
            )
            if before_replace is not None:
                before_replace()
            changed = _apply_literal_mutation_unlocked(config_data, mutation)
        except Exception as error:
            logger.error(
                "Configuration mutation failed "
                "(phase=before_replace, config_path={}, error_type={}).",
                config_path,
                type(error).__name__,
            )
            return LiteralConfigMutationResult(False, False, None, "before_replace")

        if not changed and not publish_noop:
            return LiteralConfigMutationResult(False, False, None, None)

        raw_written: Mapping[str, Any] | None = None
        if changed:
            persisted: Mapping[str, Any] | None = None
            expected_raw: Mapping[str, Any] | None = None
            try:
                persisted = _config_data_for_persistence(config_data)
                expected_raw = tomllib.loads(dumps_cli_config(dict(persisted)))
                raw_written = _write_raw_cli_config_unlocked(config_path, persisted)
            except Exception as error:
                committed_content_visible = False
                if expected_raw is not None:
                    try:
                        committed_content_visible = (
                            _read_raw_cli_config_unlocked(config_path) == expected_raw
                        )
                    except Exception:
                        committed_content_visible = False
                if committed_content_visible:
                    logger.error(
                        "Configuration mutation failed after the requested "
                        "content became visible "
                        "(phase=cache_reload, config_path={}, error_type={}).",
                        config_path,
                        type(error).__name__,
                    )
                    return LiteralConfigMutationResult(
                        True, False, None, "cache_reload"
                    )
                logger.error(
                    "Configuration mutation failed "
                    "(phase=before_replace, config_path={}, error_type={}).",
                    config_path,
                    type(error).__name__,
                )
                return LiteralConfigMutationResult(False, False, None, "before_replace")
            logger.success(f"Successfully replaced settings file at {config_path}")
        else:
            _invalidate_config_caches()
            raw_written = config_data

        if after_replace is not None:
            try:
                after_replace()
            except Exception as error:
                logger.error(
                    "Configuration mutation failed "
                    "(phase=after_replace_callback, error_type={}).",
                    type(error).__name__,
                )
                return LiteralConfigMutationResult(changed, False, None, "cache_reload")

        try:
            _publish_runtime_config_unlocked(
                raw_config=raw_written, config_path=config_path
            )
            settings_view = _current_settings_view()
        except Exception as error:
            logger.error(
                "Configuration mutation failed "
                "(phase=cache_reload, config_path={}, error_type={}).",
                config_path,
                type(error).__name__,
            )
            return LiteralConfigMutationResult(changed, False, None, "cache_reload")

        logger.info("Global configuration caches invalidated and reloaded.")
        return LiteralConfigMutationResult(changed, True, settings_view, None)


def apply_literal_settings_transaction_to_cli_config(
    mutation_builder: Callable[
        [AtomicLiteralMutationSnapshot], LiteralSettingsMutation
    ],
    *,
    mutation_precondition: Callable[[], bool] | None = None,
) -> LiteralConfigMutationResult:
    """Apply a builder-produced mutation whose tuple paths stay fully literal."""

    return _apply_literal_settings_transaction_locked(
        mutation_builder,
        mutation_precondition=mutation_precondition,
        publish_noop=True,
        validate_literal_targets=True,
    )


def refresh_runtime_config_from_cli_config() -> LiteralConfigMutationResult:
    """Republish the existing on-disk config without writing the file."""

    try:
        config_path = _get_effective_config_path()
    except Exception as error:
        logger.error(
            "Configuration refresh failed (phase=resolve_path, error_type={}).",
            type(error).__name__,
        )
        return LiteralConfigMutationResult(False, False, None, "cache_reload")

    try:
        with _config_write_lock(config_path):
            raw = _read_raw_cli_config_unlocked(config_path)
            _invalidate_config_caches()
            _publish_runtime_config_unlocked(raw_config=raw, config_path=config_path)
            return LiteralConfigMutationResult(
                False,
                True,
                _current_settings_view(),
                None,
            )
    except Exception as error:
        logger.error(
            "Configuration refresh failed "
            "(phase=cache_reload, config_path={}, error_type={}).",
            config_path,
            type(error).__name__,
        )
        return LiteralConfigMutationResult(False, False, None, "cache_reload")


def apply_settings_mutation_to_cli_config(
    section_values: Mapping[str, Mapping[Any, Any]],
    *,
    delete_keys: Mapping[str, Collection[str]] | None = None,
    mutation_precondition: Callable[[], bool] | None = None,
    locked_snapshot_precondition: Callable[[AtomicConfigSnapshot], bool] | None = None,
    before_replace: Callable[[], None] | None = None,
    after_replace: Callable[[], None] | None = None,
) -> ConfigMutationResult:
    """Atomically apply exact config sets/deletes, then refresh caches."""
    requested_deletes = {} if delete_keys is None else delete_keys
    try:
        if not isinstance(section_values, Mapping) or not isinstance(
            requested_deletes,
            Mapping,
        ):
            raise TypeError("Configuration mutations must use mappings")
        owned_section_values: dict[str, dict[Any, Any]] = {}
        for section, values in section_values.items():
            if not isinstance(values, Mapping):
                raise TypeError("Configuration section values must be mappings")
            owned_section_values[section] = copy.deepcopy(dict(values.items()))
        owned_delete_keys: dict[str, tuple[str, ...]] = {}
        for section, keys in requested_deletes.items():
            if isinstance(keys, (str, bytes)) or not isinstance(keys, Collection):
                raise TypeError("Configuration delete keys must be collections")
            owned_delete_keys[section] = tuple(keys)

        _validate_config_mutation_targets(owned_section_values, owned_delete_keys)
        literal = _detach_literal_settings_mutation(
            LiteralSettingsMutation(
                section_values={
                    tuple(section.split(".")): values
                    for section, values in owned_section_values.items()
                },
                delete_keys={
                    tuple(section.split(".")): keys
                    for section, keys in owned_delete_keys.items()
                },
            )
        )
    except Exception as error:
        logger.error(
            "Configuration mutation failed "
            "(phase=validation, config_path=unresolved, error_type={}).",
            type(error).__name__,
        )
        return ConfigMutationResult(False, False, "before_replace")
    result = _apply_literal_settings_transaction_locked(
        lambda _snapshot: literal,
        mutation_precondition=mutation_precondition,
        locked_snapshot_precondition=locked_snapshot_precondition,
        before_replace=before_replace,
        after_replace=after_replace,
        publish_noop=False,
        validate_literal_targets=False,
    )
    if result.failure_phase == "identity_changed":
        return ConfigMutationResult(
            False,
            False,
            None,
            conflict=True,
            conflict_reason="identity_changed",
        )
    return ConfigMutationResult(
        result.file_replaced,
        result.caches_reloaded,
        result.failure_phase,
    )


def save_settings_to_cli_config(
    section_values: Mapping[str, Mapping[Any, Any]],
    *,
    delete_keys: Mapping[str, Collection[str]] | None = None,
) -> bool:
    """Persist multiple config values with one atomic mutation and cache reload."""
    result = apply_settings_mutation_to_cli_config(
        section_values,
        delete_keys=delete_keys,
    )
    if result.conflict:
        return False
    if result.failure_phase is None and not result.file_replaced:
        return True
    return result.fully_applied


def delete_settings_from_cli_config(section: str, keys: List[str]) -> bool:
    """Remove exact keys through the structured atomic mutation primitive.

    Args:
        section: Dotted config section path (e.g. ``"console.rail_state"``).
        keys: Keys to remove from that section.

    Returns:
        True on success, including when the file, section, or keys are absent.
    """
    config_path = _get_effective_config_path()
    if not config_path.exists():
        return True

    result = apply_settings_mutation_to_cli_config(
        {},
        delete_keys={section: tuple(keys)},
    )
    if result.failure_phase is None and not result.file_replaced:
        return True
    return result.fully_applied


def save_setting_to_cli_config(section: str, key: str, value: Any) -> bool:
    """
    Saves a specific setting to the user's CLI TOML configuration file.

    This function reads the current config, updates a specific key within a
    section (handling nested sections like 'api_settings.openai'), and writes
    the entire configuration back to the file. It then forces a reload of the
    config cache.

    Args:
        section: The name of the TOML section (e.g., "general", "api_settings.openai").
        key: The key within the section to update.
        value: The new value for the key.

    Returns:
        True if the setting was saved successfully, False otherwise.
    """
    logger.info(
        f"Attempting to save setting: [{section}].{key} = "
        f"{_setting_value_for_log(key, value)}"
    )
    return save_settings_to_cli_config({section: {key: value}})


# --- CLI Setting Getter ---
# Sentinel distinguishing "no default argument was supplied at all" from an
# explicitly-passed `None` -- see the dotted-form disambiguation below.
_CLI_SETTING_DEFAULT_UNSET = object()


def current_config_identity() -> tuple[int, str]:
    """Return a cheap key that changes whenever the effective config changes.

    Two independent things can change what `get_cli_setting` returns, and a
    memoisation key must track BOTH:

    * ``_CONFIG_GENERATION`` -- bumped by every mutation path, so it catches
      writes to the config currently in force.
    * the effective config PATH -- retargeting ``TLDW_CONFIG_PATH`` selects a
      different file, and the loader serves the new file's values WITHOUT
      advancing the generation. Verified empirically while reviewing
      task-24456: after a retarget, `get_cli_setting` returned the new file's
      value while the generation stayed at 1, so a generation-only key served
      stale data. Caught in review by Qodo on PR #2217.

    Both reads are cheap -- an atomic reference load and an env-var lookup plus
    a lexical path expansion. No lock, no file access, no copy, which is what
    makes this usable on a hot path. ``get_runtime_config_snapshot`` and
    ``get_atomic_config_snapshot`` expose the same generation but take locks
    and deep-copy the values, so they are not substitutes here.

    Intended use (task-24456): a caller that derives an expensive view from
    many `get_cli_setting` reads caches that view against this tuple and
    recomputes only when it moves.

    Returns:
        ``(configuration generation, effective config path)``.
    """
    return (_CONFIG_GENERATION, str(_get_effective_config_path()))


def get_cli_setting(
    section: str, key: str = None, default: Any = _CLI_SETTING_DEFAULT_UNSET
) -> Any:
    """Helper to get a specific setting from the loaded CLI configuration.

    Can be called in two ways:
    1. get_cli_setting("section", "key", default)  # Traditional format
    2. get_cli_setting("section.key", default)     # Dotted format

    Dotted sections/keys that miss the flat top-level lookup are resolved
    against the nested TOML tree (``[chat.images]`` loads as
    ``config["chat"]["images"]``); a flat top-level hit always wins.

    Disambiguating form 1 from form 2 when exactly two positional
    arguments are supplied and ``section`` contains a dot: an explicit
    third argument (even ``None``) always means "traditional form,
    honour this default". Without one, ``section`` is the complete
    dotted path and ``key`` is really the default -- for a default of
    *any* type, not just non-string ones. (TASK-1771: the previous
    heuristic keyed off ``isinstance(key, str)``, so a string default --
    the common case for provider/model/language/device names -- was
    misread as one more path segment to walk, which always missed and
    silently returned ``None`` instead of either the configured value or
    the caller's own fallback.)

    Args:
        section: Top-level section name, or a dotted path into nested tables.
        key: Setting key within the section; in the dotted call shape this
            positional slot may carry the default instead.
        default: Value returned when the setting cannot be resolved.

    Returns:
        The resolved setting value, or ``default`` when the section/key does
        not exist.
    """
    config = load_cli_config_and_ensure_existence()  # Ensures config is loaded

    default_was_given = default is not _CLI_SETTING_DEFAULT_UNSET
    resolved_default = default if default_was_given else None

    # Handle dotted notation when key is None (called with positional args)
    if key is None and "." in section:
        # Split on first dot only to handle nested keys
        parts = section.split(".", 1)
        section = parts[0]
        key = parts[1]
    elif key is None:
        # No dot found and no key provided - invalid call
        return resolved_default
    elif not default_was_given and "." in section:
        # Pure 2-arg dotted form: get_cli_setting("a.b[.c...]", default).
        # `key` is really the caller's default (whatever its type) --
        # reclaim it before re-splitting `section` the same way the
        # 1-arg branch above does.
        resolved_default = key
        parts = section.split(".", 1)
        section = parts[0]
        key = parts[1]
    elif not default_was_given and not isinstance(key, str):
        # 2-arg call on an UNDOTTED section whose second positional is a
        # default, not a key -- `get_cli_setting("database", {})`. Keys are
        # always strings, so a non-string here can only be a default. This
        # shape never resolved anything (it returned the default and ignored
        # config), and it must keep returning that default rather than
        # reaching `dict.get()` with an unhashable key, which would raise
        # TypeError -- turning a long-lived silent misread into a crash for
        # callers like Helper_Scripts/Mass-Ingestion (found reviewing the
        # TASK-1771 fix).
        logger.warning(
            "get_cli_setting({!r}, <{}>) has no key; returning the default. "
            "Use get_cli_setting(section, key, default).",
            section,
            type(key).__name__,
        )
        return key

    # Flat lookup first: preserves every previously-working shape bit-for-bit
    # (a literal dotted top-level key, while impossible from TOML, wins).
    section_data = config.get(section)
    if isinstance(section_data, dict) and isinstance(key, str) and key in section_data:
        return section_data[key]
    # Nested fallback: TOML `[chat.images]` loads as config["chat"]["images"],
    # never config["chat.images"], so dotted sections/keys that miss the flat
    # lookup are resolved by walking the real tree segment by segment.
    if (
        isinstance(section, str)
        and isinstance(key, str)
        and ("." in section or "." in key)
    ):
        node: Any = config
        for part in (*section.split("."), *key.split(".")):
            if not isinstance(node, dict) or part not in node:
                return resolved_default
            node = node[part]
        return node
    if isinstance(section_data, dict) and isinstance(key, str):
        return section_data.get(key, resolved_default)
    # A non-string `key` reaches here from a caller that passed a whole default
    # as the second positional argument while meaning "give me this section"
    # -- e.g. `get_cli_setting("database", {})`. That shape never worked (it
    # silently returned the default and ignored config), but it must not
    # CRASH: `dict.get()` on an unhashable key raises TypeError, which would
    # turn a long-standing silent misread into a hard failure in callers that
    # have lived with it for a long time (Helper_Scripts/Mass-Ingestion, found
    # in review of the TASK-1771 fix). Return the caller's default, exactly as
    # before, and let the misuse stay visible in the warning below.
    if not isinstance(key, str):
        logger.warning(
            "get_cli_setting({!r}, ...) was called with a non-string key ({}); "
            "returning the default. Use get_cli_setting(section, key, default).",
            section,
            type(key).__name__,
        )
    # If section is not a dict or not found, return default
    return resolved_default


def get_chat_defaults_streaming(default: bool = True) -> bool:
    """Return chat default streaming with canonical-first legacy fallback."""
    config = load_cli_config_and_ensure_existence()
    chat_defaults = config.get("chat_defaults")
    if not isinstance(chat_defaults, dict):
        return default
    if "streaming" in chat_defaults:
        return coerce_bool_setting(chat_defaults.get("streaming"), default)
    if "enable_streaming" in chat_defaults:
        return coerce_bool_setting(chat_defaults.get("enable_streaming"), default)
    return default


def get_chat_defaults_user_display_name(default: str = "User") -> str:
    """Return the validated global human-facing Console chat name."""
    from tldw_chatbook.Chat.console_roleplay_identity import (
        configured_chat_display_name,
    )

    return configured_chat_display_name(load_cli_config_and_ensure_existence(), default)


def get_rag_citation_canonical_writes_enabled() -> bool:
    """Return the canonical citation write switch (on by default, TASK-33620.8).

    Returns:
        ``True`` unless turned off; a malformed value fails closed (off).
    """

    section = load_cli_config_and_ensure_existence().get("rag_citations")
    if not isinstance(section, dict):
        return False
    value = section.get("canonical_writes_enabled", True)  # shipped default: on
    return coerce_bool_setting(value, False)


def get_media_ingestion_defaults(media_type: str) -> Dict[str, Any]:
    """
    Get default chunking settings for a specific media type.

    Args:
        media_type: Type of media ('pdf', 'ebook', 'document', 'plaintext', 'web_article')

    Returns:
        Dictionary containing chunking configuration for the media type
    """
    # First check if user has custom settings in config
    config = load_cli_config_and_ensure_existence()
    media_ingestion_config = config.get("media_ingestion", {})

    # Get media-specific config if it exists
    if media_type in media_ingestion_config and isinstance(
        media_ingestion_config[media_type], dict
    ):
        # Use deep merge to combine with defaults, allowing partial overrides
        return deep_merge_dicts(
            DEFAULT_MEDIA_INGESTION_CONFIG.get(media_type, {}),
            media_ingestion_config[media_type],
        )

    # Fall back to hardcoded defaults
    return DEFAULT_MEDIA_INGESTION_CONFIG.get(
        media_type,
        {
            "chunk_method": "paragraphs",
            "chunk_size": 500,
            "chunk_overlap": 200,
            "use_adaptive_chunking": False,
            "use_multi_level_chunking": False,
            "chunk_language": "",
        },
    )


def get_ocr_backend_config(backend_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific OCR backend.

    Args:
        backend_name: Name of the OCR backend (e.g., 'docext', 'tesseract')

    Returns:
        Dictionary containing backend configuration
    """
    # First check if user has custom settings in config
    config = load_cli_config_and_ensure_existence()
    ocr_backend_config = config.get("ocr_backends", {})

    # Get backend-specific config if it exists
    if backend_name in ocr_backend_config and isinstance(
        ocr_backend_config[backend_name], dict
    ):
        # Use deep merge to combine with defaults, allowing partial overrides
        return deep_merge_dicts(
            DEFAULT_OCR_BACKEND_CONFIG.get(backend_name, {}),
            ocr_backend_config[backend_name],
        )

    # Fall back to hardcoded defaults
    return DEFAULT_OCR_BACKEND_CONFIG.get(backend_name, {})


# --- CLI Providers and Models Getter ---
#: TASK-35010 (OD-1 (a)): the sample ids older templates wrote into the vLLM,
#: TabbyAPI, Aphrodite, Custom and Custom 2 lists. No server serves them, so
#: the saved-list loader and Settings' default choice ignore them, and a
#: Settings write drops them from the list it rewrites; nothing rewrites the
#: file at startup. Frozen: the template no longer ships them.
# ponytail: matched by value, so a server that really serves one of these
# exact ids cannot have it listed; a template marker cannot reach files that
# older versions wrote without one.
SHIPPED_SAMPLE_MODEL_IDS: frozenset[str] = frozenset(
    {
        *("vllm-model-z", "vllm-model-x", "vllm-model-y", "vllm-model-a"),
        *("tabby-model", "tabby-model-2", "tabby-model-3"),
        *("aphrodite-engine", "aphrodite-engine-2"),
        *("custom-model-alpha", "custom-model-beta"),
        *("custom-model-gamma", "custom-model-delta"),
    }
)


def get_cli_providers_and_models() -> Dict[str, List[str]]:
    """Return each ``[providers]`` list, minus ``SHIPPED_SAMPLE_MODEL_IDS``.

    Every key stays, an emptied list included, so discovery and list-key
    resolution see the same keys as before.
    """
    config = load_settings()
    providers_data = config.get(
        "providers", {}
    )  # Default to empty dict if "providers" isn't there
    valid_providers: Dict[str, List[str]] = {}
    if isinstance(providers_data, dict):
        for provider, models in providers_data.items():
            if isinstance(models, list) and all(isinstance(m, str) for m in models):
                valid_providers[provider] = [
                    model for model in models if model not in SHIPPED_SAMPLE_MODEL_IDS
                ]
            else:
                logger.warning(
                    f"Invalid model list for provider '{provider}' in CLI config [providers]. Models: {models}. Skipping."
                )
    else:
        logger.error(
            f"CLI Config 'providers' section is not a dictionary. Found: {type(providers_data)}. No provider/model data available."
        )
    return valid_providers


def _normalize_provider_lookup_key(provider: Any) -> str:
    """Return the canonical lookup form used for provider key comparisons."""
    return normalize_provider_config_key(provider)


def resolve_provider_name(
    provider: Any,
    providers_models: Dict[str, List[str]],
) -> str:
    """Return the configured provider name as it appears in provider options.

    Config files commonly use API-setting keys such as ``llama_cpp`` while the
    selectable provider list may expose display keys such as ``Llama_cpp``.
    This keeps UI defaults from falling back to the first provider.
    """
    provider_name = str(provider or "").strip()
    if not provider_name:
        return provider_name
    if provider_name in providers_models:
        return provider_name

    normalized_provider = _normalize_provider_lookup_key(provider_name)
    for available_provider in providers_models:
        if _normalize_provider_lookup_key(available_provider) == normalized_provider:
            return available_provider
    return provider_name


def check_encryption_needed() -> bool:
    """
    Check if the config has API keys that should be encrypted.

    Returns:
        True if API keys are detected and encryption is not enabled
    """
    config = load_cli_config_and_ensure_existence()

    # Check if encryption is already enabled
    if config.get("encryption", {}).get("enabled", False):
        return False

    # Check for API keys
    enc_module = get_encryption_module()
    return enc_module.detect_api_keys(config)


def get_detected_api_providers() -> List[str]:
    """
    Get list of providers with detected API keys.

    Returns:
        List of provider names with API keys
    """
    config = load_cli_config_and_ensure_existence()
    providers = []

    # TASK-32811.3: a nested TOML load produces `config["api_settings"]` as a
    # nested table, never flat `"api_settings.<provider>"` top-level keys, so
    # the old iteration matched nothing and this always returned []. Read the
    # nested table, and screen each key through `resolve_provider_api_key`
    # (the shared validity rule) rather than the ad-hoc `<...>` check, so a
    # padded or otherwise-invalid key is not counted as configured.
    api_settings = config.get("api_settings")
    if isinstance(api_settings, Mapping):
        for provider_name, provider_config in api_settings.items():
            if not isinstance(provider_config, Mapping):
                continue
            if resolve_provider_api_key(provider_config.get("api_key")) is not None:
                providers.append(provider_name)

    return providers


# --- Config key-encryption lifecycle (TASK-34100.4) ---------------------------
#
# One owner for the four states (off / on / locked / unlocked). Every change
# below follows the same contract, because a mistake here locks people out of
# their own app (first-run review 2026-10-02, protect-summary-04):
#
# * validate BEFORE writing -- the replacement document must strict-decrypt
#   with the password the session will hold afterwards, and must not leave a
#   plaintext secret behind in an encrypted file;
# * if anything fails after the write (the runtime publish, a binding check),
#   restore the previous file bytes -- or remove a file this call created --
#   and the previous in-process password, so a failure never leaves the disk
#   changed while the caller is told nothing happened;
# * never decrypt non-strictly: a file whose keys do not match its verifier is
#   refused untouched instead of having its ciphertext written back as values.


def _set_session_encryption_password(password: Optional[str]) -> None:
    """Install (or clear) the in-process password and drop stale caches."""

    global _SETTINGS_CACHE, _CONFIG_CACHE
    if password is None:
        clear_encryption_password()
        _SETTINGS_CACHE = None
        _CONFIG_CACHE = None
        return
    set_encryption_password(password)


def _parse_raw_cli_config_text(serialized: Optional[str]) -> Dict[str, Any]:
    if serialized is None:
        return {}
    loaded = tomllib.loads(serialized)
    if not isinstance(loaded, dict):
        raise TypeError("The CLI config must contain a top-level table")
    return loaded


def _password_verifier(config_data: Mapping[str, Any]) -> Optional[str]:
    encryption = config_data.get("encryption", {})
    if not isinstance(encryption, Mapping):
        return None
    verifier = encryption.get("password_verifier")
    return verifier if isinstance(verifier, str) and verifier else None


def _validate_encrypted_document(
    document: Mapping[str, Any],
    password: str,
) -> None:
    """Raise unless ``document`` is a safe encrypted config for ``password``.

    The check mirrors what the next publish and the next startup do: the
    merged view must strict-decrypt, and no sensitive value may stay plain.
    """

    merged = deep_merge_dicts(DEFAULT_CONFIG_FROM_TOML, dict(document))
    get_encryption_module().decrypt_config_strict(merged, password)
    if _contains_unencrypted_sensitive_value(document):
        raise ValueError("An encrypted config would keep a plaintext secret")


def _strip_encrypted_values(config_data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy with every ``enc:`` string value removed, in every table.

    Arrays are walked too (Qodo round, PR #3000): an ``enc:`` element is
    dropped from its array and a table inside an array is stripped like any
    other, so a reset never leaves ciphertext behind in a TOML array.
    """

    stripped: Dict[str, Any] = {}
    for key, value in config_data.items():
        if is_encrypted_config_value(value):
            continue
        stripped[key] = _strip_encrypted_item(value)
    return stripped


def _strip_encrypted_item(value: Any) -> Any:
    """Strip ``enc:`` strings from one table or array value (copied)."""

    if isinstance(value, Mapping):
        return _strip_encrypted_values(value)
    if isinstance(value, list):
        return [
            _strip_encrypted_item(item)
            for item in value
            if not is_encrypted_config_value(item)
        ]
    return copy.deepcopy(value)


def _encrypted_value_paths(
    config_data: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> list[str]:
    """Dotted paths of every ``enc:`` value outside the ``[encryption]`` table."""

    paths: list[str] = []
    for key, value in config_data.items():
        if not prefix and key == "encryption":
            continue
        if isinstance(value, Mapping):
            paths.extend(_encrypted_value_paths(value, prefix + (str(key),)))
        elif is_encrypted_config_value(value):
            paths.append(".".join(prefix + (str(key),)))
    return paths


def _contains_encrypted_value(config_data: Mapping[str, Any]) -> bool:
    return bool(_encrypted_value_paths(config_data))


def _already_encrypted_with(config_data: Mapping[str, Any], password: str) -> bool:
    """Whether an encryption-on file is already exactly what enabling with
    ``password`` would produce: the verifier accepts it, every saved value
    strict-decrypts with it, and no sensitive value is left in plain text."""

    verifier = _password_verifier(config_data)
    if verifier is None:
        return False
    if not get_encryption_module().verify_password(password, verifier):
        return False
    try:
        _validate_encrypted_document(config_data, password)
    except ValueError:
        return False
    return True


def _restore_previous_config_unlocked(
    config_path: Path,
    previous_serialized: Optional[str],
) -> None:
    """Put the exact previous bytes back (or remove a file this call created).

    Args:
        config_path: The selected config file the failed change wrote.
        previous_serialized: The file's exact text before the change, or
            None when the change created the file.

    Raises:
        ValueError: The path fails ``validate_path_simple``; nothing is
            removed.
        OSError: Reading, removing or rewriting the file failed.
    """

    current = _try_read_cli_config_serialized_unlocked(config_path)
    if current == previous_serialized:
        return
    if previous_serialized is None:
        # The only raw filesystem call here; the restore write below goes
        # through the private atomic writer. Same validation as the hooks
        # snapshot's lock target: the path only reaches unlink(), never a
        # shell, and unlink() removes a link itself, not what it points to.
        removable = validate_path_simple(
            config_path,
            require_exists=False,
            probe_existing=False,
            reject_shell_metacharacters=False,
        )
        removable.unlink(missing_ok=True)
        _invalidate_config_caches()
        return
    _commit_serialized_cli_config_unlocked(
        config_path,
        previous_serialized,
        _prepare_config_parent(config_path),
    )


def _unlock_session_with_password_unlocked(config_path: Path, password: str) -> bool:
    """Install ``password`` for an unchanged file and republish settings.

    The same-password enable writes nothing, but a session that loaded the
    file without the password published its ciphertext in ``settings``;
    installing the password alone left every reader of that global on the
    locked view (Qodo round, PR #3000). Callers hold the write lock and have
    already checked that ``password`` reads the file.

    Args:
        config_path: The selected config file (used for log context only).
        password: The master password the file is encrypted under.

    Returns:
        True when the session holds ``password`` and the published settings
        are decrypted with it; False when the publish failed, in which case
        the previous session password is put back.
    """

    previous_password = get_encryption_password()
    _set_session_encryption_password(password)
    try:
        _publish_runtime_config_unlocked()
        return True
    except Exception as error:
        logger.error(
            "Publishing the unlocked config failed; the session was left as "
            "it was (action=enable, config_path={}, error_type={}).",
            redact_user_paths(str(config_path)),
            type(error).__name__,
        )
    _set_session_encryption_password(previous_password)
    return False


def _session_password_for_file(
    config_path: Path,
    previous_serialized: Optional[str],
    *,
    previous_password: Optional[str],
    written_password: Optional[str],
) -> Optional[str]:
    """The password that reads the file as it is after a rollback attempt."""

    try:
        current = _try_read_cli_config_serialized_unlocked(config_path)
    except Exception:
        return None
    if current == previous_serialized:
        return previous_password
    return written_password


def _commit_encryption_change_unlocked(
    config_path: Path,
    document: Mapping[str, Any],
    previous_serialized: Optional[str],
    *,
    session_password: Optional[str],
    action: str,
) -> bool:
    """Write ``document``, install ``session_password`` and publish -- or roll back.

    Callers hold the write lock and have already validated ``document``.
    """

    previous_password = get_encryption_password()
    try:
        raw_written = _write_raw_cli_config_unlocked(config_path, document)
        _set_session_encryption_password(session_password)
        _publish_runtime_config_unlocked(
                raw_config=raw_written, config_path=config_path
            )
        return True
    except Exception as error:
        logger.error(
            "Config encryption change failed; restoring the previous config "
            "(action={}, error_type={}).",
            action,
            type(error).__name__,
        )
    try:
        _restore_previous_config_unlocked(config_path, previous_serialized)
    except Exception as restore_error:
        logger.error(
            "Restoring the previous config after a failed encryption change "
            "failed (action={}, error_type={}).",
            action,
            type(restore_error).__name__,
        )
    # The session password must match what the file holds NOW (review round
    # 1, F7): the previous one after a restore, the new one when the restore
    # failed and the new document stayed. Reverting it then would encrypt the
    # next save under the old password beside the new verifier -- the
    # stranded state. An unreadable file locks the session rather than guess.
    _set_session_encryption_password(
        _session_password_for_file(
            config_path,
            previous_serialized,
            previous_password=previous_password,
            written_password=session_password,
        )
    )
    try:
        _publish_runtime_config_unlocked()
    except Exception as publish_error:
        logger.warning(
            "Republishing the restored config failed (action={}, error_type={}).",
            action,
            type(publish_error).__name__,
        )
    return False


def config_encryption_enabled_on_disk() -> Optional[bool]:
    """Whether the selected config file currently has encryption turned on.

    Returns:
        True or False from the file, or None when it could not be read (the
        caller keeps whatever state it already showed).
    """

    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            serialized = _try_read_cli_config_serialized_unlocked(config_path)
        return _encryption_enabled(_parse_raw_cli_config_text(serialized))
    except Exception as error:
        logger.warning(
            "Reading the config encryption state failed (error_type={}).",
            type(error).__name__,
        )
        return None


def encrypted_value_paths_on_disk() -> list[str]:
    """Dotted paths of the ``enc:`` values in the selected config file.

    Used to name a value that is still encrypted under an earlier password
    when encrypting is refused over it. Empty when there are none or the file
    cannot be read.
    """

    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            serialized = _try_read_cli_config_serialized_unlocked(config_path)
        return _encrypted_value_paths(_parse_raw_cli_config_text(serialized))
    except Exception as error:
        logger.warning(
            "Listing encrypted config values failed (error_type={}).",
            type(error).__name__,
        )
        return []


def verify_config_encryption_password(password: str) -> bool:
    """Whether ``password`` matches the saved master-password verifier.

    Returns False when encryption is off or the verifier is missing. Runs
    scrypt, so call it off the UI thread.
    """

    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            serialized = _try_read_cli_config_serialized_unlocked(config_path)
        config_data = _parse_raw_cli_config_text(serialized)
        verifier = _password_verifier(config_data)
        if not _encryption_enabled(config_data) or verifier is None:
            return False
        return bool(get_encryption_module().verify_password(password, verifier))
    except Exception as error:
        logger.warning(
            "Checking the master password failed (error_type={}).",
            type(error).__name__,
        )
        return False


def enable_config_encryption(password: str) -> bool:
    """
    Enable encryption for the config file and encrypt existing API keys.

    Refuses (returns False without writing) when encryption is already on
    with a different password: re-running enable used to rewrite the
    verifier over keys still encrypted under the first password. Use
    `change_encryption_password` to rotate. Enabling again with the SAME
    password, on a file that is already fully encrypted under it, writes
    nothing and returns True (review round 2, R2-F4); when this session did
    not hold that password it is installed and the runtime settings are
    republished, so a locked session is fully unlocked (Qodo round).

    Args:
        password: The master password to use for encryption

    Returns:
        True if encryption was enabled successfully, or already was with
        this password
    """
    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            previous = _try_read_cli_config_serialized_unlocked(config_path)
            config_data = _parse_raw_cli_config_text(previous)
            if _encryption_enabled(config_data):
                if _already_encrypted_with(config_data, password):
                    if get_encryption_password() != password and not (
                        _unlock_session_with_password_unlocked(config_path, password)
                    ):
                        return False
                    logger.info(
                        "Config encryption is already enabled with this "
                        "password; nothing was changed."
                    )
                    return True
                logger.warning(
                    "Config encryption is already enabled; refusing to enable "
                    "it again. Change the master password instead."
                )
                return False
            encrypted_config = encrypt_api_keys_in_config(config_data, password)
            _validate_encrypted_document(encrypted_config, password)
            if not _commit_encryption_change_unlocked(
                config_path,
                encrypted_config,
                previous,
                session_password=password,
                action="enable",
            ):
                return False

        logger.success("Config encryption enabled successfully")
        return True

    except Exception as e:
        logger.error(
            "Failed to enable config encryption (error_type={}).", type(e).__name__
        )
        return False


def disable_config_encryption(password: str) -> bool:
    """
    Disable encryption for the config file and decrypt all values.

    Every saved value must decrypt with ``password`` (strict); otherwise the
    file is left untouched and False is returned.

    Args:
        password: The master password to verify before disabling

    Returns:
        True if encryption was disabled successfully
    """
    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            previous = _try_read_cli_config_serialized_unlocked(config_path)
            config_data = _parse_raw_cli_config_text(previous)
            if _encryption_enabled(config_data):
                verifier = _password_verifier(config_data)
                if verifier is None:
                    logger.error("No password verifier found in encryption config")
                    return False
                enc_module = get_encryption_module()
                if not enc_module.verify_password(password, verifier):
                    logger.error("Invalid password provided")
                    return False
                try:
                    decrypted_config = enc_module.decrypt_config_strict(
                        config_data, password
                    )
                except ValueError:
                    logger.error(
                        "Some saved values do not decrypt with this password; "
                        "encryption was left on (action=disable, config_path={}).",
                        redact_user_paths(str(config_path)),
                    )
                    return False
            elif "encryption" in config_data:
                decrypted_config = copy.deepcopy(config_data)
            else:
                _set_session_encryption_password(None)
                logger.info("Config encryption is already disabled")
                return True
            decrypted_config.pop("encryption", None)
            if _contains_encrypted_value(decrypted_config):
                logger.error(
                    "Encrypted values would remain after disabling encryption; "
                    "encryption was left on (action=disable, config_path={}).",
                    redact_user_paths(str(config_path)),
                )
                return False
            if not _commit_encryption_change_unlocked(
                config_path,
                decrypted_config,
                previous,
                session_password=None,
                action="disable",
            ):
                return False

        logger.success("Config encryption disabled successfully")
        return True

    except Exception as e:
        logger.error(
            "Failed to disable config encryption (error_type={}).", type(e).__name__
        )
        return False


def change_encryption_password(old_password: str, new_password: str) -> bool:
    """
    Change the encryption password.

    Every saved value must decrypt with ``old_password`` (strict), and the
    re-encrypted document must decrypt with ``new_password`` before it is
    written; otherwise the file is left untouched.

    Args:
        old_password: The current password
        new_password: The new password to set

    Returns:
        True if password was changed successfully
    """
    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            previous = _try_read_cli_config_serialized_unlocked(config_path)
            config_data = _parse_raw_cli_config_text(previous)
            if not _encryption_enabled(config_data):
                logger.error("Encryption is not enabled")
                return False

            enc_module = get_encryption_module()
            verifier = _password_verifier(config_data)
            if verifier is None:
                logger.error("No password verifier found in encryption config")
                return False
            if not enc_module.verify_password(old_password, verifier):
                logger.error("Invalid current password provided")
                return False
            try:
                decrypted_config = enc_module.decrypt_config_strict(
                    config_data, old_password
                )
            except ValueError:
                logger.error(
                    "Some saved values do not decrypt with the current password; "
                    "the password was not changed (action=change, config_path={}).",
                    redact_user_paths(str(config_path)),
                )
                return False
            encrypted_config = encrypt_api_keys_in_config(
                decrypted_config,
                new_password,
            )
            _validate_encrypted_document(encrypted_config, new_password)
            if not _commit_encryption_change_unlocked(
                config_path,
                encrypted_config,
                previous,
                session_password=new_password,
                action="change",
            ):
                return False

        logger.success("Encryption password changed successfully")
        return True

    except Exception as e:
        logger.error(
            "Failed to change encryption password (error_type={}).",
            type(e).__name__,
        )
        return False


def rekey_encryption_verifier(password: str) -> bool:
    """Make ``password`` the master password again when it reads every key.

    For an encryption-on file whose verifier is missing, or was rewritten
    for another password while the keys stayed under this one (the stranded
    state an old second enable left behind). Without this the only way past
    either was a reset that deleted keys this password still reads (TASK-
    34100.4 review round 2, R2-F5). Every ``enc:`` value outside
    ``[encryption]`` must strict-decrypt with ``password``, and there must be
    at least one -- with none, any guess would "read every key". The file is
    re-encrypted under ``password`` with a new verifier, through the same
    validate-then-write and rollback as `change_encryption_password`.

    Args:
        password: The password the user typed at the startup prompt.

    Returns:
        True when the file now answers to ``password`` and the session holds
        it; False (file untouched) otherwise.
    """
    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            previous = _try_read_cli_config_serialized_unlocked(config_path)
            config_data = _parse_raw_cli_config_text(previous)
            if not _encryption_enabled(config_data):
                return False
            values = {
                key: value for key, value in config_data.items() if key != "encryption"
            }
            if not _contains_encrypted_value(values):
                return False
            try:
                decrypted_config = get_encryption_module().decrypt_config_strict(
                    values, password, log_failure=False
                )
            except ValueError:
                return False
            decrypted_config["encryption"] = copy.deepcopy(config_data["encryption"])
            encrypted_config = encrypt_api_keys_in_config(decrypted_config, password)
            _validate_encrypted_document(encrypted_config, password)
            if not _commit_encryption_change_unlocked(
                config_path,
                encrypted_config,
                previous,
                session_password=password,
                action="rekey",
            ):
                return False

        logger.success("The master-password check was repaired")
        return True

    except Exception as e:
        logger.error(
            "Repairing the master-password check failed (error_type={}).",
            type(e).__name__,
        )
        return False


def reset_encrypted_config_values() -> bool:
    """Forgot-password reset: drop every encrypted value and turn encryption off.

    Strips every ``enc:`` value from every table and array (sensitive or
    not -- none of them can be read without the password) and removes
    ``[encryption]``. Chats, notes, documents and every plain setting are
    untouched; the user re-enters API keys afterwards.

    Returns:
        True when the file now holds no encrypted value and encryption is off.
    """
    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            previous = _try_read_cli_config_serialized_unlocked(config_path)
            if previous is None:
                _set_session_encryption_password(None)
                return True
            config_data = _parse_raw_cli_config_text(previous)
            stripped = _strip_encrypted_values(config_data)
            stripped.pop("encryption", None)
            if not _commit_encryption_change_unlocked(
                config_path,
                stripped,
                previous,
                session_password=None,
                action="reset",
            ):
                return False

        logger.success("Encrypted config values were reset")
        return True

    except Exception as e:
        logger.error(
            "Resetting encrypted config values failed (error_type={}).",
            type(e).__name__,
        )
        return False


# --- CLI Database and Log File Path Getters ---
BASE_DATA_DIR_CLI = Path.home() / ".local" / "share" / "tldw_cli"  # Renamed for clarity
_DEFAULT_DATA_FALLBACK_DIRECTORY = ".tldw_cli-data"
# NOTE: BASE_DATA_DIR_CLI is a module-level constant frozen at IMPORT time
# (kept for backward compatibility -- some callers reference it directly).
# get_user_data_dir()'s fallback below does NOT use it; it resolves the
# default at CALL time via _default_base_data_dir() instead, so per-test
# HOME monkeypatches (applied well after this module is first imported) are
# actually honored. See task-519. (XDG_DATA_HOME is deliberately NOT
# consulted -- see _default_base_data_dir()'s docstring.)


def _default_base_data_dir() -> Path:
    """Default data dir resolved at CALL time (honors post-import HOME changes
    for test isolation; task-519). Deliberately does NOT honor XDG_DATA_HOME:
    the pre-existing default never did, and adding it would silently relocate
    an XDG user's data dir on upgrade with no migration (task-519 review).

    Uses os.environ["HOME"] explicitly (falling back to Path.home()) because
    Path.home() is not guaranteed to re-read a post-import HOME monkeypatch
    on every platform/Python version, whereas os.environ is always read live.
    """
    return profile_paths.default_base_data_dir()


def _data_root_entry_exists(path: Path) -> bool:
    """Count links as existing data; never interpret access errors as absence."""
    path = validate_path_simple(path, require_exists=False, probe_existing=False)
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _selected_default_base_data_dir() -> Path:
    """Read the durable default-root selection without creating directories."""
    conventional = _default_base_data_dir()
    fallback = conventional.parents[2] / _DEFAULT_DATA_FALLBACK_DIRECTORY
    if not _data_root_entry_exists(fallback):
        return conventional
    if _data_root_entry_exists(conventional):
        raise PrivatePathError(
            PrivatePathResult(
                conventional,
                PrivatePathStatus.OPERATION_FAILED,
                reason="ambiguous_default_data_roots",
            )
        )
    return fallback


@contextmanager
def _default_data_root_lock() -> Iterator[None]:
    """Serialize root selection and profile creation across starts (ADR-127)."""
    lock_path = validate_path_simple(
        _default_base_data_dir().parents[2] / profile_paths.DATA_ROOT_LOCK_NAME,
        require_exists=False,
        probe_existing=False,
    )
    with _config_participants.operation(
        sys.modules[__name__], route="config_data_lock", target=lock_path
    ) as operation:
        try:
            create_private_text(lock_path, "")
        except FileExistsError:
            pass
        # Keep this inode stable across releases and across config-file choices.
        with closing(open_private_text_append_stream(lock_path)) as stream:
            while True:
                _config_participants.check_lock_wait(sys.modules[__name__], operation)
                try:
                    portalocker.lock(
                        stream,
                        portalocker.LockFlags.EXCLUSIVE
                        | portalocker.LockFlags.NON_BLOCKING,
                    )
                    break
                except portalocker.exceptions.AlreadyLocked:
                    time.sleep(0.05)
            try:
                _config_participants.check_lock_wait(sys.modules[__name__], operation)
                yield
            finally:
                portalocker.unlock(stream)


def _secure_default_data_dir() -> Path:
    """Recover a fresh root while the caller holds _default_data_root_lock."""
    selected = _selected_default_base_data_dir()
    try:
        with _config_participants.operation(
            sys.modules[__name__], route="config_default_root", target=selected
        ):
            return secure_private_directory(
                selected, create=True, application_owned=True
            ).lexical_path
    except PrivatePathError as exc:
        conventional = _default_base_data_dir()
        if (
            selected != conventional
            or exc.result.status is not PrivatePathStatus.UNSAFE_PARENT
            or exc.result.reason != "shared_writable_parent"
            or _data_root_entry_exists(conventional)
        ):
            raise
        # Only a missing conventional root can select a new location. Existing
        # data, explicit overrides and unrelated failures must never be hidden.
        # The unchanged guard also refuses a shared/foreign/symlinked HOME.
        fallback = conventional.parents[2] / _DEFAULT_DATA_FALLBACK_DIRECTORY
        with _config_participants.operation(
            sys.modules[__name__], route="config_default_root", target=fallback
        ):
            return secure_private_directory(
                fallback, create=True, application_owned=True
            ).lexical_path


def get_api_key(api_name: str) -> Optional[str]:
    """
    Get API key for a given provider.

    Args:
        api_name: The API provider name (e.g., 'openai', 'anthropic', 'groq')

    Returns:
        A usable API key, or None.

        TASK-32806.1: every branch below is screened by
        ``resolve_provider_api_key``, the same rule the rest of the codebase
        uses, so this can no longer hand a caller a value that rule rejects.
        It previously screened one literal (``<API_KEY_HERE>``) on one branch
        and returned everything raw, so a configured ``YOUR_KEY``, a
        whitespace-padded key, or a blank string reached five live spend and
        readiness paths -- and the realtime pre-connect gate, which tested
        truthiness, passed the placeholder straight to the provider session.
        Screening here also strips, so a padded key now WORKS rather than
        being sent with its spaces.
    """
    # Normalize the API name
    api_name_lower = api_name.lower()

    # First try the newer api_settings.{provider} structure
    try:
        settings = load_settings()
        api_settings_key = f"api_settings.{api_name_lower}"

        # `load_settings()` returns a NESTED dict -- `{"api_settings":
        # {"openai": {...}}}` -- so the flat dotted membership test this
        # used to do (`api_settings_key in settings`) was False for every
        # real config, making this whole branch dead code. Consequence,
        # found at the realtime engine's live gate: a key entered through
        # the Settings screen (which writes exactly here, the correct
        # modern location) was invisible to every caller of this function,
        # and the realtime pre-connect check refused with "no OpenAI API
        # key is configured" against a config that had one. Same root
        # cause as TASK-229's `get_cli_setting` fix, one accessor over.
        #
        # The flat lookup is kept as a fallback: it costs one dict miss,
        # and some settings shapes elsewhere may yet be flattened.
        api_settings = None
        nested_api_settings = settings.get("api_settings")
        if isinstance(nested_api_settings, dict):
            api_settings = nested_api_settings.get(api_name_lower)
        if not isinstance(api_settings, dict):
            api_settings = settings.get(api_settings_key)

        if isinstance(api_settings, dict):
            # TASK-32806.2: the STORED key outranks the environment variable
            # here, matching `_normalize_legacy_provider_api_key` and the
            # rule CLAUDE.md already records ("except provider API keys,
            # where an explicit api_settings.<provider>.api_key now outranks
            # the matching environment variable"). This accessor resolved
            # env-first, so with both set the chat path spent the Settings
            # key while MCP tools and Console realtime spent the env key --
            # the same readiness-and-spend disagreement ADR-012 exists to
            # prevent, moved from two readers to two accessors.
            #
            # The env var is still read; it is simply the fallback, which is
            # what makes a key typed into Settings take effect without the
            # user having to find and unset a shell variable first.
            if (
                "api_key" in api_settings
                and api_settings["api_key"] != "<API_KEY_HERE>"
            ):
                resolved = resolve_provider_api_key(api_settings["api_key"])
                if resolved is not None:
                    return resolved

            if "api_key_env_var" in api_settings:
                env_var = api_settings["api_key_env_var"]
                resolved = resolve_provider_api_key(os.getenv(env_var))
                if resolved is not None:
                    return resolved
    except Exception as e:
        logger.debug(f"Error accessing api_settings for {api_name}: {e}")

    # Try the legacy approach used elsewhere in the codebase
    try:
        # This is the pattern used in other files like Summarization_General_Lib.py
        resolved = resolve_provider_api_key(
            get_cli_setting("API", f"{api_name_lower}_api_key", "")
        )
        if resolved is not None:
            return resolved
    except Exception as e:
        logger.debug(f"Error getting API key via get_cli_setting for {api_name}: {e}")

    # Try direct environment variable access with common patterns
    env_var_names = [f"{api_name_lower.upper()}_API_KEY", f"{api_name.upper()}_API_KEY"]

    for env_var in env_var_names:
        resolved = resolve_provider_api_key(os.getenv(env_var))
        if resolved is not None:
            return resolved

    # No API key found
    logger.debug(f"No API key found for provider: {api_name}")
    return None


def get_user_folder_name() -> str:
    """Get the current user folder name from configuration."""
    default_user = DEFAULT_CONFIG_FROM_TOML.get("general", {}).get(
        "users_name", "default_user"
    )
    user_name = get_cli_setting("general", "users_name", default_user)
    # Sanitize user name to make it safe for folder names
    # Replace spaces and special characters with underscores
    return profile_paths.user_folder_name(user_name)


#: PERF-07 (TASK-33266; ADR-126 amendment D2, 2026-09-29): the verified user
#: data directory, reused while its inputs and every stamped path component are
#: unchanged. ``(key, stamps, path)`` or None; guarded by the lock below.
_USER_DATA_DIR_MEMO: tuple | None = None
_USER_DATA_DIR_MEMO_LOCK = _threading.Lock()


def _user_data_dir_inputs() -> tuple[tuple, tuple[Path, ...]]:
    """The memo key and every path whose posture the resolution depends on.

    Returns:
        ``(key, stamped_paths)``. The key starts with the config cache object
        (compared by identity) and its generation, so any reload or write
        invalidates the memo; then the settings the resolution reads and the
        default data base, which follows HOME. The stamped paths are every
        component of each candidate user directory's chain plus the entries
        the default-root selection reads (conventional root, fallback root,
        root lock file).
    """
    from tldw_chatbook.Backup_Recovery.storage_admission import _chain

    user_folder = get_user_folder_name()
    configured_data_dir = get_cli_setting("paths", "data_dir", None)
    if configured_data_dir is None:
        configured_data_dir = get_cli_setting("Paths", "data_dir", None)
    if configured_data_dir:
        candidates = (lexical_path(configured_data_dir) / user_folder,)
        entries: tuple[Path, ...] = ()
        # Expanded, not as written: a relative value follows the working dir.
        base = str(candidates[0].parent)
    else:
        conventional = _default_base_data_dir()
        fallback = conventional.parents[2] / _DEFAULT_DATA_FALLBACK_DIRECTORY
        candidates = (conventional / user_folder, fallback / user_folder)
        entries = (
            conventional,
            fallback,
            conventional.parents[2] / profile_paths.DATA_ROOT_LOCK_NAME,
        )
        base = str(conventional)
    key = (_CONFIG_CACHE, _CONFIG_GENERATION, _CONFIG_CACHE_SOURCE, user_folder, base)
    paths = (*(p for c in candidates for p in _chain(c)), *entries)
    return key, tuple(dict.fromkeys(paths))


def _bind_user_data_dir_stamps():
    """Keep one finite optional Windows source capture; acquire no native state."""
    close_errors = ()
    batch = source_current = None
    if os.name == "nt":
        import inspect
        import stat
        from pathlib import WindowsPath
        from types import FunctionType, MethodType, ModuleType

        from tldw_chatbook.Backup_Recovery import storage_admission as storage
        from tldw_chatbook.Utils import platform_files, windows_files

        facade, owner = platform_files.os, windows_files.WindowsOS
        native = windows_files._Native
        close_error = windows_files._AdmissionMetadataCloseError
        close_errors = (close_error,)
        original_posture = storage._posture
        projection = (stat.S_IFMT, stat.S_IMODE)
        modules = tuple(
            (module, vars(module), module.__file__, module.__spec__, module.__spec__.origin)
            for module in (storage, platform_files, windows_files)
        )
        bindings = (
            (vars(storage), "_posture", original_posture),
            (vars(storage), "os", facade),
            (vars(storage), "stat", stat),
            (vars(platform_files), "os", facade),
            (vars(platform_files), "WindowsOS", owner),
            (vars(windows_files), "WindowsOS", owner),
            (vars(windows_files), "_Native", native),
            (vars(windows_files), "_AdmissionMetadataCloseError", close_error),
            *((vars(stat), name, value) for name, value in zip(("S_IFMT", "S_IMODE"), projection)),
            *((vars(windows_files), name, vars(windows_files)[name]) for name in (
                "_os", "_stat", "Path", "ntpath", "contextlib", "functools",
                "_parent", "_component", "_native", "_acl_mode", "_AdmissionMetadataHandle",
            )),
            *((vars(owner), name, vars(owner)[name]) for name in (
                "__getattr__", "stat", "_stat_handle", "stat_many_for_admission",
            )),
            *((vars(native), name, vars(native)[name]) for name in (
                "__init__", "_user_sid", "reopen", "open_handle", "info", "ntfs",
                "security", "_decoded_security", "_decode_security", "sid_string", "check", "ntcheck",
            )),
            (vars(close_error), "__init__", close_error.__init__),
            (vars(windows_files._AdmissionMetadataHandle), "__init__", windows_files._AdmissionMetadataHandle.__init__),
        )
        wrappers = (
            windows_files._parent, windows_files._native,
            native.reopen, native._decoded_security,
        )
        functions = (
            original_posture, owner.__getattr__, owner.stat, owner._stat_handle,
            owner.stat_many_for_admission, windows_files._component, windows_files._acl_mode,
            native.__init__, native._user_sid, native.open_handle, native.info, native.ntfs,
            native.security, native._decode_security, native.sid_string, native.check, native.ntcheck,
            close_error.__init__, windows_files._AdmissionMetadataHandle.__init__,
            windows_files._parent, windows_files._parent.__wrapped__,
            windows_files._native.__wrapped__, native.reopen, native.reopen.__wrapped__,
            native._decoded_security.__wrapped__,
        )
        records = tuple(
            (function, function.__code__, function.__globals__, function.__defaults__,
             function.__kwdefaults__, tuple((function.__kwdefaults__ or {}).items()),
             function.__closure__, tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
             vars(function).get("__wrapped__"))
            for function in functions
        )
        wrapped = tuple((wrapper, vars(wrapper).get("__wrapped__")) for wrapper in wrappers)
        lookup = tuple((cls, name, inspect.getattr_static(cls, name, None))
                       for cls in (owner, native) for name in ("__getattribute__", "__getattr__", "__dict__"))
        cached = tuple((name, vars(facade).get(name), vars(owner)[name])
                       for name in ("stat", "_stat_handle", "stat_many_for_admission"))
        batch = cached[-1][1]

        def source_current(posture):
            # Direct custom slots decline before any facade lookup or producer.
            if posture is not original_posture or storage.os is not facade or storage.stat is not stat:
                return False
            if any(namespace.get(name) is not value for namespace, name, value in bindings):
                return False
            if (
                type(facade) is not owner or type(owner) is not type or type(native) is not type
                or owner.__bases__ != (object,) or native.__bases__ != (object,)
                or inspect.getattr_static(owner, "__getattribute__") is not object.__getattribute__
                or inspect.getattr_static(native, "__getattribute__") is not object.__getattribute__
                or any(inspect.getattr_static(cls, name, None) is not value for cls, name, value in lookup)
            ):
                return False
            for module, namespace, path, spec, origin in modules:
                if (type(module) is not ModuleType or sys.modules.get(namespace["__name__"]) is not module
                    or vars(module) is not namespace or namespace.get("__file__") != path
                    or namespace.get("__spec__") is not spec or spec.origin != origin):
                    return False
            fields = vars(facade)
            if any(type(bound) is not MethodType or fields.get(name) is not bound
                   or bound.__self__ is not facade or bound.__func__ is not function
                   for name, bound, function in cached):
                return False
            for function, code, namespace, defaults, keywords, items, closure, cells, body in records:
                if (type(function) is not FunctionType or function.__code__ is not code
                    or function.__globals__ is not namespace or function.__defaults__ is not defaults
                    or function.__kwdefaults__ is not keywords
                    or len(function.__kwdefaults__ or {}) != len(items)
                    or any((function.__kwdefaults__ or {}).get(key) is not value for key, value in items)
                    or function.__closure__ is not closure
                    or vars(function).get("__wrapped__") is not body):
                    return False
                try:
                    if any(cell.cell_contents is not value for cell, value in cells):
                        return False
                except ValueError:  # An emptied captured cell is nonstock too.
                    return False
            return all(vars(wrapper).get("__wrapped__") is body for wrapper, body in wrapped)

    def _user_data_dir_stamps(paths: tuple[Path, ...]) -> tuple | None:
        """Observe ordered five-field postures; preserve retained-close custody.

        The stock Windows route performs one fresh metadata batch. Customized
        sources retain the original scalar calls, and ordinary OSError leaves
        resolution to report the original private-path error.
        """
        from tldw_chatbook.Backup_Recovery.storage_admission import _posture

        try:
            if type(paths) is tuple and not paths:
                return ()
            if (source_current is not None and source_current(_posture)
                and type(paths) is tuple and all(type(path) is WindowsPath and path.is_absolute()
                    and len(path.drive) == 2 and path.drive[1] == ":" and path.drive[0].isalpha()
                    for path in paths) and source_current(_posture)):
                try:
                    observed = batch(paths)
                except close_errors:
                    raise
                except OSError as error:
                    # The call has unwound its native finally before selection.
                    current = source_current(_posture)
                    if not current or type(error) is not PermissionError or error.errno != 13 or error.strerror != "windows_admission_descriptor_unavailable":
                        return None
                else:
                    # Any observed drift discards the whole sample permanently.
                    current = source_current(_posture)
                    if not current or type(observed) is not dict:
                        return None
                    result = []
                    for path in paths:
                        value = observed[path]
                        if value is None:
                            result.append(None)
                        else:
                            info = value[0]
                            result.append((info.st_dev, info.st_ino, projection[0](info.st_mode),
                                           projection[1](info.st_mode), info.st_uid))
                    return tuple(result) if source_current(_posture) else None
            return tuple(_posture(path) for path in paths)
        except close_errors:
            raise
        except OSError:
            return None

    return _user_data_dir_stamps


_user_data_dir_stamps = _bind_user_data_dir_stamps()
del _bind_user_data_dir_stamps


@_config_participants.guarded
def get_user_data_dir() -> Path:
    """Return the secured lexical user-specific data directory.

    PERF-07: after the guarded handshake, a resolution is reused while the
    config generation, the settings it reads and the posture (identity, type,
    mode, owner) of every component from ``/`` to each candidate directory
    are unchanged, re-observed on every call. Anything else runs the
    unmodified resolution, the only place that creates, hardens or refuses.
    A result is kept only when stamps taken before and after that resolution
    are identical, so it describes exactly the state that was verified.

    Returns:
        The lexical (unresolved) user data directory, created if absent and
        hardened to owner-only permissions.

    Raises:
        PrivatePathError: A component of the path is unsafe (for example a
            group-writable ancestor); the resolution refuses it.
    """
    global _USER_DATA_DIR_MEMO
    verified = _config_participants.verified_user_data_directory(sys.modules[__name__])
    if verified is not None:
        return verified
    key, stamped = _user_data_dir_inputs()
    with _USER_DATA_DIR_MEMO_LOCK:
        memo = _USER_DATA_DIR_MEMO
    if (
        memo is not None
        and memo[0][0] is key[0]
        and memo[0][1:] == key[1:]
        and memo[1] == _user_data_dir_stamps(stamped)
    ):
        return memo[2]
    before = _user_data_dir_stamps(stamped)
    result = _resolve_user_data_dir()
    after = _user_data_dir_stamps(stamped)
    import stat

    from tldw_chatbook.Backup_Recovery.storage_admission import _chain

    walked = set(_chain(result))
    # The result must be one of the stamped candidates: a working directory
    # changed mid-resolution makes a relative data dir resolve elsewhere.
    if (
        before is not None
        and before == after
        and walked <= set(stamped)
        and all(
            stamp is not None and not stat.S_ISLNK(stamp[2])
            for path, stamp in zip(stamped, after)
            if path in walked
        )
    ):
        with _USER_DATA_DIR_MEMO_LOCK:
            _USER_DATA_DIR_MEMO = (key, after, result)
    return result


def _resolve_user_data_dir() -> Path:
    """The unmodified resolution: select, create, harden or refuse."""
    user_folder = get_user_folder_name()
    configured_data_dir = get_cli_setting("paths", "data_dir", None)
    if configured_data_dir is None:
        configured_data_dir = get_cli_setting("Paths", "data_dir", None)
    with ExitStack() as stack:
        if configured_data_dir:
            base_data_dir = lexical_path(configured_data_dir)
        else:
            stack.enter_context(_default_data_root_lock())
            base_data_dir = _secure_default_data_dir()
        user_dir = base_data_dir / user_folder
        with _config_participants.operation(
            sys.modules[__name__], route="config_data", target=user_dir
        ):
            if configured_data_dir:
                verify_trusted_directory(base_data_dir, allow_shared_sticky=False)
            return secure_private_directory(
                user_dir,
                create=True,
                application_owned=True,
            ).lexical_path


def _get_custom_database_path(
    setting_name: str,
    *,
    expand_before_validation: bool = True,
) -> Path | None:
    """Return a validated lexical custom DB path, if explicitly configured.

    This is a non-mutating selection boundary: it intentionally does not
    probe, create, resolve, or chmod the user-selected parent. The consuming
    private SQLite owner must verify that the parent already satisfies the
    trusted-namespace contract before opening the database. Keeping the
    lexical spelling here preserves symlink evidence for that no-follow check.
    """
    custom_path = get_cli_setting("database", setting_name, None)
    default_path = DEFAULT_CONFIG_FROM_TOML.get("database", {}).get(setting_name)
    selected_input = profile_paths.custom_database_input(
        custom_path, default_path, expand_before_validation=expand_before_validation
    )
    if selected_input is None:
        return None
    validated = validate_path_simple(
        selected_input,
        require_exists=False,
        # Preserve lexical evidence and defer filesystem authority to the
        # private SQLite owner; see ADR-029.
        probe_existing=False,
    )
    return lexical_path(validated)


def get_chachanotes_db_path(*, ignore_override: bool = False) -> Path:
    """Get the resolved path for the ChaChaNotes database.

    Args:
        ignore_override: When True, skip any explicitly-configured custom
            path and always return the profile-aware default. Used by the
            Settings "Reset" action, which must discard a user's
            customization rather than reflect it back (TASK-927 follow-up).

    Returns:
        The resolved database path -- either a configured custom path
        (unless ``ignore_override``) or the default filename under the
        current profile's user data directory.
    """
    if ignore_override:
        return get_user_data_dir() / profile_paths.database_leaf("chachanotes_db_path")
    return _get_custom_database_path(
        "chachanotes_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf("chachanotes_db_path")


def get_tts_profiles_db_path() -> Path:
    """Return the validated local TTS generation-profile database path."""

    custom_path = get_cli_setting("database", "tts_profiles_db_path", None)
    if custom_path:
        candidate = Path(str(custom_path))
        if ".." in candidate.parts:
            raise ValueError(
                "TTS profiles database path cannot contain parent traversal"
            )
        candidate = candidate.expanduser()
        return validate_path_simple(candidate, require_exists=False).resolve()
    return get_user_data_dir() / profile_paths.database_leaf("tts_profiles_db_path")


def get_notes_sync_state_db_path() -> Path:
    """Return the device-private Notes import/sync state database path.

    Returns:
        The profile-local path for Notes import receipts and sync state.
    """

    return get_user_data_dir() / "tldw_chatbook_notes_sync_state.db"


NOTES_SYNC_RECOVERY_CAPACITY_BYTES_DEFAULT = 256 * 1024 * 1024
_NOTES_SYNC_RECOVERY_CAPACITY_ENV = "TLDW_NOTES_SYNC_RECOVERY_CAPACITY_BYTES"


def get_notes_sync_recovery_capacity_bytes(
    config_data: Mapping[str, Any] | None = None,
) -> int:
    """Return the one bounded device-private sync recovery capacity."""

    selected = (
        load_cli_config_and_ensure_existence() if config_data is None else config_data
    )
    env_value = os.getenv(_NOTES_SYNC_RECOVERY_CAPACITY_ENV)
    notes = selected.get("notes")
    capacity: object
    if env_value is not None:
        try:
            capacity = int(env_value)
        except ValueError:
            raise ValueError(
                "notes.recovery_capacity_bytes must be a positive integer."
            ) from None
    elif isinstance(notes, Mapping):
        capacity = notes.get(
            "recovery_capacity_bytes",
            NOTES_SYNC_RECOVERY_CAPACITY_BYTES_DEFAULT,
        )
    else:
        capacity = NOTES_SYNC_RECOVERY_CAPACITY_BYTES_DEFAULT
    if type(capacity) is not int or not 1 <= capacity <= 2**63 - 1:
        raise ValueError("notes.recovery_capacity_bytes must be a positive integer.")
    return capacity


NOTES_SYNC_WATCHER_INTERVAL_SECONDS_DEFAULT = 1.0
NOTES_SYNC_WATCHER_MAX_INTERVAL_SECONDS_DEFAULT = 30.0
_NOTES_SYNC_WATCHER_INTERVAL_CEILING_SECONDS = 3600.0


def get_notes_sync_watcher_intervals(
    config_data: Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    """Return the notes-sync watcher's (base, max) polling intervals.

    TASK-21112: the lasting-sync watcher polls at the base interval and backs
    off toward the max while roots are quiet. Read from
    ``[notes] sync_watcher_interval_seconds`` (default 1.0) and
    ``[notes] sync_watcher_max_interval_seconds`` (default 30.0; backed-off
    sleeps are jittered by up to +/-50 percent around it).

    Args:
        config_data: Optional pre-loaded settings mapping; loads the CLI
            config when omitted.

    Returns:
        A ``(interval_seconds, max_interval_seconds)`` pair.

    Raises:
        ValueError: If either configured value is not a positive number in
            range, or the max is below the base interval.
    """

    selected = (
        load_cli_config_and_ensure_existence() if config_data is None else config_data
    )
    notes = selected.get("notes") if isinstance(selected, Mapping) else None
    base: object = NOTES_SYNC_WATCHER_INTERVAL_SECONDS_DEFAULT
    peak: object = NOTES_SYNC_WATCHER_MAX_INTERVAL_SECONDS_DEFAULT
    if isinstance(notes, Mapping):
        base = notes.get("sync_watcher_interval_seconds", base)
        peak = notes.get("sync_watcher_max_interval_seconds", peak)
    for label, value in (
        ("notes.sync_watcher_interval_seconds", base),
        ("notes.sync_watcher_max_interval_seconds", peak),
    ):
        if (
            type(value) not in (int, float)
            or not 0.05 <= value <= _NOTES_SYNC_WATCHER_INTERVAL_CEILING_SECONDS
        ):
            raise ValueError(f"{label} must be a number between 0.05 and 3600 seconds.")
    if peak < base:
        raise ValueError(
            "notes.sync_watcher_max_interval_seconds must be at least "
            "notes.sync_watcher_interval_seconds."
        )
    return float(base), float(peak)


def load_console_library_migration_seed(
    app_config: Mapping[str, Any] | None = None,
) -> "ConsoleLibraryMigrationSeed":
    """Return the sanitized pre-upgrade automatic-retrieval migration seed.

    Args:
        app_config: Optional already-loaded application configuration.

    Returns:
        The strict typed seed required by a legacy database migration.
    """
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryMigrationSeed

    selected = (
        load_cli_config_and_ensure_existence() if app_config is None else app_config
    )
    chat_defaults = (
        selected.get("chat_defaults") if isinstance(selected, Mapping) else None
    )
    raw_value = (
        chat_defaults.get("rag_auto_retrieve_on_send", False)
        if isinstance(chat_defaults, Mapping)
        else False
    )
    return ConsoleLibraryMigrationSeed(
        auto_retrieve_on_send=raw_value if type(raw_value) is bool else False
    )


def get_prompts_db_path(*, ignore_override: bool = False) -> Path:
    """Get the resolved path for the Prompts database.

    Args:
        ignore_override: When True, skip any explicitly-configured custom
            path and always return the profile-aware default. Used by the
            Settings "Reset" action, which must discard a user's
            customization rather than reflect it back (TASK-927 follow-up).

    Returns:
        The resolved database path -- either a configured custom path
        (unless ``ignore_override``) or the default filename under the
        current profile's user data directory.
    """
    if ignore_override:
        return get_user_data_dir() / profile_paths.database_leaf("prompts_db_path")
    return _get_custom_database_path(
        "prompts_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf("prompts_db_path")


def get_media_db_path(*, ignore_override: bool = False) -> Path:
    """Get the resolved path for the Media database.

    Args:
        ignore_override: When True, skip any explicitly-configured custom
            path and always return the profile-aware default. Used by the
            Settings "Reset" action, which must discard a user's
            customization rather than reflect it back (TASK-927 follow-up).

    Returns:
        The resolved database path -- either a configured custom path
        (unless ``ignore_override``) or the default filename under the
        current profile's user data directory.
    """
    if ignore_override:
        return get_user_data_dir() / profile_paths.database_leaf("media_db_path")
    return _get_custom_database_path(
        "media_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf("media_db_path")


def get_library_collections_db_path() -> Path:
    return _get_custom_database_path(
        "library_collections_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf(
        "library_collections_db_path"
    )


def get_dreams_db_path() -> Path:
    """Return the canonical path for the Dreams database (dreams phase 1)."""
    return (
        _get_custom_database_path("dreams_db_path")
        or get_user_data_dir() / "dreams.sqlite"
    )


def get_library_ingest_jobs_db_path() -> Path:
    return _get_custom_database_path(
        "library_ingest_jobs_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf(
        "library_ingest_jobs_db_path"
    )


def get_workspaces_db_path() -> Path:
    return _get_custom_database_path(
        "workspaces_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf("workspaces_db_path")


def get_subscriptions_db_path() -> Path:
    return _get_custom_database_path(
        "subscriptions_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf("subscriptions_db_path")


def get_evals_db_path() -> Path:
    """Return the canonical path for the Evals database."""
    return _get_custom_database_path(
        "evals_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf("evals_db_path")


def get_rag_indexing_db_path() -> Path:
    """Return the canonical path for the RAG indexing-state database."""
    return _get_custom_database_path(
        "rag_indexing_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf("rag_indexing_db_path")


def get_notifications_db_path() -> Path:
    return _get_custom_database_path(
        "notifications_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf("notifications_db_path")


def get_research_db_path() -> Path:
    return _get_custom_database_path(
        "research_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf("research_db_path")


def get_workflows_db_path() -> Path:
    """Return the local authoring store; called only on Workflows entry."""
    return (
        _get_custom_database_path("workflows_db_path")
        or get_user_data_dir() / "tldw_chatbook_workflows.db"
    )


def get_writing_db_path() -> Path:
    return _get_custom_database_path(
        "writing_db_path"
    ) or get_user_data_dir() / profile_paths.database_leaf("writing_db_path")


def get_scheduled_tasks_db_path() -> Path:
    return _get_custom_database_path(
        "scheduled_tasks_db_path",
        expand_before_validation=False,
    ) or get_user_data_dir() / profile_paths.database_leaf("scheduled_tasks_db_path")


def get_cli_log_file_path() -> Path:
    """Return the configured log file beneath the secured user data directory."""

    user_dir = get_user_data_dir()
    default_log_filename = DEFAULT_CONFIG_FROM_TOML.get("logging", {}).get(
        "log_filename", "tldw_cli_app.log"
    )
    log_filename = get_cli_setting("logging", "log_filename", default_log_filename)
    if (
        not isinstance(log_filename, str)
        or not log_filename.strip()
        or log_filename in {".", ".."}
        or "/" in log_filename
        or "\\" in log_filename
        or Path(log_filename).is_absolute()
        or Path(log_filename).name != log_filename
    ):
        raise ValueError("Configured log filename must be a non-empty basename")
    return user_dir / log_filename


def get_cli_data_dir() -> Path:
    """Get the CLI data directory for storing application data."""
    # Return user-specific directory
    return get_user_data_dir()


@_config_participants.guarded
def get_model_cache_dir() -> Path:
    """Get the user-specific model cache directory for embeddings."""
    # Check if a custom cache dir is configured
    default_cache_dir = DEFAULT_CONFIG_FROM_TOML.get("embedding_config", {}).get(
        "model_cache_dir", None
    )
    custom_cache_dir = get_cli_setting(
        "embedding_config", "model_cache_dir", default_cache_dir
    )

    cache_path: Path | None = None
    if custom_cache_dir and custom_cache_dir != default_cache_dir:
        # Use custom path if explicitly configured. task-32901: the
        # configured value is config input, so it goes through central
        # validation before any filesystem creation (probe off: the
        # no-follow boundary below owns link handling). An unusable value
        # refuses loudly: silently relocating the cache would diverge from
        # the config-participant binding for this route.
        cache_path = validate_path_simple(
            Path(custom_cache_dir).expanduser(),
            require_exists=False,
            probe_existing=False,
        )
        if not cache_path.is_absolute():
            raise ValueError(
                f"Configured model cache directory {custom_cache_dir!r} is not"
                " a usable path: it must be absolute"
            )
    if cache_path is None:
        # Use user-specific folder
        user_dir = get_user_data_dir()
        cache_path = user_dir / "models" / "embeddings"

    # Create directory if it doesn't exist. task-32901: umask-derived modes
    # land 0775 on umask-002 machines. The default data-root path uses the
    # private lifecycle (creates 0700, hardens an existing 0775 instance);
    # an explicitly configured custom directory stays under the user's
    # ownership policy, so only the creation mode is pinned there.
    try:
        with _config_participants.operation(
            sys.modules[__name__], route="config_models", target=cache_path
        ):
            if custom_cache_dir and custom_cache_dir != default_cache_dir:
                cache_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            else:
                secure_private_directory(
                    cache_path, create=True, application_owned=True
                )
    except PrivatePathError as e:
        repair = private_path_repair_hint(e)
        suffix = f" Repair once with: {repair}" if repair else ""
        logger.error(
            f"Could not create model cache directory {cache_path}: {e}.{suffix}"
        )
    except OSError as e:
        logger.opt(exception=True).error(
            f"Could not create model cache directory {cache_path}: {e}"
        )

    return cache_path


# --- Global CLI Database Instances ---
chachanotes_db: Optional[CharactersRAGDB] = None
prompts_db: Optional[PromptsDatabase] = None
media_db: Optional[MediaDatabase] = None


def seed_builtin_content(db: CharactersRAGDB) -> CharactersRAGDB:
    """Seed restart-safe bundled profile content after database initialization.

    Args:
        db: Initialized profile-local character database.

    Returns:
        The same database instance after best-effort built-in seeding.
    """
    try:
        from tldw_chatbook.Character_Chat.visual_identity import ensure_builtin_samira

        ensure_builtin_samira(db)
    except Exception as exc:  # noqa: BLE001 - bundled content cannot prevent boot
        logger.warning("builtin_profile_seed_failed category={}", type(exc).__name__)
    try:
        from tldw_chatbook.Character_Chat.builtin_pixel_migu import (
            ensure_builtin_pixel_migu,
        )

        ensure_builtin_pixel_migu(db)
    except Exception as exc:  # noqa: BLE001 - bundled content cannot prevent boot
        logger.warning("pixel_migu_profile_seed_failed category={}", type(exc).__name__)
    return db


# --- Database Initialization Function (remains largely the same) ---
def initialize_all_databases():
    global chachanotes_db, prompts_db, media_db
    logger.debug("CRITICAL DEBUG: INSIDE initialize_all_databases() NOW.")
    logger.info("Initializing CLI databases...")
    # ChaChaNotes DB
    chachanotes_path = get_chachanotes_db_path()
    logger.info(f"Attempting to initialize ChaChaNotes_DB at: {chachanotes_path}")
    try:
        chachanotes_db = CharactersRAGDB(
            db_path=chachanotes_path,
            client_id=CLI_APP_CLIENT_ID,
            console_library_migration_seed=load_console_library_migration_seed(),
        )
        seed_builtin_content(chachanotes_db)
        logger.success(f"ChaChaNotes_DB initialized successfully at {chachanotes_path}")
    except Exception as e:
        logger.opt(exception=True).error(
            f"Failed to initialize ChaChaNotes_DB at {chachanotes_path}: {e}"
        )
        chachanotes_db = None
    # Prompts DB
    prompts_path = get_prompts_db_path()
    logger.info(f"Attempting to initialize Prompts_DB at: {prompts_path}")
    try:
        prompts_db = PromptsDatabase(db_path=prompts_path, client_id=CLI_APP_CLIENT_ID)
        logger.success(f"Prompts_DB initialized successfully at {prompts_path}")
    except Exception as e:
        logger.opt(exception=True).error(
            f"Failed to initialize Prompts_DB at {prompts_path}: {e}"
        )
        prompts_db = None
    # Media DB
    media_path = get_media_db_path()
    logger.info(f"Attempting to initialize Media_DB_v2 at: {media_path}")
    try:
        media_db = MediaDatabase(db_path=media_path, client_id=CLI_APP_CLIENT_ID)
        logger.success(f"Media_DB_v2 initialized successfully at {media_path}")
    except Exception as e:
        logger.opt(exception=True).error(
            f"Failed to initialize Media_DB_v2 at {media_path}: {e}"
        )
        media_db = None
    logger.info("CLI database initialization complete.")


# --- Lazy Database Getters ---
def get_chachanotes_db_lazy() -> Optional[CharactersRAGDB]:
    """Get the ChaChaNotes database instance, initializing it lazily if needed."""
    global chachanotes_db
    if chachanotes_db is None:
        chachanotes_path = get_chachanotes_db_path()
        logger.info(f"Lazy-initializing ChaChaNotes_DB at: {chachanotes_path}")
        try:
            # Get integrity check configuration
            config = load_settings()
            check_integrity = config.get("database", {}).get(
                "check_integrity_on_startup", False
            )

            chachanotes_db = CharactersRAGDB(
                db_path=chachanotes_path,
                client_id=CLI_APP_CLIENT_ID,
                check_integrity_on_startup=check_integrity,
                console_library_migration_seed=load_console_library_migration_seed(),
            )
            seed_builtin_content(chachanotes_db)
            logger.success(
                f"ChaChaNotes_DB lazy-initialized successfully at {chachanotes_path}"
            )
        except Exception as e:
            logger.opt(exception=True).error(
                f"Failed to lazy-initialize ChaChaNotes_DB at {chachanotes_path}: {e}"
            )
            if chachanotes_db is not None:
                # Seeding can fail after construction has acquired this worker's
                # connection. Retire it before discarding the lazy owner.
                chachanotes_db.close_connection()
            chachanotes_db = None
    return chachanotes_db


def get_prompts_db_lazy() -> Optional[PromptsDatabase]:
    """Get the Prompts database instance, initializing it lazily if needed."""
    global prompts_db
    if prompts_db is None:
        prompts_path = get_prompts_db_path()
        logger.info(f"Lazy-initializing Prompts_DB at: {prompts_path}")
        try:
            # Get integrity check configuration
            config = load_settings()
            check_integrity = config.get("database", {}).get(
                "check_integrity_on_startup", False
            )

            prompts_db = PromptsDatabase(
                db_path=prompts_path,
                client_id=CLI_APP_CLIENT_ID,
                check_integrity_on_startup=check_integrity,
            )
            logger.success(
                f"Prompts_DB lazy-initialized successfully at {prompts_path}"
            )
        except Exception as e:
            logger.opt(exception=True).error(
                f"Failed to lazy-initialize Prompts_DB at {prompts_path}: {e}"
            )
            prompts_db = None
    return prompts_db


def get_media_db_lazy() -> Optional[MediaDatabase]:
    """Get the Media database instance, initializing it lazily if needed."""
    global media_db
    if media_db is None:
        media_path = get_media_db_path()
        logger.info(f"Lazy-initializing Media_DB_v2 at: {media_path}")
        try:
            media_db = MediaDatabase(db_path=media_path, client_id=CLI_APP_CLIENT_ID)
            logger.success(f"Media_DB_v2 lazy-initialized successfully at {media_path}")
        except Exception as e:
            logger.opt(exception=True).error(
                f"Failed to lazy-initialize Media_DB_v2 at {media_path}: {e}"
            )
            media_db = None
    return media_db


# --- API Models (should be defined based on CONFIG_TOML_CONTENT or loaded from it) ---
# These can be loaded dynamically from the config or kept as fallback statics
# For simplicity, if CONFIG_TOML_CONTENT has [providers], use that.
# Reuse the [providers] table already parsed into DEFAULT_CONFIG_FROM_TOML
# above instead of re-parsing the entire embedded TOML string a second time.
# Deep-copied (not aliased) so the per-provider model lists stay independent
# of DEFAULT_CONFIG_FROM_TOML's own tree.
API_MODELS_BY_PROVIDER: Dict[str, List[str]] = {}
LOCAL_PROVIDERS: Dict[str, List[str]] = {}

_config_providers = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML.get("providers", {}))
# ADR-179: cloud classification is derived from the provider registry
# (single source of truth) instead of a hand-typed list here, so a newly
# registered cloud provider is classified without touching this module.
_cloud_provider_keys = CLOUD_PROVIDER_CONFIG_KEYS

for provider_name, models_list in _config_providers.items():
    if isinstance(models_list, list):
        if (
            provider_name in _cloud_provider_keys
        ):  # Crude way to separate, adjust as needed
            API_MODELS_BY_PROVIDER[provider_name] = models_list
        else:
            LOCAL_PROVIDERS[provider_name] = models_list
    else:
        logger.warning(
            f"Models for provider '{provider_name}' in CONFIG_TOML_CONTENT is not a list. Skipping."
        )

if (
    not API_MODELS_BY_PROVIDER and not LOCAL_PROVIDERS
):  # Fallback if [providers] was empty or malformed
    logger.warning(
        "No providers found in CONFIG_TOML_CONTENT's [providers] section. Using hardcoded fallbacks for API_MODELS_BY_PROVIDER and LOCAL_PROVIDERS."
    )
    API_MODELS_BY_PROVIDER = {"OpenAI": ["gpt-5.6-terra"]}  # Minimal fallback
    LOCAL_PROVIDERS = {"Ollama": ["llama3"]}  # Minimal fallback


# --- Global default_api_endpoint (example of using the new settings) ---

# --- Global Settings Object ---
# task-32900: a private-path refusal here (e.g. a group-writable ~/.config on
# a fresh install) used to surface as a bare `unsafe_parent:
# shared_writable_parent` traceback. Emit the plain-language diagnostic once,
# then keep failing closed exactly as before (ADR-029/ADR-127 posture).
try:
    load_cli_config_and_ensure_existence()
except PrivatePathError as _private_path_startup_error:
    from tldw_chatbook.Utils.startup_errors import (
        emit_private_path_startup_error,
    )

    emit_private_path_startup_error(_private_path_startup_error)
    raise
settings = load_settings()
if _CONFIG_GENERATION == 0:
    _CONFIG_GENERATION = 1

try:
    # Accessing deeply nested key safely
    default_api_endpoint = settings.get("llm_api_settings", {}).get(
        "default_api", "openai"
    )
    logger.info(
        f"Default API Endpoint (from config.py global scope): {default_api_endpoint}"
    )
except Exception as e:
    logger.opt(exception=True).error(
        f"Critical error setting default_api_endpoint in config.py global scope: {str(e)}"
    )
    default_api_endpoint = "openai"  # Fallback

# --- Optional: Export individual variables if needed (generally prefer using settings dict) ---
# SINGLE_USER_MODE = settings["SINGLE_USER_MODE"]
# OPENAI_API_KEY = settings["OPENAI_API_KEY"]

# Make APP_CONFIG, DATABASE_CONFIG, RAG_SEARCH_CONFIG available globally if needed
# These are now loaded from TOML into the `settings` dictionary.
APP_CONFIG = settings.get(
    "APP_TTS_CONFIG", DEFAULT_APP_TTS_CONFIG
)  # Fallback if not in settings for some reason
DATABASE_CONFIG = settings.get("APP_DATABASE_CONFIG", DEFAULT_DATABASE_CONFIG)
RAG_SEARCH_CONFIG = settings.get("APP_RAG_SEARCH_CONFIG", DEFAULT_RAG_SEARCH_CONFIG)

# --- Load CLI Config and Initialize Databases on module import ---
# The `settings` global variable is now the result of the unified load_settings()
logger.debug(
    "CRITICAL DEBUG: Database initialization is now lazy - will initialize on first access"
)

# Databases will be initialized lazily on first access
# This significantly improves startup time by deferring expensive DB operations

# Make APP_CONFIG available globally if needed by other modules that import from config.py
# This will be the same as `settings` if `load_settings` is the sole config loader.
APP_CONFIG_GLOBAL = settings

#
# End of tldw_cli/config.py
#######################################################################################################################


def create_mcp_credential_service(data_root: Path | None = None):
    """Build the MCP-only keyring owner for the active data root.

    No arbitrary HTTP header values enter TOML or provider-account namespaces.
    Unsupported/insecure keyring backends fail closed on first credential use.
    """
    from tldw_chatbook.MCP.credential_bindings import (
        CredentialBindingService,
        KeyringCredentialBackend,
    )

    return CredentialBindingService(
        KeyringCredentialBackend(data_root or get_user_data_dir())
    )


# Definition-time originals for the finite stock run-log availability read.
_RUN_LOG_PROBE_SOURCE = (
    globals(),
    __file__,
    __spec__,
    getattr(__spec__, "origin", None),
    (
        *(
            (globals(), name, globals()[name])
            for name in (
                "get_workspaces_db_path",
                "_get_custom_database_path",
                "get_user_data_dir",
                "get_cli_setting",
            )
        ),
    ),
    tuple(
        (
            function,
            function.__code__,
            function.__globals__,
            function.__defaults__,
            function.__kwdefaults__,
            tuple((function.__kwdefaults__ or {}).items()),
            function.__closure__,
            tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
            vars(function).get("__wrapped__"),
        )
        for _owner, _name, descriptor in (
            *(
                (globals(), name, globals()[name])
                for name in (
                    "get_workspaces_db_path",
                        "_get_custom_database_path",
                    "get_user_data_dir",
                    "get_cli_setting",
                )
            ),
        )
        if callable(descriptor) or isinstance(descriptor, (staticmethod, classmethod))
        for outer in (
            descriptor.__func__
            if isinstance(descriptor, (staticmethod, classmethod))
            else descriptor,
        )
        if hasattr(outer, "__code__")
        for function in (
            outer,
            *((outer.__wrapped__,) if hasattr(outer, "__wrapped__") else ()),
            *(
                (outer.__wrapped__.__wrapped__,)
                if hasattr(outer, "__wrapped__")
                and hasattr(outer.__wrapped__, "__wrapped__")
                else ()
            ),
        )
    ),
)


# Defining callbacks for checked stock Console preparation; no IO.
_COMPACT_MODEL_CONFIG_SOURCE = (
    globals(),
    __file__,
    __spec__,
    getattr(__spec__, "origin", None),
    tuple(
        (globals(), name, globals()[name])
        for name in (
            "get_cli_providers_and_models",
            "load_settings",
            "_load_settings_guarded",
            "_load_settings_uncached",
            "_settings_cache_hit",
            "_get_effective_config_path",
            "current_config_identity",
            "resolve_provider_name",
            "_normalize_provider_lookup_key",
            "normalize_provider_config_key",
            "_config_participants",
        )
    )
    + tuple(
        (vars(_config_participants), name, vars(_config_participants)[name])
        for name in ("operation",)
    )
    + ((sys.modules, _config_participants.__name__, _config_participants),),
    tuple(
        (
            namespace,
            name,
            function,
            function.__code__,
            function.__globals__,
            function.__defaults__,
            function.__kwdefaults__,
            tuple((function.__kwdefaults__ or {}).items()),
            function.__closure__,
            tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
            vars(function).get("__wrapped__"),
        )
        for namespace, name, function in (
            *(
                (globals(), name, globals()[name])
                for name in (
                    "get_cli_providers_and_models",
                    "load_settings",
                    "_load_settings_guarded",
                    "_load_settings_uncached",
                    "_settings_cache_hit",
                    "_get_effective_config_path",
                    "current_config_identity",
                    "resolve_provider_name",
                    "_normalize_provider_lookup_key",
                    "normalize_provider_config_key",
                )
            ),
            *(
                (vars(function), "__wrapped__", function.__wrapped__)
                for function in (_load_settings_guarded, _load_settings_uncached)
            ),
            (vars(_config_participants), "operation", _config_participants.operation),
            (
                vars(_config_participants.operation),
                "__wrapped__",
                _config_participants.operation.__wrapped__,
            ),
        )
    ),
)


# Definition-time inputs for optional in-memory pending display.
_CONSOLE_PENDING_FACTS_IDENTITY_SOURCE = (
    current_config_identity,
    current_config_identity.__code__,
    current_config_identity.__globals__,
    current_config_identity.__defaults__,
    current_config_identity.__kwdefaults__,
    current_config_identity.__closure__,
)



# Original callbacks eligible for finite deferred Collections setup only.
def _record_collections_setup_source(entries):
    from types import FunctionType

    rows = []
    for owner, name in entries:
        original = owner[name] if type(owner) is dict else getattr(owner, name)  # exact stock compatibility boundary
        function = getattr(original, "__func__", original)
        records = []
        while type(function) is FunctionType:
            records.append(
                (
                    function,
                    function.__code__,
                    function.__globals__,
                    function.__defaults__,
                    function.__kwdefaults__,
                    tuple((function.__kwdefaults__ or {}).items()),
                    function.__closure__,
                    tuple(
                        (cell, cell.cell_contents)
                        for cell in function.__closure__ or ()
                    ),
                    vars(function).get("__wrapped__"),
                )
            )
            function = vars(function).get("__wrapped__")
        rows.append((owner, name, original, tuple(records)))
    return globals(), tuple(rows)


_COLLECTIONS_SETUP_SOURCE = _record_collections_setup_source(
    (
        (globals(), "current_config_identity"),
        (globals(), "get_library_collections_db_path"),
        (globals(), "get_user_data_dir"),
        (globals(), "_get_custom_database_path"),
        (globals(), "_get_effective_config_path"),
    )
)
del _record_collections_setup_source


# --- Moved policy (TASK-36945.2): old import paths resolve lazily ----------
# Canvas policy lives in Canvas/config_policy.py and Console capture/trace
# policy in Chat/console_capture_policy.py. Both import this module, so the
# re-export is lazy; it also keeps this module from loading Chat at import.
_MOVED_POLICY_MODULES = {
    "tldw_chatbook.Canvas.config_policy": (
        "CanvasRemoteAccessStatus",
        "CanvasConfigPolicy",
        "_CANVAS_CONFIG_KEYS",
        "_CANVAS_ENABLED_ENV",
        "_CANVAS_AUTO_OPEN_ENV",
        "_strict_canvas_bool",
        "_resolve_canvas_bool",
        "_normalize_canvas_execution",
        "_summarize_canvas_web_auth_policy",
        "_canvas_remote_access_status",
        "build_canvas_config_policy",
        "get_canvas_config_policy",
        "get_canvas_execution_enabled",
    ),
    "tldw_chatbook.Chat.console_capture_policy": (
        "RuntimeCapturePolicy",
        "TraceRolloutSettings",
        "resolve_trace_rollout_settings",
        "resolve_trace_compaction_policy",
        "runtime_capture_policy",
        "apply_console_capture_settings",
        "_publish_runtime_capture_policy",
        "_register_runtime_custom_pii_ruleset",
        "_trace_rollout_environment",
        "_trace_rollout_environment_mapping",
        "_TRACE_ROLLOUT_ENV_NAMES",
        "_TRACE_ROLLOUT_BOOLEAN_ADAPTER",
    ),
}
_MOVED_POLICY_NAMES = {
    name: module for module, names in _MOVED_POLICY_MODULES.items() for name in names
}


def __getattr__(name: str) -> Any:
    module_name = _MOVED_POLICY_NAMES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
