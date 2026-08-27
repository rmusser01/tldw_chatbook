# tldw_cli/config.py
# Description: Configuration management for the tldw_cli application.
#
from __future__ import annotations

# Imports
import copy
from contextlib import ExitStack, contextmanager
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib
import os
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
    TYPE_CHECKING,
    Iterator,
)

#
# Third-Party Imports
from loguru import logger


#
# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail
from tldw_chatbook.Utils.adaptive_reader_state import (
    normalize_adaptive_reader_preferences,
)
from tldw_chatbook.Utils.console_background_effects import (
    normalize_console_background_effects,
)
from tldw_chatbook.Utils.path_validation import validate_path_simple
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
from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryMigrationSeed
#
#######################################################################################################################
#
# Functions:

logger.debug("CRITICAL DEBUG: config.py module is being imported/executed NOW.")
# --- Constants ---
# Client ID used by the Server API itself when writing to sync logs
SERVER_CLIENT_ID = "SERVER_API_V1"
# Client ID for the CLI application instance for its local databases
CLI_APP_CLIENT_ID = "tldw_cli_local_instance_v1"

# --- Path to the CLI's configuration file ---
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "tldw_cli" / "config.toml"


def _get_effective_config_path() -> Path:
    """Return the lexical active CLI config path."""
    override = os.environ.get("TLDW_CONFIG_PATH")
    candidate = Path(override).expanduser() if override else DEFAULT_CONFIG_PATH
    return lexical_path(candidate)


def get_cli_config_path() -> Path:
    """Return the effective config path, including any environment override."""

    return _get_effective_config_path()


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
_ENCRYPTION_PASSWORD = None  # Cached password for the session
_ENCRYPTION_MODULE = None  # Lazily loaded encryption module
_CONFIG_GENERATION = 0

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
    # UI Configuration for all media types
    "ui_style": "default",  # Options: "default", "redesigned", "new", "grid", "wizard", "split"
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
    _ENCRYPTION_PASSWORD = password
    _SETTINGS_CACHE = None
    _CONFIG_CACHE = None
    logger.info("Encryption password set for current session")


def get_encryption_password() -> Optional[str]:
    """Get the encryption password for the current session."""
    return _ENCRYPTION_PASSWORD


def clear_encryption_password():
    """Clear the encryption password from memory."""
    global _ENCRYPTION_PASSWORD
    _ENCRYPTION_PASSWORD = None
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


def resolve_provider_api_key(value: object) -> Optional[str]:
    """Return `value` stripped, or `None` if it is not a usable provider API
    key (not a string, blank, or one of `PROVIDER_API_KEY_PLACEHOLDERS`)."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped in PROVIDER_API_KEY_PLACEHOLDERS:
        return None
    return stripped or None


def is_valid_provider_api_key(value: object) -> bool:
    """Return whether `value` is a usable provider API key."""
    return resolve_provider_api_key(value) is not None


def normalize_provider_config_key(provider: object) -> str:
    """Return the canonical lookup form used for provider config tables.

    Args:
        provider: Provider label or key to normalize.

    Returns:
        A stripped, lowercase provider key with spaces and hyphens replaced by
        underscores.
    """
    normalized = str(provider or "").strip().lower().replace(" ", "_").replace("-", "_")
    return "zai" if normalized == "z.ai" else normalized


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
    correct), google's legacy dict is `google_generative_api` which the
    handler no longer reads, and `chat_with_google` has no env fallback of
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
    modern_value = resolve_provider_api_key(
        provider_table.get("api_key") if isinstance(provider_table, dict) else None
    )
    if modern_value:
        return modern_value  # explicit modern config always wins

    env_value = resolve_provider_api_key(os.getenv(env_var))
    if env_value:
        return env_value

    legacy_value = resolve_provider_api_key(raw_api_section.get(toml_key))
    if legacy_value:
        if not isinstance(provider_table, dict):
            provider_table = {}
            api_settings_section[provider_key] = provider_table
        provider_table["api_key"] = legacy_value
        return legacy_value

    return None


def load_settings(
    force_reload: bool = False,
    *,
    reload_bootstrap: bool | None = None,
) -> Dict:
    """Return the merged application settings, rebuilding at most once.

    Thin wrapper over :func:`_load_settings_uncached` that serializes the
    cache-miss rebuild (task-3503). The cache-hit path is unchanged: one
    short lock, no rebuild lock taken at all.

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
    global _SETTINGS_CACHE_LOCK

    if _SETTINGS_CACHE_LOCK is None:
        import threading

        _SETTINGS_CACHE_LOCK = threading.Lock()

    active_config_path = _get_effective_config_path()

    def _cache_hit():
        with _SETTINGS_CACHE_LOCK:
            if (
                _SETTINGS_CACHE is not None
                and _SETTINGS_CACHE_SOURCE == active_config_path
            ):
                return _SETTINGS_CACHE
        return None

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
    active_config_path = _get_effective_config_path()

    # Initialize lock on first use to avoid import issues
    if _SETTINGS_CACHE_LOCK is None:
        import threading

        _SETTINGS_CACHE_LOCK = threading.Lock()

    # Thread-safe cache check
    with _SETTINGS_CACHE_LOCK:
        if (
            _SETTINGS_CACHE is not None
            and _SETTINGS_CACHE_SOURCE == active_config_path
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
    final_console_settings_cli["status_chips_collapsed"] = coerce_bool_setting(
        final_console_settings_cli.get("status_chips_collapsed", False),
        False,
    )
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
    final_mcp_settings_cli["expose_local_tools"] = coerce_bool_setting(
        final_mcp_settings_cli.get("expose_local_tools", False),
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
    _bridged_provider_api_keys = {
        provider_key: _normalize_legacy_provider_api_key(
            final_api_settings,
            provider_key,
            raw_api_section=api_section_legacy,
            toml_key=toml_key,
            env_var=env_var,
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
    google_api_key = _bridged_provider_api_keys["google"]
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
        "conversations_reader",
        "notes_reader",
        "prompts_reader",
        "skills_reader",
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
        "providers": final_providers_settings,  # For UI dropdowns
        "model_catalog": final_model_catalog_settings_cli,
        "chat_defaults": final_chat_defaults_cli,
        "character_defaults": final_character_defaults_cli,
        "appearance": copy.deepcopy(toml_config_data.get("appearance", {})),
        "notes": final_notes_settings_cli,  # For notes auto-save settings
        "chunking": final_chunking_settings_cli,  # Template default for ingest (§9.1)
        "console": final_console_settings_cli,  # For Console behavior settings
        "first_run": final_first_run_settings_cli,  # Wizard setup_started/setup_completed flags
        "image_generation": final_image_generation_settings_cli,  # For Image_Generation/config.py loader
        "video_generation": final_video_generation_settings_cli,  # For Video_Generation/config.py loader
        "mcp": final_mcp_settings_cli,  # For MCP server settings
        "persona_buddy": copy.deepcopy(toml_config_data.get("persona_buddy", {})),
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
        "google_generative_api": {  # Renamed to avoid confusion with Google Search API
            "api_key": google_api_key,
            "model": api_section_legacy.get("google_model", "gemini-2.5-flash"),
            "streaming": api_section_legacy.get("google_streaming", False),
            "temperature": api_section_legacy.get("google_temperature", 0.7),
            "top_p": api_section_legacy.get("google_top_p", 0.95),
            "min_p": api_section_legacy.get(
                "google_min_p", 0.05
            ),  # Check if 'min_p' is valid for Gemini
            "max_tokens": api_section_legacy.get("google_max_tokens", 4096),
            "api_timeout": api_section_legacy.get("google_api_timeout", 90),
            "api_retries": api_section_legacy.get("google_api_retry", 3),
            "api_retry_delay": api_section_legacy.get("google_api_retry_delay", 5),
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
            "temperature": api_section_legacy.get("kobold_temperature", 0.7),
            "top_p": api_section_legacy.get("kobold_top_p", 0.95),
            "top_k": api_section_legacy.get("kobold_top_k", 100),
            "max_tokens": api_section_legacy.get("kobold_max_tokens", 4096),
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
                search_settings_section, "search_provider_default", "google"
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
                search_engines_section, "search_engine_searx_api", ""
            ),
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

    chat_dicts_folder = get_user_data_dir() / "chat_dicts"
    config_dict["chat_dictionaries"]["chat_dicts_folder"] = str(chat_dicts_folder)

    # Create the chat dictionaries folder if it doesn't exist
    try:
        chat_dicts_folder.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured chat dictionaries folder exists: {chat_dicts_folder}")
    except Exception as e:
        logger.error(
            f"Could not create chat dictionaries folder {chat_dicts_folder}: {e}"
        )

    if bootstrap.succeeded:
        with _SETTINGS_CACHE_LOCK:
            _SETTINGS_CACHE = config_dict
            _SETTINGS_CACHE_SOURCE = active_config_path
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
LOCAL_PROVIDERS = {
    "llama_cpp": ["None"],
    "Oobabooga": ["None"],
    "koboldcpp": ["None"],
    "Ollama": [
        "gemma3:12b",
        "gemma3:4b",
        "gemma3:27b",
        "qwen3:4b",
        "qwen3:8b",
        "qwen3:14b",
        "qwen3:30b",
        "qwen3:32b",
        "qwen3:235b",
        "devstral:24b",
        "deepseek-r1:671b",
    ],
    "vLLM": ["vllm-model-z", "vllm-model-x", "vllm-model-y", "vllm-model-a"],
    "TabbyAPI": ["tabby-model", "tabby-model-2", "tabby-model-3"],
    "Aphrodite": ["aphrodite-engine", "aphrodite-engine-2"],
    "Custom": ["custom-model-alpha", "custom-model-beta"],
    "Custom-2": ["custom-model-gamma", "custom-model-delta"],
}

#######################################################################################################################
# --- CLI User Configuration Section ---
#######################################################################################################################

# --- Configuration File Content (for reference or auto-creation for the CLI) ---
CONFIG_TOML_CONTENT = """
# Configuration for tldw-chatbook TUI App
# Located at: ~/.config/tldw_cli/config.toml
[general]
default_tab = "chat"  # "chat", "character", "logs", "media", "search", "ingest", "stats"
focus_mode = false  # Start the Console chrome-free (no nav bar / workbench header; one-line status bar kept)
default_theme = "textual-dark"  # Default theme on startup ("textual-dark", "textual-light", or any theme name from themes.py)
palette_theme_limit = 1  # Maximum number of themes to show in command palette (0 = show all)
log_level = "INFO" # TUI Log Level: DEBUG, INFO, WARNING, ERROR, CRITICAL
users_name = "default_user" # Default user name for the TUI
# How long shutdown may take (Textual unmount + interpreter teardown) before a
# hard exit is forced. Clamped to 1-300 seconds. A quiet exit measures ~0.6s,
# so this deadline is never reached by a normal quit.
# NOTE: it is enforced against healthy work too. A background job still
# running when you quit -- media ingest, notes/character export, library
# export, an embedding batch -- runs on a thread that cannot be interrupted.
# If it is still going 120 seconds after you quit, the process is killed and
# that job's database write is abandoned (not rolled back to a clean earlier
# state -- simply lost). Raise this if you routinely quit while long jobs are
# running; lower it only if you would rather lose such a write than wait.
shutdown_grace_seconds = 120.0

[console]
collapse_large_pastes = true  # Display large pasted chunks compactly in Console composer
show_model_thinking = true  # Presentation only; capture and replay are unchanged
thinking_history_policy_default = "auto"  # auto, include, exclude for new conversations
stack_collapsed_rail_labels = false  # Use compact stacked labels on collapsed Console rails
rail_layout_scope = "global"  # Share Console rail disclosure across workspaces; use "workspace" for per-workspace layouts
assistant_library_access_default = false  # New Console sessions block assistant Library access
paste_collapse_threshold = 50  # Collapse pasted/inserted chunks only when longer than this many characters
local_tools_enabled = true      # workspace, web, and Watchlists agent tools; every call still uses MCP Ask/Allow/Off permissions
# Root-source byte limit; allowed range is 1-1048576 (1 MiB).
project_instructions_startup_max_bytes = 32768
# Cumulative nested-source byte limit per dispatch; allowed range is 1-1048576 (1 MiB).
project_instructions_nested_max_bytes = 32768
# Conversation-memory defaults (ADR-052). Model capacity remains capability data,
# not a persisted policy value.
conversation_budget_mode = "automatic"  # automatic, custom
# conversation_budget_tokens = 32000     # required only when mode = custom
compaction_mode = "ask"                  # ask, automatic, off
compaction_representation = "text_summary"  # text_summary, visual_transcript, hybrid
compaction_trigger_ratio = 0.80
compaction_target_ratio = 0.55
compaction_summary_max_tokens = 1024
compaction_failure_behavior = "stop_and_ask"  # stop_and_ask, omit_older_context
compaction_carry_forward_mode = "memory_with_recent_turns"  # memory_with_recent_turns, memory_with_latest_exchange
# Confinement root for the fs_*/git_* agent tools (ADR-032). Empty = the app's
# cwd at startup, so the boundary MOVES with where you launch the app: start it
# from your home directory and every personal file under it is inside the
# agent's reach. Credential, gate-state and app-state paths (~/.ssh, ~/.aws,
# this file, mcp_permissions.json, the app's databases) are refused regardless
# of this setting -- see Utils/sensitive_paths.py, enforced for these tools in
# Tools/local_tool_impls.py's resolve_workspace_path (TASK-19551) -- but that
# denylist is a guardrail, not a substitute for pointing this at the one
# project directory you actually want an agent working in.
# workspace_root = ""

# Agent run budget (Settings > Console Behavior > Agent run budget).
# Applies to ONE run = one user message; sub-agents inherit turns/steps/tokens
# unchanged (their wall comes from [agents] child_max_wall_seconds instead).
# agent_max_total_tokens is the limit that actually stops a long run: the whole
# conversation is re-sent every turn, so spend grows quadratically and 25M is
# typically reached around turn 250. The turn and step caps are backstops.
agent_max_model_turns = 2000        # Provider turns (tool-calling rounds) per message
agent_max_steps = 25000             # Step backstop; a fence tool round costs 3 steps
agent_max_wall_seconds = 86400.0    # 24h ceiling on one run; Stop always works
agent_max_total_tokens = 25000000   # Per-run prompt+completion spend ceiling; 0 = unlimited
agent_max_tool_call_seconds = 3600.0  # Ceiling on ONE tool call; 0 = unlimited
# Ephemeral side chat (selection menu) — empty model = session model
sidechat_model = ""  # e.g. "openai/gpt-5-mini"; empty = follow the current session's model
sidechat_prompt_template = "Give me more details about: {selection}"  # {selection} = the quoted text
# Conversation Inspector: capture each provider exchange (request/response)
# locally per turn. Local-only; never synced. Set false to disable.
exchange_capture = true
# Safe is the application default; no UI exposes Full activation.
exchange_capture_detail = "safe"

[console.background_effects]
enabled = false  # Optional Console ambience. Off by default for readability.
effect = "none"  # none, snow, rain, matrix
scope = "transcript"  # transcript, workbench
intensity = "low"  # low, medium, high
fps = 6  # 1-12

[skills]
# project_skills_prompt_enabled = true  # offer .SKILLS/ import at startup; spec 2026-08-17

[appearance]
density = "normal"  # compact, normal, or comfortable default control density
animations_enabled = true  # Enable optional UI animations where supported
smooth_scrolling = true  # Enable smooth scrolling where supported
reduce_motion = false  # Render animations as static frames (splash screen, Console setup backdrop)
ascii_glyphs = false  # Substitute ASCII status markers for unicode glyphs (narrow-font terminals)
console_transcript_style = "role_accents"  # neutral, role_accents, or immersive_rp

[acp.runtime]
# ACP owns runtime launch/setup. Leave command empty to keep ACP honestly blocked.
command = ""
args = []
cwd = ""
runtime_id = "local-acp-runtime"
runtime_label = "Local ACP Runtime"
runtime_version = ""
startup_timeout_seconds = 2.0

[tldw_api]
base_url = "http://127.0.0.1:8000" # Or your actual default remote endpoint
# Default auth token can be stored here, or leave empty if user must always provide
auth_token = "default-secret-key-for-single-user"

[library]
# Maximum files scanned when analysing a directory for Library ingestion.
ingest_directory_scan_limit = 1000
# Check a staged ingest URL by fetching its headers before the import runs.
# OFF by default (TASK-19556). A link check is not free: it contacts the host
# before you have asked for anything to be imported, and it used to run from
# the ingest field's typing debounce, which turned a pasted link into a probe
# of whatever the address pointed at -- including hosts on your own network.
# With this on, the check runs only from the deliberate triggers (leaving the
# field, pressing Enter, Browse..., the retry button), never while you type;
# it is routed through the [web_security] egress policy, follows no
# redirects, and reports one identical "could not be checked" note for every
# address the policy declines. A link that cannot actually be fetched is
# still reported by the import job itself, with a real reason.
ingest_url_preflight_probe = false
# Parallel ingest parse workers. Default: min(3, cpu-1). Uncomment to override.
# ingest_parse_workers = 3
# Max concurrent heavy (audio/video transcription) parses; document parses fan
# out past this cap to fill the remaining pool workers. Default: 1.
# ingest_heavy_lane_max_workers = 1

[library.reader]
# Shared Library-pane visibility and width are written here by Settings and
# Library pane toggles. Older configs may omit these keys; each missing key
# independently falls back to its legacy value under [library.media_reader].
# Environment overrides use TLDW_LIBRARY_READER_<KEY>.

[library.media_reader]
# Destination Items-pane preferences. The three shared keys below remain as
# read-only compatibility fallbacks for older config files.
# Items environment overrides use TLDW_LIBRARY_MEDIA_READER_<KEY>.
library_open = true
items_open = true
custom_widths_enabled = false
# Compatibility fallback for fresh profiles; matches LIBRARY_REFERENCE_WIDTH.
library_width = 31
items_width = 40

[library.conversations_reader]
# Environment overrides use TLDW_LIBRARY_CONVERSATIONS_READER_<KEY>.
items_open = true
items_width = 40

[library.notes_reader]
# Environment overrides use TLDW_LIBRARY_NOTES_READER_<KEY>.
items_open = true
items_width = 40

[library.prompts_reader]
# Environment overrides use TLDW_LIBRARY_PROMPTS_READER_<KEY>.
items_open = true
items_width = 40

[library.skills_reader]
# Environment overrides use TLDW_LIBRARY_SKILLS_READER_<KEY>.
items_open = true
items_width = 40

# Per-type ingestion options are persisted here by the Library ingest canvas.
[library.ingest_options]

[caching]
# Anthropic prompt caching (cache_control breakpoints on the system prompt and
# the tool list for every Anthropic call; plus one on the latest message for
# Console sends only, which are the multi-turn ones). Cache writes bill at
# 1.25x input and reads at ~0.1x, so multi-turn chat wins after two sends
# inside the 5-minute TTL.
# The loss condition: sends more than ~5 minutes apart never hit a live cache,
# so every send re-pays the 1.25x write premium on the conversation prefix
# with no reads at all -- the cost ticker will show it (writes in the
# breakdown, no cache-read line). Set false if that is your usage pattern.
# Set false to disable the cache_control breakpoints this client adds;
# caller-supplied native Anthropic tool dicts already carrying cache_control
# still pass through verbatim.
anthropic_enabled = true

[agents]
# Sub-agent fleet knobs. Every key here is COMMENTED OUT on purpose: the
# authoritative defaults live in `Agents/agent_service.py`
# (DEFAULT_MAX_LIVE_SUBAGENTS / DEFAULT_CHILD_MAX_WALL_SECONDS /
# DEFAULT_SUBAGENTS_OUTLIVE_TURN / DEFAULT_AUTOWAKE_ENABLED), and shipping a live copy here would give
# the same value two homes that can silently drift apart. Uncomment a line
# to override. See Docs/User_Guide/console/agent-runs-and-tools.md.
#
# How many sub-agents of ONE conversation may run at once, counting any
# still working from an earlier message. 1 disables the fleet (sub-agents
# run inline, one at a time). The cap is per conversation AND per running
# app -- N conversations can hold N * this between them.
# max_live_subagents = 3
#
# Whether a sub-agent may keep working after the reply that spawned it has
# finished. false settles every sub-agent at the end of its own turn.
# subagents_outlive_turn = true
#
# How long ONE background sub-agent may keep working, in seconds. Checked
# between the child's own steps, so it does not interrupt a provider call
# already in flight.
# child_max_wall_seconds = 1800.0
#
# Whether a background sub-agent finishing after its turn WAKES its
# supervisor so it can act on the result (an injected, clearly machine-
# marked notice -- never user input, never approval). false still records
# every completion (toast, badge, durable mark); the wake turn just never
# fires.
# autowake_enabled = true

[splash_screen]
# Splash screen configuration for startup animations
# See Docs/Examples/SPLASH_SCREENS_CATALOG.md for all available splash screens
enabled = true  # Enable/disable splash screen
duration = 1.5  # Duration in seconds to display splash screen
skip_on_keypress = true  # Allow users to skip with any keypress

# Card selection mode:
# - "random": Randomly selects from active_cards list (default)
# - "sequential": Cycles through active_cards in order (not yet implemented)
# - "<card_name>": Always use a specific card (e.g., "matrix", "glitch", etc.)
card_selection = "random"

show_progress = true  # Show initialization progress bar

# List of splash cards to use when card_selection is "random"
# Full catalog of 40+ cards available - see documentation for descriptions
# Categories:
#   Static: default, classic, compact, minimal, blueprint
#   Classic Animated: matrix, glitch, retro, typewriter
#   Visual Effects: tech_pulse, code_scroll, arcade_high_score, digital_rain, loading_bar, starfield
#   Interactive: terminal_boot, glitch_reveal, ascii_morph, game_of_life, scrolling_credits, spotlight_reveal
#   Creative: sound_bars, raindrops_pond, pixel_zoom, text_explosion, old_film, maze_generator, dwarf_fortress
#   Tech-Themed: neural_network, quantum_particles, ascii_wave, binary_matrix, constellation_map, circuit_trace
#   More: typewriter_news, dna_sequence, plasma_field, ascii_fire, rubiks_cube, data_stream, fractal_zoom, 
#         ascii_spinner, hacker_terminal
active_cards = [
    "default", "matrix", "glitch", "retro", "classic", "compact", "minimal",
    "tech_pulse", "code_scroll", "minimal_fade", "blueprint", "arcade_high_score",
    "digital_rain", "loading_bar", "starfield", "terminal_boot", "glitch_reveal",
    "ascii_morph", "game_of_life", "scrolling_credits", "spotlight_reveal", "sound_bars",
    "spy_vs_spy", "phonebooths", "emoji_face",
    "ascii_aquarium", "bookshelf_browser", "train_journey", "clock_mechanism",
    "weather_system", "music_visualizer", "origami_folding", "ant_colony",
    "neon_sign_flicker", "zen_garden"
    # Note: "custom_image" is not included by default as it requires user configuration
]

[splash_screen.effects]
# Animation effect settings
fade_in_duration = 0.3  # Fade in time in seconds
fade_out_duration = 0.2  # Fade out time in seconds
animation_speed = 1.0  # Animation playback speed multiplier

custom_image_path = ""  # Set path to your image file for custom_image splash screen

[logging]
# Log file will be placed in the same directory as the chachanotes_db_path below.
log_filename = "tldw_cli_app.log"
file_log_level = "INFO" # File Log Level: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_max_bytes = 10485760 # 10 MB
log_backup_count = 5

[database]
# scheduled_tasks_db_path = "/custom/path.db"  # optional override
# tts_profiles_db_path = "/custom/path.db"  # optional override
# Path to the ChaChaNotes (Character, Chat, Notes) database.
chachanotes_db_path = "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
# Path to the Prompts database.
prompts_db_path = "~/.local/share/tldw_cli/tldw_cli_prompts.db"
# Path to the Media V2 database.
media_db_path = "~/.local/share/tldw_cli/tldw_cli_media_v2.db"
# Path to the local research sessions/runs database.
research_db_path = "~/.local/share/tldw_cli/tldw_chatbook_research.db"
# Path to the local writing suite database.
writing_db_path = "~/.local/share/tldw_cli/tldw_chatbook_writing.db"
# Path to the local Library Collections database.
library_collections_db_path = "~/.local/share/tldw_cli/tldw_chatbook_library_collections.db"
# Path to the local Workspaces database.
workspaces_db_path = "~/.local/share/tldw_cli/tldw_chatbook_workspaces.db"
# Path to the Evals database.
evals_db_path = "~/.local/share/tldw_cli/evals.db"
# Path to the RAG indexing-state database (rag_indexing.db; tracks
# incremental RAG indexing state -- it is not a vector store).
rag_indexing_db_path = "~/.local/share/tldw_cli/rag_indexing.db"
# Path to the Subscriptions database.
subscriptions_db_path = "~/.local/share/tldw_cli/tldw_chatbook_subscriptions.db"
USER_DB_BASE_DIR = "~/.local/share/tldw_cli/"

# Database integrity checking
check_integrity_on_startup = false  # Enable/disable automatic integrity checks on startup
integrity_check_timeout = 30  # Maximum seconds to wait for integrity check

[scheduling]
# Background sync and scheduler defaults for the scheduling module.
sync_interval_seconds = 300
sync_retry_max_attempts = 10
sync_retry_max_delay_seconds = 300
sync_retry_jitter = true
scheduler_poll_interval_seconds = 30
# A dispatch more than this many seconds after its scheduled time counts as
# late and is recorded on the task (missed_at/missed_count, task-18937).
# Default is 2x the poll interval: an idle scheduler dispatches within one
# poll. Beyond 2x has several possible causes and the row cannot tell them
# apart (task-19562): the scheduler was not running at the scheduled time
# (app closed), the machine slept or the loop was starved while the app
# stayed open, or the scheduler was busy -- tick awaits every due handler
# serially, so a slow handler holds the loop and pushes the NEXT poll past
# the grace. The loop logs which it was and counts it as
# scheduler_dispatch_late with a cause label (away / stalled / busy). A
# task created mid-session reloads the queue immediately, so that case does
# not false-positive here.
missed_fire_grace_seconds = 60
# Handler execution timeout (task-18939): a scheduled-task handler still
# running after this many seconds is cancelled and its dispatch records
# "timed_out" -- the schedule advances, so a wedged handler (e.g. a hung
# watchlist URL check) can never wedge the whole scheduler. 0 or negative
# disables the bound; a reminder's own timeout_seconds column overrides
# this per task (same semantics: <=0 disables for that task).
handler_timeout_seconds = 300
reminder_catchup_hours = 24
# Feature flags for the watchlist-to-unified-scheduler migration (ADR-019).
# Both were staged for a shadow-mode dual-run against the legacy SubscriptionScheduler.
# That scheduler is gone, so shadow mode has nothing to compare against and leaving
# these at their staging values meant nothing ever checked a watchlist (TASK-1210).
watchlist_checks_enabled = true   # Run watchlist checks on their configured cadence
# Diagnostics only: fetch but DISCARD results, ignoring cadence. Shadow mode probes
# feed and url/url_list sources directly; it CANNOT probe sitemap or api sources,
# which need the execution path it exists to avoid, and reports those as
# "shadow_unsupported" rather than pretending they were checked (TASK-1383).
watchlist_checks_shadow = false
# Run each watchlist's own briefing on its configured cadence
# (`watchlists.briefing_cadence_seconds`, opt-in per watchlist, NULL/unset
# means never -- briefings phase 4, Locked Decision 4). Only fires while
# the app is open; a schedule spends the user's own LLM tokens unattended,
# so this is the one flag that turns that on at all.
briefing_schedules_enabled = true

[media_cleanup]
# Media cleanup settings for automatic hard deletion of soft-deleted items
enabled = true  # Enable/disable automatic cleanup
cleanup_days = 30  # Number of days after soft deletion before hard deletion
cleanup_interval_hours = 24  # How often to run cleanup (in hours)
cleanup_on_startup = true  # Run cleanup check on application startup
max_items_per_cleanup = 100  # Maximum items to delete in one cleanup run
notify_before_cleanup = true  # Show notification before performing cleanup

[api_endpoints]
# Optional: Specify URLs for local/custom endpoints if they differ from library defaults
# These keys should match the provider names used in the app (adjust if needed)
llama_cpp = "http://localhost:8080" # Check if your API provider uses this address
koboldcpp = "http://localhost:5001/api" # Check if your API provider uses this address
Oobabooga = "http://localhost:5000/api" # Check if your API provider uses this address
Ollama = "http://localhost:11434"
vLLM = "http://localhost:8000" # Check if your API provider uses this address
Custom = "http://localhost:1234/v1"
Custom_2 = "http://localhost:5678/v1"
Custom_3 = "http://localhost:5678/v1"
Custom_4 = "http://localhost:5678/v1"
Custom_5 = "http://localhost:5678/v1"
Custom_6 = "http://localhost:5678/v1"

# Add other local URLs if needed

[providers]
# This section primarily lists providers and their *available* models for the UI dropdown.
# Actual default model/settings used for calls are defined in [api_settings.*] or [chat_defaults]/[character_defaults].
OpenAI = ["gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "o3-2025-04-16", "o3-mini-2025-01-31", "o1-2024-12-17", "chatgpt-4o-latest", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14", "gpt-4o-mini-2024-07-18", ]
Anthropic = ["claude-sonnet-5", "claude-opus-5", "claude-fable-5", "claude-haiku-4-5", "claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1", "claude-2.0"]
Cohere = ["command-a-03-2025", "command-r7b-12-2024", "command-r-plus-04-2024", "command-r-plus", "command-r-08-2024", "command-r-03-2024", "command", "command-nightly", "command-light", "command-light-nightly"]
DeepSeek = ["deepseek-v4-flash", "deepseek-v4-pro"]
Groq = ["gemma2-9b-it", "mmeta-llama/Llama-Guard-4-12B", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-70b-8192", "llama3-8b-8192",]
Google = ["gemini-2.5-flash", "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-05-06", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", ]
HuggingFace = ["openai/gpt-oss-120b", "meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct",]
MistralAI = ["open-mistral-nemo", "mistral-medium-2505", "codestral-2501", "mistral-saba-2502", "mistral-large-2411", "ministral-3b-2410", "ministral-8b-2410", "mistral-moderation-2411", "devstral-small-2505", "mistral-small-2503", ]
Moonshot = ["kimi-k3", "kimi-latest", "kimi-thinking-preview", "moonshot-v1-auto", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k", "moonshot-v1-8k-vision-preview", "moonshot-v1-32k-vision-preview", "moonshot-v1-128k-vision-preview", "kimi-k2-0711-preview"]
OpenRouter = ["openai/gpt-4o-mini", "anthropic/claude-3.7-sonnet", "google/gemini-2.0-flash-001", "google/gemini-2.5-pro-preview", "google/gemini-2.5-flash-preview", "deepseek/deepseek-chat-v3-0324:free", "deepseek/deepseek-chat-v3-0324", "openai/gpt-4.1", "anthropic/claude-sonnet-4", "deepseek/deepseek-r1:free", "anthropic/claude-3.7-sonnet:thinking", "google/gemini-flash-1.5-8b", "mistralai/mistral-nemo", "google/gemini-2.5-flash-preview-05-20", ]
QwenCloud = ["qwen3.8-max"]
ZAI = ["glm-5.2", "glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4.5v", "glm-4-32b-0414-128k"]
# Local Providers
Llama_cpp = ["None"]
koboldcpp = ["None"]
Oobabooga = ["None"]
Ollama = ["gemma3:12b", "gemma3:4b", "gemma3:27b", "qwen3:4b", "qwen3:8b", "qwen3:14b", "qwen3:30b", "qwen3:32b", "qwen3:235b", "devstral:24b", "deepseek-r1:671b"]
vLLM = ["vllm-model-z", "vllm-model-x", "vllm-model-y", "vllm-model-a"]
Custom = ["custom-model-alpha", "custom-model-beta"]
Custom_2 = ["custom-model-gamma", "custom-model-delta"]
TabbyAPI = ["tabby-model", "tabby-model-2", "tabby-model-3"]
Aphrodite = ["aphrodite-engine", "aphrodite-engine-2"]
local-llm = ["None"] # Add if you have a specific local-llm provider entry
local_llamacpp = ["None"]
local_llamafile = ["None"]
local_ollama = ["None"]
local_vllm = ["None"]
local_onnx = ["None"]
local_transformers = ["None"]
local_mlx_lm = ["None"]

[model_catalog]
# Automatic model-list refresh for cloud providers (ADR-020).
auto_refresh_enabled = true
# The startup check is confirm-first: nothing is contacted online until the
# user answers the one-time consent dialog (which sets this to true).
refresh_consent_recorded = false
stale_after_hours = 24 # 0 = refetch every launch
auto_refresh_disabled = [] # exact [providers] keys to opt out, e.g. ["ZAI"]
write_to_config = [] # exact [providers] keys whose new models append to this file

[api_settings] # Parent section for all API provider specific settings

    # --- Cloud Providers ---
    [api_settings.openai]
    api_key_env_var = "OPENAI_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "gpt-5.6-terra" # Default model for direct calls (if not overridden)
    temperature = 0.7
    top_p = 1.0 # OpenAI uses top_p (represented as maxp sometimes in UI)
    max_tokens = 4096
    timeout = 60 # seconds
    retries = 3
    retry_delay = 5 # seconds (backoff factor)
    streaming = false

    [api_settings.anthropic]
    api_key_env_var = "ANTHROPIC_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "claude-sonnet-5"
    temperature = 0.7
    top_p = 1.0 # Anthropic uses top_p (represented as topp in UI)
    top_k = 0 # Anthropic specific, 0 or -1 usually disables it
    max_tokens = 4096
    timeout = 90
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.cohere]
    api_key_env_var = "COHERE_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "command-a-03-2025"
    temperature = 0.3
    top_p = 0.75 # Cohere uses 'p' (represented as topp in UI)
    top_k = 0 # Cohere uses 'k'
    max_tokens = 4096 # Cohere uses max_tokens
    timeout = 90
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.deepseek]
    api_key_env_var = "DEEPSEEK_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "deepseek-v4-flash"
    temperature = 0.7
    top_p = 1.0 # Deepseek uses top_p (represented as topp in UI)
    max_tokens = 4096
    timeout = 60
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.groq]
    api_key_env_var = "GROQ_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "llama-3.3-70b-versatile"
    temperature = 0.7
    top_p = 1.0 # Groq uses top_p (represented as maxp in UI)
    max_tokens = 8192
    timeout = 60
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.google]
    api_key_env_var = "GOOGLE_API_KEY"
    api_key = "<API_KEY_HERE>"
    model = "gemini-2.5-flash"
    temperature = 0.7
    top_p = 0.9 # Google uses topP (represented as topp in UI)
    top_k = 100 # Google uses topK
    max_tokens = 8192 # Google uses maxOutputTokens
    timeout = 120
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.huggingface]
    api_key_env_var = "HUGGINGFACE_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "openai/gpt-oss-120b"
    api_base_url = "https://router.huggingface.co/v1"
    api_chat_path = "chat/completions"
    temperature = 0.7
    top_p = 1.0 # HF Inference API uses top_p
    top_k = 50  # HF Inference API uses top_k
    max_tokens = 4096 # HF Inf API uses max_tokens / max_new_tokens
    timeout = 60
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.mistralai] # Matches key in [providers]
    api_key_env_var = "MISTRAL_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "mistral-large-latest"
    temperature = 0.7
    top_p = 1.0 # Mistral uses top_p (represented as topp in UI)
    max_tokens = 4096
    timeout = 60
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.openrouter]
    api_key_env_var = "OPENROUTER_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "meta-llama/Llama-3.1-8B-Instruct"
    temperature = 0.7
    top_p = 1.0 # OpenRouter uses top_p
    top_k = 0   # OpenRouter uses top_k
    min_p = 0.0 # OpenRouter uses min_p
    max_tokens = 4096
    timeout = 120
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.moonshot]
    api_key_env_var = "MOONSHOT_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "kimi-k3"
    temperature = 0.7
    top_p = 0.95 # Moonshot uses top_p (OpenAI compatible)
    max_tokens = 4096
    api_region = "international" # "international" or "china"
    api_base_url = "https://api.moonshot.ai/v1" # Default for international; use https://api.moonshot.cn/v1 for China
    timeout = 90
    retries = 3
    retry_delay = 1.0
    streaming = true

    [api_settings.qwencloud]
    api_mode = "responses"
    api_key_env_var = "DASHSCOPE_API_KEY"
    api_base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    model = "qwen3.8-max"
    timeout = 120
    retries = 3
    retry_delay = 1
    streaming = true

    [api_settings.zai] # Matches key in [providers]
    api_key_env_var = "ZAI_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "glm-5.2"
    temperature = 0.7
    top_p = 0.95
    max_tokens = 4096
    api_base_url = "https://api.z.ai/api/paas/v4"
    timeout = 90
    retries = 3
    retry_delay = 5
    streaming = true

    # --- Local Providers ---
    # Local providers default to streaming = true so slow generations render
    # incrementally instead of timing out on one long blocking completion.
    [api_settings.llama_cpp] # Matches key in [providers]
    api_key_env_var = "LLAMA_CPP_API_KEY" # If you set one on the server
    # api_key = ""
    api_url = "http://localhost:8080" # llama.cpp server root; the OpenAI-compatible /v1/chat/completions path is appended automatically
    model = "" # Often not needed if server serves one model
    temperature = 0.7
    top_p = 0.95
    top_k = 40
    min_p = 0.05
    max_tokens = 4096 # llama.cpp uses n_predict
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.oobabooga] # Matches key in [providers]
    api_key_env_var = "OOBABOOGA_API_KEY" # If API extension needs one
    api_url = "http://localhost:5000/v1/chat/completions" # Ooba OpenAI compatible endpoint
    model = "" # Model loaded in Ooba UI
    temperature = 0.7
    top_p = 0.9
    # top_k = 50 # Check Ooba endpoint docs for OpenAI compatibility params
    # min_p = 0.0
    max_tokens = 4096
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.koboldcpp] # Matches key in [providers]
    # api_key = "" # Kobold doesn't use keys
    api_url = "http://localhost:5001/api/v1/generate" # Kobold non-streaming API
    # api_streaming_url = "http://localhost:5001/api/v1/stream" # Kobold streaming API (different format)
    model = "" # Model loaded in Kobold UI
    temperature = 0.7
    top_p = 0.9
    top_k = 50
    max_tokens = 4096 # Kobold uses max_context_length / max_length
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = true # Kobold streaming is non-standard, handle carefully
    system_prompt = "You are a helpful AI assistant"

    [api_settings.ollama]
    # No API Key usually needed
    api_url = "http://localhost:11434/v1/chat/completions" # Default Ollama OpenAI endpoint
    model = "llama3:latest"
    temperature = 0.7
    top_p = 0.9
    top_k = 40 # Ollama supports top_k via OpenAI endpoint
    # min_p = 0.05 # Ollama OpenAI endpoint doesn't support min_p directly
    max_tokens = 4096
    timeout = 300 # Longer timeout for local models
    retries = 1
    retry_delay = 2
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.vllm] # Matches key in [providers]
    api_key_env_var = "VLLM_API_KEY" # If served behind auth
    api_url = "http://localhost:8000/v1/chat/completions" # vLLM OpenAI compatible endpoint
    model = "" # Model specified when starting vLLM server
    temperature = 0.7
    top_p = 0.95
    top_k = 50
    min_p = 0.05
    max_tokens = 4096
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.aphrodite] # Matches key in [providers]
    api_key_env_var = "APHRODITE_API_KEY" # If served behind auth
    api_url = "http://localhost:2242/v1/chat/completions" # Default Aphrodite port
    model = "aphrodite-engine" # Model loaded in Aphrodite
    temperature = 0.7
    top_p = 0.95
    top_k = 50
    min_p = 0.05
    max_tokens = 4096
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.tabbyapi] # Matches key in [providers]
    api_key_env_var = "TABBYAPI_API_KEY"
    api_url = "http://localhost:8080/v1/chat/completions" # Check TabbyAPI docs for exact URL
    model = "tabby-model" # Model configured in TabbyAPI
    temperature = 0.7
    top_p = 0.95
    top_k = 50
    min_p = 0.05
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 3
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.custom] # Matches key in [providers]
    api_key_env_var = "CUSTOM_API_KEY"
    api_url = "http://localhost:1234/v1/chat/completions"
    model = "custom-model-alpha"
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = false
    system_prompt = "You are a helpful AI assistant"

    [api_settings.custom_2] # Matches key in [providers]
    api_key_env_var = "CUSTOM_2_API_KEY"
    api_url = "http://localhost:5678/v1/chat/completions"
    model = "custom-model-gamma"
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = false
    system_prompt = "You are a helpful AI assistant"

    [api_settings.local-llm] # Matches key in [providers]
    api_url = "http://localhost:8000/v1/chat/completions"
    model = ""
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.local_llamafile] # Matches key in [providers]
    api_url = "http://localhost:8001/v1/chat/completions"
    model = ""
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.local_llamacpp] # Matches key in [providers]
    #api_key_env_var = "local_llamacpp_API_KEY"
    api_url = "http://localhost:8001/v1/chat/completions"
    model = "custom-model-gamma"
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.local_vllm] # Matches key in [providers]
    #api_key_env_var = "local_vllm_API_KEY" # If served behind auth
    api_url = "http://localhost:8008/v1/chat/completions"
    model = "custom-model-gamma"
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.local_ollama] # Matches key in [providers]
    api_key_env_var = "local_ollama_API_KEY" # If served behind auth
    api_url = "http://localhost:5678/v1/chat/completions"
    model = "custom-model-gamma"
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.local_onnx] # Matches key in [providers]
    api_url = "http://localhost:8000/v1/chat/completions"
    model = ""
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.local_transformers] # Matches key in [providers]
    api_url = "http://localhost:8000/v1/chat/completions"
    model = ""
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = true
    system_prompt = "You are a helpful AI assistant"

    [api_settings.local_mlx_lm] # Matches key in [providers]
    api_url = "http://localhost:5678/v1/chat/completions"
    model = "custom-model-gamma"
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = true
    system_prompt = "You are a helpful AI assistant"
    # ... etc ...

[chat_defaults]
# Default settings specifically for the 'Chat' tab
user_display_name = "User"
provider = "OpenAI"
model = "gpt-5.6-terra"
system_prompt = "You are a helpful AI assistant."
temperature = 0.6
top_p = 0.95
min_p = 0.05
top_k = 50
strip_thinking_tags = true

# Console transcript view pruning: when the transcript's virtual height (in
# terminal rows) exceeds prune_high_watermark, the oldest message rows are
# dropped from the view until the remaining height fits under
# prune_low_watermark. Scroll position is preserved, the in-progress
# streaming row is never pruned, and the store keeps the full history
# (pruning is view-only). Set prune_high_watermark <= 0 to disable pruning.
prune_high_watermark = 20000
prune_low_watermark = 12000

# Console transcript load window: opening or switching to a conversation
# mounts only a tail of it, and scrolling to the top of that tail prepends
# more. Both values are FLOORS in terminal rows -- the effective budget is the
# larger of the floor and the mounted viewport (6x for the initial window, 4x
# per scrollback step). Set transcript_window_lines <= 0 to mount the whole
# history at load, as before. With windowing on and sane watermarks the
# window is two-sided (TASK-15777): sustained scroll-back keeps loading older
# history while trimming the newest end back out of the view (never a
# selected message, which pauses the sliding while pinned at that end), so
# the session stays reachable by scrolling at a bounded mounted size, and a
# jump to a far-away message mounts a fresh window around it instead of
# everything between it and the tail.
transcript_window_lines = 144
transcript_scrollback_lines = 96

# Render assistant replies as full markdown (code blocks, tables, lists,
# links) in the Console transcript. Set false to restore the lightweight
# span renderer, which keeps roleplay speech/action flavor colors.
assistant_markdown = true

# Image attachment settings for chat
[chat.images]
enabled = true
show_attach_button = true  # Show/hide the attach file button in chat
# show_character_avatar = true  # show the active character's avatar in the Console left rail
# react_character_expressions = true  # swap the Console character avatar among idle/thinking/speaking/error as it generates a reply (requires per-state images on the character); set false to keep a static avatar
default_render_mode = "auto"  # auto, pixels, regular
max_size_mb = 10.0
auto_resize = true
resize_max_dimension = 2048
save_location = "~/Downloads"
supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg"]

[chat.images.terminal_overrides]
kitty = "regular"
wezterm = "regular"
iterm2 = "regular"
default = "pixels"

[web_security]
# Egress guard (SSRF protection) for web scraping / ingestion / subscription
# fetches. When enabled, content-derived URLs (redirects, sitemap/crawl
# discoveries, feed items) must resolve to public IPs; URLs you explicitly
# configure (feed sources, Confluence base_url, ingest URLs) may be private.
# Cloud metadata endpoints (169.254.169.254 etc.) are always blocked unless
# the exact host is listed in allowed_hosts.
enabled = true
allowed_hosts = []
# Default (connect, read) timeout applied by `Utils/egress.py`'s
# `create_default_session()` to any request through that session which does
# not pass its own `timeout=` (task-19830). Per-provider `api_timeout`
# settings under [api_settings.<provider>] still take precedence wherever
# they're read explicitly -- these are only the floor for call sites that
# don't set one.
# request_connect_timeout_seconds = 10
# request_read_timeout_seconds = 30

[image_generation]
# Backend-specific fields (model, base_url, timeout_seconds, api_key, ...) go
# ONLY under the matching [image_generation.<backend>] table below -- e.g.
# [image_generation.openrouter] default_model = "...". A flat key such as
# `openrouter_image_default_model` written directly in THIS [image_generation]
# table is NOT read; it logs a startup warning and is ignored (task-621).
default_backend = "swarmui"          # local SwarmUI instance is the friendliest zero-key default
enabled_backends = ["swarmui"]
max_width = 1024
max_height = 1024
max_pixels = 1048576
max_steps = 50
max_prompt_length = 1000
inline_max_bytes = 4000000
default_batch = 1
max_variants_per_message = 8

# `/generate-image` with no prompt text composes from conversation context.
# When context_llm_enabled, an LLM call (the session's active provider+
# model, same as a normal Console send) composes a richer prompt from the
# last context_llm_turns messages; ANY failure (kill-switch off, no ready
# provider, error, timeout, empty response) falls back to the built-in
# keyword extractor -- generation is never blocked by this.
context_llm_enabled = true
context_llm_turns = 10
context_llm_timeout_seconds = 15

[image_generation.stable_diffusion_cpp]
binary_path = ""                      # local `sd` CLI; empty = backend unusable
diffusion_model_path = ""             # OR model_path
model_path = ""
llm_path = ""
vae_path = ""
lora_paths = []
device = "auto"
default_steps = 25
default_cfg_scale = 7.5
default_sampler = "euler_a"
timeout_seconds = 120
allowed_extra_params = []

[image_generation.swarmui]
base_url = "http://127.0.0.1:7801"
default_model = ""
timeout_seconds = 120
allowed_extra_params = []
# swarm_token: secret, resolved via env/keyring precedence, not stored plaintext here

# ComfyUI H3 image editing is explicit opt-in: add "comfyui" to
# enabled_backends above after reviewing this server boundary. Saving a base_url
# in F9 Settings consents to sending the source image and instruction to that
# exact origin. ComfyUI retains uploaded inputs and saved outputs according to
# the server operator's policy.
# [image_generation.comfyui]
# base_url = "http://127.0.0.1:8188"
# request_timeout_seconds = 30
# connect_timeout_seconds = 5
# poll_interval_seconds = 1
# total_deadline_seconds = 1800
# default_seed = -1                # omit to use packaged workflow
# default_steps = 28               # omit to use packaged workflow
# default_sampler = "euler"        # omit to use packaged workflow

[image_generation.openrouter]
base_url = "https://openrouter.ai/api/v1"
default_model = "google/gemini-2.5-flash-image"  # task-620: "openai/gpt-image-1" was retired upstream and 404s
timeout_seconds = 120
allowed_extra_params = []
api_key = "<API_KEY_HERE>"

[image_generation.novita]
base_url = "https://api.novita.ai"
default_model = "sd_xl_base_1.0.safetensors"
timeout_seconds = 180
poll_interval_seconds = 2
allowed_extra_params = []
api_key = "<API_KEY_HERE>"

[image_generation.together]
base_url = "https://api.together.xyz/v1"
default_model = "black-forest-labs/FLUX.1-schnell-Free"
timeout_seconds = 120
allowed_extra_params = []
api_key = "<API_KEY_HERE>"

[image_generation.modelstudio]
base_url = ""                         # region-derived if empty
default_model = "qwen-image"
region = "sg"                         # sg|cn|us
mode = "auto"                         # sync|async|auto
poll_interval_seconds = 2
timeout_seconds = 180
allowed_extra_params = []
api_key = "<API_KEY_HERE>"

# fal.ai and Gemini backends -- uncomment and add enabled_backends/
# default_backend above to use.
# [image_generation.fal]
# base_url = "https://queue.fal.run"
# default_model = "fal-ai/flux/schnell"
# poll_interval_seconds = 2
# timeout_seconds = 120
# api_key = "<API_KEY_HERE>"          # or set FAL_KEY in the environment

# [image_generation.gemini]
# base_url = "https://generativelanguage.googleapis.com/v1beta"
# default_model = "gemini-2.5-flash-image"
# timeout_seconds = 120
# api_key = "<API_KEY_HERE>"          # or set GEMINI_API_KEY / GOOGLE_API_KEY

# User-defined `/generate-image` @style templates, layered over the 13
# built-ins (Media_Creation/generation_templates.py). A user template with
# the same id as a builtin overrides it; new ids extend the set. Fields
# mirror a builtin's shape: name/category/base_prompt are required,
# everything else optional. Uncomment and edit to add your own, or drop
# one *.toml file per template (same fields, no [image_generation.styles.*]
# wrapper) under <user data dir>/image_generation_styles/ instead -- see
# that module's docstring for the full precedence rule.
# [image_generation.styles.my_glow]
# name = "My Glow"
# category = "Custom"
# description = "Soft dreamy glow"
# base_prompt = "{{subject}}, soft glow lighting, dreamy atmosphere"
# negative_prompt = "harsh lighting, low quality"
# tags = ["custom", "glow"]
# [image_generation.styles.my_glow.default_params]
# width = 768
# height = 768
# steps = 28
# cfg_scale = 7.5
# [image_generation.styles.my_glow.context_mappings]
# subject = "last_message"

[character_defaults]
# Default settings specifically for the 'Character' tab
provider = "Anthropic"
model = "claude-haiku-4-5" # Make sure this exists in [providers.Anthropic]
system_prompt = "You are roleplaying as a witty pirate captain."
temperature = 0.8
top_p = 0.9
min_p = 0.0 # Check if API supports this
top_k = 100 # Check if API supports this

[analysis_defaults]
# Default settings specifically for the Media Analysis feature
provider = "OpenAI"
model = "gpt-4o"
temperature = 0.7
top_p = 0.95
min_p = 0.05
top_k = 50
system_prompt = "You are an AI assistant specialized in analyzing and summarizing media content. Provide comprehensive, insightful analysis with clear structure and key takeaways."
max_tokens = 4096
# Prompt search/filter defaults
default_prompt_search = ""
default_keyword_filter = ""
# Auto-save analysis after generation
auto_save = false
# Show analysis button in media viewer by default
show_analysis_button = true

[llm_management]
# LLM Management settings
model_download_dir = "~/Downloads/tldw_models"  # Legacy read-only scan root for Installed models

[notes]
# Device-private lasting-sync settings. Legacy sync keys are intentionally not
# emitted for fresh profiles; already-present keys remain migration input only.
recovery_capacity_bytes = 268435456  # Device-private lasting-sync recovery capacity (256 MiB)
# Lasting-sync change watcher: base poll interval, and the cap it backs off to
# while roots are quiet (backed-off sleeps are jittered by up to +/-50%).
sync_watcher_interval_seconds = 1.0
sync_watcher_max_interval_seconds = 10.0

# Auto-save settings
auto_save_enabled = true             # Enable auto-save feature
auto_save_delay_ms = 3000           # Delay in milliseconds before auto-saving (3 seconds)
auto_save_on_every_key = false      # If true, saves on every keystroke; if false, uses delay


# ==========================================================
# Default/Template Prompts
# ==========================================================
[Prompts]
# Default prompts used by various functions. These can be overridden by user settings.
sub_question_generation_prompt = "Based on the user query and chat history, generate up to 3 sub-questions to gather more specific information. Format as a numbered list."
search_result_relevance_eval_prompt = "Evaluate the relevance of the following search result snippet to the query. Score from 1 (not relevant) to 5 (highly relevant)."
analyze_search_results_prompt = "Analyze the provided search results and synthesize a comprehensive answer to the original query."
situate_chunk_context_prompt = "You are an AI assistant. Please follow the instructions provided in the input text carefully and accurately."

[prompts.document_generation.timeline]
prompt = "Create a detailed text-based timeline based on our conversation/materials being referenced. Include key dates, events, and their relationships in chronological order."
temperature = 0.3
max_tokens = 2000

[prompts.document_generation.study_guide]
prompt = "Create a detailed and well produced study guide based on the current focus of our conversation/materials in reference. Include key concepts, definitions, learning objectives, and potential exam questions."
temperature = 0.5
max_tokens = 3000

[prompts.document_generation.briefing]
prompt = "Create a detailed and well produced executive briefing document regarding this conversation and the subject material. Include key points, actionable insights, strategic implications, and recommendations."
temperature = 0.4
max_tokens = 2500


# ==========================================================
# Embedding Configuration
# ==========================================================
[embedding_config]
default_model_id = "e5-small-v2"
default_llm_for_contextualization = "gpt-3.5-turbo"
model_cache_dir = "~/.local/share/tldw_cli/models/embeddings"
auto_download = true
cache_size_limit_gb = 10.0

    # --- HuggingFace Models ---
    [embedding_config.models.e5-small-v2]
    provider = "huggingface"
    model_name_or_path = "intfloat/e5-small-v2"
    dimension = 384
    trust_remote_code = false
    max_length = 512

    [embedding_config.models.multilingual-e5-large-instruct]
    provider = "huggingface"
    model_name_or_path = "intfloat/multilingual-e5-large-instruct"
    dimension = 1024
    trust_remote_code = false
    max_length = 512

    [embedding_config.models.e5-base-v2]
    provider = "huggingface"
    model_name_or_path = "intfloat/e5-base-v2"
    dimension = 768
    trust_remote_code = false
    max_length = 512

    [embedding_config.models.e5-large-v2]
    provider = "huggingface"
    model_name_or_path = "intfloat/e5-large-v2"
    dimension = 1024
    trust_remote_code = false
    max_length = 512

    [embedding_config.models.all-MiniLM-L6-v2]
    provider = "huggingface"
    model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 384
    trust_remote_code = false
    max_length = 256

    [embedding_config.models.all-mpnet-base-v2]
    provider = "huggingface"
    model_name_or_path = "sentence-transformers/all-mpnet-base-v2"
    dimension = 768
    trust_remote_code = false
    max_length = 384

    [embedding_config.models.bge-small-en-v1.5]
    provider = "huggingface"
    model_name_or_path = "BAAI/bge-small-en-v1.5"
    dimension = 384
    trust_remote_code = false
    max_length = 512

    [embedding_config.models.bge-base-en-v1.5]
    provider = "huggingface"
    model_name_or_path = "BAAI/bge-base-en-v1.5"
    dimension = 768
    trust_remote_code = false
    max_length = 512

    [embedding_config.models.gte-small]
    provider = "huggingface"
    model_name_or_path = "thenlper/gte-small"
    dimension = 384
    trust_remote_code = false
    max_length = 512

    # --- Official OpenAI Models ---
    [embedding_config.models.openai-ada-002]
    provider = "openai"
    model_name_or_path = "text-embedding-ada-002"
    dimension = 1536
    api_key = "YOUR_OPENAI_API_KEY_OR_LEAVE_BLANK_IF_ENV_VAR_SET" # User fills this or sets ENV

    [embedding_config.models.openai-text-embedding-3-small]
    provider = "openai"
    model_name_or_path = "text-embedding-3-small" # Common model name
    dimension = 3072 # Or 256,, 1536, 2048 3072 depending on how you use it
    api_key = "YOUR_OPENAI_API_KEY_OR_LEAVE_BLANK_IF_ENV_VAR_SET"

    [embedding_config.models.openai-text-embedding-3-large]
    provider = "openai"
    model_name_or_path = "text-embedding-3-large" # Common model name
    dimension = 1536 # Or 512, 1536 depending on how you use it
    api_key = "YOUR_OPENAI_API_KEY_OR_LEAVE_BLANK_IF_ENV_VAR_SET"

    # --- Placeholder for a Local OpenAI-Compatible Server ---
    # The user needs to edit this section for their specific local setup.
    # The 'key' (e.g., "my-local-nomic-model") is what will appear in the UI's model dropdown
    # when "Local OpenAI-Compliant Server" provider is selected.
    [embedding_config.models.my-local-nomic-model]
    provider = "openai" # CRITICAL: This tells EmbeddingFactory to use OpenAICfg
    model_name_or_path = "nomic-ai/nomic-embed-text-v1" # The actual model name the LOCAL SERVER uses/expects
    base_url = "http://localhost:8080/v1" # The base URL of THE LOCAL SERVER's OpenAI-compatible API
    dimension = 768 # CRITICAL: User MUST provide the correct dimension for this model
    # api_key can be omitted if the local server doesn't require one, or set to a dummy value.
    # api_key = "not-needed-for-local"

    # --- Another Local Example (e.g., for a Llama.cpp server with embeddings) ---
    [embedding_config.models.local-llama-cpp-embeddings]
    provider = "openai"
    model_name_or_path = "llama-2-7b-chat.Q4_K_M.gguf" # Or whatever model name the server endpoint expects
    base_url = "http://localhost:8000/v1" # Common port for Llama.cpp server's OpenAI API
    dimension = 4096 # Example dimension for Llama-2 base models
    # api_key = "sk-xxxxxxxxxxxxxxxxx" # If your Llama.cpp server is configured with an API key

# You can add more local model configurations following the pattern above.
# The key part is `provider = "openai"` and providing the correct `base_url` and `dimension`.


# ==========================================================
# RAG (Retrieval-Augmented Generation) Configuration
# ==========================================================
[rag_citations]
canonical_writes_enabled = false

[rag]
# Comprehensive configuration for the RAG system

    # --- Retrieval Settings ---
    [rag.retriever]
    fts_top_k = 10              # Number of results from full-text search
    vector_top_k = 10           # Number of results from vector search
    hybrid_alpha = 0.7          # Hybrid RRF fusion: vector-leg weight (0=FTS only, 1=vector only); 0.7 = tldw_server default
    chunk_size = 512            # Size of text chunks for indexing
    chunk_overlap = 128         # Overlap between chunks
    
    # Collection names for different data types
    media_collection = "media_embeddings"
    chat_collection = "chat_embeddings"
    notes_collection = "notes_embeddings"
    character_collection = "character_embeddings"
    
    # --- Processing Settings ---
    [rag.processor]
    enable_reranking = true         # Enable result reranking
    reranker_model = "cohere"       # Reranker model: "cohere", "flashrank", or null
    reranker_top_k = 5             # Number of results to rerank
    deduplication_threshold = 0.85  # Similarity threshold for deduplication
    max_context_length = 4096      # Maximum context length for LLM
    combination_method = "weighted" # "weighted", "round_robin", "score_based"
    
    # --- Generation Settings ---
    [rag.generator]
    default_model = ""             # Default LLM model (empty = use chat defaults)
    default_temperature = 0.7      # Default temperature for RAG responses
    max_tokens = 1024              # Maximum tokens for RAG responses
    enable_streaming = true        # Enable streaming responses
    stream_chunk_size = 10         # Tokens per stream chunk
    
    # --- ChromaDB Settings ---
    [rag.chroma]
    persist_directory = ""         # Directory for ChromaDB (empty = auto)
    collection_prefix = "tldw_rag" # Prefix for collection names
    embedding_model = "all-MiniLM-L6-v2"  # Default embedding model
    embedding_dimension = 384      # Embedding dimension
    distance_metric = "cosine"     # "cosine", "euclidean", "ip"
    
    # --- Caching Settings ---
    [rag.cache]
    enable_cache = true            # Enable result caching
    cache_ttl = 3600              # Cache TTL in seconds (1 hour)
    max_cache_size = 1000         # Maximum cached items
    cache_embedding_results = true # Cache embedding results
    cache_search_results = true   # Cache search results
    cache_llm_responses = false   # Cache LLM responses (usually want fresh)
    
    # --- Memory Management Settings ---
    [rag.memory_management]
    max_total_size_mb = 1024.0         # Maximum total ChromaDB size (MB)
    max_collection_size_mb = 512.0     # Maximum size per collection (MB)
    max_documents_per_collection = 100000  # Maximum documents per collection
    max_age_days = 90                  # Maximum age of documents (days)
    inactive_collection_days = 30      # Days before cleaning inactive collections
    enable_automatic_cleanup = true    # Enable automatic cleanup
    cleanup_interval_hours = 24        # Hours between cleanup runs
    cleanup_batch_size = 1000         # Documents to delete per batch
    enable_lru_cache = true           # Enable ChromaDB LRU cache
    memory_limit_bytes = 2147483648   # Memory limit for ChromaDB (2GB)
    min_documents_to_keep = 100       # Minimum documents to always keep
    cleanup_confirmation_required = false  # Require confirmation for cleanup
    
    # --- Performance Settings ---
    [rag.performance]
    lazy_load_embeddings = true        # Defer loading embeddings models until first use
    preload_models = false             # Preload embedding models on startup
    unload_models_after_idle = true    # Unload models after idle timeout
    model_idle_timeout_seconds = 900   # Idle timeout before unloading (15 minutes)
    eager_dependency_check = false     # Check all dependencies on startup

# Legacy RAG settings (for backwards compatibility)
[rag_search]
fts_top_k = 10
vector_top_k = 10
web_vector_top_k = 10
llm_context_document_limit = 10

# ==========================================================
# Chunking Template Configuration
# ==========================================================
[chunking]
# Default chunking template for imports that did not pick one in the
# Library ingest form. Empty (the default) means plain chunk options
# (method/size/overlap) -- exactly today's behavior. The name must match
# a live template row in the media DB (RAG Admin: chunking templates);
# an unresolvable name fails the import with a named error rather than
# silently falling back to different chunking.
default_template = ""


# --- Model Capabilities Configuration ---
[model_capabilities]
# This section defines which models have specific capabilities like vision support.
# Users can override or extend these patterns in their config file.

# Direct model-to-capability mappings (highest priority)
[model_capabilities.models]
# OpenAI models
"gpt-4-vision-preview" = { vision = true, max_images = 1 }
"gpt-4-turbo" = { vision = true, max_images = 10 }
"gpt-4-turbo-2024-04-09" = { vision = true, max_images = 10 }
"gpt-4o" = { vision = true, max_images = 10 }
"gpt-4o-mini" = { vision = true, max_images = 10 }
"gpt-5.6-terra" = { vision = true, max_images = 10 }

# Anthropic models
"claude-3-opus-20240229" = { vision = true, max_images = 5 }
"claude-3-sonnet-20240229" = { vision = true, max_images = 5 }
"claude-3-haiku-20240307" = { vision = true, max_images = 5 }
"claude-3-5-sonnet-20240620" = { vision = true, max_images = 5 }
"claude-3-5-sonnet-20241022" = { vision = true, max_images = 5 }
"claude-sonnet-5" = { vision = true, max_images = 5 }

# Google models
"gemini-pro-vision" = { vision = true, max_images = 1 }
"gemini-1.5-pro" = { vision = true, max_images = 10 }
"gemini-1.5-flash" = { vision = true, max_images = 10 }
"gemini-2.0-flash" = { vision = true, max_images = 10 }

# Pattern-based matching for model families (fallback if not in direct mappings)
[model_capabilities.patterns]
# OpenAI patterns
OpenAI = [
    { pattern = "^gpt-4.*vision", vision = true },
    { pattern = "^gpt-4[o0](?:-mini)?", vision = true },  # gpt-4o, gpt-40, gpt-4o-mini
    { pattern = "^gpt-4.*turbo", vision = true }
]

# Anthropic patterns
Anthropic = [
    { pattern = "^claude-3", vision = true },             # All Claude 3 models have vision
    { pattern = "^claude.*opus-4", vision = true },      # Claude Opus 4 series
    { pattern = "^claude.*sonnet-4", vision = true }     # Claude Sonnet 4 series
]

# Google patterns
Google = [
    { pattern = "gemini.*vision", vision = true },
    { pattern = "gemini-[0-9.]+-(pro|flash)", vision = true },  # Modern Gemini models
    { pattern = "gemini-2\\\\.", vision = true }                 # Gemini 2.x series
]

# OpenRouter patterns (uses provider/model format)
OpenRouter = [
    { pattern = "openai/gpt-4.*vision", vision = true },
    { pattern = "openai/gpt-4[o0]", vision = true },
    { pattern = "anthropic/claude-3", vision = true },
    { pattern = "google/gemini.*vision", vision = true },
    { pattern = "google/gemini-[0-9.]+-(pro|flash)", vision = true }
]

# Default behavior for unknown models
[model_capabilities.defaults]
unknown_models_vision = false  # Whether to assume unknown models have vision capabilities
log_unknown_models = true      # Whether to log when an unknown model is queried

# --- Sections below are placeholders based on config.txt, integrate as needed ---
# [tts_settings]
# default_provider = "kokoro"
# ...

# ==========================================================
# Deep-Search Configuration
# ==========================================================
[tools]
# web_deep_search_enabled = false    # Opt-in deep-search tool; requires app restart; each call makes ~2x-results+3 LLM calls plus page fetches (real money on paid providers)

[SearchSettings]
# Deep-search (web_deep_search tool) defaults. Enable the tool itself with
# [tools] web_deep_search_enabled = true (requires app restart; each call makes
# ~2x-results+3 LLM calls plus page fetches -- real money on paid providers).
# search_provider_default = "google"
# relevance_analysis_llm = "openai"
# final_answer_llm = "openai"
# search_enable_subquery = false   # generate sub-questions from the query and
#   search those too. Costs one LLM call plus up to search_default_max_queries-1
#   extra searches, each carrying its own per-result relevance calls. Qodo
#   (PR 1772): with BOTH lanes active each generated facet costs TWO provider
#   calls, not one -- a web search and an academic-provider search -- so the
#   worst case is 2 x (search_default_max_queries - 1) extra calls per round,
#   plus the per-result relevance and summarization calls each returned source
#   then needs. Since task-17372 the facets ALSO drive academic-provider
#   searches, so enabling this changes both what evidence is retrieved and how
#   it is judged
#   (the facets reach the relevance prompt either way). Before that fix it
#   changed only the judging, which is why the recorded gate measurement for
#   fan-out says nothing about retrieval -- see
#   Docs/Development/research-report-eval-baseline.md.
# search_default_max_queries = 5
# search_result_max = 10
# relevance_llm_timeout_s = 30   # per relevance/summarization LLM call. Measured
#   (task-17370): per-result summarization against a local 27B took 42-131s, so
#   30 guarantees the fallback-to-source-text path on local models. Raise it if
#   you want summarized evidence rather than raw page text.
# relevance_scrape_timeout_s = 30
# research_max_iterations = 2   # rounds a local research RUN performs when it
#   does not set its own limit: round 1 researches the question, later rounds
#   research the gaps the previous synthesis left open (task-16324). 2 is the
#   shipped default because a second round measurably improved evidence --
#   resolved citation markers 24 -> 39, citation density 0.77 -> 0.95
#   (task-17370). It costs one extra search per gap, each with its own
#   per-result relevance and summarization calls, plus another synthesis and gap
#   analysis per round: the measured arm went from 3 to 12 search calls across
#   three questions and roughly tripled wall-clock. Set to 1 for single-pass
#   runs; a run's own limits always override this.
# deep_search_timeout_s = 240   # the agent runtime automatically allots this tool its own per-call timeout of this value plus ~50s slack (wait_for grace + thread-join + scheduling jitter), via LocalToolProvider.timeout_for -- independent of max_tool_call_seconds, any value here is safe

[webfetch]
# Governs the web_fetch and web_crawl tools (task-2833). When true (the
# default), every hop -- web_fetch redirects, web_crawl pages, and sitemap
# fetches -- is checked against its host's robots.txt before being
# fetched; a disallowed web_fetch hop is refused, a disallowed web_crawl
# page/sitemap is skipped and counted, not fatal. A robots.txt that can't
# be fetched or parsed fails OPEN (no restrictions applied), so a robots
# outage never bricks fetching. Set to false to disable enforcement
# entirely (no robots.txt fetches are made).
# respect_robots_txt = true

# ==========================================================
# Search Engines Configuration
# ==========================================================
[SearchEngines]
# API Keys for various search engines
bing_search_api_key = ""
google_search_api_key = ""
brave_search_api_key = ""
brave_search_ai_api_key = ""
kagi_search_api_key = ""
tavily_search_api_key = ""
# Serper (google.serper.dev) API key
serper_search_api_key = ""
# Exa (exa.ai) API key
exa_search_api_key = ""
# Yandex Cloud Search API key
yandex_search_api_key = ""
# Yandex Cloud folder id for Search API v2
yandex_search_folder_id = ""

# API URLs
bing_search_api_url = "https://api.bing.microsoft.com/v7.0/search"
google_search_api_url = "https://www.googleapis.com/customsearch/v1"
searx_search_api_url = "https://searx.example.com/search"

# General search settings
search_result_max = 10

# Country and language settings
bing_country_code = "US"
search_engine_country_code_brave = "US"
google_search_country = "US"
google_search_engine_id = ""

# Google-specific settings
google_simp_trad_chinese = false
limit_google_search_to_country = false
google_safe_search = false

# ============================================================================
# Media Processing Configuration
# ============================================================================

[media_processing]
# Maximum file sizes for processing
max_audio_file_size_mb = 500
max_video_file_size_mb = 2000

# FFmpeg path (optional - will try to find automatically if not set)
# ffmpeg_path = "/usr/bin/ffmpeg"

# Temporary file cleanup
cleanup_temp_files = true
temp_dir = ""  # Empty means use system temp

[transcription]
# Default transcription provider
# Options: "faster-whisper", "parakeet-onnx", "qwen2audio", "parakeet", "canary", "parakeet-mlx", "lightning-whisper-mlx", "remote-whisper"
# Resolved for this install when this file was first created: parakeet-mlx or
# lightning-whisper-mlx on macOS when installed, otherwise faster-whisper.
# Edit this line to pin a different provider yourself -- your choice always
# wins over the platform preference.
default_provider = "__DEFAULT_TRANSCRIPTION_PROVIDER__"

# Default model for transcription
# For faster-whisper: large-v1, large-v2, large-v3, large, distil-large-v2, distil-large-v3,
#                     distil-medium.en, distil-small.en, deepdml/faster-distil-whisper-large-v3.5,
#                     deepdml/faster-whisper-large-v3-turbo-ct2, nyrahealth/faster_CrisperWhisper
#   Note: faster-whisper supports translation to English for non-English audio
# For qwen2audio: Qwen2-Audio-7B-Instruct
# For parakeet: nvidia/parakeet-tdt-1.1b, nvidia/parakeet-rnnt-1.1b, nvidia/parakeet-ctc-1.1b,
#               nvidia/parakeet-tdt-0.6b, nvidia/parakeet-rnnt-0.6b, nvidia/parakeet-ctc-0.6b,
#               nvidia/parakeet-tdt-0.6b-v2
# For canary: nvidia/canary-1b-flash, nvidia/canary-1b
#   Note: Canary supports multilingual ASR and translation between en, de, es, fr
# For remote-whisper: Depends on your server (e.g., "whisper-1" for OpenAI)
default_model = "distil-large-v3"

# Default language for transcription (use "auto" for automatic detection)
# For source language in transcription
default_language = "en"

# Default source language (overrides default_language if specified)
# Used for explicitly setting the audio's language
default_source_language = ""

# Default target language for translation (leave empty for no translation)
# Supported by:
#   - faster-whisper: Only supports translation to English ("en")
#   - canary: Supports translation between en, de, es, fr
default_target_language = ""

# Device to use for transcription
# Options: "cpu", "cuda", "mps" (Apple Silicon)
device = "cpu"

# Compute type for faster-whisper
# Options: "int8", "float16", "float32"
compute_type = "int8"

# Explicit local directory containing the Parakeet v2 INT8 ONNX bundle.
# parakeet-onnx never downloads model files implicitly.
parakeet_onnx_model_dir = ""

# Exact v2/v3 × INT8/F32 external-source records. External files remain
# user-owned and are never copied or managed implicitly.
parakeet_external_sources = {}

# Voice Activity Detection
use_vad_by_default = false

# Speaker diarization (not yet fully implemented)
use_diarization_by_default = false

# Chunk length for long audio processing (in seconds)
# Used by Canary model for efficient processing of long audio files
chunk_length_seconds = 40.0

[diarization]
# Speaker diarization configuration
enabled = false  # Enable/disable diarization by default

# Voice Activity Detection (VAD) settings
vad_threshold = 0.5  # Sensitivity for speech detection (0.0-1.0)
vad_min_speech_duration = 0.25  # Minimum speech duration in seconds
vad_min_silence_duration = 0.25  # Minimum silence duration in seconds

# Segmentation settings for speaker analysis
segment_duration = 2.0  # Duration of analysis segments in seconds
segment_overlap = 0.5  # Overlap between segments in seconds
min_segment_duration = 1.0  # Minimum segment duration
max_segment_duration = 3.0  # Maximum segment duration

# Speaker embedding model
embedding_model = "speechbrain/spkrec-ecapa-voxceleb"  # Model for speaker embeddings
embedding_device = "auto"  # Device: "auto", "cuda", "cpu"

# Clustering settings
clustering_method = "spectral"  # Method: "spectral" or "agglomerative"
similarity_threshold = 0.85  # Threshold for speaker similarity (0.0-1.0)
min_speakers = 1  # Minimum expected speakers
max_speakers = 10  # Maximum expected speakers

# Post-processing
merge_threshold = 0.5  # Time gap in seconds to merge same-speaker segments
min_speaker_duration = 3.0  # Minimum total duration per speaker in seconds

[transcription.remote_whisper]
# Remote Whisper OpenAI API compatible transcription backend
# Enable this to use a custom transcription server instead of local models
enabled = false

# API endpoint URL
# Examples:
#   - OpenAI: "https://api.openai.com/v1/audio/transcriptions"
#   - Local server: "http://localhost:8000/v1/audio/transcriptions"
#   - Custom endpoint: "https://your-whisper-server.com/transcribe"
api_endpoint = "http://localhost:8000/v1/audio/transcriptions"

# API authentication (leave empty if not required)
api_key_env_var = "REMOTE_WHISPER_API_KEY"  # Environment variable name
# api_key = ""  # Direct API key (less secure - use env var instead)

# Model to use (e.g., "whisper-1" for OpenAI, or your custom model name)
model = "whisper-1"

# Request timeout in seconds
timeout = 300

# Response format: "json", "text", "srt", "verbose_json", "vtt"
response_format = "json"

# Optional parameters
temperature = 0.0  # Sampling temperature (0-1)
# prompt = ""  # Optional prompt to guide transcription style

# Additional custom parameters (will be passed as-is to the API)
# Format: key = "value"
# Example:
# [transcription.remote_whisper.additional_params]
# custom_param = "custom_value"

[local_ingestion]
# YouTube/URL download settings
enable_url_downloads = true
use_cookies_for_downloads = false
cookie_file_path = ""

# Audio extraction settings for videos
extract_audio_format = "mp3"
audio_bitrate = "192k"
audio_sample_rate = 44100

# Processing defaults
keep_original_files = false
auto_analyze_transcripts = true

# Parallel processing
max_concurrent_processes = 2

[mcp]
# Model Context Protocol (MCP) settings
enabled = false  # Enable MCP server functionality
server_name = "tldw_chatbook"
server_version = "0.1.0"
transport = "stdio"  # "stdio" for Claude Desktop, "http" for web-based clients
http_port = 3000  # Port for HTTP transport
allowed_clients = ["claude-desktop", "localhost"]  # List of allowed client identifiers

# Feature toggles
expose_tools = true  # Expose tools (chat, search, etc.)
expose_resources = true  # Expose resources (conversations, notes, etc.)
expose_prompts = true  # Expose prompt templates

# Security settings
require_auth = false  # Require authentication (not implemented yet)
rate_limit = 100  # Max requests per minute per client
max_concurrent_requests = 10  # Max concurrent requests
# approval_timeout_seconds = 0  # Console approval-card auto-deny ceiling: 0 (default) waits indefinitely; e.g. 120 auto-denies undecided calls after 120s

# expose_local_tools = false   # expose workspace, web, and Watchlists agent tools (fs_*/git_*/web_*/watchlists_*) to external MCP clients; each tool remains permission-gated

# Tool-specific settings
[mcp.tools]
chat_default_provider = "openai"
chat_default_temperature = 0.7
chat_default_max_tokens = 4096
search_default_limit = 10
enable_media_ingestion = true  # Allow media ingestion via MCP

# Resource-specific settings
[mcp.resources]
max_list_limit = 100  # Maximum items to return in list operations
default_list_limit = 10  # Default items to return in list operations
enable_binary_resources = false  # Allow serving binary resources (images, etc.)

# Prompt-specific settings
[mcp.prompts]
enable_custom_prompts = true  # Allow custom prompt creation
max_prompt_length = 10000  # Maximum prompt length in characters

# Subscription system configuration
[subscriptions]
enabled = true
default_check_interval = 3600  # 1 hour in seconds
max_concurrent_checks = 10
check_timeout_seconds = 30
auto_pause_after_failures = 10  # Seeds auto_pause_threshold for NEW subscriptions only (task-1410); an explicit per-subscription value always overrides, and existing subscriptions keep their stored value
enable_background_checking = true
default_priority = 3  # 1-5, higher is more important

# Rate limiting
[subscriptions.rate_limiting]
global_requests_per_minute = 60
per_domain_requests_per_minute = 10
retry_after_rate_limit = true

# Performance settings
[subscriptions.performance]
use_connection_pooling = true
enable_response_caching = true
cache_ttl_seconds = 300
use_http2 = true
enable_compression = true

# Content processing
[subscriptions.content_processing]
default_analysis_prompt = "Summarize the key points and provide actionable insights."
auto_analyze_new_items = false
save_analysis_only = false
extract_keywords = true
max_keywords = 15

# Briefing/newsletter generation
[subscriptions.briefings]
enabled = true
default_format = "markdown"  # markdown, html, pdf
save_to_notes = true
email_notifications = false
morning_digest_time = "06:00"  # 24-hour format

# Default templates for common subscription types
[subscriptions.templates.tech_news]
name = "Tech News Feed"
type = "rss"
check_frequency = 1800  # 30 minutes
priority = 4
extraction_method = "auto"
auto_ingest = true
tags = ["tech", "news"]

[subscriptions.templates.documentation]
name = "Documentation Monitor"
type = "url"
check_frequency = 86400  # Daily
change_threshold = 0.05  # 5% change threshold
priority = 3
tags = ["docs", "reference"]

# GitHub configuration for repository browsing
[github]
# Personal access token for accessing private repositories
# Create a token at: https://github.com/settings/tokens
# Required scopes: repo (for private repos), public_repo (for public only)
api_token = ""  # Leave empty to only access public repositories
api_token_env_var = "GITHUB_API_TOKEN"  # Environment variable to check first

# Rate limiting settings
enable_rate_limit_handling = true  # Automatically handle rate limit responses
cache_ttl_seconds = 300  # Cache API responses for 5 minutes
max_retries = 3  # Maximum retries for failed requests

# Default behavior
default_branch = "main"  # Default branch to use if not specified
show_hidden_files = false  # Show files starting with . (except .gitignore)
respect_gitignore = true  # Respect .gitignore rules when displaying files

# Performance settings
lazy_load_tree = true  # Load repository tree on-demand
max_tree_depth = 10  # Maximum depth for recursive tree loading
batch_file_requests = true  # Batch multiple file content requests
max_concurrent_requests = 5  # Maximum concurrent API requests

# UI preferences
auto_expand_small_folders = true  # Auto-expand folders with < 5 items
highlight_binary_files = true  # Visually distinguish binary files
show_file_sizes = true  # Display file sizes in the tree
default_preview_language = "auto"  # Language for syntax highlighting

# Export settings
default_export_format = "compilation"  # Default format: compilation, zip, markdown
include_file_metadata = false  # Include metadata in exports
max_export_size_mb = 100  # Maximum size for exports in MB

# Selection profiles directory
profiles_directory = "~/.config/tldw_cli/github_profiles"  # Where to store selection profiles

[briefings_feed_server]
# Opt-in, session-only static file server for ONE exported briefings podcast
# feed directory at a time (task-1760). This is its own section on purpose --
# it is unrelated to [web_server] below: that runs the whole chatbook UI in a
# browser instead of a terminal (a separate, mutually exclusive process mode)
# and has no route that serves a directory you choose. Nothing here ever
# auto-starts anything -- these are only the defaults the Watchlists
# Artifacts pane's Serve action uses when you press it; serving always
# requires that explicit action. See Docs/User_Guide/watchlists.md's
# "Serving an exported feed" section for the full security posture
# (no authentication, localhost-only unless you widen bind, recursive
# file serving with directory listings disabled, so point this at a
# dedicated export folder rather than a general-purpose one like $HOME).
# A blank/invalid bind value here falls back to loopback rather than
# silently widening exposure.
bind = "127.0.0.1"  # Loopback only. Widen only if you understand the exposure -- there is no authentication.
port = 0  # 0 = pick any free port each time; set a fixed port to reuse the same URL

[web_server]
# Web server configuration for running tldw_chatbook in a browser
enabled = true  # Enable web server functionality
host = "localhost"  # Host address to bind to
port = 8000  # Port to bind to
title = "tldw chatbook"  # Title for the web page
font_size = 12  # Browser terminal font size; 12 keeps Textual Web close to native terminal density
debug = false  # Enable debug mode for development
"""

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

try:
    DEFAULT_CONFIG_FROM_TOML: Dict[str, Any] = tomllib.loads(CONFIG_TOML_CONTENT)
except tomllib.TOMLDecodeError as e:
    logger.critical(
        f"FATAL: Could not parse internal DEFAULT_CONFIG_TOML_CONTENT: {e}. Application cannot start correctly."
    )
    DEFAULT_CONFIG_FROM_TOML = {}  # Should not happen with valid TOML string


# --- Primary Configuration Loading Logic for the CLI ---
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_CACHE_SOURCE: Optional[Path] = None
_FIRST_PROFILE_CREATED_THIS_SESSION = False


def first_profile_created_this_session() -> bool:
    """Return whether this process created the active profile config.

    Returns:
        ``True`` when this process created the active profile configuration;
        otherwise ``False``.
    """
    return _FIRST_PROFILE_CREATED_THIS_SESSION


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


def _load_cli_config_bootstrap_unlocked(
    force_reload: bool = False,
) -> _ConfigBootstrapResult:
    global _CONFIG_CACHE, _CONFIG_CACHE_SOURCE, _LAST_CONFIG_LOAD_FAILURE
    global _FIRST_PROFILE_CREATED_THIS_SESSION
    config_path = _get_effective_config_path()
    if (
        _CONFIG_CACHE is not None
        and _CONFIG_CACHE_SOURCE == config_path
        and not force_reload
    ):
        return _ConfigBootstrapResult(_CONFIG_CACHE, True)

    _CONFIG_CACHE = None
    _CONFIG_CACHE_SOURCE = None

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
        loaded_config = deep_merge_dicts(loaded_config, user_config_from_file)
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
        _LAST_CONFIG_LOAD_FAILURE = ConfigLoadFailure(path=config_path, message=str(e))
    except Exception as e:
        logger.opt(exception=True).error(
            f"An unexpected error occurred while loading CLI config {config_path}: {e}. Using internal defaults + any previous successful load."
        )

    if bootstrap_succeeded:
        _CONFIG_CACHE = loaded_config
        _CONFIG_CACHE_SOURCE = config_path
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


def load_cli_config_and_ensure_existence(
    force_reload: bool = False,
) -> Dict[str, Any]:  # Renamed from load_cli_config
    """
    Loads settings for the CLI application from ~/.config/tldw_cli/config.toml.
    If the file doesn't exist, it's created with default values from CONFIG_TOML_CONTENT.
    Uses programmatic defaults (from CONFIG_TOML_CONTENT) as a base.
    """
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

_CONFIG_FILE_LOCK = _threading.RLock()


def _config_file_lock():
    """Return the shared config-file read-modify-write lock.

    Returns:
        The process-wide lock serializing config file saves/deletes across
        Textual thread workers so concurrent cycles cannot drop updates.
    """
    return _CONFIG_FILE_LOCK


@contextmanager
def _config_interprocess_lock(config_path: Path) -> Iterator[None]:
    """Hold one OS-backed lock across a whole-file config transaction."""

    lock_path = config_path.with_name(f"{config_path.name}.lock")
    application_directory = application_owned_config_directory(config_path)
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
    locked = False
    try:
        portalocker.lock(stream, portalocker.LockFlags.EXCLUSIVE)
        locked = True
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


@contextmanager
def _config_write_lock(config_path: Path) -> Iterator[None]:
    """Serialize one config write transaction within and across processes."""

    with (
        _settings_rebuild_lock(),
        _config_file_lock(),
        _config_interprocess_lock(config_path),
    ):
        yield


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

    if not force_reload:
        config_path = _get_effective_config_path()
        generation_before = _CONFIG_GENERATION
        cached_config = _CONFIG_CACHE
        cached_source = _CONFIG_CACHE_SOURCE
        if (
            cached_config is not None
            and cached_source == config_path
            and _CONFIG_CACHE is cached_config
            and _CONFIG_GENERATION == generation_before
        ):
            return _ConfigBootstrapResult(cached_config, True)

    with _config_file_lock():
        return _load_cli_config_bootstrap_unlocked(force_reload=force_reload)


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
    first-run wizard, and -- notably -- the full default+user re-merge every
    process shutdown persists via ``persist_cli_config_for_shutdown``) ends
    up here. The in-memory ``config_data`` is always a plain ``dict`` with
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
    serialized = toml.dumps(dict(config_data))
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
    result = atomic_private_write_text(
        config_path,
        serialized,
        application_owned_directory=application_directory,
    )
    _report_config_path_posture(result)
    _invalidate_config_caches()
    return parsed_back


def _install_bootstrap_cache_from_raw(
    raw_config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Install the bootstrap cache from an already-parsed raw config mapping.

    TASK-21124: replicates exactly the success tail of
    `_load_cli_config_bootstrap_unlocked` (merge the programmatic defaults,
    decrypt strictly, publish cache + source, retire any recorded parse
    failure) without re-reading or re-parsing the file. Used by
    `_publish_runtime_config_unlocked` with the verify parse-back a write
    just produced. Must be called with `_config_file_lock` held.

    Returns:
        The installed merged+decrypted config, or ``None`` when strict
        decryption failed -- the caller then falls back to the full locked
        reload, which reproduces the historical failure handling
        (cache left empty, in-memory defaults returned).
    """

    global _CONFIG_CACHE, _CONFIG_CACHE_SOURCE, _LAST_CONFIG_LOAD_FAILURE

    config_path = _get_effective_config_path()
    merged = deep_merge_dicts(DEFAULT_CONFIG_FROM_TOML, dict(raw_config))
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
    return loaded_config


def _publish_runtime_config_unlocked(
    raw_config: Mapping[str, Any] | None = None,
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

    The `_CONFIG_GENERATION` bump is kept last for consistency, but the
    ordering is not load-bearing: the fast path's soundness rests on its
    cache-identity re-check, and the generation sandwich only adds
    conservatism (see `_load_cli_config_bootstrap` -- the TASK-21124
    review proved a bump-first mutant equivalent). Callers hold the write
    lock.
    """

    global settings, _CONFIG_GENERATION

    loaded: Optional[Dict[str, Any]] = None
    if raw_config is not None:
        loaded = _install_bootstrap_cache_from_raw(raw_config)
    if loaded is None:
        loaded = _load_cli_config_bootstrap_unlocked(force_reload=True).config
    settings = load_settings(force_reload=True, reload_bootstrap=False)
    _CONFIG_GENERATION += 1
    return loaded


class RuntimeConfigSnapshot(NamedTuple):
    """A defensive normalized config view paired with its write generation."""

    generation: int
    values: Dict[str, Any]


class AtomicConfigSnapshot(NamedTuple):
    """A locked authoritative config view paired with its published generation."""

    generation: int
    values: Dict[str, Any]


def _atomic_config_values_from_raw(
    config_data: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge and decrypt the exact raw mapping already read under the write lock."""

    merged = deep_merge_dicts(DEFAULT_CONFIG_FROM_TOML, dict(config_data))
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


def get_runtime_config_snapshot(
    *,
    force_reload: bool = False,
) -> RuntimeConfigSnapshot:
    """Return a defensive current runtime config view."""

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


def _try_read_cli_config_serialized_unlocked(config_path: Path) -> str | None:
    try:
        with open_private_binary(config_path) as opened:
            _report_config_path_posture(opened.result)
            return opened.stream.read().decode("utf-8")
    except FileNotFoundError:
        return None


def _read_cli_config_serialized_unlocked(config_path: Path) -> str:
    serialized = _try_read_cli_config_serialized_unlocked(config_path)
    if serialized is None:
        _load_cli_config_bootstrap_unlocked(force_reload=True)
        with open_private_binary(config_path) as opened:
            _report_config_path_posture(opened.result)
            return opened.stream.read().decode("utf-8")
    return serialized


def read_cli_config_serialized() -> str:
    """Return the effective config's exact serialized on-disk representation."""

    with _config_file_lock():
        return _read_cli_config_serialized_unlocked(get_cli_config_path())


def _advanced_backup_path(config_path: Path) -> Path:
    return config_path.with_suffix(config_path.suffix + ".bak")


def _write_serialized_config_artifact_unlocked(
    path: Path,
    serialized: str,
    *,
    config_path: Path,
) -> Path:
    application_directory = _prepare_config_parent(config_path)
    result = atomic_private_write_text(
        path,
        serialized,
        application_owned_directory=application_directory,
    )
    _report_config_path_posture(result, target_kind="snapshot")
    return result.lexical_path


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
    """Validate and replace raw TOML without downgrading encryption."""

    replacement = tomllib.loads(serialized)
    if not isinstance(replacement, dict):
        raise TypeError("The CLI config must contain a top-level table")

    config_path = get_cli_config_path()
    with _config_write_lock(config_path):
        current_serialized = _try_read_cli_config_serialized_unlocked(config_path)
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
        return _publish_runtime_config_unlocked(raw_config=raw_written), backup_path


def persist_cli_config_for_shutdown() -> bool:
    """Re-persist the effective config through the normal encrypted owner."""

    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            bootstrap = _load_cli_config_bootstrap_unlocked(force_reload=True)
            if not bootstrap.succeeded:
                logger.warning(
                    "Shutdown config persistence skipped because the current "
                    "file could not be loaded safely."
                )
                return False
            current = bootstrap.config
            persisted = _config_data_for_persistence(current)
            raw_written = _write_raw_cli_config_unlocked(config_path, persisted)
            _publish_runtime_config_unlocked(raw_config=raw_written)
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
        return _publish_runtime_config_unlocked(raw_config=raw_written)


def export_cli_config_snapshot(
    config_data: Mapping[str, Any] | None = None,
    *,
    timestamp: str | None = None,
) -> Path:
    """Create an owner-only snapshot beside the effective config file."""

    config_path = get_cli_config_path()
    snapshot_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = config_path.parent / f"config_backup_{snapshot_timestamp}.toml"
    with _config_file_lock():
        serialized = _try_read_cli_config_serialized_unlocked(config_path)
        if serialized is None:
            if config_data is None:
                raise FileNotFoundError(config_path)
            serialized = toml.dumps(_config_data_for_persistence(config_data))
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
            _publish_runtime_config_unlocked(raw_config=raw_written)
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
    changed = False
    missing = object()
    for section, keys in delete_keys.items():
        current_level: Any = config_data
        section_missing = False
        for part in section.split("."):
            if not isinstance(current_level, dict):
                raise TypeError(part)
            if part not in current_level:
                section_missing = True
                break
            current_level = current_level[part]
        if section_missing:
            continue
        if not isinstance(current_level, dict):
            raise TypeError(section)
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
    global _CONFIG_CACHE, _SETTINGS_CACHE, settings
    requested_deletes = {} if delete_keys is None else delete_keys
    try:
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
        return ConfigMutationResult(False, False, "before_replace")

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
            return ConfigMutationResult(False, False, "before_replace")
        try:
            _validate_config_mutation_targets(section_values, requested_deletes)
            logged_keys = {
                section: list(values.keys())
                for section, values in section_values.items()
            }
            logged_deletes = {
                section: list(keys) for section, keys in requested_deletes.items()
            }
            logger.info(
                "Attempting to apply settings mutation: "
                f"sets={logged_keys!r}, deletes={logged_deletes!r}"
            )
            config_data = _read_raw_cli_config_unlocked(config_path)
        except tomllib.TOMLDecodeError as error:
            logger.error(
                "Configuration mutation failed "
                "(phase=read, config_path={}, error_type={}).",
                config_path,
                type(error).__name__,
            )
            return ConfigMutationResult(False, False, "before_replace")
        except Exception as error:
            logger.opt(exception=True).error(
                "Configuration mutation failed "
                "(phase=read, config_path={}, error_type={}).",
                config_path,
                type(error).__name__,
            )
            return ConfigMutationResult(False, False, "before_replace")

        if mutation_precondition is not None:
            try:
                is_current = mutation_precondition()
            except Exception as error:
                logger.error(
                    "Configuration mutation failed "
                    "(phase=precondition, error_type={}).",
                    type(error).__name__,
                )
                return ConfigMutationResult(False, False, "before_replace")
            if is_current is not True:
                return ConfigMutationResult(
                    False,
                    False,
                    None,
                    conflict=True,
                    conflict_reason="identity_changed",
                )

        if locked_snapshot_precondition is not None:
            try:
                locked_snapshot = AtomicConfigSnapshot(
                    generation=_CONFIG_GENERATION,
                    values=_atomic_config_values_from_raw(config_data),
                )
                is_current = locked_snapshot_precondition(locked_snapshot)
            except Exception as error:
                logger.error(
                    "Configuration mutation failed "
                    "(phase=locked_precondition, error_type={}).",
                    type(error).__name__,
                )
                return ConfigMutationResult(False, False, "before_replace")
            if is_current is not True:
                return ConfigMutationResult(
                    False,
                    False,
                    None,
                    conflict=True,
                    conflict_reason="identity_changed",
                )

        if before_replace is not None:
            try:
                before_replace()
            except Exception as error:
                logger.error(
                    "Configuration mutation failed "
                    "(phase=before_replace_callback, error_type={}).",
                    type(error).__name__,
                )
                return ConfigMutationResult(False, False, "before_replace")

        try:
            deleted_any = _delete_config_keys(config_data, requested_deletes)
            for section, values in section_values.items():
                if not values:
                    continue
                current_level = _target_config_section(config_data, section)
                for key, value in values.items():
                    current_level[key] = _maybe_encrypt_setting_value(
                        config_data, key, value
                    )
        except Exception as error:
            logger.error(
                "Configuration mutation failed "
                "(phase=before_replace, config_path={}, error_type={}).",
                config_path,
                type(error).__name__,
            )
            return ConfigMutationResult(False, False, "before_replace")
        set_any = any(bool(values) for values in section_values.values())
        if not set_any and not deleted_any:
            return ConfigMutationResult(False, False, None)

        try:
            persisted = _config_data_for_persistence(config_data)
            raw_written = _write_raw_cli_config_unlocked(
                config_path,
                persisted,
            )
        except Exception as error:
            logger.error(
                "Configuration mutation failed "
                "(phase=before_replace, config_path={}, error_type={}).",
                config_path,
                type(error).__name__,
            )
            return ConfigMutationResult(False, False, "before_replace")

        file_replaced = True
        logger.success(f"Successfully replaced settings file at {config_path}")

        if after_replace is not None:
            try:
                after_replace()
            except Exception as error:
                logger.error(
                    "Configuration mutation failed "
                    "(phase=after_replace_callback, error_type={}).",
                    type(error).__name__,
                )
                return ConfigMutationResult(True, False, "cache_reload")

        try:
            _publish_runtime_config_unlocked(raw_config=raw_written)
        except Exception as error:
            logger.error(
                "Configuration mutation failed "
                "(phase=cache_reload, config_path={}, error_type={}).",
                config_path,
                type(error).__name__,
            )
            return ConfigMutationResult(file_replaced, False, "cache_reload")

        logger.info("Global configuration caches invalidated and reloaded.")
        return ConfigMutationResult(file_replaced, True, None)


@dataclass(frozen=True, slots=True)
class RuntimeCapturePolicy:
    """Canonical process projection for future Console capture admission."""

    enabled: bool
    detail: CaptureDetail
    generation: int


_RUNTIME_CAPTURE_POLICY_LOCK = _threading.RLock()
_RUNTIME_CAPTURE_POLICY: RuntimeCapturePolicy | None = None


def _publish_runtime_capture_policy(
    enabled: bool,
    detail: CaptureDetail,
    generation: int,
) -> RuntimeCapturePolicy:
    """Publish one validated capture policy without touching general caches."""
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail

    if not isinstance(detail, CaptureDetail):
        raise TypeError("detail must be CaptureDetail")
    policy = RuntimeCapturePolicy(bool(enabled), detail, generation)
    global _RUNTIME_CAPTURE_POLICY
    with _RUNTIME_CAPTURE_POLICY_LOCK:
        _RUNTIME_CAPTURE_POLICY = policy
    return policy


def runtime_capture_policy() -> RuntimeCapturePolicy:
    """Return the shared runtime capture policy, resolving invalid detail Safe.

    Returns:
        The canonical enabled/detail projection and its config generation.
    """
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail

    global _RUNTIME_CAPTURE_POLICY
    with _RUNTIME_CAPTURE_POLICY_LOCK:
        current = _RUNTIME_CAPTURE_POLICY
        if current is not None and current.generation == _CONFIG_GENERATION:
            return current
    snapshot = _published_runtime_config_snapshot()
    with _RUNTIME_CAPTURE_POLICY_LOCK:
        current = _RUNTIME_CAPTURE_POLICY
        if current is not None and current.generation == snapshot.generation:
            return current
        console = snapshot.values.get("console", {})
        if not isinstance(console, Mapping):
            console = {}
        try:
            detail = CaptureDetail(console.get("exchange_capture_detail", "safe"))
        except (TypeError, ValueError):
            detail = CaptureDetail.SAFE
        current = RuntimeCapturePolicy(
            coerce_bool_setting(console.get("exchange_capture", True), True),
            detail,
            snapshot.generation,
        )
        _RUNTIME_CAPTURE_POLICY = current
        return current


def apply_console_capture_settings(
    *,
    enabled: bool,
    detail: CaptureDetail,
    expected_generation: int,
) -> ConfigMutationResult:
    """Apply the kill switch/detail with privacy-safe publication ordering.

    Args:
        enabled: Future-capture kill-switch state.
        detail: Safe or Full global capture detail.
        expected_generation: Config generation observed by the caller.

    Returns:
        Structured replacement/cache-publication status, including conflicts
        and partial post-replacement cache failures.
    """
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail

    if type(enabled) is not bool or not isinstance(detail, CaptureDetail):
        return ConfigMutationResult(False, False, "before_replace")

    def generation_is_current(snapshot: AtomicConfigSnapshot) -> bool:
        return snapshot.generation == expected_generation

    def publish_before_replace() -> None:
        _publish_runtime_capture_policy(enabled, detail, expected_generation)

    def publish_after_replace() -> None:
        # The general config generation advances only after cache publication
        # succeeds. Publishing the committed capture owner at the still-current
        # generation keeps it authoritative if that later step fails; on
        # success, the generation bump makes the next read rebuild from the new
        # canonical snapshot. This callback runs while the config write lock is
        # still held, so a newer writer cannot be overwritten afterward.
        _publish_runtime_capture_policy(enabled, detail, expected_generation)

    privacy_safe = not enabled or detail is CaptureDetail.SAFE
    result = apply_settings_mutation_to_cli_config(
        {
            "console": {
                "exchange_capture": enabled,
                "exchange_capture_detail": detail.value,
            }
        },
        locked_snapshot_precondition=generation_is_current,
        before_replace=publish_before_replace if privacy_safe else None,
        after_replace=None if privacy_safe else publish_after_replace,
    )
    return result


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
    """Return the validated global human-facing Console chat name.

    Args:
        default: Fallback used when ``chat_defaults.user_display_name`` is
            absent.

    Returns:
        The configured validated name, or the neutral ``"User"`` fallback
        when the configured/default value is invalid.
    """
    # Import lazily because ``tldw_chatbook.Chat.__init__`` reaches runtime
    # policy modules that import this configuration module during startup.
    from tldw_chatbook.Chat.console_roleplay_identity import (
        ChatDisplayNameError,
        normalize_chat_display_name,
    )

    config = load_cli_config_and_ensure_existence()
    chat_defaults = config.get("chat_defaults")
    raw_value = (
        chat_defaults.get("user_display_name", default)
        if isinstance(chat_defaults, dict)
        else default
    )
    try:
        return normalize_chat_display_name(raw_value, blank_means_none=False) or "User"
    except ChatDisplayNameError:
        logger.warning(
            "Invalid chat display name in [chat_defaults]; using the neutral default."
        )
        return "User"


def get_rag_citation_canonical_writes_enabled() -> bool:
    """Return the typed, fail-closed canonical citation write switch.

    Returns:
        ``True`` when canonical citation writes are explicitly enabled.
    """

    config = load_cli_config_and_ensure_existence()
    section = config.get("rag_citations")
    if not isinstance(section, dict):
        return False
    return coerce_bool_setting(section.get("canonical_writes_enabled"), False)


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


def get_ingest_ui_style() -> str:
    """
    Get the configured UI style for media ingestion.

    Returns:
        UI style string: "simplified", "grid", "wizard", or "split"
    """
    config = load_cli_config_and_ensure_existence()
    media_ingestion_config = config.get("media_ingestion", {})

    # Get UI style from config, fall back to default
    ui_style = media_ingestion_config.get(
        "ui_style", DEFAULT_MEDIA_INGESTION_CONFIG.get("ui_style", "default")
    )

    # Validate the UI style
    valid_styles = ["default", "redesigned", "new", "grid", "wizard", "split"]
    if ui_style not in valid_styles:
        logger.warning(
            f"Invalid ingest UI style '{ui_style}', falling back to 'default'"
        )
        return "default"

    return ui_style


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
def get_cli_providers_and_models() -> Dict[str, List[str]]:
    config = load_settings()
    providers_data = config.get(
        "providers", {}
    )  # Default to empty dict if "providers" isn't there
    valid_providers: Dict[str, List[str]] = {}
    if isinstance(providers_data, dict):
        for provider, models in providers_data.items():
            if isinstance(models, list) and all(isinstance(m, str) for m in models):
                valid_providers[provider] = models
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

    for section_name, section_value in config.items():
        if section_name.startswith("api_settings.") and isinstance(section_value, dict):
            api_key = section_value.get("api_key", "")
            # Check if API key exists and is not a placeholder
            if api_key and not api_key.startswith("<") and not api_key.endswith(">"):
                provider_name = section_name.replace("api_settings.", "")
                providers.append(provider_name)

    return providers


def enable_config_encryption(password: str) -> bool:
    """
    Enable encryption for the config file and encrypt existing API keys.

    Args:
        password: The master password to use for encryption

    Returns:
        True if encryption was enabled successfully
    """
    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            config_data = _read_raw_cli_config_unlocked(config_path)
            encrypted_config = encrypt_api_keys_in_config(config_data, password)
            raw_written = _write_raw_cli_config_unlocked(config_path, encrypted_config)
            set_encryption_password(password)
            _publish_runtime_config_unlocked(raw_config=raw_written)

        logger.success("Config encryption enabled successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to enable config encryption: {e}")
        return False


def disable_config_encryption(password: str) -> bool:
    """
    Disable encryption for the config file and decrypt all values.

    Args:
        password: The master password to verify before disabling

    Returns:
        True if encryption was disabled successfully
    """
    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            config_data = _read_raw_cli_config_unlocked(config_path)
            encryption_config = config_data.get("encryption", {})
            if encryption_config.get("enabled", False):
                enc_module = get_encryption_module()
                password_verifier = encryption_config.get("password_verifier", "")
                if not password_verifier:
                    logger.error("No password verifier found in encryption config")
                    return False
                if not enc_module.verify_password(password, password_verifier):
                    logger.error("Invalid password provided")
                    return False

            set_encryption_password(password)
            decrypted_config = decrypt_config_section(config_data)
            decrypted_config.pop("encryption", None)
            raw_written = _write_raw_cli_config_unlocked(config_path, decrypted_config)
            clear_encryption_password()
            _publish_runtime_config_unlocked(raw_config=raw_written)

        logger.success("Config encryption disabled successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to disable config encryption: {e}")
        return False


def change_encryption_password(old_password: str, new_password: str) -> bool:
    """
    Change the encryption password.

    Args:
        old_password: The current password
        new_password: The new password to set

    Returns:
        True if password was changed successfully
    """
    try:
        config_path = get_cli_config_path()
        with _config_write_lock(config_path):
            config_data = _read_raw_cli_config_unlocked(config_path)
            encryption_config = config_data.get("encryption", {})
            if not encryption_config.get("enabled", False):
                logger.error("Encryption is not enabled")
                return False

            enc_module = get_encryption_module()
            password_verifier = encryption_config.get("password_verifier", "")
            if not password_verifier:
                logger.error("No password verifier found in encryption config")
                return False
            if not enc_module.verify_password(old_password, password_verifier):
                logger.error("Invalid current password provided")
                return False

            set_encryption_password(old_password)
            decrypted_config = decrypt_config_section(config_data)
            encrypted_config = encrypt_api_keys_in_config(
                decrypted_config,
                new_password,
            )
            raw_written = _write_raw_cli_config_unlocked(config_path, encrypted_config)
            set_encryption_password(new_password)
            _publish_runtime_config_unlocked(raw_config=raw_written)

        logger.success("Encryption password changed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to change encryption password: {e}")
        return False


# --- CLI Database and Log File Path Getters ---
BASE_DATA_DIR_CLI = Path.home() / ".local" / "share" / "tldw_cli"  # Renamed for clarity
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
    home = os.environ.get("HOME")
    base = Path(home).expanduser() if home else Path.home()
    return base / ".local" / "share" / "tldw_cli"


def get_api_key(api_name: str) -> Optional[str]:
    """
    Get API key for a given provider.

    Args:
        api_name: The API provider name (e.g., 'openai', 'anthropic', 'groq')

    Returns:
        The API key if found, None otherwise
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
            # Check environment variable first if specified
            if "api_key_env_var" in api_settings:
                env_var = api_settings["api_key_env_var"]
                env_value = os.getenv(env_var)
                if env_value:
                    return env_value

            # Fall back to config file API key
            if (
                "api_key" in api_settings
                and api_settings["api_key"] != "<API_KEY_HERE>"
            ):
                return api_settings["api_key"]
    except Exception as e:
        logger.debug(f"Error accessing api_settings for {api_name}: {e}")

    # Try the legacy approach used elsewhere in the codebase
    try:
        # This is the pattern used in other files like Summarization_General_Lib.py
        api_key = get_cli_setting("API", f"{api_name_lower}_api_key", "")
        if api_key:
            return api_key
    except Exception as e:
        logger.debug(f"Error getting API key via get_cli_setting for {api_name}: {e}")

    # Try direct environment variable access with common patterns
    env_var_names = [f"{api_name_lower.upper()}_API_KEY", f"{api_name.upper()}_API_KEY"]

    for env_var in env_var_names:
        env_value = os.getenv(env_var)
        if env_value:
            return env_value

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
    import re

    safe_user_name = re.sub(r"[^a-zA-Z0-9_-]", "_", user_name)
    return safe_user_name if safe_user_name else "default_user"


def get_user_data_dir() -> Path:
    """Return the secured lexical user-specific data directory."""
    user_folder = get_user_folder_name()
    configured_data_dir = get_cli_setting("paths", "data_dir", None)
    if configured_data_dir is None:
        configured_data_dir = get_cli_setting("Paths", "data_dir", None)
    if configured_data_dir:
        base_data_dir = lexical_path(configured_data_dir)
        verify_trusted_directory(base_data_dir, allow_shared_sticky=False)
    else:
        base_data_dir = secure_private_directory(
            _default_base_data_dir(),
            create=True,
            application_owned=True,
        ).lexical_path
    user_dir = base_data_dir / user_folder
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
    if not custom_path or custom_path == default_path:
        return None
    selected_input = Path(str(custom_path))
    if expand_before_validation:
        selected_input = selected_input.expanduser()
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
        return get_user_data_dir() / "tldw_chatbook_ChaChaNotes.db"
    return (
        _get_custom_database_path("chachanotes_db_path")
        or get_user_data_dir() / "tldw_chatbook_ChaChaNotes.db"
    )


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
    return get_user_data_dir() / "tldw_chatbook_tts_profiles.db"


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
NOTES_SYNC_WATCHER_MAX_INTERVAL_SECONDS_DEFAULT = 10.0
_NOTES_SYNC_WATCHER_INTERVAL_CEILING_SECONDS = 3600.0


def get_notes_sync_watcher_intervals(
    config_data: Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    """Return the notes-sync watcher's (base, max) polling intervals.

    TASK-21112: the lasting-sync watcher polls at the base interval and backs
    off toward the max while roots are quiet. Read from
    ``[notes] sync_watcher_interval_seconds`` (default 1.0) and
    ``[notes] sync_watcher_max_interval_seconds`` (default 10.0; backed-off
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
        return get_user_data_dir() / "tldw_chatbook_prompts.db"
    return (
        _get_custom_database_path("prompts_db_path")
        or get_user_data_dir() / "tldw_chatbook_prompts.db"
    )


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
        return get_user_data_dir() / "tldw_chatbook_media_v2.db"
    return (
        _get_custom_database_path("media_db_path")
        or get_user_data_dir() / "tldw_chatbook_media_v2.db"
    )


def get_library_collections_db_path() -> Path:
    return (
        _get_custom_database_path("library_collections_db_path")
        or get_user_data_dir() / "tldw_chatbook_library_collections.db"
    )


def get_library_ingest_jobs_db_path() -> Path:
    return (
        _get_custom_database_path("library_ingest_jobs_db_path")
        or get_user_data_dir() / "tldw_chatbook_library_ingest_jobs.db"
    )


def get_workspaces_db_path() -> Path:
    return (
        _get_custom_database_path("workspaces_db_path")
        or get_user_data_dir() / "tldw_chatbook_workspaces.db"
    )


def get_subscriptions_db_path() -> Path:
    return (
        _get_custom_database_path("subscriptions_db_path")
        or get_user_data_dir() / "tldw_chatbook_subscriptions.db"
    )


def get_evals_db_path() -> Path:
    """Return the canonical path for the Evals database."""
    return (
        _get_custom_database_path("evals_db_path") or get_user_data_dir() / "evals.db"
    )


def get_rag_indexing_db_path() -> Path:
    """Return the canonical path for the RAG indexing-state database."""
    return (
        _get_custom_database_path("rag_indexing_db_path")
        or get_user_data_dir() / "rag_indexing.db"
    )


def get_notifications_db_path() -> Path:
    return (
        _get_custom_database_path("notifications_db_path")
        or get_user_data_dir() / "tldw_chatbook_notifications.db"
    )


def get_research_db_path() -> Path:
    return (
        _get_custom_database_path("research_db_path")
        or get_user_data_dir() / "tldw_chatbook_research.db"
    )


def get_writing_db_path() -> Path:
    return (
        _get_custom_database_path("writing_db_path")
        or get_user_data_dir() / "tldw_chatbook_writing.db"
    )


def get_scheduled_tasks_db_path() -> Path:
    return (
        _get_custom_database_path(
            "scheduled_tasks_db_path",
            expand_before_validation=False,
        )
        or get_user_data_dir() / "tldw_chatbook_scheduled_tasks.db"
    )


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


def get_model_cache_dir() -> Path:
    """Get the user-specific model cache directory for embeddings."""
    # Check if a custom cache dir is configured
    default_cache_dir = DEFAULT_CONFIG_FROM_TOML.get("embedding_config", {}).get(
        "model_cache_dir", None
    )
    custom_cache_dir = get_cli_setting(
        "embedding_config", "model_cache_dir", default_cache_dir
    )

    if custom_cache_dir and custom_cache_dir != default_cache_dir:
        # Use custom path if explicitly configured
        cache_path = Path(custom_cache_dir).expanduser().resolve()
    else:
        # Use user-specific folder
        user_dir = get_user_data_dir()
        cache_path = user_dir / "models" / "embeddings"

    # Create directory if it doesn't exist
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
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
_cloud_provider_keys = [
    "OpenAI",
    "Anthropic",
    "Cohere",
    "DeepSeek",
    "Groq",
    "Google",
    "HuggingFace",
    "MistralAI",
    "Moonshot",
    "OpenRouter",
    "QwenCloud",
    "ZAI",
]  # Example list

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
load_cli_config_and_ensure_existence()
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
