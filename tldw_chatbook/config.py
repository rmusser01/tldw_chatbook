# tldw_cli/config.py
# Description: Configuration management for the tldw_cli application.
#
# Imports
import copy
import json
import sys
if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib
import os
from pathlib import Path
import toml
from typing import Dict, Any, List, Optional
#
# Third-Party Imports
from loguru import logger
#
# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, SchemaError as ChaChaSchemaError, ConflictError as ChaChaConflictError
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase, DatabaseError as MediaDBError, SchemaError as MediaSchemaError, ConflictError as MediaConflictError
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase, DatabaseError as PromptsDBError, SchemaError as PromptsSchemaError, ConflictError as PromptsConflictError
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

# --- Encryption support ---
_ENCRYPTION_PASSWORD = None  # Cached password for the session
_ENCRYPTION_MODULE = None    # Lazily loaded encryption module

# --- Chunking Settings (Default, can be overridden by TOML) ---
global_default_chunk_language = "en"

# --- Default Fallback Configurations (if not found in TOML) ---
# These will be populated from TOML or use these hardcoded dicts as fallbacks.
DEFAULT_APP_TTS_CONFIG = {
    "OPENAI_API_KEY_fallback": "sk-...", # Note: API keys should primarily come from [API] section or ENV
    "KOKORO_ONNX_MODEL_PATH_DEFAULT": "path/to/your/downloaded/kokoro-v0_19.onnx",
    "KOKORO_ONNX_VOICES_JSON_DEFAULT": "path/to/your/downloaded/voices.json",
    "KOKORO_DEVICE_DEFAULT": "cpu", # or "cuda"
    "ELEVENLABS_API_KEY_fallback": "el-...", # Note: API keys should primarily come from [API] section or ENV
    "local_kokoro_default_onnx": {
        "KOKORO_DEVICE": "cuda:0"
    },
    "global_tts_settings": {
        # shared settings
    }
}

DEFAULT_DATABASE_CONFIG = {} # Example, can be populated if needed

DEFAULT_RAG_SEARCH_CONFIG = {
    # Legacy settings for backwards compatibility
    "fts_top_k": 10,
    "vector_top_k": 10,
    "web_vector_top_k": 10,
    "llm_context_document_limit": 10,
    "chat_context_limit": 10,
    
    # New comprehensive RAG settings
    "retriever": {
        "fts_top_k": 10,
        "vector_top_k": 10,
        "hybrid_alpha": 0.5,
        "chunk_size": 512,
        "chunk_overlap": 128,
        "media_collection": "media_embeddings",
        "chat_collection": "chat_embeddings",
        "notes_collection": "notes_embeddings",
        "character_collection": "character_embeddings"
    },
    "processor": {
        "enable_reranking": True,
        "reranker_model": None,
        "reranker_top_k": 5,
        "deduplication_threshold": 0.85,
        "max_context_length": 4096,
        "combination_method": "weighted"
    },
    "generator": {
        "default_model": None,
        "default_temperature": 0.7,
        "max_tokens": 1024,
        "enable_streaming": True,
        "stream_chunk_size": 10
    },
    "chroma": {
        "persist_directory": None,
        "collection_prefix": "tldw_rag",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "distance_metric": "cosine"
    },
    "cache": {
        "enable_cache": True,
        "cache_ttl": 3600,
        "max_cache_size": 1000,
        "cache_embedding_results": True,
        "cache_search_results": True,
        "cache_llm_responses": False
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
        "cleanup_confirmation_required": False
    }
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
        "ocr_confidence_threshold": 0.8  # Minimum confidence score
    },
    "ebook": {
        "chunk_method": "ebook_chapters",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": ""
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
        "ocr_confidence_threshold": 0.8  # Minimum confidence score
    },
    "plaintext": {
        "chunk_method": "paragraphs",
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": ""
    },
    "web_article": {
        "chunk_method": "paragraphs",
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": ""
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
        "diarize": False
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
        "extract_audio_only": True
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
        "max_image_size": 4096  # Max dimension in pixels
    }
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
        "openai_api_key": "123"
    },
    "tesseract": {
        "config": "",  # Additional tesseract config options
        "lang": "eng"  # Default language
    },
    "easyocr": {
        "use_gpu": True,
        "languages": ["en"]
    },
    "paddleocr": {
        "use_gpu": True,
        "lang": "en"
    }
}

def load_openai_mappings() -> Dict:
    current_file_path = Path(__file__).resolve()
    api_component_root = current_file_path.parent.parent.parent
    mapping_path = api_component_root / "Config_Files" / "openai_tts_mappings.json"
    logger.info(f"Attempting to load OpenAI TTS mappings from: {str(mapping_path)}")
    try:
        with open(mapping_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OpenAI TTS mappings from {mapping_path}: {e}", exc_info=True)
        return {
            "models": {"tts-1": "openai_official_tts-1"},
            "voices": {"alloy": "alloy"}
        }

_openai_mappings = load_openai_mappings()

# This hardcoded mapping can also be moved to TOML or be a fallback for the JSON loaded one
openai_tts_mappings = {
    "models": {
        "tts-1": "openai_official_tts-1",
        "tts-1-hd": "openai_official_tts-1-hd",
        "eleven_monolingual_v1": "elevenlabs_english_v1",
        "kokoro": "local_kokoro_default_onnx"
    },
    "voices": {
        "alloy": "alloy", "echo": "echo", "fable": "fable",
        "onyx": "onyx", "nova": "nova", "shimmer": "shimmer",
        "RachelEL": "21m00Tcm4TlvDq8ikWAM",
        "k_bella": "af_bella",
        "k_adam" : "am_v0adam"
    }
}
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
    """Set the encryption password for the current session."""
    global _ENCRYPTION_PASSWORD
    _ENCRYPTION_PASSWORD = password
    logger.info("Encryption password set for current session")


def get_encryption_password() -> Optional[str]:
    """Get the encryption password for the current session."""
    return _ENCRYPTION_PASSWORD


def clear_encryption_password():
    """Clear the encryption password from memory."""
    global _ENCRYPTION_PASSWORD
    _ENCRYPTION_PASSWORD = None
    logger.info("Encryption password cleared from memory")


def decrypt_config_section(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decrypt encrypted values in the config if encryption is enabled.
    
    Args:
        config_data: The config dictionary potentially containing encrypted values
        
    Returns:
        Config dictionary with decrypted values
    """
    # Check if encryption is enabled
    encryption_config = config_data.get("encryption", {})
    if not encryption_config.get("enabled", False):
        return config_data
    
    password = get_encryption_password()
    if not password:
        logger.warning("Encryption is enabled but no password is set. Cannot decrypt config.")
        return config_data
    
    salt_b64 = encryption_config.get("salt")
    if not salt_b64:
        logger.error("Encryption is enabled but no salt found in config.")
        return config_data
    
    try:
        import base64
        salt = base64.b64decode(salt_b64)
        enc_module = get_encryption_module()
        
        # Decrypt api_settings sections
        decrypted_config = copy.deepcopy(config_data)
        for section_name, section_value in config_data.items():
            if section_name.startswith('api_settings.') and isinstance(section_value, dict):
                decrypted_section = enc_module.decrypt_config_section(section_value, password, salt)
                decrypted_config[section_name] = decrypted_section
        
        return decrypted_config
    except Exception as e:
        logger.error(f"Failed to decrypt config: {e}")
        return config_data


def encrypt_api_keys_in_config(config_data: Dict[str, Any], password: str) -> Dict[str, Any]:
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
    
    # Generate salt if not present
    encryption_config = encrypted_config.get("encryption", {})
    if not encryption_config.get("salt"):
        salt = enc_module.generate_salt()
        import base64
        encryption_config["salt"] = base64.b64encode(salt).decode('utf-8')
    else:
        import base64
        salt = base64.b64decode(encryption_config["salt"])
    
    # Set encryption metadata
    encryption_config["enabled"] = True
    encryption_config["method"] = "AES-256-CBC"
    encryption_config["password_hash"] = enc_module.hash_password(password)
    encrypted_config["encryption"] = encryption_config
    
    # Encrypt api_settings sections
    for section_name, section_value in config_data.items():
        if section_name.startswith('api_settings.') and isinstance(section_value, dict):
            encrypted_section, _ = enc_module.encrypt_config_section(section_value, password, salt)
            encrypted_config[section_name] = encrypted_section
    
    return encrypted_config


def _get_typed_value(data_dict: Dict, key: str, default: Any, target_type: type = str) -> Any:
    """Helper to get value from dict and cast to type, with logging for type errors."""
    value = data_dict.get(key, default)
    if value is default and default is not None : # if value is the default, it's already typed
        return value
    if value is None: # If key is missing and default is None
        return None

    try:
        if target_type == bool:
            if isinstance(value, bool):
                return value
            # For bools from TOML strings (shouldn't happen if TOML is well-formed)
            return str(value).lower() in ['true', '1', 't', 'y', 'yes']
        if target_type == Path:
             return Path(value) if value else default
        return target_type(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Config key '{key}' has value '{value}' which could not be converted to {target_type}. Using default: '{default}'. Error: {e}")
        return default

# Global cache for load_settings to avoid redundant file I/O
_SETTINGS_CACHE: Optional[Dict[str, Any]] = None
_SETTINGS_CACHE_LOCK = None  # Will be initialized when needed

def load_settings(force_reload: bool = False) -> Dict:
    """
    Loads all settings from TOML config files, environment variables, or defaults into a dictionary.
    It first loads a base config (e.g., server-local), then attempts to load a user-specific
    CLI config which can override or extend the base settings.
    
    Args:
        force_reload: If True, bypasses the cache and reloads from disk.
    
    Returns:
        Dictionary containing all configuration settings.
    """
    global _SETTINGS_CACHE, _SETTINGS_CACHE_LOCK
    
    # Initialize lock on first use to avoid import issues
    if _SETTINGS_CACHE_LOCK is None:
        import threading
        _SETTINGS_CACHE_LOCK = threading.Lock()
    
    # Thread-safe cache check
    with _SETTINGS_CACHE_LOCK:
        if _SETTINGS_CACHE is not None and not force_reload:
            logger.debug("load_settings: Returning cached configuration (cache hit)")
            return _SETTINGS_CACHE

    current_file_path = Path(__file__).resolve()
    # config.py is in project_root/tldw_server_api/app/core/config.py
    ACTUAL_PROJECT_ROOT = current_file_path.parent # /project_root/
    APP_COMPONENT_ROOT = current_file_path.parent # /project_root/tldw_server_api/
    logger.info(f"Determined ACTUAL_PROJECT_ROOT for general paths: {ACTUAL_PROJECT_ROOT}")
    logger.info(f"Determined APP_COMPONENT_ROOT for config files: {APP_COMPONENT_ROOT}")

    # --- Load Comprehensive Config from TOML ---
    base_config_data = {}
    user_override_config_data = {}

    # 1. Load the primary (e.g., server/application component) config file
    # This path is assumed to be the original target for load_settings()
    primary_config_toml_path = APP_COMPONENT_ROOT / "Config_Files" / "config.toml"
    logger.info(f"Attempting to load primary TOML config from: {str(primary_config_toml_path)}")
    try:
        with open(primary_config_toml_path, "rb") as f: # Use "rb" for tomllib.load
            base_config_data = tomllib.load(f)
        logger.info(f"Successfully loaded primary TOML config from: {str(primary_config_toml_path)}")
    except FileNotFoundError:
        logger.warning(f"Primary TOML Config file not found at {primary_config_toml_path}. Proceeding without it.")
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error decoding primary TOML config file {primary_config_toml_path}: {e}. Proceeding with potentially empty base config.", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading primary TOML {primary_config_toml_path}: {e}. Proceeding with potentially empty base config.", exc_info=True)

    # 2. Load the user-specific CLI config file (as potential overrides or additions)
    # This is the path DEFAULT_CONFIG_PATH used by load_cli_config()
    user_cli_config_toml_path = Path.home() / ".config" / "tldw_cli" / "config.toml"
    logger.info(f"Attempting to load user-specific CLI TOML config for overrides from: {str(user_cli_config_toml_path)}")
    if user_cli_config_toml_path.exists():
        try:
            with open(user_cli_config_toml_path, "rb") as f: # Use "rb" for tomllib.load
                user_override_config_data = tomllib.load(f)
            logger.info(f"Successfully loaded user-specific CLI TOML config from: {str(user_cli_config_toml_path)}")
        except tomllib.TOMLDecodeError as e:
            logger.error(f"Error decoding user-specific CLI TOML config file {user_cli_config_toml_path}: {e}. User overrides will not be applied from this file.", exc_info=True)
            user_override_config_data = {} # Ensure it's empty if decode fails, to prevent merging bad data
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading user-specific CLI TOML {user_cli_config_toml_path}: {e}. User overrides will not be applied from this file.", exc_info=True)
            user_override_config_data = {} # Ensure it's empty on other errors
    else:
        logger.info(f"User-specific CLI TOML config file not found at {user_cli_config_toml_path}. No user overrides will be applied from this file.")

    # 3. Merge configs: user_override_config_data will overwrite/extend keys in base_config_data
    #    Start with DEFAULT_CONFIG_FROM_TOML as the absolute base to ensure all CLI sections are present.
    toml_config_data = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML) # Start with CLI defaults
    toml_config_data = deep_merge_dicts(toml_config_data, base_config_data) # Merge primary config
    toml_config_data = deep_merge_dicts(toml_config_data, user_override_config_data) # Merge user CLI overrides
    logger.info("Merged all configurations: CLI Defaults < Primary Config < User CLI Config.")
    logger.debug("load_settings: Configuration loaded from disk (cache miss or forced reload)")
    # logger.debug(f"Final toml_config_data after potential merge: {toml_config_data}") # Optional: for verbose debugging

    # --- Extract settings from the (potentially merged) TOML, with fallbacks ---
    # Helper to get values from specific TOML sections within the final toml_config_data
    def get_toml_section(section_name: str, default_val: Optional[Dict] = None) -> Dict:
        return toml_config_data.get(section_name, default_val if default_val is not None else {})

    api_section = get_toml_section('API') # This will now check the merged config
    # If [API] exists in user_override_config_data, it would have merged with/overridden base_config_data's [API]
    # Same applies to all other sections retrieved below.

    paths_section = get_toml_section('Paths')
    logging_section_server = get_toml_section('Logging')
    processing_section = get_toml_section('Processing')
    chunking_section = get_toml_section('Chunking')
    embeddings_section = get_toml_section('Embeddings')
    embedding_config_section = get_toml_section('embedding_config')  # Get the [embedding_config] table
    chat_dicts_section = get_toml_section('ChatDictionaries')
    auto_save_section = get_toml_section('AutoSave')
    stt_settings_section = get_toml_section('STTSettings')
    tts_settings_section = get_toml_section('TTSSettings')
    search_engines_section = get_toml_section('SearchEngines')
    search_settings_section = get_toml_section('SearchSettings')
    web_scraper_section = get_toml_section('WebScraper')
    confluence_section = get_toml_section('Confluence')
    file_validation_section = get_toml_section('FileValidation')
    providers_section_from_toml = get_toml_section('providers')  # Get the [providers] table

    final_api_settings = get_toml_section('api_settings')
    final_logging_settings = get_toml_section('logging')
    final_providers_settings = get_toml_section('providers')
    final_general_settings_cli = get_toml_section('general')
    final_database_settings_cli = get_toml_section('database')
    final_chat_defaults_cli = get_toml_section('chat_defaults')
    final_character_defaults_cli = get_toml_section('character_defaults')
    final_notes_settings_cli = get_toml_section('notes')

    # --- Application Mode ---
    single_user_mode_str = os.getenv("APP_MODE", _get_typed_value(processing_section, "app_mode", "single")).lower()
    single_user_mode = single_user_mode_str != "multi"

    # --- Single-User Settings ---
    single_user_fixed_id = int(os.getenv("SINGLE_USER_FIXED_ID", _get_typed_value(processing_section, "single_user_fixed_id", "0", int)))
    single_user_api_key = os.getenv("API_KEY", _get_typed_value(api_section, "single_user_api_key", "default-secret-key-for-single-user"))

    # --- Paths ---
    api_section_legacy = get_toml_section('API')  # For legacy direct API key access if any
    paths_section_legacy = get_toml_section('Paths')
    processing_section_legacy = get_toml_section('Processing')
    chunking_section_legacy = get_toml_section('Chunking')

    # --- User Name ---
    default_users_name_fallback = "default_user"
    users_name_from_toml_general = _get_typed_value(final_general_settings_cli, "users_name",
                                                    default_users_name_fallback, str)
    users_name = os.getenv("USERS_NAME", users_name_from_toml_general)

    users_db_configured = os.getenv("USERS_DB_ENABLED", _get_typed_value(processing_section, "users_db_enabled", "false", str)).lower() == "true"
    log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_toml = _get_typed_value(logging_section_server, "log_level", log_level_env, str).upper()



    # --- Load specific configurations from TOML or use defaults ---
    app_tts_config = get_toml_section('AppTTSConfig') # For APP_CONFIG related values
    app_database_config = get_toml_section('AppDatabaseConfig') # For DATABASE_CONFIG
    app_rag_search_config = get_toml_section('AppRAGSearchConfig') # For RAG_SEARCH_CONFIG

    # API Keys (Prioritize ENV, then TOML, then None)
    def get_api_key(toml_key: str, env_var: str, section: Dict = api_section_legacy) -> Optional[str]:
        return os.getenv(env_var, section.get(toml_key))

    openai_api_key = get_api_key('openai_api_key', 'OPENAI_API_KEY')
    anthropic_api_key = get_api_key('anthropic_api_key', 'ANTHROPIC_API_KEY')
    cohere_api_key = get_api_key('cohere_api_key', 'COHERE_API_KEY')
    groq_api_key = get_api_key('groq_api_key', 'GROQ_API_KEY')
    huggingface_api_key = get_api_key('huggingface_api_key', 'HUGGINGFACE_API_KEY')
    openrouter_api_key = get_api_key('openrouter_api_key', 'OPENROUTER_API_KEY')
    deepseek_api_key = get_api_key('deepseek_api_key', 'DEEPSEEK_API_KEY')
    mistral_api_key = get_api_key('mistral_api_key', 'MISTRAL_API_KEY')
    google_api_key = get_api_key('google_api_key', 'GOOGLE_API_KEY')
    elevenlabs_api_key = get_api_key('elevenlabs_api_key', 'ELEVENLABS_API_KEY')


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
        "chat_defaults": final_chat_defaults_cli,
        "character_defaults": final_character_defaults_cli,
        "notes": final_notes_settings_cli,  # For notes auto-save settings

        # Single User
        "SINGLE_USER_FIXED_ID": single_user_fixed_id,

        # Auth
        "SINGLE_USER_API_KEY": get_api_key("single_user_api_key", "API_KEY", section=api_section_legacy) or "default-secret-key-for-single-user",
        "DATABASE_URL": os.getenv("DATABASE_URL", paths_section_legacy.get("database_url",
                                                                           f"sqlite:///{ACTUAL_PROJECT_ROOT / 'user_databases' / 'single_user' / 'tldw.db'}")),
        "USERS_DB_CONFIGURED": users_db_configured,

        # --- Configurations migrated from load_and_log_configs ---
        "anthropic_api": {
            'api_key': anthropic_api_key,
            'model': api_section_legacy.get('anthropic_model', 'claude-3-5-sonnet-20240620'),
            'streaming': api_section_legacy.get("anthropic_streaming", False),
            'temperature': api_section_legacy.get('anthropic_temperature', 0.7),
            'top_p': api_section_legacy.get('anthropic_top_p', 0.95),
            'top_k': api_section_legacy.get('anthropic_top_k', 100),
            'max_tokens': api_section_legacy.get('anthropic_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('anthropic_api_timeout', 90),
            'api_retries': api_section_legacy.get('anthropic_api_retry', 3), # Key name consistency
            'api_retry_delay': api_section_legacy.get('anthropic_api_retry_delay', 5)
        },
        "cohere_api": {
            'api_key': cohere_api_key,
            'model': api_section_legacy.get('cohere_model', 'command-r-plus'),
            'streaming': api_section_legacy.get('cohere_streaming', False),
            'temperature': api_section_legacy.get('cohere_temperature', 0.7),
            'max_p': api_section_legacy.get('cohere_max_p', 0.95), # Note: check param name, Cohere might use 'p' or 'top_p'
            'top_k': api_section_legacy.get('cohere_top_k', 100),
            'max_tokens': api_section_legacy.get('cohere_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('cohere_api_timeout', 90),
            'api_retries': api_section_legacy.get('cohere_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('cohere_api_retry_delay', 5)
        },
        "deepseek_api": {
            'api_key': deepseek_api_key,
            'model': api_section_legacy.get('deepseek_model', 'deepseek-chat'),
            'streaming': api_section_legacy.get('deepseek_streaming', False),
            'temperature': api_section_legacy.get('deepseek_temperature', 0.7),
            'top_p': api_section_legacy.get('deepseek_top_p', 0.95),
            'min_p': api_section_legacy.get('deepseek_min_p', 0.05),
            'max_tokens': api_section_legacy.get('deepseek_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('deepseek_api_timeout', 90),
            'api_retries': api_section_legacy.get('deepseek_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('deepseek_api_retry_delay', 5)
        },
        "google_generative_api": { # Renamed to avoid confusion with Google Search API
            'api_key': google_api_key,
            'model': api_section_legacy.get('google_model', 'gemini-2.5-pro'),
            'streaming': api_section_legacy.get('google_streaming', False),
            'temperature': api_section_legacy.get('google_temperature', 0.7),
            'top_p': api_section_legacy.get('google_top_p', 0.95),
            'min_p': api_section_legacy.get('google_min_p', 0.05), # Check if 'min_p' is valid for Gemini
            'max_tokens': api_section_legacy.get('google_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('google_api_timeout', 90),
            'api_retries': api_section_legacy.get('google_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('google_api_retry_delay', 5)
        },
        "groq_api": {
            'api_key': groq_api_key,
            'model': api_section_legacy.get('groq_model', 'llama3-70b-8192'),
            'streaming': api_section_legacy.get('groq_streaming', False),
            'temperature': api_section_legacy.get('groq_temperature', 0.7),
            'top_p': api_section_legacy.get('groq_top_p', 0.95),
            'max_tokens': api_section_legacy.get('groq_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('groq_api_timeout', 90),
            'api_retries': api_section_legacy.get('groq_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('groq_api_retry_delay', 5)
        },
        "huggingface_api": {
            'api_key': huggingface_api_key,
            'huggingface_use_router_url_format': api_section_legacy.get('huggingface_use_router_url_format', False),
            'huggingface_router_base_url': api_section_legacy.get('huggingface_router_base_url', 'https://router.huggingface.co/hf-inference'),
            'api_base_url': api_section_legacy.get('huggingface_api_base_url', 'https://router.huggingface.co/hf-inference/models'), # Redundant if router_base_url is used for construction
            'model': api_section_legacy.get('huggingface_model', '/Qwen/Qwen3-235B-A22B'),
            'streaming': api_section_legacy.get('huggingface_streaming', False),
            'temperature': api_section_legacy.get('huggingface_temperature', 0.7),
            'top_p': api_section_legacy.get('huggingface_top_p', 0.95),
            'min_p': api_section_legacy.get('huggingface_min_p', 0.05),
            'max_tokens': api_section_legacy.get('huggingface_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('huggingface_api_timeout', 90),
            'api_retries': api_section_legacy.get('huggingface_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('huggingface_api_retry_delay', 5)
        },
        "mistral_api": {
            'api_key': mistral_api_key,
            'model': api_section_legacy.get('mistral_model', 'mistral-large-latest'),
            'streaming': api_section_legacy.get('mistral_streaming', False),
            'temperature': api_section_legacy.get('mistral_temperature', 0.7),
            'top_p': api_section_legacy.get('mistral_top_p', 0.95),
            'max_tokens': api_section_legacy.get('mistral_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('mistral_api_timeout', 90),
            'api_retries': api_section_legacy.get('mistral_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('mistral_api_retry_delay', 5)
        },
        "openrouter_api": {
            'api_key': openrouter_api_key,
            'model': api_section_legacy.get('openrouter_model', 'microsoft/wizardlm-2-8x22b'),
            'streaming': api_section_legacy.get('openrouter_streaming', False),
            'temperature': api_section_legacy.get('openrouter_temperature', 0.7),
            'top_p': api_section_legacy.get('openrouter_top_p', 0.95),
            'min_p': api_section_legacy.get('openrouter_min_p', 0.05),
            'top_k': api_section_legacy.get('openrouter_top_k', 100),
            'max_tokens': api_section_legacy.get('openrouter_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('openrouter_api_timeout', 90),
            'api_retries': api_section_legacy.get('openrouter_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('openrouter_api_retry_delay', 5)
        },
        "openai_api": { # OpenAI specific model params, API key is separate
            'api_key': openai_api_key, # This is now the primary OpenAI API key
            'model': api_section_legacy.get('openai_model', 'gpt-4o'),
            'streaming': api_section_legacy.get('openai_streaming', False),
            'temperature': api_section_legacy.get('openai_temperature', 0.7),
            'top_p': api_section_legacy.get('openai_top_p', 0.95),
            'max_tokens': api_section_legacy.get('openai_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('openai_api_timeout', 90),
            'api_retries': api_section_legacy.get('openai_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('openai_api_retry_delay', 5)
        },
        "elevenlabs_api": { # Primarily for the API key, other settings in TTS
            'api_key': elevenlabs_api_key,
        },
        # Local APIs from LocalAPI section
        "kobold_api": {
            'api_ip': api_section_legacy.get('kobold_api_IP', 'http://127.0.0.1:5000/api/v1/generate'),
            'api_streaming_ip': api_section_legacy.get('kobold_openai_api_IP', 'http://127.0.0.1:5001/v1/chat/completions'),
            'api_key': api_section_legacy.get('kobold_api_key', ''),
            'streaming': api_section_legacy.get('kobold_streaming', False),
            'temperature': api_section_legacy.get('kobold_temperature', 0.7),
            'top_p': api_section_legacy.get('kobold_top_p', 0.95),
            'top_k': api_section_legacy.get('kobold_top_k', 100),
            'max_tokens': api_section_legacy.get('kobold_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('kobold_api_timeout', 90),
            'api_retries': api_section_legacy.get('kobold_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('kobold_api_retry_delay', 5)
        },
        "llama_cpp_api": { # Renamed for clarity, assuming llama.cpp server
            'api_ip': api_section_legacy.get('llama_api_IP', 'http://127.0.0.1:8080/v1/chat/completions'),
            'api_key': api_section_legacy.get('llama_api_key', ''),
            'streaming': api_section_legacy.get('llama_streaming', False),
            'temperature': api_section_legacy.get('llama_temperature', 0.7),
            'top_p': api_section_legacy.get('llama_top_p', 0.95),
            'min_p': api_section_legacy.get('llama_min_p', 0.05),
            'top_k': api_section_legacy.get('llama_top_k', 100),
            'max_tokens': api_section_legacy.get('llama_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('llama_api_timeout', 90),
            'api_retries': api_section_legacy.get('llama_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('llama_api_retry_delay', 5)
        },
        "ooba_api": {
            'api_ip': api_section_legacy.get('ooba_api_IP', 'http://127.0.0.1:5000/v1/chat/completions'),
            'api_key': api_section_legacy.get('ooba_api_key', ''),
            'streaming': api_section_legacy.get('ooba_streaming', False),
            'temperature': api_section_legacy.get('ooba_temperature', 0.7),
            'top_p': api_section_legacy.get('ooba_top_p', 0.95),
            'min_p': api_section_legacy.get('ooba_min_p', 0.05),
            'top_k': api_section_legacy.get('ooba_top_k', 100),
            'max_tokens': api_section_legacy.get('ooba_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('ooba_api_timeout', 90),
            'api_retries': api_section_legacy.get('ooba_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('ooba_api_retry_delay', 5)
        },
         "tabby_api": {
            'api_ip': api_section_legacy.get('tabby_api_IP', 'http://127.0.0.1:5000/api/v1/generate'),
            'api_key': api_section_legacy.get('tabby_api_key', None),
            'model': api_section_legacy.get('tabby_model', None), # Tabby model might be part of URL or configured in Tabby
            'streaming': api_section_legacy.get('tabby_streaming', False),
            'temperature': api_section_legacy.get('tabby_temperature', 0.7),
            'top_p': api_section_legacy.get('tabby_top_p', 0.95),
            'top_k': api_section_legacy.get('tabby_top_k', 100),
            'min_p': api_section_legacy.get('tabby_min_p', 0.05),
            'max_tokens': api_section_legacy.get('tabby_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('tabby_api_timeout', 90),
            'api_retries': api_section_legacy.get('tabby_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('tabby_api_retry_delay', 5)
        },
        "vllm_api": {
            'api_ip': api_section_legacy.get('vllm_api_IP', 'http://127.0.0.1:5000/v1/chat/completions'), # Corrected key
            'api_key': api_section_legacy.get('vllm_api_key', None),
            'model': api_section_legacy.get('vllm_model', None),
            'streaming': api_section_legacy.get('vllm_streaming', False),
            'temperature': api_section_legacy.get('vllm_temperature', 0.7),
            'top_p': api_section_legacy.get('vllm_top_p', 0.95),
            'top_k': api_section_legacy.get('vllm_top_k', 100),
            'min_p': api_section_legacy.get('vllm_min_p', 0.05),
            'max_tokens': api_section_legacy.get('vllm_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('vllm_api_timeout', 90),
            'api_retries': api_section_legacy.get('vllm_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('vllm_api_retry_delay', 5)
        },
        "ollama_api": {
            'api_url': api_section_legacy.get('ollama_api_IP', 'http://127.0.0.1:11434/api/generate'), # ollama_api_url or IP
            'api_key': api_section_legacy.get('ollama_api_key', None), # Ollama doesn't typically use API keys
            'model': api_section_legacy.get('ollama_model', None),
            'streaming': api_section_legacy.get('ollama_streaming', False),
            'temperature': api_section_legacy.get('ollama_temperature', 0.7),
            'top_p': api_section_legacy.get('ollama_top_p', 0.95),
            'max_tokens': api_section_legacy.get('ollama_max_tokens', 4096), # Ollama might handle max_tokens differently (num_predict)
            'api_timeout': api_section_legacy.get('ollama_api_timeout', 90),
            'api_retries': api_section_legacy.get('ollama_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('ollama_api_retry_delay', 5)
        },
        "aphrodite_api": {
            'api_ip': api_section_legacy.get('aphrodite_api_IP', 'http://127.0.0.1:8080/v1/chat/completions'),
            'api_key': api_section_legacy.get('aphrodite_api_key', ''),
            'model': api_section_legacy.get('aphrodite_model', ''),
            'max_tokens': api_section_legacy.get('aphrodite_max_tokens', 4096),
            'streaming': api_section_legacy.get('aphrodite_streaming', False),
            'api_timeout': api_section_legacy.get('aphrodite_api_timeout', 90), # Original used llama_api_timeout
            'api_retries': api_section_legacy.get('aphrodite_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('aphrodite_api_retry_delay', 5)
        },
        "custom_openai_api": {
            'api_ip': api_section_legacy.get('custom_openai_api_ip', 'http://127.0.0.1:5000/v1/chat/completions'),
            'api_key': api_section_legacy.get('custom_openai_api_key', None),
            'model': api_section_legacy.get('custom_openai_api_model', None),
            'streaming': api_section_legacy.get('custom_openai_api_streaming', False),
            'temperature': api_section_legacy.get('custom_openai_api_temperature', 0.7),
            'top_p': api_section_legacy.get('custom_openai_api_top_p', 0.95),
            'min_p': api_section_legacy.get('custom_openai_api_min_p', 0.05), # Original used top_k, ensure consistency
            'max_tokens': api_section_legacy.get('custom_openai_api_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('custom_openai_api_timeout', 90),
            'api_retries': api_section_legacy.get('custom_openai_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('custom_openai_api_retry_delay', 5)
        },
        "custom_openai_api_2": { # Ensure key names are consistent e.g. custom_openai2_api_min_p
            'api_ip': api_section_legacy.get('custom_openai2_api_ip', 'http://127.0.0.1:5000/v1/chat/completions'),
            'api_key': api_section_legacy.get('custom_openai2_api_key', None),
            'model': api_section_legacy.get('custom_openai2_api_model', None),
            'streaming': api_section_legacy.get('custom_openai2_api_streaming', False),
            'temperature': api_section_legacy.get('custom_openai2_api_temperature', 0.7),
            'top_p': api_section_legacy.get('custom_openai2_api_top_p', 0.95), # original had custom_openai_api2_top_p
            'min_p': api_section_legacy.get('custom_openai2_api_min_p', 0.05), # original had custom_openai_api2_top_k
            'max_tokens': api_section_legacy.get('custom_openai2_api_max_tokens', 4096),
            'api_timeout': api_section_legacy.get('custom_openai2_api_timeout', 90),
            'api_retries': api_section_legacy.get('custom_openai2_api_retry', 3),
            'api_retry_delay': api_section_legacy.get('custom_openai2_api_retry_delay', 5)
        },
        "llm_api_settings": { # General LLM settings
            'default_api': api_section_legacy.get('default_api', 'openai'),
            'local_api_timeout': api_section_legacy.get('local_api_timeout', 90), # Note: this was also in Local-API Settings before
            'local_api_retries': api_section_legacy.get('local_api_retry', 3), # Key name consistency
            'local_api_retry_delay': api_section_legacy.get('local_api_retry_delay', 5),
        },
        "output_path": _get_typed_value(paths_section, 'output_path', 'results', Path),
        "system_preferences": {
            'save_video_transcripts': _get_typed_value(paths_section, 'save_video_transcripts', True, bool),
        },
        "processing_choice": _get_typed_value(processing_section, 'processing_choice', 'cpu'),

        "chat_dictionaries": {
            'enable_chat_dictionaries': _get_typed_value(chat_dicts_section, 'enable_chat_dictionaries', False, bool),
            'post_gen_replacement': _get_typed_value(chat_dicts_section, 'post_gen_replacement', False, bool),
            'post_gen_replacement_dict': _get_typed_value(chat_dicts_section, 'post_gen_replacement_dict', ''),
            'chat_dict_chat_prompts': _get_typed_value(chat_dicts_section, 'chat_dictionary_chat_prompts', ''),
            'chat_dict_RAG_prompts': _get_typed_value(chat_dicts_section, 'chat_dictionary_RAG_prompts', ''),
            'chat_dict_replacement_strategy': _get_typed_value(chat_dicts_section, 'chat_dictionary_replacement_strategy', 'character_lore_first'),
            'chat_dict_max_tokens': _get_typed_value(chat_dicts_section, 'chat_dictionary_max_tokens', 1000, int),
            'default_rag_prompt': _get_typed_value(chat_dicts_section, 'default_rag_prompt', ''),
            'chat_dicts_folder': ''  # Will be set dynamically below
        },
        "chunking_config": {
            # Global defaults
            'chunking_method': _get_typed_value(chunking_section, 'chunking_method', 'words'),
            'chunk_max_size': _get_typed_value(chunking_section, 'chunk_max_size', 400, int),
            'chunk_overlap': _get_typed_value(chunking_section, 'chunk_overlap', 200, int),
            'adaptive_chunking': _get_typed_value(chunking_section, 'adaptive_chunking', False, bool),
            'multi_level': _get_typed_value(chunking_section, 'chunking_multi_level', False, bool),
            'chunk_language': _get_typed_value(chunking_section, 'chunk_language', global_default_chunk_language), # Use global default
            # Per-type overrides (example for article, repeat for others: audio, book, etc.)
            'article_chunking_method': _get_typed_value(chunking_section, 'article_chunking_method', 'words'),
            'article_chunk_max_size': _get_typed_value(chunking_section, 'article_chunk_max_size', 400, int),
            'article_chunk_overlap': _get_typed_value(chunking_section, 'article_chunk_overlap', 200, int),
            'article_adaptive_chunking': _get_typed_value(chunking_section, 'article_adaptive_chunking', False, bool),
            'article_chunking_multi_level': _get_typed_value(chunking_section,'article_chunking_multi_level', False, bool),
            'article_language': _get_typed_value(chunking_section,'article_language', 'en'),
            'audio_chunking_method': _get_typed_value(chunking_section,'audio_chunking_method', 'words'),
            'audio_chunk_max_size': _get_typed_value(chunking_section,'audio_chunk_max_size', 400, int),
            'audio_chunk_overlap': _get_typed_value(chunking_section,'audio_chunk_overlap', 200, int),
            'audio_adaptive_chunking': _get_typed_value(chunking_section,'audio_adaptive_chunking', False, bool),
            'audio_chunking_multi_level': _get_typed_value(chunking_section,'audio_chunking_multi_level', False, bool),
            'audio_language': _get_typed_value(chunking_section,'audio_language', 'en'),
            'book_chunking_method': _get_typed_value(chunking_section,'book_chunking_method', 'ebook_chunk_by_chapter'),
            'book_chunk_max_size': _get_typed_value(chunking_section,'book_chunk_max_size', 400, int),
            'book_chunk_overlap': _get_typed_value(chunking_section,'book_chunk_overlap', 200, int),
            'book_adaptive_chunking': _get_typed_value(chunking_section,'book_adaptive_chunking', False, bool),
            'book_chunking_multi_level': _get_typed_value(chunking_section,'book_chunking_multi_level', False, bool),
            'book_language': _get_typed_value(chunking_section,'book_language', 'en'),
            'document_chunking_method': _get_typed_value(chunking_section,'document_chunking_method', 'words'),
            'document_chunk_max_size': _get_typed_value(chunking_section,'document_chunk_max_size', 400, int),
            'document_chunk_overlap': _get_typed_value(chunking_section,'document_chunk_overlap', 200, int),
            'document_adaptive_chunking': _get_typed_value(chunking_section,'document_adaptive_chunking', False, bool),
            'document_chunking_multi_level': _get_typed_value(chunking_section,'document_chunking_multi_level', False, bool),
            'document_language': _get_typed_value(chunking_section,'document_language', 'en'),
            'mediawiki_article_chunking_method': _get_typed_value(chunking_section,'mediawiki_article_chunking_method', 'words'),
            'mediawiki_article_chunk_max_size': _get_typed_value(chunking_section,'mediawiki_article_chunk_max_size', 400, int),
            'mediawiki_article_chunk_overlap': _get_typed_value(chunking_section,'mediawiki_article_chunk_overlap', 200, int),
            'mediawiki_article_adaptive_chunking': _get_typed_value(chunking_section,'mediawiki_article_adaptive_chunking', False, bool),
            'mediawiki_article_chunking_multi_level': _get_typed_value(chunking_section,'mediawiki_article_chunking_multi_level', False, bool),
            'mediawiki_article_language': _get_typed_value(chunking_section,'mediawiki_article_language', 'en'),
            'mediawiki_dump_chunking_method': _get_typed_value(chunking_section,'mediawiki_dump_chunking_method', 'words'),
            'mediawiki_dump_chunk_max_size': _get_typed_value(chunking_section,'mediawiki_dump_chunk_max_size', 400, int),
            'mediawiki_dump_chunk_overlap': _get_typed_value(chunking_section,'mediawiki_dump_chunk_overlap', 200, int),
            'mediawiki_dump_adaptive_chunking': _get_typed_value(chunking_section,'mediawiki_dump_adaptive_chunking', False, bool),
            'mediawiki_dump_chunking_multi_level': _get_typed_value(chunking_section,'mediawiki_dump_chunking_multi_level', False, bool),
            'mediawiki_dump_language': _get_typed_value(chunking_section,'mediawiki_dump_language', 'en'),
            'obsidian_note_chunking_method': _get_typed_value(chunking_section,'obsidian_note_chunking_method', 'words'),
            'obsidian_note_chunk_max_size': _get_typed_value(chunking_section,'obsidian_note_chunk_max_size', 400, int),
            'obsidian_note_chunk_overlap': _get_typed_value(chunking_section,'obsidian_note_chunk_overlap', 200, int),
            'obsidian_note_adaptive_chunking': _get_typed_value(chunking_section,'obsidian_note_adaptive_chunking', False, bool),
            'obsidian_note_chunking_multi_level': _get_typed_value(chunking_section,'obsidian_note_chunking_multi_level', False, bool),
            'obsidian_note_language': _get_typed_value(chunking_section,'obsidian_note_language', 'en'),
            'podcast_chunking_method': _get_typed_value(chunking_section,'podcast_chunking_method', 'sentences'),
            'podcast_chunk_max_size': _get_typed_value(chunking_section,'podcast_chunk_max_size', 300, int),
            'podcast_chunk_overlap': _get_typed_value(chunking_section,'podcast_chunk_overlap', 30, int),
            'podcast_adaptive_chunking': _get_typed_value(chunking_section,'podcast_adaptive_chunking', False, bool),
            'podcast_chunking_multi_level': _get_typed_value(chunking_section,'podcast_chunking_multi_level', False, bool),
            'podcast_language': _get_typed_value(chunking_section,'podcast_language', 'en'),
            'text_chunking_method': _get_typed_value(chunking_section,'text_chunking_method', 'words'),
            'text_chunk_max_size': _get_typed_value(chunking_section,'text_chunk_max_size', 400, int),
            'text_chunk_overlap': _get_typed_value(chunking_section,'text_chunk_overlap', 200, int),
            'text_adaptive_chunking': _get_typed_value(chunking_section,'text_adaptive_chunking', False, bool),
            'text_chunking_multi_level': _get_typed_value(chunking_section,'text_chunking_multi_level', False, bool),
            'text_language': _get_typed_value(chunking_section,'text_language', 'en'),
            'video_chunking_method': _get_typed_value(chunking_section,'video_chunking_method', 'words'),
            'video_chunk_max_size': _get_typed_value(chunking_section,'video_chunk_max_size', 400, int),
            'video_chunk_overlap': _get_typed_value(chunking_section,'video_chunk_overlap', 200, int),
            'video_adaptive_chunking': _get_typed_value(chunking_section,'video_adaptive_chunking', False, bool),
            'video_chunking_multi_level': _get_typed_value(chunking_section,'video_chunking_multi_level', False, bool),
            'video_language': _get_typed_value(chunking_section,'video_language', 'en'),
        },
        "embedding_config": {
            'embedding_provider': _get_typed_value(embeddings_section, 'embedding_provider', 'openai'),
            'embedding_model': _get_typed_value(embeddings_section, 'embedding_model', 'text-embedding-3-large'),
            'onnx_model_path': _get_typed_value(embeddings_section, 'onnx_model_path', "./Models/onnx_models/text-embedding-3-small.onnx", Path),
            'model_dir': _get_typed_value(embeddings_section, 'model_dir', "./Models", Path),
            'embedding_api_url': _get_typed_value(embeddings_section, 'embedding_api_url', "http://localhost:8080/v1/embeddings"),
            'embedding_api_key': _get_typed_value(embeddings_section, 'embedding_api_key', ''),
            'chunk_size': _get_typed_value(embeddings_section, 'chunk_size', 400, int), # This was 'chunk_size' in old Embeddings, also in Chunking
            'chunk_overlap': _get_typed_value(embeddings_section, 'overlap', 200, int), # This was 'overlap' in old Embeddings
            'models': embedding_config_section.get('models', {})  # Include the models from the embedding_config section
        },
        "auto_save": {
            'save_character_chats': _get_typed_value(auto_save_section, 'save_character_chats', False, bool),
            'save_rag_chats': _get_typed_value(auto_save_section, 'save_rag_chats', False, bool),
        },
        "STT_settings": { # Corrected key from STT-Settings
            'default_stt_provider': _get_typed_value(stt_settings_section, 'default_stt_provider', 'faster_whisper'),
        },
        "tts_settings": {
            'default_tts_provider': _get_typed_value(tts_settings_section, 'default_tts_provider', 'openai'),
            'tts_voice': _get_typed_value(tts_settings_section, 'default_tts_voice', 'shimmer'), # General default voice
            'local_tts_device': _get_typed_value(tts_settings_section, 'local_tts_device', 'cpu'),
            # OpenAI TTS
            'default_openai_tts_model': _get_typed_value(tts_settings_section, 'default_openai_tts_model', 'tts-1-hd'),
            'default_openai_tts_voice': _get_typed_value(tts_settings_section, 'default_openai_tts_voice', 'shimmer'),
            'default_openai_tts_speed': _get_typed_value(tts_settings_section, 'default_openai_tts_speed', 1.0, float),
            'default_openai_tts_output_format': _get_typed_value(tts_settings_section, 'default_openai_tts_output_format', 'mp3'),
            'default_openai_tts_streaming': _get_typed_value(tts_settings_section, 'default_openai_tts_streaming', False, bool),
             # Google TTS
            'default_google_tts_model': _get_typed_value(tts_settings_section, 'default_google_tts_model', 'en'), # FIXME: Review defaults
            'default_google_tts_voice': _get_typed_value(tts_settings_section, 'default_google_tts_voice', 'en'), # FIXME: Review defaults
            'default_google_tts_speed': _get_typed_value(tts_settings_section, 'default_google_tts_speed', 1.0, float), # FIXME: Review defaults
            # ElevenLabs TTS
            'default_eleven_tts_model': _get_typed_value(tts_settings_section, 'default_eleven_tts_model', 'eleven_multilingual_v2'), # FIXME: Placeholder
            'default_eleven_tts_voice': _get_typed_value(tts_settings_section, 'default_eleven_tts_voice', 'Rachel'), # FIXME: Placeholder
            'default_eleven_tts_language_code': _get_typed_value(tts_settings_section, 'default_eleven_tts_language_code', 'en-US'), # FIXME
            'default_eleven_tts_voice_stability': _get_typed_value(tts_settings_section, 'default_eleven_tts_voice_stability', 0.5, float), # FIXME
            'default_eleven_tts_voice_similiarity_boost': _get_typed_value(tts_settings_section, 'default_eleven_tts_voice_similiarity_boost', 0.75, float), # FIXME
            'default_eleven_tts_voice_style': _get_typed_value(tts_settings_section, 'default_eleven_tts_voice_style', 0.0, float), # FIXME
            'default_eleven_tts_voice_use_speaker_boost': _get_typed_value(tts_settings_section, 'default_eleven_tts_voice_use_speaker_boost', True, bool), # FIXME
            'default_eleven_tts_output_format': _get_typed_value(tts_settings_section, 'default_eleven_tts_output_format', 'mp3_44100_192'),
            # AllTalk TTS (from load_and_log_configs, now integrated)
            'alltalk_api_ip': _get_typed_value(tts_settings_section, 'alltalk_api_ip', 'http://127.0.0.1:7851/v1/audio/speech'),
            'default_alltalk_tts_model': _get_typed_value(tts_settings_section, 'default_alltalk_tts_model', 'alltalk_model'),
            'default_alltalk_tts_voice': _get_typed_value(tts_settings_section, 'default_alltalk_tts_voice', 'alloy'),
            'default_alltalk_tts_speed': _get_typed_value(tts_settings_section, 'default_alltalk_tts_speed', 1.0, float),
            'default_alltalk_tts_output_format': _get_typed_value(tts_settings_section, 'default_alltalk_tts_output_format', 'mp3'),
            # Kokoro TTS
            'kokoro_model_path': _get_typed_value(tts_settings_section, 'kokoro_model_path', 'Databases/kokoro_models', Path),
            'default_kokoro_tts_model': _get_typed_value(tts_settings_section, 'default_kokoro_tts_model', 'pht'),
            'default_kokoro_tts_voice': _get_typed_value(tts_settings_section, 'default_kokoro_tts_voice', 'sky'),
            'default_kokoro_tts_speed': _get_typed_value(tts_settings_section, 'default_kokoro_tts_speed', 1.0, float),
            'default_kokoro_tts_output_format': _get_typed_value(tts_settings_section, 'default_kokoro_tts_output_format', 'wav'),
            # Self-hosted OpenAI API TTS
            'default_openai_api_tts_model': _get_typed_value(tts_settings_section, 'default_openai_api_tts_model', 'tts-1-hd'),
            'default_openai_api_tts_voice': _get_typed_value(tts_settings_section, 'default_openai_api_tts_voice', 'shimmer'),
            'default_openai_api_tts_speed': _get_typed_value(tts_settings_section, 'default_openai_api_tts_speed', 1.0, float), # Was '1' string
            'default_openai_api_tts_output_format': _get_typed_value(tts_settings_section, 'default_openai_api_tts_output_format', 'mp3'), # key was default_openai_tts_api_output_format
            'default_openai_api_tts_streaming': _get_typed_value(tts_settings_section, 'default_openai_api_tts_streaming', False, bool),
        },
        "search_settings_general": { # Renamed from 'search_settings' to avoid conflict with SearchEngines section for keys
            'default_search_provider': _get_typed_value(search_settings_section, 'search_provider_default', 'google'),
            'search_language_query': _get_typed_value(search_settings_section, 'search_language_query', 'en'),
            'search_language_analysis': _get_typed_value(search_settings_section, 'search_language_analysis', 'en'),
            'search_default_max_queries': _get_typed_value(search_settings_section, 'search_default_max_queries', 5, int),
            'search_enable_subquery': _get_typed_value(search_settings_section, 'search_enable_subquery', False, bool),
            'search_enable_subquery_count_max': _get_typed_value(search_settings_section, 'search_enable_subquery_count_max', 3, int),
            'search_result_rerank': _get_typed_value(search_settings_section, 'search_result_rerank', False, bool),
            'search_result_max': _get_typed_value(search_settings_section, 'search_result_max', 10, int),
            'search_result_max_per_query': _get_typed_value(search_settings_section, 'search_result_max_per_query', 10, int),
            'search_result_blacklist': _get_typed_value(search_settings_section, 'search_result_blacklist' , ''),
            'search_result_display_type': _get_typed_value(search_settings_section, 'search_result_display_type' , 'text'),
            'search_result_display_metadata': _get_typed_value(search_settings_section, 'search_result_display_metadata' , True, bool),
            'search_result_save_to_db': _get_typed_value(search_settings_section, 'search_result_save_to_db' , True, bool),
            'search_result_analysis_tone': _get_typed_value(search_settings_section, 'search_result_analysis_tone' , 'neutral'),
            'relevance_analysis_llm': _get_typed_value(search_settings_section, 'relevance_analysis_llm' , 'openai'),
            'final_answer_llm': _get_typed_value(search_settings_section, 'final_answer_llm' , 'openai'),
        },
        "search_engine_specific_settings": {  # API Keys for various search engines from 'SearchEngines' TOML table
            'baidu_search_api_key': _get_typed_value(search_engines_section, 'baidu_search_api_key', ''),
            'bing_country_code': _get_typed_value(search_engines_section, 'bing_country_code', ''),
            'bing_search_api_url': _get_typed_value(search_engines_section, 'bing_search_api_url', ''),
            'brave_country_code': _get_typed_value(search_engines_section, 'brave_country_code', ''),
            'google_search_api_url': _get_typed_value(search_engines_section, 'google_search_api_url', ''),
            'google_search_engine_id': _get_typed_value(search_engines_section, 'google_search_engine_id', ''),
            'google_simp_trad_chinese': _get_typed_value(search_engines_section, 'google_simp_trad_chinese', False, bool),
            'limit_google_search_to_country': _get_typed_value(search_engines_section, 'limit_google_search_to_country', False, bool),
            'google_search_country': _get_typed_value(search_engines_section, 'google_search_country', ''),
            'google_search_country_code': _get_typed_value(search_engines_section, 'google_search_country_code', ''),
            'google_search_filter_setting': _get_typed_value(search_engines_section, 'google_filter_setting', ''),
            'google_user_geolocation': _get_typed_value(search_engines_section, 'google_user_geolocation', False, bool),
            'google_ui_language': _get_typed_value(search_engines_section, 'google_ui_language', ''),
            'google_limit_search_results_to_language': _get_typed_value(search_engines_section, 'google_limit_search_results_to_language', False, bool),
            'google_site_search_include': _get_typed_value(search_engines_section, 'google_site_search_include', ''),
            'google_site_search_exclude': _get_typed_value(search_engines_section, 'google_site_search_exclude', ''),
            'google_sort_results_by': _get_typed_value(search_engines_section, 'google_sort_results_by', ''),
            'google_default_search_results': _get_typed_value(search_engines_section, 'google_default_search_results', 10, int),
            'google_safe_search': _get_typed_value(search_engines_section, 'google_safe_search', False, bool),
            'google_enable_site_search': _get_typed_value(search_engines_section, 'google_enable_site_search', False, bool),
            'yandex_search_engine_id': _get_typed_value(search_engines_section, 'yandex_search_engine_id', ''),
        },
        "search_engines_keys": { # API Keys for various search engines from 'SearchEngines' TOML table
            'baidu_search_api_key': _get_typed_value(search_engines_section, 'search_engine_api_key_baidu', ''),
            'bing_search_api_key': _get_typed_value(search_engines_section, 'search_engine_api_key_bing', ''),
            'brave_search_api_key': _get_typed_value(search_engines_section, 'brave_search_api_key', ''),
            'brave_search_ai_api_key': _get_typed_value(search_engines_section, 'brave_search_ai_api_key', ''),
            'duckduckgo_search_api_key': _get_typed_value(search_engines_section, 'duckduckgo_search_api_key', ''),
            'google_search_api_key': _get_typed_value(search_engines_section, 'google_search_api_key', ''),
            'kagi_search_api_key': _get_typed_value(search_engines_section, 'kagi_search_api_key', ''),
            'searx_search_api_url': _get_typed_value(search_engines_section, 'search_engine_searx_api', ''),
            'tavily_search_api_key': _get_typed_value(search_engines_section, 'tavily_search_api_key', ''),
            'yandex_search_api_key': _get_typed_value(search_engines_section, 'yandex_search_api_key', ''),
        },
        "prompts_strings": { # Specific prompt strings from 'Prompts' TOML table
            'sub_question_generation_prompt': _get_typed_value(get_toml_section('Prompts'), 'sub_question_generation_prompt', ''),
            'search_result_relevance_eval_prompt': _get_typed_value(get_toml_section('Prompts'), 'search_result_relevance_eval_prompt', ''),
            'analyze_search_results_prompt': _get_typed_value(get_toml_section('Prompts'), 'analyze_search_results_prompt', ''),
        },
        "web_scraper_settings": {
            'web_scraper_api_key': _get_typed_value(web_scraper_section, 'web_scraper_api_key', ''),
            'web_scraper_api_url': _get_typed_value(web_scraper_section, 'web_scraper_api_url', ''),
            # ... (all web scraper settings)
        },
        "confluence": {
            'base_url': _get_typed_value(confluence_section, 'base_url', os.getenv('CONFLUENCE_BASE_URL', '')),
            'auth_method': _get_typed_value(confluence_section, 'auth_method', os.getenv('CONFLUENCE_AUTH_METHOD', 'api_token')),
            'username': _get_typed_value(confluence_section, 'username', os.getenv('CONFLUENCE_USERNAME', '')),
            'api_token': _get_typed_value(confluence_section, 'api_token', os.getenv('CONFLUENCE_API_TOKEN', '')),
            'oauth_token': _get_typed_value(confluence_section, 'oauth_token', os.getenv('CONFLUENCE_OAUTH_TOKEN', '')),
            'password': _get_typed_value(confluence_section, 'password', os.getenv('CONFLUENCE_PASSWORD', '')),
            'browser': _get_typed_value(confluence_section, 'browser', 'all'),
            'space_keys': _get_typed_value(confluence_section, 'space_keys', [], list),
            'max_pages_per_space': _get_typed_value(confluence_section, 'max_pages_per_space', 100, int),
            'max_crawl_depth': _get_typed_value(confluence_section, 'max_crawl_depth', 5, int),
            'include_attachments': _get_typed_value(confluence_section, 'include_attachments', False, bool),
            'follow_links': _get_typed_value(confluence_section, 'follow_links', False, bool),
            'rate_limit_delay': _get_typed_value(confluence_section, 'rate_limit_delay', 0.5, float),
        },

        # Configurations from hardcoded dicts (now from TOML or fallback to Python dicts)
        "APP_TTS_CONFIG": {**DEFAULT_APP_TTS_CONFIG, **app_tts_config},
        "APP_DATABASE_CONFIG": {**DEFAULT_DATABASE_CONFIG, **app_database_config},
        "APP_RAG_SEARCH_CONFIG": {**DEFAULT_RAG_SEARCH_CONFIG, **app_rag_search_config},

        "COMPREHENSIVE_CONFIG_RAW": toml_config_data, # Store the raw TOML data if needed
        "OPENAI_API_KEY": openai_api_key, # Top-level convenience access
    }

    # Populate the rest of chunking_config (tedious but necessary)
    chunking_types = ['audio', 'book', 'document', 'mediawiki_article', 'mediawiki_dump',
                      'obsidian_note', 'podcast', 'text', 'video']
    for ctype in chunking_types:
        # Use direct defaults from chunking_section or hardcoded fallbacks
        default_method = _get_typed_value(chunking_section, "chunking_method", "words")
        default_max_size = _get_typed_value(chunking_section, "chunk_max_size", 400, int)
        default_overlap = _get_typed_value(chunking_section, "chunk_overlap", 200, int)
        default_adaptive = _get_typed_value(chunking_section, "adaptive_chunking", False, bool)
        default_multi_level = _get_typed_value(chunking_section, "chunking_multi_level", False, bool)
        default_language = _get_typed_value(chunking_section, "chunk_language", global_default_chunk_language)

        # Only set if not already defined in lines 494-562
        if f"{ctype}_chunking_method" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunking_method"] = _get_typed_value(
                chunking_section, f"{ctype}_chunking_method", default_method)
        if f"{ctype}_chunk_max_size" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunk_max_size"] = _get_typed_value(
                chunking_section, f"{ctype}_chunk_max_size", default_max_size, int)
        if f"{ctype}_chunk_overlap" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunk_overlap"] = _get_typed_value(
                chunking_section, f"{ctype}_chunk_overlap", default_overlap, int)
        if f"{ctype}_adaptive_chunking" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_adaptive_chunking"] = _get_typed_value(
                chunking_section, f"{ctype}_adaptive_chunking", default_adaptive, bool)
        if f"{ctype}_chunking_multi_level" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunking_multi_level"] = _get_typed_value(
                chunking_section, f"{ctype}_chunking_multi_level", default_multi_level, bool)
        if f"{ctype}_language" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_language"] = _get_typed_value(
                chunking_section, f"{ctype}_language", default_language)


    # --- Warnings ---

    # Create necessary directories if they don't exist
    # Ensure main SQLite database directory exists
    db_url_server = config_dict.get("DATABASE_URL", "")
    if db_url_server and db_url_server.startswith("sqlite:///"):
        main_db_file_path_str_server = db_url_server.replace("sqlite:///", "")
        main_db_file_path_server = Path(main_db_file_path_str_server)
        if not main_db_file_path_server.is_absolute() and ACTUAL_PROJECT_ROOT:
            main_db_file_path_server = ACTUAL_PROJECT_ROOT / main_db_file_path_server
        try:
            main_db_file_path_server.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create server database directory {main_db_file_path_server.parent}: {e}")

    user_data_base_dir_server = config_dict.get("USER_DB_BASE_DIR")
    if user_data_base_dir_server and isinstance(user_data_base_dir_server, Path):
        try:
            user_data_base_dir_server.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create server user data base directory {user_data_base_dir_server}: {e}")
    
    # Set the chat dictionaries folder path dynamically
    from .Utils.paths import get_user_data_dir
    chat_dicts_folder = get_user_data_dir() / "chat_dicts"
    config_dict["chat_dictionaries"]["chat_dicts_folder"] = str(chat_dicts_folder)
    
    # Create the chat dictionaries folder if it doesn't exist
    try:
        chat_dicts_folder.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured chat dictionaries folder exists: {chat_dicts_folder}")
    except Exception as e:
        logger.error(f"Could not create chat dictionaries folder {chat_dicts_folder}: {e}")
    
    # Cache the configuration before returning
    with _SETTINGS_CACHE_LOCK:
        _SETTINGS_CACHE = config_dict
        logger.debug("load_settings: Configuration cached for future use")
    
    return config_dict

# --- Define API Models (Combined Cloud & Local) ---
# (Keep your existing API_MODELS_BY_PROVIDER and LOCAL_PROVIDERS dictionaries)
API_MODELS_BY_PROVIDER = {
    "OpenAI": ["gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "o3-2025-04-16", "o3-mini-2025-01-31",
               "o1-2024-12-17", "chatgpt-4o-latest", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06",
               "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14", "gpt-4o-mini-2024-07-18", ],
    "Anthropic": ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219",
                  "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20240620",
                  "claude-3-haiku-20240307", "claude-3-opus-20240229", "claude-3-sonnet-20240229",
                  "claude-2.1", "claude-2.0"],
    "Cohere": ["command-a-03-2025", "command-r7b-12-2024", "command-r-plus-04-2024", "command-r-plus",
               "command-r-08-2024", "command-r-03-2024", "command", "command-nightly", "command-light",
               "command-light-nightly"],
    "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
    "Groq": ["gemma2-9b-it", "mmeta-llama/Llama-Guard-4-12B", "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
             "llama3-70b-8192", "llama3-70b-8192", "llama3-8b-8192",],
    "Google": ["gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-05-06", "gemini-2.0-flash",
               "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", ],
    "HuggingFace": ["meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct",],
    "MistralAI": ["open-mistral-nemo", "mistral-medium-2505", "codestral-2501", "mistral-saba-2502",
                  "mistral-large-2411", "ministral-3b-2410", "ministral-8b-2410", "mistral-moderation-2411",
                  "devstral-small-2505", "mistral-small-2503", ],
    "OpenRouter": ["openai/gpt-4o-mini", "anthropic/claude-3.7-sonnet", "google/gemini-2.0-flash-001",
                   "google/gemini-2.5-pro-preview", "google/gemini-2.5-flash-preview",
                   "deepseek/deepseek-chat-v3-0324:free", "deepseek/deepseek-chat-v3-0324",
                   "openai/gpt-4.1", "anthropic/claude-sonnet-4", "deepseek/deepseek-r1:free",
                   "anthropic/claude-3.7-sonnet:thinking", "google/gemini-flash-1.5-8b",
                   "mistralai/mistral-nemo", "google/gemini-2.5-flash-preview-05-20", ],
}
LOCAL_PROVIDERS = {
    "llama_cpp": ["None"],
    "Oobabooga": ["None"],
    "koboldcpp": ["None"],
    "Ollama": ["gemma3:12b", "gemma3:4b", "gemma3:27b", "qwen3:4b", "qwen3:8b", "qwen3:14b", "qwen3:30b",
               "qwen3:32b", "qwen3:235b", "devstral:24b", "deepseek-r1:671b"],
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
default_theme = "textual-dark"  # Default theme on startup ("textual-dark", "textual-light", or any theme name from themes.py)
palette_theme_limit = 1  # Maximum number of themes to show in command palette (0 = show all)
log_level = "INFO" # TUI Log Level: DEBUG, INFO, WARNING, ERROR, CRITICAL
users_name = "default_user" # Default user name for the TUI

[tldw_api]
base_url = "http://127.0.0.1:8000" # Or your actual default remote endpoint
# Default auth token can be stored here, or leave empty if user must always provide
auth_token = "default-secret-key-for-single-user"

[splash_screen]
# Splash screen configuration for startup animations
enabled = true  # Enable/disable splash screen
duration = 1.5  # Duration in seconds to display splash screen
skip_on_keypress = true  # Allow users to skip with any keypress

# Card selection mode:
# - "random": Randomly selects from active_cards list (default)
# - "sequential": Cycles through active_cards in order (not yet implemented)
# - "<card_name>": Always use a specific card (e.g., "matrix", "glitch", etc.)
card_selection = "random"

show_progress = true  # Show initialization progress bar

# All available splash cards are enabled by default for variety
# To customize: Remove cards you don't want, or replace the entire list with your preferred cards
# Static cards: default, classic, compact, minimal, blueprint
# Animated cards: matrix, glitch, retro, tech_pulse, code_scroll, minimal_fade, arcade_high_score,
#                digital_rain, loading_bar, starfield, terminal_boot, glitch_reveal, ascii_morph,
#                game_of_life, scrolling_credits, spotlight_reveal, sound_bars
active_cards = [
    "default", "matrix", "glitch", "retro", "classic", "compact", "minimal",
    "tech_pulse", "code_scroll", "minimal_fade", "blueprint", "arcade_high_score",
    "digital_rain", "loading_bar", "starfield", "terminal_boot", "glitch_reveal",
    "ascii_morph", "game_of_life", "scrolling_credits", "spotlight_reveal", "sound_bars"
]

[splash_screen.effects]
# Animation effect settings
fade_in_duration = 0.3  # Fade in time in seconds
fade_out_duration = 0.2  # Fade out time in seconds
animation_speed = 1.0  # Animation playback speed multiplier

[logging]
# Log file will be placed in the same directory as the chachanotes_db_path below.
log_filename = "tldw_cli_app.log"
file_log_level = "INFO" # File Log Level: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_max_bytes = 10485760 # 10 MB
log_backup_count = 5

[database]
# Path to the ChaChaNotes (Character, Chat, Notes) database.
chachanotes_db_path = "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
# Path to the Prompts database.
prompts_db_path = "~/.local/share/tldw_cli/tldw_cli_prompts.db"
# Path to the Media V2 database.
media_db_path = "~/.local/share/tldw_cli/tldw_cli_media_v2.db"
USER_DB_BASE_DIR = "~/.local/share/tldw_cli/"

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
OpenAI = ["gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "o3-2025-04-16", "o3-mini-2025-01-31", "o1-2024-12-17", "chatgpt-4o-latest", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14", "gpt-4o-mini-2024-07-18", ]
Anthropic = ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1", "claude-2.0"]
Cohere = ["command-a-03-2025", "command-r7b-12-2024", "command-r-plus-04-2024", "command-r-plus", "command-r-08-2024", "command-r-03-2024", "command", "command-nightly", "command-light", "command-light-nightly"]
DeepSeek = ["deepseek-chat", "deepseek-reasoner"]
Groq = ["gemma2-9b-it", "mmeta-llama/Llama-Guard-4-12B", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-70b-8192", "llama3-8b-8192",]
Google = ["gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-05-06", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", ]
HuggingFace = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct",]
MistralAI = ["open-mistral-nemo", "mistral-medium-2505", "codestral-2501", "mistral-saba-2502", "mistral-large-2411", "ministral-3b-2410", "ministral-8b-2410", "mistral-moderation-2411", "devstral-small-2505", "mistral-small-2503", ]
OpenRouter = ["openai/gpt-4o-mini", "anthropic/claude-3.7-sonnet", "google/gemini-2.0-flash-001", "google/gemini-2.5-pro-preview", "google/gemini-2.5-flash-preview", "deepseek/deepseek-chat-v3-0324:free", "deepseek/deepseek-chat-v3-0324", "openai/gpt-4.1", "anthropic/claude-sonnet-4", "deepseek/deepseek-r1:free", "anthropic/claude-3.7-sonnet:thinking", "google/gemini-flash-1.5-8b", "mistralai/mistral-nemo", "google/gemini-2.5-flash-preview-05-20", ]
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

[api_settings] # Parent section for all API provider specific settings

    # --- Cloud Providers ---
    [api_settings.openai]
    api_key_env_var = "OPENAI_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "gpt-4o" # Default model for direct calls (if not overridden)
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
    model = "claude-3-haiku-20240307"
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
    model = "command-r-plus"
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
    model = "deepseek-chat"
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
    model = "llama3-70b-8192"
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
    model = "gemini-1.5-pro-latest"
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
    model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
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

    # --- Local Providers ---
    [api_settings.llama_cpp] # Matches key in [providers]
    api_key_env_var = "LLAMA_CPP_API_KEY" # If you set one on the server
    # api_key = ""
    api_url = "http://localhost:8080/completion" # llama.cpp /completion endpoint
    model = "" # Often not needed if server serves one model
    temperature = 0.7
    top_p = 0.95
    top_k = 40
    min_p = 0.05
    max_tokens = 4096 # llama.cpp uses n_predict
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = false
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
    streaming = false
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
    streaming = false # Kobold streaming is non-standard, handle carefully
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
    streaming = false
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
    streaming = false
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
    streaming = false
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
    streaming = false
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
    streaming = false
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
    streaming = false
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
    streaming = false
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
    streaming = false
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
    streaming = false
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
    streaming = false
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
    streaming = false
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
    streaming = false
    system_prompt = "You are a helpful AI assistant"
    # ... etc ...

[chat_defaults]
# Default settings specifically for the 'Chat' tab
provider = "OpenAI"
model = "gpt-4o"
system_prompt = "You are a helpful AI assistant."
temperature = 0.6
top_p = 0.95
min_p = 0.05
top_k = 50
strip_thinking_tags = true
use_enhanced_window = false  # Enable enhanced chat window with image support
enable_tabs = false  # Enable tabbed chat interface (experimental)
max_tabs = 10  # Maximum number of chat tabs allowed

# Image support settings (when use_enhanced_window = true)
[chat.images]
enabled = true
show_attach_button = true  # Show/hide the attach file button in chat
default_render_mode = "auto"  # auto, pixels, regular
max_size_mb = 10.0
auto_resize = true
resize_max_dimension = 2048
save_location = "~/Downloads"
supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]

[chat.images.terminal_overrides]
kitty = "regular"
wezterm = "regular"
iterm2 = "regular"
default = "pixels"

[character_defaults]
# Default settings specifically for the 'Character' tab
provider = "Anthropic"
model = "claude-3-haiku-20240307" # Make sure this exists in [providers.Anthropic]
system_prompt = "You are roleplaying as a witty pirate captain."
temperature = 0.8
top_p = 0.9
min_p = 0.0 # Check if API supports this
top_k = 100 # Check if API supports this

[notes]
# Default settings for the Notes tab
sync_directory = "~/Documents/Notes"  # Default directory for notes synchronization
auto_sync_enabled = false            # Enable automatic sync on startup
sync_on_close = false               # Sync when closing the app
conflict_resolution = "newer_wins"   # Default conflict resolution: newer_wins, ask, disk_wins, db_wins
sync_direction = "bidirectional"     # Default sync direction: bidirectional, disk_to_db, db_to_disk

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
[rag]
# Comprehensive configuration for the RAG system

    # --- Retrieval Settings ---
    [rag.retriever]
    fts_top_k = 10              # Number of results from full-text search
    vector_top_k = 10           # Number of results from vector search
    hybrid_alpha = 0.5          # Weight for hybrid search (0=FTS only, 1=vector only)
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

# Legacy RAG settings (for backwards compatibility)
[rag_search]
fts_top_k = 10
vector_top_k = 10
web_vector_top_k = 10
llm_context_document_limit = 10
chat_context_limit = 10


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

# Anthropic models
"claude-3-opus-20240229" = { vision = true, max_images = 5 }
"claude-3-sonnet-20240229" = { vision = true, max_images = 5 }
"claude-3-haiku-20240307" = { vision = true, max_images = 5 }
"claude-3-5-sonnet-20240620" = { vision = true, max_images = 5 }
"claude-3-5-sonnet-20241022" = { vision = true, max_images = 5 }

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

# [search_settings]
# default_provider = "google"
# ...

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
# Options: "faster-whisper", "qwen2audio", "parakeet", "canary"
default_provider = "faster-whisper"

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

# Voice Activity Detection
use_vad_by_default = false

# Speaker diarization (not yet fully implemented)
use_diarization_by_default = false

# Chunk length for long audio processing (in seconds)
# Used by Canary model for efficient processing of long audio files
chunk_length_seconds = 40.0

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
"""

try:
    DEFAULT_CONFIG_FROM_TOML: Dict[str, Any] = tomllib.loads(CONFIG_TOML_CONTENT)
except tomllib.TOMLDecodeError as e:
    logger.critical(f"FATAL: Could not parse internal DEFAULT_CONFIG_TOML_CONTENT: {e}. Application cannot start correctly.")
    DEFAULT_CONFIG_FROM_TOML = {} # Should not happen with valid TOML string

# --- Helper for deep merging dictionaries ---
def deep_merge_dicts(base: Dict, update: Dict) -> Dict:
    """Recursively merges update_dict into base_dict."""
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

# --- Primary Configuration Loading Logic for the CLI ---
_CONFIG_CACHE: Optional[Dict[str, Any]] = None

def load_cli_config_and_ensure_existence(force_reload: bool = False) -> Dict[str, Any]: # Renamed from load_cli_config
    """
    Loads settings for the CLI application from ~/.config/tldw_cli/config.toml.
    If the file doesn't exist, it's created with default values from CONFIG_TOML_CONTENT.
    Uses programmatic defaults (from CONFIG_TOML_CONTENT) as a base.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    # Start with the programmatic defaults defined in CONFIG_TOML_CONTENT
    loaded_config = copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)

    if not DEFAULT_CONFIG_PATH.exists():
        logger.info(f"CLI Config file not found at {DEFAULT_CONFIG_PATH}. Creating with default values from CONFIG_TOML_CONTENT.")
        try:
            DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(CONFIG_TOML_CONTENT) # Write the raw TOML string
            logger.info(f"Created default CLI config file at {DEFAULT_CONFIG_PATH}")
            # Set a flag to notify the user on first run
            loaded_config["_first_run"] = True
            # loaded_config is already correct as it's from DEFAULT_CONFIG_FROM_TOML
        except PermissionError as e:
            # Try alternative location in user's home directory
            logger.warning(f"Permission denied creating config at {DEFAULT_CONFIG_PATH}: {e}")
            alt_config_path = Path.home() / ".tldw_cli_config.toml"
            logger.info(f"Attempting to create config at alternative location: {alt_config_path}")
            try:
                with open(alt_config_path, "w", encoding="utf-8") as f:
                    f.write(CONFIG_TOML_CONTENT)
                logger.warning(f"Created config file at alternative location: {alt_config_path}")
                logger.warning("Please move this file to the standard location when possible.")
                # Note: We don't update DEFAULT_CONFIG_PATH here to maintain consistency
            except Exception as alt_e:
                logger.error(f"Could not create config file at alternative location either: {alt_e}")
                logger.error("Application will use internal defaults only.")
        except OSError as e:
            logger.error(f"Could not create default CLI config file {DEFAULT_CONFIG_PATH}: {e}. Using internal defaults.")
            # Log more helpful information for the user
            logger.info(f"You may need to manually create the directory: {DEFAULT_CONFIG_PATH.parent}")
            logger.info("Or check that you have write permissions to this location.")
    else:
        logger.info(f"Attempting to load CLI config from: {DEFAULT_CONFIG_PATH}")
        try:
            with open(DEFAULT_CONFIG_PATH, "rb") as f:
                user_config_from_file = tomllib.load(f)
            # Merge user's file settings on top of the programmatic defaults
            loaded_config = deep_merge_dicts(loaded_config, user_config_from_file)
            logger.info(f"Successfully loaded and merged CLI config from {DEFAULT_CONFIG_PATH}")
            
            # Decrypt config if encryption is enabled
            loaded_config = decrypt_config_section(loaded_config)
        except tomllib.TOMLDecodeError as e:
            logger.error(f"Error decoding CLI TOML config file {DEFAULT_CONFIG_PATH}: {e}. Using internal defaults + any previous successful load.", exc_info=True)
            # `loaded_config` remains the programmatic defaults in this case.
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading CLI config {DEFAULT_CONFIG_PATH}: {e}. Using internal defaults + any previous successful load.", exc_info=True)
            # `loaded_config` remains the programmatic defaults.

    _CONFIG_CACHE = loaded_config
    # Log the keys of the configuration being returned to verify its structure
    logger.debug(f"load_cli_config_and_ensure_existence returning config with top-level keys: {list(loaded_config.keys())}")
    if "api_settings" in loaded_config:
        logger.debug(f"  'api_settings' found with keys: {list(loaded_config.get('api_settings', {}).keys())}")
    else:
        logger.warning("  'api_settings' key NOT FOUND in the loaded configuration for load_cli_config_and_ensure_existence.")

    return _CONFIG_CACHE


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
    global _CONFIG_CACHE, _SETTINGS_CACHE, settings
    logger.info(f"Attempting to save setting: [{section}].{key} = {repr(value)}")

    # Ensure the parent directory for the config file exists.
    try:
        DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create config directory {DEFAULT_CONFIG_PATH.parent}: {e}")
        return False

    # Step 1: Read the current configuration from the user's file.
    # If the file doesn't exist, we start with an empty dictionary.
    config_data: Dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        try:
            with open(DEFAULT_CONFIG_PATH, "rb") as f:
                config_data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            logger.error(f"Corrupted config file at {DEFAULT_CONFIG_PATH}. Cannot save. Please fix or delete it. Error: {e}")
            # Consider creating a backup of the corrupt file for the user.
            return False
        except Exception as e:
            logger.error(f"Unexpected error reading {DEFAULT_CONFIG_PATH}: {e}", exc_info=True)
            return False

    # Step 2: Modify the configuration data in memory.
    # This handles nested sections by splitting the section string.
    keys = section.split('.')
    current_level = config_data

    try:
        for part in keys:
            # Traverse or create the nested dictionary structure.
            current_level = current_level.setdefault(part, {})
        # Assign the new value to the key in the target section.
        current_level[key] = value
    except (TypeError, AttributeError):
        # This error occurs if a key in the path (e.g., 'api_settings') is a value, not a table.
        logger.error(
            f"Configuration structure conflict. Could not set '{key}' in section '{section}' "
            f"because a part of the path is not a table/dictionary. Please check your config file."
        )
        return False

    # Step 3: Check if we need to encrypt the value
    # If we're saving to an api_settings section and encryption is enabled, encrypt the value
    encryption_config = config_data.get("encryption", {})
    if (encryption_config.get("enabled", False) and 
        section.startswith("api_settings.") and 
        key == "api_key" and 
        isinstance(value, str) and 
        value and 
        not value.startswith("enc:")):
        
        password = get_encryption_password()
        if password:
            try:
                enc_module = get_encryption_module()
                salt_b64 = encryption_config.get("salt")
                if salt_b64:
                    import base64
                    salt = base64.b64decode(salt_b64)
                    encrypted_value, _ = enc_module.encrypt_value(value, password, salt)
                    current_level[key] = encrypted_value
                    logger.info(f"Encrypted API key for {section}")
            except Exception as e:
                logger.error(f"Failed to encrypt value: {e}")
                # Continue with unencrypted value
    
    # Step 4: Write the updated configuration back to the TOML file.
    try:
        with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
            toml.dump(config_data, f)
        logger.success(f"Successfully saved setting to {DEFAULT_CONFIG_PATH}")

        # Step 4: Invalidate and reload global config caches to reflect changes immediately.
        # Clear both caches
        _CONFIG_CACHE = None
        _SETTINGS_CACHE = None
        load_cli_config_and_ensure_existence(force_reload=True)
        settings = load_settings(force_reload=True)
        logger.info("Global configuration caches invalidated and reloaded.")

        return True
    except (IOError, toml.TomlDecodeError) as e:
        logger.error(f"Failed to write updated config to {DEFAULT_CONFIG_PATH}: {e}", exc_info=True)
        return False


# --- CLI Setting Getter ---
def get_cli_setting(section: str, key: str, default: Any = None) -> Any:
    """Helper to get a specific setting from the loaded CLI configuration."""
    config = load_cli_config_and_ensure_existence() # Ensures config is loaded
    # Use `config.get(section, {})` to safely access potentially missing sections
    section_data = config.get(section)
    if isinstance(section_data, dict):
        return section_data.get(key, default)
    # If section is not a dict or not found, return default
    return default


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
    if media_type in media_ingestion_config and isinstance(media_ingestion_config[media_type], dict):
        # Use deep merge to combine with defaults, allowing partial overrides
        return deep_merge_dicts(
            DEFAULT_MEDIA_INGESTION_CONFIG.get(media_type, {}),
            media_ingestion_config[media_type]
        )
    
    # Fall back to hardcoded defaults
    return DEFAULT_MEDIA_INGESTION_CONFIG.get(media_type, {
        "chunk_method": "paragraphs",
        "chunk_size": 500,
        "chunk_overlap": 200,
        "use_adaptive_chunking": False,
        "use_multi_level_chunking": False,
        "chunk_language": ""
    })


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
    if backend_name in ocr_backend_config and isinstance(ocr_backend_config[backend_name], dict):
        # Use deep merge to combine with defaults, allowing partial overrides
        return deep_merge_dicts(
            DEFAULT_OCR_BACKEND_CONFIG.get(backend_name, {}),
            ocr_backend_config[backend_name]
        )
    
    # Fall back to hardcoded defaults
    return DEFAULT_OCR_BACKEND_CONFIG.get(backend_name, {})


# --- CLI Providers and Models Getter ---
def get_cli_providers_and_models() -> Dict[str, List[str]]:
    config = load_settings()
    providers_data = config.get("providers", {}) # Default to empty dict if "providers" isn't there
    valid_providers: Dict[str, List[str]] = {}
    if isinstance(providers_data, dict):
        for provider, models in providers_data.items():
            if isinstance(models, list) and all(isinstance(m, str) for m in models):
                valid_providers[provider] = models
            else:
                logger.warning(f"Invalid model list for provider '{provider}' in CLI config [providers]. Models: {models}. Skipping.")
    else:
        logger.error(f"CLI Config 'providers' section is not a dictionary. Found: {type(providers_data)}. No provider/model data available.")
    return valid_providers


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
        if section_name.startswith('api_settings.') and isinstance(section_value, dict):
            api_key = section_value.get('api_key', '')
            # Check if API key exists and is not a placeholder
            if api_key and not api_key.startswith('<') and not api_key.endswith('>'):
                provider_name = section_name.replace('api_settings.', '')
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
        # Load current config
        config_data = {}
        if DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH, "rb") as f:
                config_data = tomllib.load(f)
        
        # Encrypt the config
        encrypted_config = encrypt_api_keys_in_config(config_data, password)
        
        # Save the encrypted config
        with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
            toml.dump(encrypted_config, f)
        
        # Set the password for the current session
        set_encryption_password(password)
        
        # Clear and reload caches
        global _CONFIG_CACHE, _SETTINGS_CACHE
        _CONFIG_CACHE = None
        _SETTINGS_CACHE = None
        
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
        # Load current config
        config_data = {}
        if DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH, "rb") as f:
                config_data = tomllib.load(f)
        
        # Verify password
        encryption_config = config_data.get("encryption", {})
        if encryption_config.get("enabled", False):
            enc_module = get_encryption_module()
            stored_hash = encryption_config.get("password_hash", "")
            if not enc_module.verify_password(password, stored_hash):
                logger.error("Invalid password provided")
                return False
        
        # Set password temporarily for decryption
        set_encryption_password(password)
        
        # Decrypt the config
        decrypted_config = decrypt_config_section(config_data)
        
        # Remove encryption metadata
        if "encryption" in decrypted_config:
            del decrypted_config["encryption"]
        
        # Save the decrypted config
        with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
            toml.dump(decrypted_config, f)
        
        # Clear password
        clear_encryption_password()
        
        # Clear and reload caches
        global _CONFIG_CACHE, _SETTINGS_CACHE
        _CONFIG_CACHE = None
        _SETTINGS_CACHE = None
        
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
        # Load current config
        config_data = {}
        if DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH, "rb") as f:
                config_data = tomllib.load(f)
        
        # Verify old password
        encryption_config = config_data.get("encryption", {})
        if encryption_config.get("enabled", False):
            enc_module = get_encryption_module()
            stored_hash = encryption_config.get("password_hash", "")
            if not enc_module.verify_password(old_password, stored_hash):
                logger.error("Invalid current password provided")
                return False
        else:
            logger.error("Encryption is not enabled")
            return False
        
        # Set old password temporarily for decryption
        set_encryption_password(old_password)
        
        # Decrypt the config
        decrypted_config = decrypt_config_section(config_data)
        
        # Re-encrypt with new password
        encrypted_config = encrypt_api_keys_in_config(decrypted_config, new_password)
        
        # Save the re-encrypted config
        with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
            toml.dump(encrypted_config, f)
        
        # Set the new password for the current session
        set_encryption_password(new_password)
        
        # Clear and reload caches
        global _CONFIG_CACHE, _SETTINGS_CACHE
        _CONFIG_CACHE = None
        _SETTINGS_CACHE = None
        
        logger.success("Encryption password changed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to change encryption password: {e}")
        return False


# --- CLI Database and Log File Path Getters ---
BASE_DATA_DIR_CLI = Path.home() / ".local" / "share" / "tldw_cli" # Renamed for clarity

def get_chachanotes_db_path() -> Path:
    default_db_path_str = DEFAULT_CONFIG_FROM_TOML.get("database", {}).get("chachanotes_db_path", str(BASE_DATA_DIR_CLI / "tldw_chatbook_ChaChaNotes.db"))
    db_path_str = get_cli_setting("database", "chachanotes_db_path", default_db_path_str)
    db_path = Path(db_path_str).expanduser().resolve()
    return db_path

def get_prompts_db_path() -> Path:
    default_db_path_str = DEFAULT_CONFIG_FROM_TOML.get("database", {}).get("prompts_db_path", str(BASE_DATA_DIR_CLI / "tldw_chatbook_prompts.db"))
    db_path_str = get_cli_setting("database", "prompts_db_path", default_db_path_str)
    db_path = Path(db_path_str).expanduser().resolve()
    return db_path

def get_media_db_path() -> Path:
    default_db_path_str = DEFAULT_CONFIG_FROM_TOML.get("database", {}).get("media_db_path", str(BASE_DATA_DIR_CLI / "tldw_chatbook_media_v2.db"))
    db_path_str = get_cli_setting("database", "media_db_path", default_db_path_str)
    db_path = Path(db_path_str).expanduser().resolve()
    return db_path

def get_cli_log_file_path() -> Path:
    chachanotes_parent_dir = get_chachanotes_db_path().parent
    default_log_filename = DEFAULT_CONFIG_FROM_TOML.get("logging", {}).get("log_filename", "tldw_cli_app.log")
    log_filename = get_cli_setting("logging", "log_filename", default_log_filename)
    log_file_path = chachanotes_parent_dir / log_filename
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create log directory {log_file_path.parent}: {e}", exc_info=True)
    return log_file_path


def get_cli_data_dir() -> Path:
    """Get the CLI data directory for storing application data."""
    # Create directory if it doesn't exist
    BASE_DATA_DIR_CLI.mkdir(parents=True, exist_ok=True)
    return BASE_DATA_DIR_CLI

# --- Global CLI Database Instances ---
chachanotes_db: Optional[CharactersRAGDB] = None
prompts_db: Optional[PromptsDatabase] = None
media_db: Optional[MediaDatabase] = None

# --- Database Initialization Function (remains largely the same) ---
def initialize_all_databases():
    global chachanotes_db, prompts_db, media_db
    logger.debug("CRITICAL DEBUG: INSIDE initialize_all_databases() NOW.")
    logger.info("Initializing CLI databases...")
    # ChaChaNotes DB
    chachanotes_path = get_chachanotes_db_path()
    logger.info(f"Attempting to initialize ChaChaNotes_DB at: {chachanotes_path}")
    try:
        chachanotes_db = CharactersRAGDB(db_path=chachanotes_path, client_id=CLI_APP_CLIENT_ID)
        logger.success(f"ChaChaNotes_DB initialized successfully at {chachanotes_path}")
    except Exception as e:
        logger.error(f"Failed to initialize ChaChaNotes_DB at {chachanotes_path}: {e}", exc_info=True)
        chachanotes_db = None
    # Prompts DB
    prompts_path = get_prompts_db_path()
    logger.info(f"Attempting to initialize Prompts_DB at: {prompts_path}")
    try:
        prompts_db = PromptsDatabase(db_path=prompts_path, client_id=CLI_APP_CLIENT_ID)
        logger.success(f"Prompts_DB initialized successfully at {prompts_path}")
    except Exception as e:
        logger.error(f"Failed to initialize Prompts_DB at {prompts_path}: {e}", exc_info=True)
        prompts_db = None
    # Media DB
    media_path = get_media_db_path()
    logger.info(f"Attempting to initialize Media_DB_v2 at: {media_path}")
    try:
        media_db = MediaDatabase(db_path=media_path, client_id=CLI_APP_CLIENT_ID)
        logger.success(f"Media_DB_v2 initialized successfully at {media_path}")
    except Exception as e:
        logger.error(f"Failed to initialize Media_DB_v2 at {media_path}: {e}", exc_info=True)
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
            chachanotes_db = CharactersRAGDB(db_path=chachanotes_path, client_id=CLI_APP_CLIENT_ID)
            logger.success(f"ChaChaNotes_DB lazy-initialized successfully at {chachanotes_path}")
        except Exception as e:
            logger.error(f"Failed to lazy-initialize ChaChaNotes_DB at {chachanotes_path}: {e}", exc_info=True)
            chachanotes_db = None
    return chachanotes_db


def get_prompts_db_lazy() -> Optional[PromptsDatabase]:
    """Get the Prompts database instance, initializing it lazily if needed."""
    global prompts_db
    if prompts_db is None:
        prompts_path = get_prompts_db_path()
        logger.info(f"Lazy-initializing Prompts_DB at: {prompts_path}")
        try:
            prompts_db = PromptsDatabase(db_path=prompts_path, client_id=CLI_APP_CLIENT_ID)
            logger.success(f"Prompts_DB lazy-initialized successfully at {prompts_path}")
        except Exception as e:
            logger.error(f"Failed to lazy-initialize Prompts_DB at {prompts_path}: {e}", exc_info=True)
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
            logger.error(f"Failed to lazy-initialize Media_DB_v2 at {media_path}: {e}", exc_info=True)
            media_db = None
    return media_db


# --- API Models (should be defined based on CONFIG_TOML_CONTENT or loaded from it) ---
# These can be loaded dynamically from the config or kept as fallback statics
# For simplicity, if CONFIG_TOML_CONTENT has [providers], use that.
_temp_loaded_config_for_models = tomllib.loads(CONFIG_TOML_CONTENT)
API_MODELS_BY_PROVIDER: Dict[str, List[str]] = {}
LOCAL_PROVIDERS: Dict[str, List[str]] = {}

_config_providers = _temp_loaded_config_for_models.get("providers", {})
_cloud_provider_keys = ["OpenAI", "Anthropic", "Cohere", "DeepSeek", "Groq", "Google", "HuggingFace", "MistralAI", "OpenRouter"] # Example list

for provider_name, models_list in _config_providers.items():
    if isinstance(models_list, list):
        if provider_name in _cloud_provider_keys: # Crude way to separate, adjust as needed
            API_MODELS_BY_PROVIDER[provider_name] = models_list
        else:
            LOCAL_PROVIDERS[provider_name] = models_list
    else:
        logger.warning(f"Models for provider '{provider_name}' in CONFIG_TOML_CONTENT is not a list. Skipping.")

if not API_MODELS_BY_PROVIDER and not LOCAL_PROVIDERS: # Fallback if [providers] was empty or malformed
    logger.warning("No providers found in CONFIG_TOML_CONTENT's [providers] section. Using hardcoded fallbacks for API_MODELS_BY_PROVIDER and LOCAL_PROVIDERS.")
    API_MODELS_BY_PROVIDER = { "OpenAI": ["gpt-4o"] } # Minimal fallback
    LOCAL_PROVIDERS = { "Ollama": ["llama3"] } # Minimal fallback


# --- Global default_api_endpoint (example of using the new settings) ---

# --- Global Settings Object ---
load_cli_config_and_ensure_existence()
settings = load_settings()

try:
    # Accessing deeply nested key safely
    default_api_endpoint = settings.get('llm_api_settings', {}).get('default_api', 'openai')
    logger.info(f"Default API Endpoint (from config.py global scope): {default_api_endpoint}")
except Exception as e:
    logger.error(f"Critical error setting default_api_endpoint in config.py global scope: {str(e)}", exc_info=True)
    default_api_endpoint = "openai"  # Fallback

# --- Optional: Export individual variables if needed (generally prefer using settings dict) ---
# SINGLE_USER_MODE = settings["SINGLE_USER_MODE"]
# OPENAI_API_KEY = settings["OPENAI_API_KEY"]

# Make APP_CONFIG, DATABASE_CONFIG, RAG_SEARCH_CONFIG available globally if needed
# These are now loaded from TOML into the `settings` dictionary.
APP_CONFIG = settings.get("APP_TTS_CONFIG", DEFAULT_APP_TTS_CONFIG) # Fallback if not in settings for some reason
DATABASE_CONFIG = settings.get("APP_DATABASE_CONFIG", DEFAULT_DATABASE_CONFIG)
RAG_SEARCH_CONFIG = settings.get("APP_RAG_SEARCH_CONFIG", DEFAULT_RAG_SEARCH_CONFIG)

# --- Default Prompts ---
CONFIG_PROMPT_SITUATE_CHUNK_CONTEXT = settings.get("prompts_strings", {}).get("situate_chunk_context_prompt", "You are an AI assistant. Please follow the instructions provided in the input text carefully and accurately.")

# --- Load CLI Config and Initialize Databases on module import ---
# The `settings` global variable is now the result of the unified load_settings()
logger.debug("CRITICAL DEBUG: Database initialization is now lazy - will initialize on first access")

# Databases will be initialized lazily on first access
# This significantly improves startup time by deferring expensive DB operations

# Make APP_CONFIG available globally if needed by other modules that import from config.py
# This will be the same as `settings` if `load_settings` is the sole config loader.
APP_CONFIG_GLOBAL = settings

#
# End of tldw_cli/config.py
#######################################################################################################################
