"""RAG storage directory policy without importing the retrieval engine."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from ..config import get_cli_setting, get_user_data_dir

_RAG_PROFILES_DIRNAME = "rag_profiles"


def _read_rag_setting(section: str, key: str, getter: Callable[..., Any]) -> Any | None:
    """Read an explicit `[AppRAGSearchConfig.rag.<section>].<key>` user setting.

    Args:
        section: Subsection name within the rag config (e.g. 'vector_store').
        key: Setting name within that subsection (e.g. 'type').
        getter: Live configuration accessor from the calling module.

    Returns:
        The configured value, or None when not explicitly set (or on any
        config read failure).
    """
    try:
        rag_section = getter("AppRAGSearchConfig", "rag", {}) or {}
        if not isinstance(rag_section, dict):
            return None
        subsection = rag_section.get(section, {})
        if not isinstance(subsection, dict):
            return None
        return subsection.get(key)
    except Exception as e:  # noqa: BLE001 - preserve the existing config-read fallback
        logger.debug(f"Could not read rag.{section}.{key} from user config: {e}")
        return None


def _cleaned_path_setting(value: Any) -> str | None:
    """Strip an explicit path setting, treating blank values as unset.

    Args:
        value: Raw setting value from env/config.

    Returns:
        Stripped path string, or None when unset or whitespace-only.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _explicit_rag_setting(section: str, key: str) -> Any | None:
    return _read_rag_setting(section, key, get_cli_setting)


def _chroma_directory(
    explicit_reader: Callable[[str, str], Any],
    user_data_dir: Callable[[], Path],
    clean_path: Callable[[Any], str | None],
) -> Path:
    """Resolve with live collaborators, retaining short-circuit and error behavior."""
    explicit = (
        clean_path(os.getenv("RAG_PERSIST_DIR"))
        or clean_path(explicit_reader("vector_store", "persist_directory"))
        or clean_path(explicit_reader("chroma", "persist_directory"))  # Legacy location
    )
    if explicit:
        return Path(explicit).expanduser()
    return user_data_dir() / "chromadb"


def default_chroma_persist_directory() -> Path:
    """Resolve env, explicit vector/legacy config, then the user-data default."""
    return _chroma_directory(
        _explicit_rag_setting, get_user_data_dir, _cleaned_path_setting
    )


def _rag_profiles_dir(user_data_dir: Callable[[], Path]) -> Path:
    return user_data_dir() / _RAG_PROFILES_DIRNAME


def default_rag_profiles_dir() -> Path:
    """Return the profile directory beneath the current user-data directory."""
    return _rag_profiles_dir(get_user_data_dir)
