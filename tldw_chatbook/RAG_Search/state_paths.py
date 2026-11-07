"""Live RAG state-directory policy without optional RAG runtime imports."""

from collections.abc import Callable
import os
from pathlib import Path
from typing import Any


def _explicit_rag_setting(
    section: str, key: str, get_cli_setting: Callable[[str, str, Any], Any]
) -> Any:
    """Read one explicit setting, retaining per-read failure fallback."""
    try:
        rag_section = get_cli_setting("AppRAGSearchConfig", "rag", {}) or {}
        if not isinstance(rag_section, dict):
            return None
        subsection = rag_section.get(section, {})
        if not isinstance(subsection, dict):
            return None
        return subsection.get(key)
    except Exception as exc:
        from loguru import logger

        logger.bind(exception_type=type(exc).__name__).debug(
            "Could not read RAG state path setting"
        )
        return None


def _cleaned_path_setting(value: Any) -> str | None:
    """Strip a non-null setting while retaining its existing coercion."""
    if value is None:
        return None
    return str(value).strip() or None


def _resolve_chroma_persist_directory(
    read_setting: Callable[[str, str], Any], get_user_data_dir: Callable[[], Path]
) -> Path:
    """Resolve once through the caller's live configuration accessors."""
    explicit = (
        _cleaned_path_setting(os.getenv("RAG_PERSIST_DIR"))
        or _cleaned_path_setting(read_setting("vector_store", "persist_directory"))
        or _cleaned_path_setting(read_setting("chroma", "persist_directory"))
    )
    if explicit:
        return Path(explicit).expanduser()
    return get_user_data_dir() / "chromadb"


def _resolve_rag_profiles_dir(get_user_data_dir: Callable[[], Path]) -> Path:
    """Resolve the current profile directory through its owner's accessor."""
    return get_user_data_dir() / "rag_profiles"


def default_chroma_persist_directory() -> Path:
    """Return the live environment, current, legacy or user-data Chroma path.

    Returns:
        Selected path, expanding home shorthand but retaining relative paths.
    """
    from .. import config

    return _resolve_chroma_persist_directory(
        lambda section, key: _explicit_rag_setting(
            section, key, config.get_cli_setting
        ),
        config.get_user_data_dir,
    )


def default_rag_profiles_dir() -> Path:
    """Return the profile directory under the currently selected user data.

    Returns:
        Current user-data directory joined with the RAG-profile leaf name.
    """
    from .. import config

    return _resolve_rag_profiles_dir(config.get_user_data_dir)
