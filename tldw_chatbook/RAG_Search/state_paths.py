"""Compatible RAG entry points for canonical lower-layer path policy."""

from pathlib import Path

from ..Utils.rag_state_paths import (
    _cleaned_path_setting as _cleaned_path_setting,
    _explicit_rag_setting,
    _resolve_chroma_persist_directory,
    _resolve_rag_profiles_dir,
)


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
