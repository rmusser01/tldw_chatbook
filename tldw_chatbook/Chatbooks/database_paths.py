"""Canonical runtime database paths used by local Chatbook workflows."""

from pathlib import Path

from .. import config
from ..Utils.private_paths import secure_private_directory


def get_chatbook_database_paths() -> dict[str, str]:
    """Return database paths using the key names required by Chatbook services."""

    return {
        "ChaChaNotes": str(config.get_chachanotes_db_path()),
        "Prompts": str(config.get_prompts_db_path()),
        "Media": str(config.get_media_db_path()),
    }


def get_private_chatbooks_dir() -> Path:
    """Return the secured app-owned directory for local Chatbook archives."""

    return secure_private_directory(
        config.get_user_data_dir() / "chatbooks",
        create=True,
        application_owned=True,
    ).lexical_path
