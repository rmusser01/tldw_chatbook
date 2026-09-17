"""Canonical runtime database paths used by local Chatbook workflows."""

from pathlib import Path

from .. import config
from ..Backup_Recovery.local_content_lifetime import operation
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

    path = config.get_user_data_dir() / "chatbooks"
    with operation((path,)):
        return secure_chatbook_directory(path)


def secure_chatbook_directory(path: str | Path) -> Path:
    """Create or harden an application-owned Chatbook directory."""

    with operation((Path(path),)):
        return secure_private_directory(
            path,
            create=True,
            application_owned=True,
        ).lexical_path
