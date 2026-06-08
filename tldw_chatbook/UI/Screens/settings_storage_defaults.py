"""Storage path defaults exposed by the Settings hub."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .settings_config_models import SettingsValidationResult


DEFAULT_USER_DB_BASE_DIR = "~/.local/share/tldw_cli/"
DEFAULT_CHACHANOTES_DB_PATH = "~/.local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
DEFAULT_PROMPTS_DB_PATH = "~/.local/share/tldw_cli/tldw_cli_prompts.db"
DEFAULT_MEDIA_DB_PATH = "~/.local/share/tldw_cli/tldw_cli_media_v2.db"
DEFAULT_RESEARCH_DB_PATH = "~/.local/share/tldw_cli/tldw_chatbook_research.db"
DEFAULT_WRITING_DB_PATH = "~/.local/share/tldw_cli/tldw_chatbook_writing.db"
DEFAULT_LIBRARY_COLLECTIONS_DB_PATH = (
    "~/.local/share/tldw_cli/tldw_chatbook_library_collections.db"
)
DEFAULT_WORKSPACES_DB_PATH = "~/.local/share/tldw_cli/tldw_chatbook_workspaces.db"
DATABASE_FILE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})


@dataclass(frozen=True)
class SettingsStorageDefaults:
    """Editable Storage defaults exposed in Settings."""

    user_db_base_dir: str = DEFAULT_USER_DB_BASE_DIR
    chachanotes_db_path: str = DEFAULT_CHACHANOTES_DB_PATH
    prompts_db_path: str = DEFAULT_PROMPTS_DB_PATH
    media_db_path: str = DEFAULT_MEDIA_DB_PATH
    research_db_path: str = DEFAULT_RESEARCH_DB_PATH
    writing_db_path: str = DEFAULT_WRITING_DB_PATH
    library_collections_db_path: str = DEFAULT_LIBRARY_COLLECTIONS_DB_PATH
    workspaces_db_path: str = DEFAULT_WORKSPACES_DB_PATH


STORAGE_FIELD_CONFIG_KEYS: Mapping[str, str] = {
    "user_db_base_dir": "USER_DB_BASE_DIR",
    "chachanotes_db_path": "chachanotes_db_path",
    "prompts_db_path": "prompts_db_path",
    "media_db_path": "media_db_path",
    "research_db_path": "research_db_path",
    "writing_db_path": "writing_db_path",
    "library_collections_db_path": "library_collections_db_path",
    "workspaces_db_path": "workspaces_db_path",
}

STORAGE_FIELD_LABELS: Mapping[str, str] = {
    "user_db_base_dir": "Base data directory",
    "chachanotes_db_path": "ChaChaNotes DB",
    "prompts_db_path": "Prompts DB",
    "media_db_path": "Media DB",
    "research_db_path": "Research DB",
    "writing_db_path": "Writing DB",
    "library_collections_db_path": "Library Collections DB",
    "workspaces_db_path": "Workspaces DB",
}


def _database_section(app_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the database config section when present."""
    database = app_config.get("database", {})
    return database if isinstance(database, Mapping) else {}


def _path_text(value: Any, default: str) -> str:
    """Normalize a path-like config value to editable text."""
    text = str(value if value is not None else "").strip()
    return text or default


def load_storage_defaults(app_config: Mapping[str, Any]) -> SettingsStorageDefaults:
    """Load Settings-owned Storage defaults from app configuration.

    Args:
        app_config: Application configuration mapping to read from.

    Returns:
        Storage defaults using existing config values with safe fallbacks for
        missing or malformed sections.
    """
    database = _database_section(app_config)
    return SettingsStorageDefaults(
        user_db_base_dir=_path_text(
            database.get("USER_DB_BASE_DIR"),
            DEFAULT_USER_DB_BASE_DIR,
        ),
        chachanotes_db_path=_path_text(
            database.get("chachanotes_db_path"),
            DEFAULT_CHACHANOTES_DB_PATH,
        ),
        prompts_db_path=_path_text(
            database.get("prompts_db_path"),
            DEFAULT_PROMPTS_DB_PATH,
        ),
        media_db_path=_path_text(
            database.get("media_db_path"),
            DEFAULT_MEDIA_DB_PATH,
        ),
        research_db_path=_path_text(
            database.get("research_db_path"),
            DEFAULT_RESEARCH_DB_PATH,
        ),
        writing_db_path=_path_text(
            database.get("writing_db_path"),
            DEFAULT_WRITING_DB_PATH,
        ),
        library_collections_db_path=_path_text(
            database.get("library_collections_db_path"),
            DEFAULT_LIBRARY_COLLECTIONS_DB_PATH,
        ),
        workspaces_db_path=_path_text(
            database.get("workspaces_db_path"),
            DEFAULT_WORKSPACES_DB_PATH,
        ),
    )


def _expanded_path(value: str) -> Path:
    """Return an expanded path for validation without changing saved text."""
    return Path(value).expanduser()


def _path_has_parent_traversal(value: str) -> bool:
    """Return whether a path contains explicit parent traversal."""
    return ".." in Path(value).parts


def _validate_path_text(field_name: str, value: str, *, database_file: bool) -> str | None:
    """Return a validation error for one editable Storage path."""
    label = STORAGE_FIELD_LABELS[field_name]
    text = str(value or "").strip()
    if not text:
        return f"{label} is required."
    if "\x00" in text or "\n" in text or "\r" in text:
        return f"{label} must be a single filesystem path."
    if _path_has_parent_traversal(text):
        return f"{label} cannot contain parent-directory traversal."

    path = _expanded_path(text)
    if database_file:
        if text.endswith(("/", "\\")) or not path.name:
            return f"{label} must be a database file path."
        if path.suffix.lower() not in DATABASE_FILE_SUFFIXES:
            return f"{label} must end with .db, .sqlite, or .sqlite3."
        parent = path.parent
    else:
        parent = path

    if parent.exists() and not parent.is_dir():
        return f"{label} parent must be a directory."
    return None


def validate_storage_defaults(
    values: SettingsStorageDefaults,
) -> SettingsValidationResult:
    """Validate editable Storage defaults before persistence.

    Args:
        values: Storage defaults to validate.

    Returns:
        Validation state and user-facing recovery copy.
    """
    value_map = asdict(values)
    for field_name, value in value_map.items():
        error = _validate_path_text(
            field_name,
            value,
            database_file=field_name != "user_db_base_dir",
        )
        if error:
            return SettingsValidationResult(False, error)
    return SettingsValidationResult(
        True,
        "Storage defaults are valid. Changes apply on next app launch.",
    )


def build_storage_save_sections(
    app_config: Mapping[str, Any],
    values: SettingsStorageDefaults,
) -> dict[str, dict[str, Any]]:
    """Build config sections needed to persist Storage defaults.

    Args:
        app_config: Existing application configuration mapping.
        values: Validated Storage defaults to persist.

    Returns:
        A database section suitable for ``SettingsConfigAdapter.save_sections``.
    """
    database = dict(deepcopy(_database_section(app_config)))
    value_map = asdict(values)
    for field_name, config_key in STORAGE_FIELD_CONFIG_KEYS.items():
        database[config_key] = str(value_map[field_name]).strip()
    return {"database": database}


def _parent_status(path_text: str, *, database_file: bool) -> str:
    """Return a non-mutating parent-directory readiness label."""
    path = _expanded_path(path_text)
    parent = path.parent if database_file else path
    if parent.exists() and parent.is_dir():
        return f"{parent}: ready"
    if parent.exists():
        return f"{parent}: blocked, not a directory"
    return f"{parent}: missing, create before restart"


def build_storage_check_rows(values: SettingsStorageDefaults) -> tuple[str, ...]:
    """Build non-mutating Storage check rows for the current draft values."""
    rows = ["Storage check: complete"]
    validation = validate_storage_defaults(values)
    rows.append(validation.message)
    for field_name, value in asdict(values).items():
        label = STORAGE_FIELD_LABELS[field_name]
        status = _parent_status(
            str(value),
            database_file=field_name != "user_db_base_dir",
        )
        rows.append(f"{label}: {status}")
    rows.append("Storage safety: no files were created, moved, or reconnected.")
    return tuple(rows)
