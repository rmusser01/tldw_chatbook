"""Pure canonical path selection; no bootstrap, service imports or filesystem writes.

Relative custom paths retain the application's cwd interpretation. XDG_DATA_HOME
is intentionally ignored: honoring it here would silently select a different profile.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from tldw_chatbook.Utils.platform_files import os


def lexical_path(value: str | Path) -> Path:
    """Expand home and make absolute without resolving links or probing parents."""
    return Path(os.path.abspath(os.path.expanduser(str(value))))


def default_config_path() -> Path:
    return Path.home() / ".config" / "tldw_cli" / "config.toml"


def effective_config_path(default: Path | None = None) -> Path:
    """Select TLDW_CONFIG_PATH ahead of the supplied canonical default."""
    return lexical_path(
        os.environ.get("TLDW_CONFIG_PATH") or default or default_config_path()
    )


def default_base_data_dir() -> Path:
    home = os.environ.get("HOME")
    base = Path(home).expanduser() if home else Path.home()
    return base / ".local" / "share" / "tldw_cli"


def user_folder_name(user_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", user_name) or "default_user"


def setting(
    config: Mapping[str, object], section: str, key: str, default: object = None
) -> object:
    values = config.get(section, {})
    if not isinstance(values, Mapping):
        raise ValueError("invalid_config_shape")
    return values.get(key, default)


def data_base(config: Mapping[str, object]) -> Path:
    selected = setting(config, "paths", "data_dir")
    if selected is None:
        selected = setting(config, "Paths", "data_dir")
    if selected and not isinstance(selected, str):
        raise ValueError("invalid_config_path")
    return lexical_path(selected) if selected else default_base_data_dir()


def user_data_dir(config: Mapping[str, object]) -> Path:
    name = setting(config, "general", "users_name", "default_user")
    if not isinstance(name, str):
        raise ValueError("invalid_profile_name")
    return data_base(config) / user_folder_name(name)


# owner, setting, canonical profile leaf, legacy shipped sentinel (None = no sentinel).
DATABASE_PATHS = (
    (
        "db.chachanotes.primary",
        "chachanotes_db_path",
        "tldw_chatbook_ChaChaNotes.db",
        "tldw_chatbook_ChaChaNotes.db",
    ),
    (
        "db.prompts.primary",
        "prompts_db_path",
        "tldw_chatbook_prompts.db",
        "tldw_cli_prompts.db",
    ),
    (
        "db.media.primary",
        "media_db_path",
        "tldw_chatbook_media_v2.db",
        "tldw_cli_media_v2.db",
    ),
    (
        "db.library_collections",
        "library_collections_db_path",
        "tldw_chatbook_library_collections.db",
        "tldw_chatbook_library_collections.db",
    ),
    (
        "db.library_ingest_jobs",
        "library_ingest_jobs_db_path",
        "tldw_chatbook_library_ingest_jobs.db",
        None,
    ),
    (
        "db.workspaces",
        "workspaces_db_path",
        "tldw_chatbook_workspaces.db",
        "tldw_chatbook_workspaces.db",
    ),
    (
        "db.subscriptions",
        "subscriptions_db_path",
        "tldw_chatbook_subscriptions.db",
        "tldw_chatbook_subscriptions.db",
    ),
    ("db.evals", "evals_db_path", "evals.db", "evals.db"),
    ("db.rag_indexing", "rag_indexing_db_path", "rag_indexing.db", "rag_indexing.db"),
    (
        "notifications.client",
        "notifications_db_path",
        "tldw_chatbook_notifications.db",
        None,
    ),
    (
        "research.local",
        "research_db_path",
        "tldw_chatbook_research.db",
        "tldw_chatbook_research.db",
    ),
    (
        "writing.local",
        "writing_db_path",
        "tldw_chatbook_writing.db",
        "tldw_chatbook_writing.db",
    ),
    (
        "db.scheduled_tasks",
        "scheduled_tasks_db_path",
        "tldw_chatbook_scheduled_tasks.db",
        None,
    ),
    (
        "tts.profile_store",
        "tts_profiles_db_path",
        "tldw_chatbook_tts_profiles.db",
        None,
    ),
)


def database_leaf(setting_name: str) -> str:
    return next(row[2] for row in DATABASE_PATHS if row[1] == setting_name)


def custom_database_input(
    value: object, default: object, *, expand_before_validation: bool = True
) -> Path | None:
    """Select the custom spelling; callers own validation and opening authority."""
    if not value or value == default:
        return None
    selected = Path(str(value))
    return selected.expanduser() if expand_before_validation else selected


def database_path(config: Mapping[str, object], setting_name: str) -> Path:
    row = next(row for row in DATABASE_PATHS if row[1] == setting_name)
    # The shipped TOML default is a literal selector, independent of OS spelling.
    legacy = "~/.local/share/tldw_cli/" + row[3] if row[3] else None
    raw = setting(config, "database", setting_name)
    if raw is not None and not isinstance(raw, str):
        raise ValueError("invalid_config_path")
    selected = custom_database_input(
        raw, legacy, expand_before_validation=setting_name != "scheduled_tasks_db_path"
    )
    if selected is None:
        return user_data_dir(config) / row[2]
    # Same lexical selection rejection as validate_path_simple(probe_existing=False).
    # Do not import that module here: its metrics dependency bootstraps configuration.
    if any(
        value in str(selected)
        for value in (
            "../..",
            "..\\..",
            "~/",
            "~\\",
            "\0",
            "|",
            ";",
            "&&",
            "`",
            "$(",
            "${",
        )
    ):
        raise ValueError("invalid_config_path")
    if setting_name == "tts_profiles_db_path" and ".." in Path(str(raw)).parts:
        raise ValueError("invalid_config_path")
    return (
        lexical_path(selected).resolve()
        if setting_name == "tts_profiles_db_path"
        else lexical_path(selected)
    )
