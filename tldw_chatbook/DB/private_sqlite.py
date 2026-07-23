"""Checked ownership inventory for Chatbook SQLite targets.

Connection and backup behavior is added in later TASK-489 slices. This module
currently defines only the immutable target classifications and owner registry
used to keep that migration complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class SQLiteTargetKind(StrEnum):
    """Supported SQLite storage target classifications."""

    PRIVATE_FILE = "private_file"
    MEMORY = "memory"
    READ_ONLY_URI = "read_only_uri"


@dataclass(frozen=True, slots=True)
class SQLiteOwnerPolicy:
    """Immutable storage policy for one registered production owner."""

    production_module: str
    allowed_target_kinds: frozenset[SQLiteTargetKind]
    reason: str
    centralized_backup_allowed: bool = False


_PRIVATE_FILE = frozenset({SQLiteTargetKind.PRIVATE_FILE})
_MEMORY = frozenset({SQLiteTargetKind.MEMORY})
_PRIVATE_OR_MEMORY = frozenset({SQLiteTargetKind.PRIVATE_FILE, SQLiteTargetKind.MEMORY})
_READ_ONLY_URI = frozenset({SQLiteTargetKind.READ_ONLY_URI})

_SQLITE_OWNER_POLICIES = {
    "app.prompts_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/app",
        _PRIVATE_FILE,
        "Prompts startup participates in configured database parent policy.",
    ),
    "config.server_sqlite_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/config",
        _PRIVATE_FILE,
        "Stale server-only SQLite directory creation has no connection consumer.",
    ),
    "config.server_user_db_base": SQLiteOwnerPolicy(
        "tldw_chatbook/config",
        _PRIVATE_FILE,
        "Stale server user-database creation has no connection consumer.",
    ),
    "config.user_data_directory": SQLiteOwnerPolicy(
        "tldw_chatbook/config",
        _PRIVATE_FILE,
        "The application-owned default data directory is the private root.",
    ),
    "cookies.chrome": SQLiteOwnerPolicy(
        "tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner",
        _READ_ONLY_URI,
        "Chrome cookie clones are validated read-only SQLite sources.",
    ),
    "cookies.edge": SQLiteOwnerPolicy(
        "tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner",
        _READ_ONLY_URI,
        "Edge cookie clones are validated read-only SQLite sources.",
    ),
    "cookies.firefox": SQLiteOwnerPolicy(
        "tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner",
        _READ_ONLY_URI,
        "Firefox cookie clones are validated read-only SQLite sources.",
    ),
    "db.base": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/base_db",
        _PRIVATE_OR_MEMORY,
        "BaseDB is the shared file and memory connection owner for subclasses.",
    ),
    "db.chachanotes.backup": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/ChaChaNotes_DB",
        _PRIVATE_FILE,
        "ChaChaNotes backup targets require centralized private creation.",
        centralized_backup_allowed=True,
    ),
    "db.chachanotes.primary": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/ChaChaNotes_DB",
        _PRIVATE_OR_MEMORY,
        "ChaChaNotes owns private file and in-memory primary databases.",
    ),
    "db.evals": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Evals_DB",
        _PRIVATE_OR_MEMORY,
        "Evaluation storage supports private files and exact in-memory targets.",
    ),
    "db.library_ingest_jobs": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Library_Ingest_Jobs_DB",
        _PRIVATE_OR_MEMORY,
        "Library ingest jobs override the shared connection owner.",
    ),
    "db.media.backup": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Client_Media_DB_v2",
        _PRIVATE_FILE,
        "Media backup targets require centralized private creation.",
        centralized_backup_allowed=True,
    ),
    "db.media.integrity": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Client_Media_DB_v2",
        _READ_ONLY_URI,
        "Media integrity checks use a validated read-only SQLite URI.",
    ),
    "db.media.primary": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Client_Media_DB_v2",
        _PRIVATE_OR_MEMORY,
        "Media owns private file and in-memory primary databases.",
    ),
    "db.prompts.backup": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Prompts_DB",
        _PRIVATE_FILE,
        "Prompts backup targets require centralized private creation.",
        centralized_backup_allowed=True,
    ),
    "db.prompts.primary": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Prompts_DB",
        _PRIVATE_OR_MEMORY,
        "Prompts owns private file and in-memory primary databases.",
    ),
    "db.rag_indexing": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/RAG_Indexing_DB",
        _PRIVATE_OR_MEMORY,
        "RAG indexing supports private files and exact in-memory targets.",
    ),
    "db.search_history": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/search_history_db",
        _PRIVATE_OR_MEMORY,
        "Search history supports private files and exact in-memory targets.",
    ),
    "db.sync_client_example": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Sync_Client",
        _PRIVATE_FILE,
        "The executable sync example must not teach unsafe parent creation.",
    ),
    "eval.events_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/Event_Handlers/eval_events",
        _PRIVATE_FILE,
        "The evaluation event factory participates in default parent setup.",
    ),
    "eval.orchestrator_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/Evals/eval_orchestrator",
        _PRIVATE_FILE,
        "The evaluation orchestrator participates in default parent setup.",
    ),
    "kanban.local": SQLiteOwnerPolicy(
        "tldw_chatbook/Kanban_Interop/local_kanban_db",
        _PRIVATE_OR_MEMORY,
        "Local Kanban supports private files and exact in-memory targets.",
    ),
    "notes.library_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/Notes/Notes_Library",
        _PRIVATE_FILE,
        "The Notes library owns a per-user database parent.",
    ),
    "notifications.client": SQLiteOwnerPolicy(
        "tldw_chatbook/Notifications/client_notifications_db",
        _MEMORY,
        "Client notifications currently use only an in-memory database.",
    ),
    "notifications.event_state": SQLiteOwnerPolicy(
        "tldw_chatbook/Notifications/event_state_repository",
        _MEMORY,
        "Event state currently uses only an in-memory database.",
    ),
    "research.local": SQLiteOwnerPolicy(
        "tldw_chatbook/Research_Interop/local_research_service",
        _PRIVATE_OR_MEMORY,
        "Local research accepts private files and Path(':memory:').",
    ),
    "runtime.server_parity_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/runtime_policy/server_parity_state",
        _PRIVATE_FILE,
        "Server parity repositories use file-backed storage below this parent.",
    ),
    "settings.bulk_backup": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _PRIVATE_FILE,
        "The Settings bulk worker backs up all three Chatbook databases.",
        centralized_backup_allowed=True,
    ),
    "settings.integrity": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _READ_ONLY_URI,
        "Settings integrity checks require validated read-only access.",
    ),
    "settings.pre_restore_backup": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _PRIVATE_FILE,
        "Settings creates a private safety backup before restoring.",
        centralized_backup_allowed=True,
    ),
    "settings.restore": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _PRIVATE_FILE,
        "Settings restore uses verified source and destination identities.",
        centralized_backup_allowed=True,
    ),
    "settings.schema": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _READ_ONLY_URI,
        "Settings schema inspection requires validated read-only access.",
    ),
    "settings.single_backup": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _PRIVATE_FILE,
        "Settings single-database backups use centralized private creation.",
        centralized_backup_allowed=True,
    ),
    "settings.vacuum": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _PRIVATE_FILE,
        "Settings VACUUM requires a checked writable private database.",
    ),
    "sync.notes_mirror": SQLiteOwnerPolicy(
        "tldw_chatbook/Sync_Interop/notes_mirror",
        _PRIVATE_OR_MEMORY,
        "Notes mirror supports an optional private file or exact memory target.",
    ),
    "sync.state": SQLiteOwnerPolicy(
        "tldw_chatbook/Sync_Interop/sync_state_repository",
        _MEMORY,
        "Sync state currently uses only an in-memory database.",
    ),
    "tamagotchi.sqlite": SQLiteOwnerPolicy(
        "tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage",
        _PRIVATE_OR_MEMORY,
        "All SQLiteStorage methods share private file and Path(':memory:') support.",
    ),
    "utils.legacy_user_database_path": SQLiteOwnerPolicy(
        "tldw_chatbook/Utils/paths",
        _PRIVATE_FILE,
        "The unused legacy user database helper is an explicit exclusion.",
    ),
    "utils.project_databases_directory": SQLiteOwnerPolicy(
        "tldw_chatbook/Utils/paths",
        _PRIVATE_FILE,
        "Project template and demonstration databases are explicit exclusions.",
    ),
    "writing.local": SQLiteOwnerPolicy(
        "tldw_chatbook/Writing_Interop/local_writing_service",
        _PRIVATE_OR_MEMORY,
        "Local writing accepts private files and Path(':memory:').",
    ),
}

SQLITE_OWNER_REGISTRY: Mapping[str, SQLiteOwnerPolicy] = MappingProxyType(
    _SQLITE_OWNER_POLICIES
)

__all__ = [
    "SQLITE_OWNER_REGISTRY",
    "SQLiteOwnerPolicy",
    "SQLiteTargetKind",
]
