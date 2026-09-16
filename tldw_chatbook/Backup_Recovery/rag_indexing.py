"""Exact installed indexing-state SQLite schema; no vector engine imports."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass

from tldw_chatbook.DB.recovery_sqlite import _checked_capture, _validate_sqlite

from .models import SchemaPolicy, discovery_context, storage_logical_id
from .profile_paths import database_path
from .recovery_files import _RawDeclaration

_SCHEMA = (
    (
        0,
        (
            "CREATE INDEX idx_indexed_items_indexed \n        ON indexed_items(last_indexed)",
            "CREATE INDEX idx_indexed_items_modified \n        ON indexed_items(last_modified)",
            "CREATE INDEX idx_indexed_items_type \n        ON indexed_items(item_type)",
            "CREATE TABLE collection_state (\n            collection_name TEXT PRIMARY KEY,\n            last_full_index DATETIME,\n            total_items INTEGER DEFAULT 0,\n            indexed_items INTEGER DEFAULT 0,\n            metadata TEXT\n        )",
            "CREATE TABLE indexed_items (\n            item_id TEXT NOT NULL,\n            item_type TEXT NOT NULL,\n            last_indexed DATETIME NOT NULL,\n            last_modified DATETIME NOT NULL,\n            chunk_count INTEGER DEFAULT 0,\n            metadata TEXT,\n            PRIMARY KEY (item_id, item_type)\n        )",
        ),
    ),
)


@dataclass(frozen=True)
class _Indexing(_RawDeclaration):
    def discover(self, config):
        from dataclasses import replace

        from .rag_inventory import _absent

        path = database_path(config, "rag_indexing_db_path")
        context = discovery_context(config)
        entries = _absent(config, self.owner_id, path) or (self._item(config, path),)
        return tuple(
            replace(
                item,
                logical_id=storage_logical_id(context, self.owner_id),
                dependencies=tuple(
                    storage_logical_id(context, owner)
                    for owner in (
                        "config",
                        "db.media.primary",
                        "db.chachanotes.primary",
                        "db.prompts.primary",
                    )
                ),
            )
            for item in entries
        )

    def schema_policy(self):
        return SchemaPolicy(self.owner_id, (0,), _SCHEMA, ())

    def _validate_connection(self, connection):
        return _validate_sqlite(connection, (0,), _SCHEMA)

    def validate(self, candidate):
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.rag_indexing", candidate, read_only=True
                )
            ) as connection:
                return self._validate_connection(connection)
        except (OSError, ValueError, sqlite3.Error):
            return ("rag_indexing_validation_unavailable",)

    def capture(self, item, destination, cancel):
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.rag_indexing", item.path, destination, progress_guard=guard
            )


def recovery_adapters():
    return (_Indexing("db.rag_indexing"),)
