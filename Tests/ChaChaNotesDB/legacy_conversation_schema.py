"""Shared helpers for legacy conversation schema migration tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence


_DB_SCHEMA_VERSION_TABLE_SQL = """
CREATE TABLE db_schema_version(
  schema_name TEXT PRIMARY KEY NOT NULL,
  version INTEGER NOT NULL
);
"""

_SYNC_LOG_TABLE_SQL = """
CREATE TABLE sync_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  operation TEXT NOT NULL,
  timestamp DATETIME NOT NULL,
  client_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  payload TEXT
);
"""

_LEGACY_V12_CONVERSATION_COLUMNS_SQL = """
  id TEXT PRIMARY KEY,
  root_id TEXT NOT NULL,
  forked_from_message_id TEXT,
  parent_conversation_id TEXT,
  character_id INTEGER,
  title TEXT,
  rating INTEGER,
  created_at DATETIME NOT NULL,
  last_modified DATETIME NOT NULL,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1
"""

_LEGACY_V13_CONVERSATION_COLUMNS_SQL = """
  id TEXT PRIMARY KEY,
  root_id TEXT NOT NULL,
  forked_from_message_id TEXT,
  parent_conversation_id TEXT,
  character_id INTEGER,
  title TEXT,
  rating INTEGER,
  created_at DATETIME NOT NULL,
  last_modified DATETIME NOT NULL,
  deleted BOOLEAN NOT NULL DEFAULT 0,
  client_id TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  assistant_kind TEXT,
  assistant_id TEXT,
  persona_memory_mode TEXT,
  scope_type TEXT NOT NULL DEFAULT 'global',
  workspace_id TEXT,
  state TEXT NOT NULL DEFAULT 'in-progress',
  topic_label TEXT,
  topic_label_source TEXT,
  topic_last_tagged_at TEXT,
  topic_last_tagged_message_id TEXT,
  cluster_id TEXT,
  source TEXT,
  external_ref TEXT
"""

_LEGACY_V12_INSERT_COLUMNS = (
    "id",
    "root_id",
    "forked_from_message_id",
    "parent_conversation_id",
    "character_id",
    "title",
    "rating",
    "created_at",
    "last_modified",
    "deleted",
    "client_id",
    "version",
)

_LEGACY_V13_INSERT_COLUMNS = (
    "id",
    "root_id",
    "forked_from_message_id",
    "parent_conversation_id",
    "character_id",
    "title",
    "rating",
    "created_at",
    "last_modified",
    "deleted",
    "client_id",
    "version",
    "assistant_kind",
    "assistant_id",
    "persona_memory_mode",
    "scope_type",
    "workspace_id",
    "state",
    "topic_label",
    "topic_label_source",
    "topic_last_tagged_at",
    "topic_last_tagged_message_id",
    "cluster_id",
    "source",
    "external_ref",
)


def _create_legacy_conversations_db(
    db_path: Path,
    *,
    schema_version: int,
    conversation_columns_sql: str,
    insert_columns: Sequence[str],
    rows: Sequence[tuple[object, ...]],
) -> None:
    connection = sqlite3.connect(str(db_path))
    insert_column_list = ",\n              ".join(insert_columns)
    placeholders = ", ".join("?" for _ in insert_columns)
    try:
        connection.executescript(
            f"""
            {_DB_SCHEMA_VERSION_TABLE_SQL}

            INSERT INTO db_schema_version(schema_name, version)
            VALUES('rag_char_chat_schema', {schema_version});

            CREATE TABLE conversations(
{conversation_columns_sql}
            );

            {_SYNC_LOG_TABLE_SQL}
            """
        )
        connection.executemany(
            f"""
            INSERT INTO conversations(
              {insert_column_list}
            )
            VALUES({placeholders})
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def create_legacy_v12_conversations_db(
    db_path: Path,
    rows: Sequence[tuple[object, ...]],
) -> None:
    _create_legacy_conversations_db(
        db_path,
        schema_version=12,
        conversation_columns_sql=_LEGACY_V12_CONVERSATION_COLUMNS_SQL,
        insert_columns=_LEGACY_V12_INSERT_COLUMNS,
        rows=rows,
    )


def create_legacy_v13_conversations_db(
    db_path: Path,
    rows: Sequence[tuple[object, ...]],
) -> None:
    _create_legacy_conversations_db(
        db_path,
        schema_version=13,
        conversation_columns_sql=_LEGACY_V13_CONVERSATION_COLUMNS_SQL,
        insert_columns=_LEGACY_V13_INSERT_COLUMNS,
        rows=rows,
    )
