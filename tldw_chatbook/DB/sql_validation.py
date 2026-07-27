"""
SQL identifier validation module for preventing SQL injection in dynamic queries.

This module provides validation for table names, column names, and other SQL identifiers
to ensure they match expected patterns and are safe to use in dynamic SQL construction.
"""

import re
from typing import Optional
from loguru import logger

# Define valid table names for each database
#
# NOTE (TASK-864): the "chachanotes" set below used to only allowlist 9 of the
# ~47 real tables the live schema creates (``ChaChaNotes_DB._FULL_SCHEMA_SQL_V4``
# plus every ``_migrate_from_vX_to_vY`` step run up to
# ``CharactersRAGDB._CURRENT_SCHEMA_VERSION``), which made
# ``update_keyword_collection()`` raise ``ValueError`` unconditionally --
# ``keyword_collections`` itself was one of the omissions. This set is kept
# hand-maintained rather than derived at import time from a live
# ``CharactersRAGDB(":memory:")`` for two reasons: (1) building one runs the
# full schema-creation + every migration step as a side effect of importing a
# lightweight identifier-validation module used by three otherwise-unrelated
# DB modules (chachanotes/media/prompts each have their own schema; deriving
# for just one would be inconsistent), and (2) it would make this module
# depend on ``ChaChaNotes_DB`` at runtime, which itself imports from this
# module -- a lazy in-function import avoids a hard cycle but the
# heavyweight, logging-generating DB construction as a side effect of
# validating a table name is worse than the alternative below.
# Instead, ``Tests/DB/test_sql_validation.py`` asserts this set against
# ``sqlite_master`` of a real, fully-migrated ``CharactersRAGDB(":memory:")``
# (excluding FTS5 shadow/virtual tables and ``sqlite_sequence`` -- see that
# test for the exact filter) so any future migration that adds, renames, or
# removes a table fails the test immediately instead of surfacing as a user
# hitting an unconditional ``ValueError`` the next time they touch the new
# table through a generic CRUD helper.
VALID_TABLES = {
    "chachanotes": {
        "character_cards",
        "character_expression_images",
        "chat_dictionaries",
        "collection_keywords",
        "conversation_dictionaries",
        "conversation_keywords",
        "conversation_local_marks",
        "conversation_world_books",
        "conversations",
        "db_schema_version",
        "decks",
        "flashcard_assets",
        "flashcard_templates",
        "flashcards",
        "keyword_collections",
        "keywords",
        "learning_paths",
        "message_attachments",
        "message_generation_metadata",
        "messages",
        "mindmap_nodes",
        "mindmaps",
        "note_keywords",
        "notes",
        "quiz_attempts",
        "quiz_questions",
        "quizzes",
        "rag_answer_attempt_payloads",
        "rag_artifact_owner_leases",
        "rag_artifact_owner_operations",
        "rag_citation_traces",
        "rag_evidence_runs",
        "rag_evidence_snapshots",
        "rag_identity_context",
        "rag_legacy_migration_journal",
        "rag_message_trace_owners",
        "rag_payload_tombstones",
        "rag_source_observations",
        "rag_trace_evidence_refs",
        "review_history",
        "study_sessions",
        "sync_conflicts",
        "sync_log",
        "sync_sessions",
        "topics",
        "world_book_entries",
        "world_books",
    },
    "media": {
        "Media",
        "Keywords",
        "MediaKeywords",
        "MediaVersion",
        "MediaModifications",
        "UnvectorizedMediaChunks",
        "DocumentVersions",
        "IngestionTriggerTracking",
        "sync_log",
        "Media_fts",
        "Keywords_fts",
        "MediaChunks",
        "MediaChunks_fts",
        "Transcripts",
    },
    "prompts": {
        "Prompts",
        "Keywords",
        "PromptKeywords",
        "sync_log",
        "Prompts_fts",
        "Keywords_fts",
    },
}

# Define valid columns for each table (for most commonly used tables)
VALID_COLUMNS = {
    # ChaChaNotes DB
    "character_cards": {
        "id",
        "uuid",
        "name",
        "alternate_greetings",
        "description",
        "personality",
        "post_history_instructions",
        "first_mes",
        "mes_example",
        "scenario",
        "system_prompt",
        "creator_notes",
        "creator",
        "character_version",
        "avatar",
        "extensions",
        "tags",
        "created_at",
        "last_modified",
        "deleted",
        "version",
        "deleted_at",
        "client_id",
    },
    "conversations": {
        "id",
        "uuid",
        "character_id",
        "title",
        "deleted",
        "created_at",
        "last_modified",
        "version",
        "deleted_at",
        "client_id",
    },
    "messages": {
        "id",
        "uuid",
        "conversation_id",
        "sender",
        "content",
        "created_at",
        "last_modified",
        "deleted",
        "version",
        "deleted_at",
        "client_id",
    },
    "notes": {
        "id",
        "uuid",
        "title",
        "content",
        "created_at",
        "last_modified",
        "deleted",
        "version",
        "deleted_at",
        "client_id",
    },
    "keywords": {
        "id",
        "uuid",
        "keyword",
        "deleted",
        "created_at",
        "last_modified",
        "version",
        "deleted_at",
        "client_id",
    },
    # TASK-864: real caller is ChaChaNotes_DB.update_keyword_collection() /
    # soft_delete_keyword_collection() via _update_generic_item /
    # _soft_delete_generic_item, both passing pk_col_name="id". Columns
    # verified against ``PRAGMA table_info(keyword_collections)`` on a live
    # migrated DB -- note there is deliberately no ``uuid``/``deleted_at``
    # here, unlike the sibling tables above.
    "keyword_collections": {
        "id",
        "name",
        "parent_id",
        "created_at",
        "last_modified",
        "deleted",
        "client_id",
        "version",
    },
    # Media DB
    "Media": {
        "id",
        "uuid",
        "title",
        "type",
        "url",
        "content",
        "author",
        "ingestion_date",
        "last_modified",
        "deleted",
        "is_trash",
        "trash_date",
        "transcription_model",
        "vector_processing",
        "vector_id",
        "book_cover",
        "file_hash",
        "version",
        "deleted_at",
        "client_id",
    },
    "Keywords": {
        "id",
        "uuid",
        "keyword",
        "deleted",
        "last_modified",
        "version",
        "deleted_at",
        "client_id",
    },
    # TASK-864: these four are Media's soft-delete/undelete cascade child
    # tables (Client_Media_DB_v2.py, ``child_tables`` cascade loops in
    # ``soft_delete_media``/``undelete_media``), which always validate
    # ``fk_col="media_id"``/``uuid_col="uuid"``. Columns verified against
    # each table's own ``CREATE TABLE`` in Client_Media_DB_v2.py.
    "Transcripts": {
        "id",
        "media_id",
        "whisper_model",
        "transcription",
        "created_at",
        "uuid",
        "last_modified",
        "version",
        "client_id",
        "deleted",
        "prev_version",
        "merge_parent_uuid",
    },
    "MediaChunks": {
        "id",
        "media_id",
        "chunk_text",
        "start_index",
        "end_index",
        "chunk_id",
        "uuid",
        "last_modified",
        "version",
        "client_id",
        "deleted",
        "prev_version",
        "merge_parent_uuid",
    },
    "UnvectorizedMediaChunks": {
        "id",
        "media_id",
        "chunk_text",
        "chunk_index",
        "start_char",
        "end_char",
        "chunk_type",
        "creation_date",
        "last_modified_orig",
        "is_processed",
        "metadata",
        "uuid",
        "last_modified",
        "version",
        "client_id",
        "deleted",
        "prev_version",
        "merge_parent_uuid",
    },
    "DocumentVersions": {
        "id",
        "media_id",
        "version_number",
        "prompt",
        "analysis_content",
        "content",
        "created_at",
        "uuid",
        "last_modified",
        "version",
        "client_id",
        "deleted",
        "prev_version",
        "merge_parent_uuid",
    },
    # Prompts DB
    "Prompts": {
        "id",
        "uuid",
        "name",
        "system_prompt",
        "user_prompt",
        "created_at",
        "last_modified",
        "deleted",
        "version",
        "deleted_at",
        "client_id",
    },
    # TASK-864: real caller is Sync_Interop/sync_state_repository.py's
    # ``_ensure_sync_v2_profile_columns``, immediately before an
    # ``ALTER TABLE ... ADD COLUMN`` f-string. Columns verified against that
    # module's own ``CREATE TABLE IF NOT EXISTS sync_profile_state`` (not a
    # ChaChaNotes/Media/Prompts DB table, but sharing this module's
    # allow-list is simplest -- ``validate_column_name`` takes no db_type).
    "sync_profile_state": {
        "source_authority",
        "server_profile_id",
        "authenticated_principal_id",
        "workspace_scope",
        "profile_mode",
        "device_id",
        "dataset_id",
        "dataset_cursors",
        "capabilities",
        "dry_run_metadata",
        "last_error",
        "last_mirror_report_id",
        "updated_at",
    },
}

# Link table columns
LINK_TABLE_COLUMNS = {
    "conversation_keywords": {"conversation_id", "keyword_id", "created_at"},
    "collection_keywords": {"collection_id", "keyword_id", "created_at"},
    "note_keywords": {"note_id", "keyword_id", "created_at"},
    "MediaKeywords": {"media_id", "keyword_id"},
    "PromptKeywords": {"prompt_id", "keyword_id"},
}

# SQL identifier pattern - allows alphanumeric, underscore, and supports Unicode
# This pattern is designed to be safe while supporting non-English identifiers
SQL_IDENTIFIER_PATTERN = re.compile(r"^[\w\u0080-\uFFFF]+$", re.UNICODE)

# Reserved SQL keywords that should not be used as identifiers
SQL_RESERVED_KEYWORDS = {
    "SELECT",
    "FROM",
    "WHERE",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "TABLE",
    "INDEX",
    "VIEW",
    "UNION",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "ORDER",
    "BY",
    "GROUP",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "AS",
    "ON",
    "AND",
    "OR",
    "NOT",
    "NULL",
    "PRIMARY",
    "KEY",
    "FOREIGN",
    "REFERENCES",
    "CASCADE",
    "SET",
    "VALUES",
    "INTO",
    "EXISTS",
    "BETWEEN",
    "LIKE",
    "IN",
    "IS",
    "DISTINCT",
    "ALL",
}


def validate_identifier(identifier: str, identifier_type: str = "identifier") -> bool:
    """
    Validates a SQL identifier (table name, column name, etc.) for safety.

    Args:
        identifier: The SQL identifier to validate
        identifier_type: Type of identifier for logging (e.g., "table", "column")

    Returns:
        bool: True if valid, False otherwise
    """
    if not identifier:
        logger.warning(f"Empty {identifier_type} provided")
        return False

    # Check length limits
    if len(identifier) > 64:  # Common SQL identifier length limit
        logger.warning(f"{identifier_type} '{identifier}' exceeds maximum length")
        return False

    # Check against pattern
    if not SQL_IDENTIFIER_PATTERN.match(identifier):
        logger.warning(f"{identifier_type} '{identifier}' contains invalid characters")
        return False

    # Check against reserved keywords
    if identifier.upper() in SQL_RESERVED_KEYWORDS:
        logger.warning(f"{identifier_type} '{identifier}' is a reserved SQL keyword")
        return False

    return True


def validate_table_name(table_name: str, db_type: str) -> bool:
    """
    Validates a table name against the whitelist for a specific database type.

    Args:
        table_name: The table name to validate
        db_type: The database type ('chachanotes', 'media', or 'prompts')

    Returns:
        bool: True if valid, False otherwise
    """
    if not validate_identifier(table_name, "table name"):
        return False

    valid_tables = VALID_TABLES.get(db_type, set())
    if table_name not in valid_tables:
        logger.warning(f"Table '{table_name}' not in whitelist for {db_type} database")
        return False

    return True


def validate_column_name(column_name: str, table_name: Optional[str] = None) -> bool:
    """
    Validates a column name, optionally against a specific table's schema.

    TASK-864: when ``table_name`` is given, this fails CLOSED for a table
    that has no entry in ``VALID_COLUMNS`` -- it used to silently no-op
    (returning whatever ``validate_identifier`` alone decided) for any
    table not among that dict's keys, so a caller documenting "validated
    against this table's schema" wasn't actually getting that check for
    ``sync_profile_state`` or the Media cascade child tables (Transcripts,
    MediaChunks, UnvectorizedMediaChunks, DocumentVersions) -- not
    exploitable at the time (every real caller passed in-file literals),
    but the promised validation wasn't being delivered. Every real caller
    that passes a concrete ``table_name`` now has a matching
    ``VALID_COLUMNS`` entry (see the table-specific comments above), so
    this tightening does not change behavior for any of them; a future
    caller that introduces a new table must add its columns here first,
    the same way ``validate_table_name`` already requires a
    ``VALID_TABLES`` entry.

    Passing ``table_name=None`` (the "no schema context" case, e.g.
    generic identifier checks with no specific table in scope) is
    unaffected and still skips the per-table check entirely.

    Args:
        column_name: The column name to validate
        table_name: Optional table name to validate against specific schema

    Returns:
        bool: True if valid, False otherwise
    """
    if not validate_identifier(column_name, "column name"):
        return False

    if table_name:
        if table_name not in VALID_COLUMNS:
            logger.warning(
                f"No column allow-list registered for table '{table_name}'; "
                f"rejecting column '{column_name}' (fail-closed)"
            )
            return False
        if column_name not in VALID_COLUMNS[table_name]:
            logger.warning(
                f"Column '{column_name}' not in schema for table '{table_name}'"
            )
            return False

    return True


def validate_column_list(columns: list[str], table_name: Optional[str] = None) -> bool:
    """
    Validates a list of column names.

    Args:
        columns: List of column names to validate
        table_name: Optional table name to validate against specific schema

    Returns:
        bool: True if all columns are valid, False otherwise
    """
    for column in columns:
        if not validate_column_name(column, table_name):
            return False
    return True


def validate_link_table(table_name: str, col1_name: str, col2_name: str) -> bool:
    """
    Validates a link table and its column names.

    Args:
        table_name: The link table name
        col1_name: First column name
        col2_name: Second column name

    Returns:
        bool: True if valid, False otherwise
    """
    if table_name not in LINK_TABLE_COLUMNS:
        logger.warning(f"Link table '{table_name}' not recognized")
        return False

    valid_columns = LINK_TABLE_COLUMNS[table_name]
    if col1_name not in valid_columns or col2_name not in valid_columns:
        logger.warning(
            f"Invalid columns for link table '{table_name}': {col1_name}, {col2_name}"
        )
        return False

    return True


def get_safe_table_name(table_name: str, db_type: str) -> Optional[str]:
    """
    Returns a validated table name or None if invalid.

    Args:
        table_name: The table name to validate
        db_type: The database type

    Returns:
        Optional[str]: The table name if valid, None otherwise
    """
    if validate_table_name(table_name, db_type):
        return table_name
    return None


def get_safe_column_name(
    column_name: str, table_name: Optional[str] = None
) -> Optional[str]:
    """
    Returns a validated column name or None if invalid.

    Args:
        column_name: The column name to validate
        table_name: Optional table name for schema validation

    Returns:
        Optional[str]: The column name if valid, None otherwise
    """
    if validate_column_name(column_name, table_name):
        return column_name
    return None


# Helper function to escape identifiers (as a last resort)
def escape_identifier(identifier: str) -> str:
    """
    Escapes a SQL identifier by wrapping it in double quotes.
    Note: This should only be used after validation, not as a replacement for validation.

    Args:
        identifier: The identifier to escape

    Returns:
        str: The escaped identifier
    """
    # Replace any existing double quotes with two double quotes (SQL escaping)
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'
