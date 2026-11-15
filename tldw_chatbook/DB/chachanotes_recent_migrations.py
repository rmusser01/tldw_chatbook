"""Recent native schema steps; owners retain their existing transaction methods."""

from __future__ import annotations


def _migrate_from_v85_to_v86(self, conn) -> None:
    """Install conversation-owned operational prompt mix-in records."""
    from tldw_chatbook.DB import ChaChaNotes_DB as host

    self._require_migration_entry_version(conn, 85, "V85→V86")
    path = (
        host.Path(host.__file__).parent / "migrations" / "chachanotes_prompt_mixins.sql"
    )
    try:
        with self.transaction() as cursor:
            self._execute_migration_statements(
                cursor, path.read_text(encoding="utf-8"), "V85→V86"
            )
            updated = cursor.execute(
                "UPDATE db_schema_version SET version=86 WHERE schema_name=? AND version=85",
                (self._SCHEMA_NAME,),
            )
            if updated.rowcount != 1:
                raise host.SchemaError("V85→V86 version update failed")
    except (
        OSError,
        host.sqlite3.Error,
        host.CharactersRAGDBError,
        host.SchemaError,
    ) as exc:
        raise host.SchemaError(
            f"Migration V85→V86 failed: {type(exc).__name__}"
        ) from exc


def _migrate_from_v84_to_v85(self, conn) -> None:
    """Add local archive chronology while retaining shared sync events."""
    from tldw_chatbook.DB import ChaChaNotes_DB as host

    self._require_migration_entry_version(conn, 84, "V84→V85")
    migration_path = (
        host.Path(host.__file__).parent
        / "migrations"
        / "chachanotes_v84_to_v85_archive_chronology.sql"
    )
    try:
        with self.transaction(immediate=True) as cursor:
            self._execute_migration_statements(
                cursor, migration_path.read_text(encoding="utf-8"), "V84→V85"
            )
            updated = cursor.execute(
                "UPDATE db_schema_version SET version = 85 WHERE schema_name = ? AND version = 84",
                (self._SCHEMA_NAME,),
            )
            if updated.rowcount != 1:
                raise host.SchemaError(
                    "Archive chronology migration version update failed"
                )
        if self._get_db_version(conn) != 85:
            raise host.SchemaError("Archive chronology migration version check failed")
    except (
        OSError,
        host.sqlite3.Error,
        host.CharactersRAGDBError,
        host.SchemaError,
    ) as exc:
        raise host.SchemaError(
            f"Migration from V84 to V85 failed for '{self._SCHEMA_NAME}': {type(exc).__name__}"
        ) from exc


def _migrate_from_v86_to_v87(self, conn) -> None:
    """Install durable consolidation batch evidence and index the eight
    voice-promotion message locators (TASK-36510)."""
    from tldw_chatbook.DB import ChaChaNotes_DB as host

    self._require_migration_entry_version(conn, 86, "V86→V87")
    path = (
        host.Path(host.__file__).parent
        / "migrations"
        / "chachanotes_v86_to_v87_mobile_consolidation.sql"
    )
    try:
        with self.transaction() as cursor:
            self._execute_migration_statements(
                cursor, path.read_text(encoding="utf-8"), "V86→V87"
            )
            updated = cursor.execute(
                "UPDATE db_schema_version SET version = 87 WHERE schema_name = ? AND version = 86",
                (self._SCHEMA_NAME,),
            )
            if updated.rowcount != 1:
                raise host.SchemaError("V86→V87 version update failed")
        if self._get_db_version(conn) != 87:
            raise host.SchemaError("V86→V87 version check failed")
    except (
        OSError,
        host.sqlite3.Error,
        host.CharactersRAGDBError,
        host.SchemaError,
    ) as exc:
        raise host.SchemaError(
            f"Migration V86→V87 failed: {type(exc).__name__}"
        ) from exc


def _migrate_from_v83_to_v84(self, conn) -> None:
    """Install independent source-bound regions and optional applied provenance."""
    from tldw_chatbook.DB import ChaChaNotes_DB as host

    self._require_migration_entry_version(conn, 83, "V83→V84")
    path = (
        host.Path(host.__file__).parent
        / "migrations"
        / "chachanotes_v83_to_v84_portrait_regions.sql"
    )
    try:
        with self.transaction() as cursor:
            self._execute_migration_statements(
                cursor, path.read_text(encoding="utf-8"), "V83→V84"
            )
            changed = cursor.execute(
                "UPDATE db_schema_version SET version=84 WHERE schema_name=? AND version=83",
                (self._SCHEMA_NAME,),
            )
            if changed.rowcount != 1:
                raise host.SchemaError("V83→V84 version update failed")
    except (OSError, host.sqlite3.Error, host.CharactersRAGDBError) as exc:
        raise host.SchemaError(
            f"Migration V83→V84 failed: {type(exc).__name__}"
        ) from exc


def _migrate_from_v82_to_v83(self, conn) -> None:
    """Install source-bound local portrait sets without sync triggers."""
    from tldw_chatbook.DB import ChaChaNotes_DB as host

    self._require_migration_entry_version(conn, 82, "V82→V83")
    path = (
        host.Path(host.__file__).parent
        / "migrations"
        / "chachanotes_v82_to_v83_portrait_performance.sql"
    )
    try:
        with self.transaction() as cursor:
            self._execute_migration_statements(
                cursor, path.read_text(encoding="utf-8"), "V82→V83"
            )
            changed = cursor.execute(
                "UPDATE db_schema_version SET version=83 WHERE schema_name=? AND version=82",
                (self._SCHEMA_NAME,),
            )
            if changed.rowcount != 1:
                raise host.SchemaError("V82→V83 version update failed")
    except (OSError, host.sqlite3.Error, host.CharactersRAGDBError) as exc:
        raise host.SchemaError(
            f"Migration V82→V83 failed: {type(exc).__name__}"
        ) from exc


def _migrate_from_v81_to_v82(self, conn) -> None:
    """Permit immutable sibling ties while preserving one root per Canvas."""
    from tldw_chatbook.DB import ChaChaNotes_DB as host

    self._require_migration_entry_version(conn, 81, "V81→V82")
    path = (
        host.Path(host.__file__).parent
        / "migrations"
        / "chachanotes_v81_to_v82_canvas_sibling_sequences.sql"
    )
    try:
        with self.transaction() as cursor:
            predecessor = cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_canvas_revisions_canvas_sequence'"
            ).fetchone()
            if (
                predecessor is None
                or predecessor[0]
                != "CREATE UNIQUE INDEX idx_canvas_revisions_canvas_sequence\n  ON canvas_revisions(canvas_id, sequence)"
            ):
                raise host.SchemaError(
                    "incompatible Canvas sequence predecessor schema"
                )
            self._execute_migration_statements(
                cursor, path.read_text(encoding="utf-8"), "V81→V82"
            )
            updated = cursor.execute(
                "UPDATE db_schema_version SET version = 82 WHERE schema_name = ? AND version = 81",
                (self._SCHEMA_NAME,),
            )
            if updated.rowcount != 1:
                raise host.SchemaError("V81→V82 version update failed")
        if self._get_db_version(conn) != 82:
            raise host.SchemaError("V81→V82 version check failed")
    except (
        OSError,
        host.sqlite3.Error,
        host.CharactersRAGDBError,
        host.SchemaError,
    ) as exc:
        raise host.SchemaError(
            f"Migration V81→V82 failed: {type(exc).__name__}"
        ) from exc
