"""Installed local Dreams/Guardian recovery owners; importing opens no stores."""
from contextlib import closing
from pathlib import Path
from threading import Event

from tldw_chatbook.Backup_Recovery.models import OwnerAdapter, SchemaPolicy, StorageItem
from tldw_chatbook.DB.guardian_schema import GUARDIAN_SCHEMAS
from tldw_chatbook.DB.recovery_operations import _SQLiteDeclaration
from tldw_chatbook.DB.recovery_sqlite import _checked_capture, _validate_sqlite


class _GuardianAdapter(_SQLiteDeclaration):
    def schema_policy(self) -> SchemaPolicy:
        migration = Path(__file__).with_name("migrations") / "guardian_v1_to_v2_crisis_provenance.sql"
        statements = tuple(part.strip() for part in migration.read_text().split(";") if part.strip())
        return SchemaPolicy(self.owner_id, self.versions, self.schemas, ((1, 2, statements),))

    def validate(self, candidate: Path) -> tuple[str, ...]:
        import sqlite3

        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(connect_private_sqlite("recovery.operations.guardian", candidate, read_only=True)) as connection:
                versions = connection.execute("SELECT version FROM schema_version").fetchmany(2)
                if len(versions) != 1 or type(versions[0][0]) is not int:
                    return ("unsupported_schema_version",)
                return _validate_sqlite(connection, self.versions, self.schemas,
                    version_query="SELECT version FROM schema_version")
        except (OSError, ValueError, sqlite3.Error):
            return ("guardian_validation_unavailable",)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _checked_capture(self.owner_id, item, destination, cancel, self.validate) as guard:
            copy_private_sqlite("recovery.operations.guardian", item.path, destination, progress_guard=guard)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_GuardianAdapter("db.guardian", "guardian_db_path", "guardian.sqlite",
        (2, 1), GUARDIAN_SCHEMAS, optional_default=True),)
