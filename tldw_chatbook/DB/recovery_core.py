"""Installed core recovery declarations, independent of runtime constructors."""

from dataclasses import dataclass
from contextlib import closing
import sqlite3
from pathlib import Path
from threading import Event
from typing import Mapping

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    SchemaPolicy,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import database_path


@dataclass(frozen=True)
class _CoreAdapter:
    owner_id: str
    setting_name: str
    dependent_owners: tuple[str, ...] = ()
    activation_required: bool = True

    def discover(self, config: Mapping[str, object]) -> tuple[StorageItem, ...]:
        context = discovery_context(config)
        path = database_path(config, self.setting_name)
        status = "included" if path.is_file() else "missing_required"
        return (
            StorageItem(
                self.owner_id,
                storage_logical_id(context, self.owner_id),
                path,
                status,
                tuple(
                    storage_logical_id(
                        context,
                        owner,
                        ""
                        if owner == "config" or owner.startswith("db.")
                        else "unresolved",
                    )
                    for owner in ("config",) + self.dependent_owners
                ),
            ),
        )

    @property
    def backup_owner_id(self) -> str:
        return "recovery.core." + self.owner_id.removeprefix("db.").removesuffix(
            ".primary"
        )

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from .private_sqlite import copy_private_sqlite
        from tldw_chatbook.Backup_Recovery.admission import _local

        if (
            item.owner != self.owner_id
            or item.path is None
            or item.status != "included"
        ):
            raise ValueError("invalid_capture_item")
        scope = getattr(_local, "capture_scope", None)
        if scope is None:
            raise ValueError("capture_requires_maintenance")
        scope.check()
        if destination.exists():
            raise FileExistsError("capture_destination_exists")

        def check_cancelled() -> None:
            if cancel.is_set():
                raise InterruptedError("cancelled")

        check_cancelled()
        copy_private_sqlite(
            self.backup_owner_id, item.path, destination, progress_guard=check_cancelled
        )
        check_cancelled()
        issues = self.validate(destination)
        if issues:
            raise ValueError(issues[0])

    def validate(self, candidate: Path) -> tuple[str, ...]:
        from .private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(self.backup_owner_id, candidate, read_only=True)
            ) as connection:
                connection.execute("PRAGMA trusted_schema = OFF")
                actual = tuple(
                    row[0]
                    for row in connection.execute(
                        "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
                    )
                )
                policy = self.schema_policy()
                if actual != policy.schema_sql[0][1]:
                    return ("unsupported_schema",)
                version_sql = (
                    "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
                    if self.owner_id == "db.chachanotes.primary"
                    else "SELECT version FROM schema_version"
                )
                if (
                    tuple(row[0] for row in connection.execute(version_sql))
                    != policy.versions
                ):
                    return ("unsupported_schema_version",)
                if (
                    connection.execute("PRAGMA foreign_key_check").fetchone()
                    is not None
                ):
                    return ("invalid_domain_reference",)
                if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    return ("invalid_sqlite_integrity",)
                return ()
        except (OSError, ValueError, sqlite3.Error):
            return ("core_validation_unavailable",)

    def relocate(self, candidate: Path, mapping: Mapping[str, Path]) -> None:
        # Core-owned attachments are BLOBs and managed visual asset locators are
        # relative. Their identities remain stable when the owning DB/asset roots
        # move. Absolute sync/dictionary/ingest locators belong to other owners or
        # user external inputs: never rewrite them or activate their bindings here.
        issues = self.validate(candidate)
        if issues:
            raise ValueError(issues[0])

    def validate_dependencies(
        self, item: StorageItem, candidate: Path, candidates: Mapping[str, Path]
    ) -> tuple[str, ...]:
        """Validate local links using only this item's declared final logical IDs.

        The recovery executor supplies a locally validated staging map. Other
        profiles' similarly named owners never satisfy a missing dependency.
        Later asset owners remain responsible for validating their byte inventories.
        """
        from .private_sqlite import connect_private_sqlite
        from .sql_validation import validate_identifier, escape_identifier

        if item.owner != self.owner_id or not item.logical_id.startswith("profile:"):
            return ("invalid_dependency_context",)
        parts = item.logical_id.split(":")
        if len(parts) != 3 or parts[2] != self.owner_id or not parts[1]:
            return ("invalid_dependency_context",)
        prefix = "profile:" + parts[1] + ":"
        issues = self.validate(candidate)
        if issues:
            return issues
        adapters = {adapter.owner_id: adapter for adapter in core_adapters()}
        try:
            with closing(
                connect_private_sqlite(self.backup_owner_id, candidate, read_only=True)
            ) as connection:
                connection.execute("PRAGMA trusted_schema=OFF")
                references = []
                if self.owner_id == "db.library_ingest_jobs":
                    references = [
                        ("db.media.primary", "Media", "id", row[0])
                        for row in connection.execute(
                            "SELECT media_id FROM ingest_jobs WHERE origin='local' AND media_id IS NOT NULL"
                        )
                    ]
                elif self.owner_id == "db.library_collections":
                    targets = {
                        "media": ("db.media.primary", "Media", "id"),
                        "note": ("db.chachanotes.primary", "notes", "id"),
                        "conversation": (
                            "db.chachanotes.primary",
                            "conversations",
                            "id",
                        ),
                        "prompt": ("db.prompts.primary", "Prompts", "id"),
                        "collection": (
                            "db.library_collections",
                            "library_collections",
                            "collection_id",
                        ),
                    }
                    for kind, identity in connection.execute(
                        "SELECT source_type,source_id FROM library_collection_items"
                    ):
                        target = targets.get(kind.lower())
                        if target is None:
                            return ("unsupported_domain_reference",)
                        references.append((*target, identity))
                elif self.owner_id == "db.chachanotes.primary":
                    # No normal constructor, filesystem probing from DB text, or
                    # invented owner/tombstone authority is involved in this check.
                    for query, owner in (
                        (
                            "SELECT 1 FROM notes WHERE file_path_on_disk IS NOT NULL OR sync_root_folder IS NOT NULL LIMIT 1",
                            "notes.file_notes",
                        ),
                        (
                            "SELECT 1 FROM chat_dictionaries WHERE file_path IS NOT NULL LIMIT 1",
                            "chat.dictionaries",
                        ),
                        (
                            "SELECT 1 FROM persona_visual_assets UNION ALL SELECT 1 FROM visual_identity_assets LIMIT 1",
                            "persona.assets",
                        ),
                    ):
                        if connection.execute(query).fetchone() is not None:
                            key = prefix + owner + ":unresolved"
                            if (
                                key not in item.dependencies
                                or key not in candidates
                                or not candidates[key].exists()
                            ):
                                return ("dependency_unavailable",)
                for owner, table, column, identity in references:
                    key = prefix + owner
                    if owner == self.owner_id:
                        target = candidate
                    else:
                        if key not in item.dependencies or key not in candidates:
                            return ("dependency_unavailable",)
                        target = candidates[key]
                    validation = adapters[owner].validate(target)
                    if validation:
                        return ("dependency_unavailable",)
                    with closing(
                        connect_private_sqlite(
                            adapters[owner].backup_owner_id, target, read_only=True
                        )
                    ) as peer:
                        peer.execute("PRAGMA trusted_schema=OFF")
                        # Identifiers above are installed literals, never DB text.
                        if not validate_identifier(table) or not validate_identifier(
                            column
                        ):
                            return ("unsupported_domain_reference",)
                        query = f"SELECT 1 FROM {escape_identifier(table)} WHERE {escape_identifier(column)}=?"
                        if peer.execute(query, (identity,)).fetchone() is None:
                            return ("invalid_domain_reference",)
                return ()
        except (OSError, ValueError, sqlite3.Error):
            return ("dependency_unavailable",)

    def schema_policy(self) -> SchemaPolicy | None:
        from .recovery_core_schema import CORE_SCHEMAS

        _, version, sql = next(row for row in CORE_SCHEMAS if row[0] == self.owner_id)
        return SchemaPolicy(self.owner_id, (version,), ((version, sql),), ())


def core_adapters() -> tuple[OwnerAdapter, ...]:
    """Return installed declarations without opening stores or importing services."""
    return (
        _CoreAdapter(
            "db.chachanotes.primary",
            "chachanotes_db_path",
            (
                "notes.file_notes",
                "notes.sync_bindings",
                "chat.dictionaries",
                "persona.assets",
                "chat.attachments",
            ),
        ),
        _CoreAdapter("db.media.primary", "media_db_path"),
        _CoreAdapter("db.prompts.primary", "prompts_db_path"),
        _CoreAdapter(
            "db.library_collections",
            "library_collections_db_path",
            (
                "db.media.primary",
                "db.chachanotes.primary",
                "db.prompts.primary",
                "skills",
            ),
        ),
        _CoreAdapter(
            "db.library_ingest_jobs",
            "library_ingest_jobs_db_path",
            ("db.media.primary",),
        ),
    )
