"""Private, versioned plugin metadata. Stored intent never constitutes trust."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Self

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

if TYPE_CHECKING:
    from tldw_chatbook.Plugins.models import PackageInspection
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

SCHEMA_VERSION = 2
PAGE_SIZE = 50
SELECT_INSTALLATIONS = """
SELECT installation_id, revision_digest, activation_default
FROM installations ORDER BY installation_id LIMIT ? OFFSET ?
"""


def validate_page(limit: int, offset: int) -> None:
    """Reject unbounded/invalid page requests before reaching SQLite."""
    if type(limit) is not int or not 1 <= limit <= PAGE_SIZE:
        raise ValueError("limit must be between 1 and 50")
    if type(offset) is not int or not 0 <= offset <= 2**63 - 1:
        raise ValueError("offset must be a nonnegative SQLite integer")


def _schema_statements(migration: str = "001_initial.sql") -> tuple[str, ...]:
    text = (
        files("tldw_chatbook.Plugins")
        .joinpath("migrations", migration)
        .read_text(encoding="utf-8")
    )
    statements, pending = [], ""
    for line in text.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statements.append(pending)
            pending = ""
    if pending.strip():
        raise sqlite3.DatabaseError("incomplete plugin migration")
    return tuple(statements)


class PluginRegistry:
    """Own a synchronous SQLite handle; disk writes require a live runtime owner.

    Args:
        path: Registry file under the resolved plugin root, or ``:memory:``.
        owner: Acquired owner of this exact root, required for disk mutation.
        read_only: Explicitly restrict an owner-backed view to browsing.
    """

    def __init__(
        self,
        path: Path,
        *,
        owner: PluginRuntimeOwner | None = None,
        read_only: bool | None = None,
    ) -> None:
        self._memory = str(path) == ":memory:"
        self.path = path if self._memory else path.parent.resolve() / path.name
        self._owner = owner
        self.read_only = (
            (not self._memory and owner is None) if read_only is None else read_only
        )
        self._closed = False
        if not self.read_only:
            self._require_write()
        self._connection = connect_private_sqlite(
            "plugins.registry",
            self.path,
            read_only=self.read_only,
            must_exist=self.read_only,
            isolation_level=None,
            # The body authorizer must run on every execution, including reuse
            # after SQLite automatically rolls back a failed statement.
            cached_statements=0,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys=ON")
            if not self.read_only:
                mode = self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if mode != ("memory" if self._memory else "wal"):
                    raise sqlite3.DatabaseError("plugin WAL unavailable")
                self._connection.execute("PRAGMA synchronous=FULL")
                self._connection.execute("PRAGMA fullfsync=ON")
                if self._connection.execute("PRAGMA synchronous").fetchone()[0] != 2:
                    raise sqlite3.DatabaseError("plugin durable commit unavailable")
                if self._connection.execute("PRAGMA fullfsync").fetchone()[0] != 1:
                    raise sqlite3.DatabaseError("plugin fullfsync unavailable")
            self._initialize()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    def _require_write(self) -> None:
        if self._closed or self.read_only:
            raise PermissionError("plugin registry is read only or closed")
        if not self._memory:
            if self._owner is None:
                raise PermissionError("plugin mutation requires runtime ownership")
            self._owner.require_owner(self.path.parent)

    def _initialize(self) -> None:
        connection = self._connection
        connection.execute("BEGIN" if self.read_only else "BEGIN IMMEDIATE")
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            objects = connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall()
            if version == 0 and not objects and not self.read_only:
                for statement in _schema_statements():
                    connection.execute(statement)
                version = 1
            if version == 1 and not self.read_only:
                # Validate the exact predecessor before applying any schema change.
                with PluginRegistry._reference_schema(version=1) as reference:
                    expected_v1 = [
                        tuple(row)
                        for row in reference.execute(
                            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                        )
                    ]
                actual_v1 = [
                    tuple(row)
                    for row in connection.execute(
                        "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                    )
                ]
                if actual_v1 != expected_v1:
                    raise sqlite3.DatabaseError("invalid plugin migration predecessor")
                for statement in _schema_statements("002_authority.sql"):
                    connection.execute(statement)
                connection.execute("PRAGMA user_version=2")
                version = 2
            if version != SCHEMA_VERSION:
                raise sqlite3.DatabaseError("unsupported plugin registry schema")
            # Exact DDL validation catches missing constraints/columns/triggers, not
            # merely a forged user_version. The reference has no filesystem owner.
            with PluginRegistry._reference_schema() as reference:
                expected = [
                    tuple(row)
                    for row in reference.execute(
                        "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                    )
                ]
            actual = [
                tuple(row)
                for row in connection.execute(
                    "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                )
            ]
            if (
                actual != expected
                or connection.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            ):
                raise sqlite3.DatabaseError("invalid plugin registry schema")
            if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise sqlite3.DatabaseError("invalid plugin registry references")
            if not self.read_only:
                self._require_write()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    @staticmethod
    @contextmanager
    def _reference_schema(
        version: int = SCHEMA_VERSION,
    ) -> Iterator[sqlite3.Connection]:
        # Reuse the registered memory seam for schema validation as well.
        connection = connect_private_sqlite("plugins.registry", ":memory:")
        try:
            for statement in _schema_statements():
                connection.execute(statement)
            if version >= 2:
                for statement in _schema_statements("002_authority.sql"):
                    connection.execute(statement)
            yield connection
        finally:
            connection.close()

    @property
    def schema_version(self) -> int:
        """Return the schema version of this validated view."""
        return self._connection.execute("PRAGMA user_version").fetchone()[0]

    def _authorize_transaction_body(self, action: int, *_args: object) -> int:
        # executescript first requests COMMIT, even before parsing the script.
        # Deny it, explicit transaction control, and any work following SQLite's
        # automatic rollback (for example a caught INSERT OR ROLLBACK failure).
        if action in (sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT):
            return sqlite3.SQLITE_DENY
        if not self._connection.in_transaction:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Commit atomically and durably before returning; rollback on any failure.

        A single worker must serialize access. Nested transactions are refused.
        Body statements cannot commit/rollback/savepoint or execute scripts.
        This low-level service seam is for trusted coordinator code, not plugins.
        """
        self._require_write()
        if self._connection.in_transaction:
            raise RuntimeError("nested plugin transaction")
        self._connection.execute("BEGIN IMMEDIATE")
        cursor = self._connection.cursor()
        try:
            self._connection.set_authorizer(self._authorize_transaction_body)
            try:
                yield cursor
                self._require_write()
                if not self._connection.in_transaction:
                    raise RuntimeError("plugin transaction ended inside its body")
            finally:
                # Only this wrapper may settle the transaction, after ownership
                # validation. Remove the body guard for its commit or rollback.
                self._connection.set_authorizer(None)
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        finally:
            cursor.close()

    def list_installations(self, *, limit: int, offset: int) -> tuple[dict, ...]:
        """Return a bounded, deterministic page of untrusted installation intent."""
        validate_page(limit, offset)
        return tuple(
            dict(row)
            for row in self._connection.execute(SELECT_INSTALLATIONS, (limit, offset))
        )

    def authority_projection(self, *, operation_result: dict | None) -> dict:
        """Read complete logical authority in one SQLite snapshot.

        This validates shape, not reviewed authorization. The coordinator must
        compare it with protected authority before admission. Package definitions
        are retained references: recovery must reinspect exact immutable material.
        A caller-owned transaction remains owned by its existing guarded wrapper.
        """
        from tldw_chatbook.Skills_Interop.skill_trust_crypto import (
            canonical_json,
            sha256_hex,
        )

        from .authority import canonical_snapshot, empty_snapshot
        from .models import ComponentRecord, PackageInspection
        from .package_files import parse_document

        started = not self._connection.in_transaction
        if started:
            self._connection.execute("BEGIN")
        try:
            result = empty_snapshot()
            result["operation_result"] = operation_result
            queries = {
                "installations": "SELECT installation_id, revision_digest, activation_default FROM installations ORDER BY installation_id",
                "selections": "SELECT installation_id, revision_digest, component_id, selected FROM selections ORDER BY installation_id, revision_digest, component_id",
                "activation": "SELECT installation_id, workspace_id, intent FROM activation ORDER BY installation_id, workspace_id",
                "authority_generations": "SELECT installation_id, scope_kind, workspace_id, generation, revoked FROM authority_generations ORDER BY installation_id, scope_kind, workspace_id",
                "revision_trust": "SELECT installation_id, revision_digest, reviewed FROM revision_trust ORDER BY installation_id, revision_digest",
                "tombstones": "SELECT installation_id, generation, operation_id FROM tombstones ORDER BY installation_id",
                "data_roots": "SELECT root_id, installation_id, workspace_id, path, generation, deletion_fenced FROM data_roots ORDER BY root_id",
            }
            booleans = {
                "activation_default",
                "selected",
                "revoked",
                "reviewed",
                "deletion_fenced",
            }
            for name, query in queries.items():
                for record in self._connection.execute(query):
                    row = dict(record)
                    for field in booleans & row.keys():
                        if type(row[field]) is not int or row[field] not in (0, 1):
                            raise ValueError("invalid registry boolean")
                        row[field] = bool(row[field])
                    result[name].append(row)
            inventories = {}
            for row in self._connection.execute(
                "SELECT installation_id, revision_digest, inspection_json FROM revisions ORDER BY installation_id, revision_digest"
            ):
                raw = parse_document(row["inspection_json"].encode())
                inspection = PackageInspection.model_validate_json(
                    json.dumps(raw), strict=True
                )
                if inspection.effective_digest != row["revision_digest"]:
                    raise ValueError("revision inspection identity mismatch")
                fields = (
                    "source_identity",
                    "dialect",
                    "format_version",
                    "adapter_version",
                    "root_manifest",
                    "overlay_identities",
                    "content_digest",
                    "source_digest",
                    "materialized_identity",
                    "link_targets",
                    "activation_blockers",
                    "rejected",
                )
                data = inspection.model_dump(mode="json")
                result["revisions"].append(
                    {
                        "installation_id": row["installation_id"],
                        "revision_digest": row["revision_digest"],
                        **{field: data[field] for field in fields},
                        "variables_digest": sha256_hex(
                            canonical_json(
                                parse_document(inspection.variables_json.encode())
                            )
                        ),
                    }
                )
                inventories[(row["installation_id"], row["revision_digest"])] = dict(
                    inspection.inventory
                )
            for row in self._connection.execute(
                "SELECT installation_id, revision_digest, component_id, definition_json FROM components ORDER BY installation_id, revision_digest, component_id"
            ):
                raw = parse_document(row["definition_json"].encode())
                component = ComponentRecord.model_validate_json(
                    json.dumps(raw), strict=True
                )
                identity = (row["installation_id"], row["revision_digest"])
                if (
                    row["component_id"] != component.component_id
                    or inventories[identity].pop(component.component_id, None)
                    != component
                ):
                    raise ValueError("component inspection identity mismatch")
                data = component.model_dump(mode="json")
                for field in ("availability", "evidence", "definition_json"):
                    data.pop(field)
                result["components"].append(
                    {
                        "installation_id": row["installation_id"],
                        "revision_digest": row["revision_digest"],
                        **data,
                        "definition_digest": sha256_hex(
                            canonical_json(
                                parse_document(
                                    b'{"definition":'
                                    + component.definition_json.encode()
                                    + b"}"
                                )["definition"]
                            )
                        ),
                    }
                )
            if any(inventories.values()):
                raise ValueError("missing recorded components")
            for row in self._connection.execute(
                "SELECT installation_id, mapping_id, mapping_json FROM mappings ORDER BY installation_id, mapping_id"
            ):
                data = parse_document(row["mapping_json"].encode())
                if (
                    data.get("installation_id") != row["installation_id"]
                    or data.get("mapping_id") != row["mapping_id"]
                ):
                    raise ValueError("mapping identity mismatch")
                result["mappings"].append(data)
            return canonical_snapshot(result)
        finally:
            if started:
                self._connection.rollback()

    def read_operation(self, operation_id: str) -> dict | None:
        """Load one untrusted operation hint with a closed intended-result shape.

        The phase is not commitment proof; F4 must authenticate exact protected
        transition evidence. This lookup never creates missing/default authority.
        """
        from .authority import OperationResult
        from .package_files import parse_document

        row = self._connection.execute(
            "SELECT operation_id, installation_id, phase, intent_json FROM operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        result = OperationResult.model_validate(
            parse_document(row["intent_json"].encode())
        ).model_dump(mode="json")
        if (
            result["operation_id"] != row["operation_id"]
            or result["installation_id"] != row["installation_id"]
        ):
            raise ValueError("operation identity mismatch")
        if row["phase"] not in {
            "prepared",
            "committed",
            "complete",
            "recovery_required",
        }:
            raise ValueError("invalid operation phase")
        return {"phase": row["phase"], "result": result}

    def list_operations(self, *, limit: int, offset: int) -> tuple[dict, ...]:
        """Return bounded operation hints; callers must authenticate each result."""
        validate_page(limit, offset)
        ids = self._connection.execute(
            "SELECT operation_id FROM operations ORDER BY operation_id LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return tuple(self.read_operation(row[0]) for row in ids)

    def insert_installation(
        self,
        cursor: sqlite3.Cursor,
        installation_id: str,
        inspection: PackageInspection,
        selection: tuple[str, ...],
    ) -> None:
        """Insert a fresh disabled installation inside the caller's transaction."""
        self._require_write()
        if not self._connection.in_transaction:
            raise RuntimeError("installation requires owned transaction")
        revision = inspection.effective_digest
        cursor.execute(
            "INSERT INTO installations VALUES (?, ?, 0)", (installation_id, revision)
        )
        cursor.execute(
            "INSERT INTO revisions VALUES (?, ?, ?)",
            (installation_id, revision, inspection.model_dump_json()),
        )
        for component in inspection.inventory.values():
            cursor.execute(
                "INSERT INTO components VALUES (?, ?, ?, ?)",
                (
                    installation_id,
                    revision,
                    component.component_id,
                    component.model_dump_json(),
                ),
            )
            cursor.execute(
                "INSERT INTO selections VALUES (?, ?, ?, ?)",
                (
                    installation_id,
                    revision,
                    component.component_id,
                    component.component_id in selection,
                ),
            )
        cursor.execute(
            "INSERT INTO revision_trust VALUES (?, ?, 0)", (installation_id, revision)
        )
        for scope in ("installation", "global_default"):
            cursor.execute(
                "INSERT INTO authority_generations VALUES (?, ?, '', 1, 0)",
                (installation_id, scope),
            )

    def write_operation(
        self, cursor: sqlite3.Cursor, result: dict, *, phase: str
    ) -> None:
        """Persist a validated hint after its authority snapshot was prepared."""
        from .authority import OperationResult

        self._require_write()
        result = OperationResult.model_validate(result).model_dump(mode="json")
        if phase not in {"committed", "complete", "recovery_required"}:
            raise ValueError("invalid operation phase")
        prior = self.read_operation(result["operation_id"])
        if prior is not None and prior["result"] != result:
            raise ValueError("operation identity conflict")
        cursor.execute(
            "INSERT INTO operations VALUES (?, ?, ?, ?) ON CONFLICT(operation_id) DO UPDATE SET phase=excluded.phase",
            (
                result["operation_id"],
                result["installation_id"],
                phase,
                json.dumps(result),
            ),
        )

    def restore_authority(
        self, snapshot: dict, inspections: dict[tuple[str, str], PackageInspection]
    ) -> None:
        """Reconstruct exact authenticated authority, retaining all process evidence.

        The coordinator authenticates the snapshot and retained bytes first.
        Reprojection inside this transaction proves definitions and references
        match. Unknown runtime provenance becomes explicit unresolved evidence;
        an empty rebuilt process table never establishes successful cleanup.
        """
        from .authority import canonical_snapshot

        snapshot = canonical_snapshot(snapshot)
        with self.transaction() as cursor:
            sources = cursor.execute("SELECT * FROM sources").fetchall()
            for table in (
                "sources",
                "selections",
                "activation",
                "mappings",
                "authority_generations",
                "revision_trust",
                "components",
                "revisions",
                "installations",
                "tombstones",
                "data_roots",
            ):
                # Closed internal table names, never user-provided identifiers.
                cursor.execute(f"DELETE FROM {table}")
            for row in snapshot["installations"]:
                cursor.execute(
                    "INSERT INTO installations VALUES (?, ?, ?)",
                    tuple(
                        row[key]
                        for key in (
                            "installation_id",
                            "revision_digest",
                            "activation_default",
                        )
                    ),
                )
            for row in snapshot["revisions"]:
                identity = row["installation_id"], row["revision_digest"]
                inspection = inspections[identity]
                cursor.execute(
                    "INSERT INTO revisions VALUES (?, ?, ?)",
                    (*identity, inspection.model_dump_json()),
                )
                for component in inspection.inventory.values():
                    cursor.execute(
                        "INSERT INTO components VALUES (?, ?, ?, ?)",
                        (
                            *identity,
                            component.component_id,
                            component.model_dump_json(),
                        ),
                    )
            columns = {
                "selections": (
                    "installation_id",
                    "revision_digest",
                    "component_id",
                    "selected",
                ),
                "activation": ("installation_id", "workspace_id", "intent"),
                "authority_generations": (
                    "installation_id",
                    "scope_kind",
                    "workspace_id",
                    "generation",
                    "revoked",
                ),
                "revision_trust": ("installation_id", "revision_digest", "reviewed"),
                "tombstones": ("installation_id", "generation", "operation_id"),
                "data_roots": (
                    "root_id",
                    "installation_id",
                    "workspace_id",
                    "path",
                    "generation",
                    "deletion_fenced",
                ),
            }
            for table, fields in columns.items():
                placeholders = ",".join("?" for _ in fields)
                for row in snapshot[table]:
                    cursor.execute(
                        f"INSERT INTO {table} VALUES ({placeholders})",
                        tuple(row[key] for key in fields),
                    )
            for row in snapshot["mappings"]:
                cursor.execute(
                    "INSERT INTO mappings VALUES (?, ?, ?)",
                    (row["installation_id"], row["mapping_id"], json.dumps(row)),
                )
            installation_ids = {
                row["installation_id"] for row in snapshot["installations"]
            }
            for row in sources:
                if row["installation_id"] in installation_ids:
                    cursor.execute("INSERT INTO sources VALUES (?, ?, ?)", tuple(row))
            if (
                self.authority_projection(operation_result=snapshot["operation_result"])
                != snapshot
            ):
                raise ValueError("retained definitions do not match complete authority")
            for row in snapshot["installations"]:
                if row["revision_digest"] is not None:
                    cursor.execute(
                        "INSERT INTO processes VALUES (?, ?, ?, NULL, ?, 'unknown:registry-reconstruction', 'unresolved', 'active_run', ?) ON CONFLICT(token) DO UPDATE SET state='unresolved'",
                        (
                            "recovery:" + row["installation_id"],
                            snapshot["operation_result"]["operation_id"],
                            row["installation_id"],
                            row["revision_digest"],
                            '{"reason":"registry_reconstructed","unknown_runtime_users":true}',
                        ),
                    )
            if snapshot["operation_result"] is not None:
                self.write_operation(
                    cursor, snapshot["operation_result"], phase="committed"
                )

    def close(self) -> None:
        """Release the database handle without deleting recovery evidence."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
