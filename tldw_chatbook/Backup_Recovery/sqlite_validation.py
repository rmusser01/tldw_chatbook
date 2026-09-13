"""Bounded inspection/migration of disposable candidates using installed SQL.

Cross-owner file locators require the executor's staged dependency map. This
entry checks SQLite content and owned BLOBs; it never opens those external paths.
"""

import sqlite3
from contextlib import closing, contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Event
from time import monotonic

from .models import OwnerAdapter

_PROGRESS_INTERVAL = 1000
_STEP_BUDGET = 5_000_000
_SECONDS = 30.0
_CATALOG_LIMIT = 10_000
_CATALOG_BYTES = 8 * 1024**2


_active_restrictions = ContextVar("recovery_sqlite_restrictions", default=None)


@contextmanager
def _recovery_restriction_scope(connection, restrictions):
    """Expose only this connection's original budget to nested owner checks."""
    token = _active_restrictions.set((connection, restrictions))
    try:
        yield
    finally:
        _active_restrictions.reset(token)


def _current_restrictions(connection):
    active = _active_restrictions.get()
    return active[1] if active is not None and active[0] is connection else None


def _installed_owner(owner_id):
    # Factories contain frozen declarations only: no config/profile/repository
    # construction and no archive-supplied policy or migration authority.
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.DB.recovery_operations import recovery_adapters as operations
    from tldw_chatbook.Evals.recovery import recovery_adapters as evals
    from tldw_chatbook.Kanban_Interop.recovery import recovery_adapters as kanban
    from tldw_chatbook.Notes.recovery import recovery_adapters as notes
    from tldw_chatbook.Notifications.recovery import recovery_adapters as notifications
    from tldw_chatbook.Research_Interop.recovery import recovery_adapters as research
    from tldw_chatbook.Scheduling.recovery import recovery_adapters as scheduling
    from tldw_chatbook.Study_Interop.recovery import recovery_adapters as study
    from tldw_chatbook.Sync_Interop.recovery import recovery_adapters as sync
    from tldw_chatbook.TTS.recovery import recovery_adapters as tts
    from tldw_chatbook.Writing_Interop.recovery import recovery_adapters as writing

    from .config_adapter import recovery_adapters as config
    from .rag_indexing import recovery_adapters as rag_indexing
    from .recovered_media import recovery_adapters as recovered

    for factory in (
        core_adapters,
        operations,
        research,
        writing,
        evals,
        study,
        notes,
        kanban,
        scheduling,
        notifications,
        sync,
        tts,
        recovered,
        config,
        rag_indexing,
    ):
        for owner in factory():
            if owner.owner_id == owner_id:
                policy = owner.schema_policy()
                if policy is not None and policy.schema_sql:
                    return owner
    raise ValueError("unsupported_sqlite_owner")


class _Restrictions:
    def __init__(self, connection, cancel=None):
        self.cancel = cancel
        self.deadline = monotonic() + _SECONDS
        self.steps = 0
        self.migrating = False
        self.canvas_schema = False
        self.changing_schema_trust = False
        connection.set_authorizer(self.authorize)
        connection.set_progress_handler(self.progress, _PROGRESS_INTERVAL)

    def progress(self):
        self.steps += _PROGRESS_INTERVAL
        return int(self.expired())

    def expired(self):
        return (
            (self.cancel is not None and self.cancel.is_set())
            or self.steps > _STEP_BUDGET
            or monotonic() >= self.deadline
        )

    def authorize(self, action, first, second, database, source):
        if database not in (None, "main"):
            return sqlite3.SQLITE_DENY
        if action in (
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_RECURSIVE,
        ):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            allowed = {
                "length",
                "typeof",
                "max",
                "min",
                "count",
                "sum",
                "avg",
                "coalesce",
                "ifnull",
                "nullif",
                "abs",
                "round",
                "lower",
                "upper",
                "substr",
                "substring",
                "like",
                "glob",
                "match",
                "hex",
            }
            if self.canvas_schema:
                allowed.add("canvas_revision_payload_valid")
            if self.migrating:
                allowed |= {"printf", "sqlite_rename_test", "sqlite_rename_quotefix"}
            return sqlite3.SQLITE_OK if second in allowed else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA:
            reads = {
                "trusted_schema",
                "user_version",
                "quick_check",
                "foreign_key_check",
                "data_version",
            }
            metadata = {"table_xinfo", "index_list", "index_xinfo", "table_list"}
            permitted = (first in reads and second is None) or first in metadata
            permitted |= first == "trusted_schema" and second in ("OFF", "0")
            permitted |= (
                self.changing_schema_trust
                and first == "trusted_schema"
                and second in ("ON", "1")
            )
            permitted |= self.migrating and first == "user_version"
            return sqlite3.SQLITE_OK if permitted else sqlite3.SQLITE_DENY
        if self.migrating:
            if action == sqlite3.SQLITE_TRANSACTION:
                return sqlite3.SQLITE_OK
            # The only installed historical migration is Research's ADD COLUMN
            # chain. Schema writes are SQLite's own ALTER implementation; callers
            # cannot submit SQL while this temporary authority is enabled.
            if (
                action == sqlite3.SQLITE_ALTER_TABLE
                and first == "main"
                and second == "research_runs"
            ):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_UPDATE and first == "sqlite_master":
                return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY


def _restrict_connection(connection, cancel=None):
    """Install and verify mandatory primitives before any candidate query."""
    try:
        connection.enable_load_extension(False)
        connection.execute("PRAGMA trusted_schema=OFF")
        if connection.execute("PRAGMA trusted_schema").fetchone() != (0,):
            raise ValueError("sqlite_security_unavailable")
        for category, bound in (
            (sqlite3.SQLITE_LIMIT_LENGTH, 64 * 1024**2),
            (sqlite3.SQLITE_LIMIT_SQL_LENGTH, 1024**2),
            (sqlite3.SQLITE_LIMIT_COLUMN, 2048),
            (sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 1000),
            (sqlite3.SQLITE_LIMIT_VDBE_OP, 250_000),
            (sqlite3.SQLITE_LIMIT_ATTACHED, 0),
            (sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 1000),
            (sqlite3.SQLITE_LIMIT_TRIGGER_DEPTH, 100),
            (sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 50),
            (sqlite3.SQLITE_LIMIT_WORKER_THREADS, 0),
        ):
            connection.setlimit(category, bound)
            if connection.getlimit(category) > bound:
                raise ValueError("sqlite_security_unavailable")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("PRAGMA temp_store=MEMORY")
        return _Restrictions(connection, cancel)
    except (AttributeError, NotImplementedError, sqlite3.Error) as error:
        raise ValueError("sqlite_security_unavailable") from error


def _catalog(connection):
    rows = []
    size = 0
    for row in connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name"
    ):
        size += sum(len(value.encode("utf-8")) for value in row if value is not None)
        if len(rows) >= _CATALOG_LIMIT or size > _CATALOG_BYTES:
            raise ValueError("sqlite_resource_limit")
        rows.append(row)
    return tuple(rows)


def _metadata(connection, catalog):
    def quote(name):
        return '"' + name.replace('"', '""') + '"'

    result = []
    for kind, name, _, _ in catalog:
        if kind in ("table", "view"):
            columns = tuple(connection.execute(f"PRAGMA table_xinfo({quote(name)})"))
            # Sequence reflects creation order, not index semantics.
            indexes = tuple(
                sorted(
                    row[1:]
                    for row in connection.execute(f"PRAGMA index_list({quote(name)})")
                )
            )
            result.append((name, columns, indexes))
        elif kind == "index":
            result.append(
                (name, tuple(connection.execute(f"PRAGMA index_xinfo({quote(name)})")))
            )
    return tuple(result)


@contextmanager
def _canvas_schema_access(connection, schema, restrictions=None):
    """Run the shipped pure CHECK only for an exact installed Canvas catalog.

    Candidate callers must first compare their entire catalog. Reference and
    reconstruction callers execute only this same frozen installed SQL. Python
    cannot mark a SQLite UDF innocuous, so trust is enabled for this bounded
    scope and removed before the restricted connection returns to its caller.
    """
    from tldw_chatbook.DB.canvas_payload_validation import (
        CANVAS_REVISION_PAYLOAD_VALIDATION_FUNCTION,
        install_canvas_revision_payload_validator,
    )
    from tldw_chatbook.DB.recovery_core_schema import (
        CHACHANOTES_DICTIONARY_UPDATE_SCHEMA,
        CORE_SCHEMAS,
    )

    function = CANVAS_REVISION_PAYLOAD_VALIDATION_FUNCTION
    if not any(function in sql for sql in schema):
        yield
        return
    from tldw_chatbook.DB.recovery_operations import _SUBSCRIPTIONS_SCHEMA

    installed = next(
        sql for owner, _, sql in CORE_SCHEMAS if owner == "db.chachanotes.primary"
    )
    frozen = (
        installed,
        CHACHANOTES_DICTIONARY_UPDATE_SCHEMA,
        *(sql for _, sql in _SUBSCRIPTIONS_SCHEMA),
    )
    if schema not in frozen:
        raise ValueError("unsupported_schema")
    if connection.execute("PRAGMA trusted_schema").fetchone() != (0,):
        raise ValueError("sqlite_security_unavailable")
    install_canvas_revision_payload_validator(connection)
    try:
        if restrictions is not None:
            restrictions.canvas_schema = True
            restrictions.changing_schema_trust = True
        connection.execute("PRAGMA trusted_schema=ON")
        if connection.execute("PRAGMA trusted_schema").fetchone() != (1,):
            raise ValueError("sqlite_security_unavailable")
        if restrictions is not None:
            restrictions.changing_schema_trust = False
        yield
    finally:
        # An expired validation deadline must not interrupt removal of trust.
        if restrictions is not None:
            connection.set_progress_handler(None, 0)
        try:
            connection.execute("PRAGMA trusted_schema=OFF")
            if connection.execute("PRAGMA trusted_schema").fetchone() != (0,):
                raise ValueError("sqlite_security_unavailable")
        finally:
            if restrictions is not None:
                restrictions.canvas_schema = False
                restrictions.changing_schema_trust = False
                connection.set_progress_handler(
                    restrictions.progress, _PROGRESS_INTERVAL
                )
            connection.create_function(function, 3, None)


def _canvas_payload_issues(connection, restrictions):
    """Check payloads explicitly after the installed CHECK function is enabled.

    SQLite may have parsed the CHECK before this connection registered the UDF;
    its cached schema can then omit it from quick_check. A direct expression
    evaluates the exact shipped validator under the same limits and progress
    handler, regardless of that schema-cache state.
    """
    if restrictions is None or not restrictions.canvas_schema:
        return ()
    if (
        connection.execute(
            "SELECT 1 FROM canvas_revisions WHERE typeof(html) != 'text' OR "
            "canvas_revision_payload_valid(CAST(html AS BLOB),content_sha256,html_bytes) != 1 LIMIT 1"
        ).fetchone()
        is not None
    ):
        return ("invalid_sqlite_integrity",)
    return ()


def _reference(schema):
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    with closing(
        connect_private_sqlite("recovery.validation_schema", ":memory:")
    ) as reference:
        reference.execute("PRAGMA trusted_schema=OFF")
        with _canvas_schema_access(reference, schema):
            # Execute only the installed catalog, never the candidate's SQL text.
            tables = [
                sql
                for sql in schema
                if sql.upper().startswith("CREATE TABLE")
                or sql.upper().startswith("CREATE VIRTUAL TABLE")
            ]
            others = [sql for sql in schema if sql not in tables]
            for sql in tables + others:
                # AUTOINCREMENT and FTS create their own internal/shadow tables.
                existing = {row[3] for row in _catalog(reference)}
                if sql in existing:
                    continue
                reference.execute(sql)
            catalog = _catalog(reference)
            return catalog, _metadata(reference, catalog)


def _version(connection, owner_id):
    if owner_id in {
        "db.chachanotes.primary",
        "study.local",
        "quiz.local",
        "notes.sync_bindings",
        "chat.attachments",
    }:
        query = "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
    elif owner_id in {
        "db.media.primary",
        "db.prompts.primary",
        "db.library_ingest_jobs",
    }:
        query = "SELECT version FROM schema_version"
    elif owner_id in {
        "db.library_collections",
        "db.workspaces",
        "db.agent_runs",
        "db.subscriptions",
        "db.scheduled_tasks",
        "notifications.client",
        "runtime.event_state",
        "runtime.sync_state",
    }:
        query = "SELECT MAX(version) FROM schema_version"
    elif owner_id == "kanban.local":
        query = "SELECT CAST(value AS INTEGER) FROM local_kanban_schema_meta WHERE key='schema_version'"
    elif owner_id == "notes.file_notes":
        query = "SELECT 0"
    else:
        query = "PRAGMA user_version"
    rows = connection.execute(query).fetchmany(2)
    return rows[0][0] if len(rows) == 1 else None


def _check(connection, owner, policy, restrictions):
    if restrictions.expired():
        raise ValueError("sqlite_resource_limit")
    actual = _catalog(connection)
    actual_sql = tuple(row[3] for row in actual if row[3] is not None)
    matched = tuple(
        (version, schema)
        for version, schema in policy.schema_sql
        if schema == actual_sql
    )
    if not matched:
        return ("unsupported_schema",), None
    # Only after exact catalog match may metadata/version/domain queries prepare
    # expressions contained in the imported schema.
    version = _version(connection, owner.owner_id)
    if version not in policy.versions or not any(v == version for v, _ in matched):
        return ("unsupported_schema_version",), None
    with _canvas_schema_access(connection, matched[0][1], restrictions):
        reference_catalog, metadata = _reference(matched[0][1])
        if actual != reference_catalog or _metadata(connection, actual) != metadata:
            return ("unsupported_schema",), None
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            return ("invalid_domain_reference",), None
        payload_issues = _canvas_payload_issues(connection, restrictions)
        if payload_issues:
            return payload_issues, None
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            return ("invalid_sqlite_integrity",), None
        if (
            owner.owner_id == "db.subscriptions"
            and any(row[1] == "db_schema_version" for row in actual)
            and connection.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()
            != (73,)
        ):
            return ("unsupported_schema_version",), None
        checker = getattr(owner, "_validate_connection", None)
        if checker is not None:
            issues = checker(connection)
            if issues:
                return issues, None
        if restrictions.expired():
            raise ValueError("sqlite_resource_limit")
        return (), version


def _validate_candidate(
    owner: OwnerAdapter, candidate: Path, cancel: Event, *, migrate: bool
) -> tuple[tuple[str, ...], int | None]:
    """Validate installed content; optionally migrate this disposable file only."""
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    restrictions = None
    if cancel.is_set():
        return (("cancelled",), None)
    try:
        installed = _installed_owner(owner.owner_id)
        policy = installed.schema_policy()
        if owner.schema_policy() != policy:
            return (("unsupported_schema_policy",), None)
        with open_recovery_validation(
            installed.owner_id,
            candidate,
            writable=migrate,
            with_restrictions=True,
            cancel=cancel,
        ) as (connection, restrictions):
            issues, version = _check(connection, installed, policy, restrictions)
            if issues:
                return (issues, None)
            if migrate and version != max(policy.versions):
                restrictions.migrating = True
                connection.execute("BEGIN IMMEDIATE")
                try:
                    while version != max(policy.versions):
                        choices = [
                            (end, sql)
                            for start, end, sql in policy.migration_steps
                            if start == version and end > start
                        ]
                        if len(choices) != 1:
                            return (("unsupported_schema_migration",), None)
                        expected, statements = choices[0]
                        for statement in statements:
                            if restrictions.expired():
                                raise InterruptedError
                            connection.execute(statement)
                        restrictions.migrating = False
                        issues, version = _check(
                            connection, installed, policy, restrictions
                        )
                        if issues or version != expected:
                            return (issues or ("unsupported_schema_migration",), None)
                        restrictions.migrating = True
                    if restrictions.expired():
                        raise InterruptedError
                    connection.commit()
                finally:
                    # Cancellation cannot interrupt the rollback needed to leave
                    # this disposable candidate at its original committed state.
                    connection.set_progress_handler(None, 0)
                    restrictions.migrating = True
                    connection.rollback()
                    restrictions.migrating = False
            if restrictions.expired():
                return (
                    ("cancelled" if cancel.is_set() else "sqlite_resource_limit",),
                    None,
                )
            return ((), version)
    except (
        sqlite3.Error,
        OSError,
        ValueError,
        TypeError,
        MemoryError,
        InterruptedError,
    ) as error:
        if cancel.is_set():
            return (("cancelled",), None)
        if restrictions is not None and restrictions.expired():
            return (("sqlite_resource_limit",), None)
        if isinstance(error, ValueError) and str(error) in {
            "sqlite_security_unavailable",
            "sqlite_resource_limit",
            "unsupported_sqlite_owner",
        }:
            return ((str(error),), None)
        return (("sqlite_validation_unavailable",), None)


def validate_candidate(
    owner: OwnerAdapter, candidate: Path, cancel: Event, *, migrate: bool
) -> tuple[str, ...]:
    """Validate with the original issue-only contract on one restricted connection."""
    return _validate_candidate(owner, candidate, cancel, migrate=migrate)[0]


def validated_schema_version(
    owner: OwnerAdapter, candidate: Path, cancel: Event
) -> int:
    """Return the exact observed installed version after full restricted validation."""
    issues, version = _validate_candidate(owner, candidate, cancel, migrate=False)
    if issues or version is None:
        raise ValueError(issues[0] if issues else "sqlite_validation_unavailable")
    return version
