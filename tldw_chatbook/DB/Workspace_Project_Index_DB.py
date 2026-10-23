"""Private per-binding keyword cache with atomic publication and durable CAS."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from functools import wraps
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from tldw_chatbook.DB.store_kernel import SQLiteStoreKernel
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.Utils.timestamps import utc_now_iso
from tldw_chatbook.Workspaces.project_index.allocation import (
    admit_connection,
    admit_keyword_connection,
    allocation_gate,
    allocation_store,
    retain_uncertain,
    verify_legacy_pre_open,
    verify_legacy_settlement,
    verify_pre_open,
    verify_settlement,
)
from tldw_chatbook.Workspaces.project_index.models import (
    MAX_CACHE_BYTES,
    MAX_ENTRIES,
    OWNER_ID,
    IndexCandidate,
    IndexRetirementVerifier,
    PreparedChunk,
    PreparedScanPage,
    PreparedScanUnit,
    ScanUnit,
    ProjectIndexJobClaim,
    ProjectIndexRetirementUnproven,
    ProjectIndexKey,
    ProjectIndexPolicy,
    ProjectIndexScope,
    ProjectIndexSearchAuthority,
    ProjectIndexStatus,
    ProjectSemanticSettings,
    ProjectSemanticState,
    ScanPage,
)

_SCHEMA_V2 = (
    "CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=2))",
    "CREATE TABLE control(singleton INTEGER PRIMARY KEY CHECK(singleton=1), namespace TEXT NOT NULL, epoch INTEGER NOT NULL, processing_revision INTEGER NOT NULL, policy TEXT NOT NULL, enabled INTEGER NOT NULL, state TEXT NOT NULL, job_revision INTEGER NOT NULL, manifest_revision INTEGER NOT NULL, last_scan_at TEXT)",
    "CREATE TABLE active_job(singleton INTEGER PRIMARY KEY CHECK(singleton=1), job_id TEXT UNIQUE NOT NULL, owner_token TEXT NOT NULL, job_revision INTEGER NOT NULL, authority_revision INTEGER NOT NULL)",
    "CREATE TABLE directories(path TEXT PRIMARY KEY, identity TEXT NOT NULL, complete INTEGER NOT NULL)",
    "CREATE TABLE entries(path TEXT PRIMARY KEY, kind TEXT NOT NULL, identity TEXT NOT NULL, mtime_ns INTEGER NOT NULL, size_bytes INTEGER NOT NULL)",
    "CREATE TABLE files(path TEXT PRIMARY KEY, identity TEXT NOT NULL, mtime_ns INTEGER NOT NULL, size_bytes INTEGER NOT NULL, content_sha256 TEXT NOT NULL, text TEXT NOT NULL)",
    "CREATE TABLE chunks(id INTEGER PRIMARY KEY, path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE, text TEXT NOT NULL, start_char INTEGER NOT NULL, end_char INTEGER NOT NULL, line_start INTEGER NOT NULL, line_end INTEGER NOT NULL)",
    "CREATE VIRTUAL TABLE chunk_fts USING fts5(text)",
    "CREATE TABLE scan_pages(job_id TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(job_id,digest))",
    "CREATE TABLE authority_witness(singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL)",
    "CREATE TABLE scan_seen(path TEXT PRIMARY KEY)",
    "CREATE TABLE legacy_retirement(singleton INTEGER PRIMARY KEY CHECK(singleton=1))",
    "CREATE TABLE cleanup_receipts(epoch INTEGER PRIMARY KEY, cleared_at TEXT NOT NULL)",
)


_SCHEMA_V3 = tuple(
    statement.replace("CHECK(version=2)", "CHECK(version=3)")
    .replace(
        "complete INTEGER NOT NULL)",
        "complete INTEGER NOT NULL, attempt_revision INTEGER NOT NULL DEFAULT 0)",
    )
    .replace("chunk_fts USING fts5(text)", "chunk_fts USING fts5(path,text)")
    for statement in _SCHEMA_V2
) + (
    "CREATE INDEX directories_eligible ON directories(complete,attempt_revision,path)",
)


_SCHEMA = tuple(
    statement.replace("CHECK(version=3)", "CHECK(version=4)")
    for statement in _SCHEMA_V3
) + (
    "CREATE TABLE semantic_control(singleton INTEGER PRIMARY KEY CHECK(singleton=1), namespace TEXT NOT NULL, epoch INTEGER NOT NULL, semantic_revision INTEGER NOT NULL, enabled INTEGER NOT NULL, settings_json TEXT, state TEXT NOT NULL, reason TEXT, ready_generation_id TEXT)",
    "CREATE TABLE semantic_generations(generation_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, epoch INTEGER NOT NULL, processing_revision INTEGER NOT NULL, semantic_revision INTEGER NOT NULL, provider_id TEXT NOT NULL, model_id TEXT NOT NULL, destination_fingerprint TEXT NOT NULL, config_fingerprint TEXT NOT NULL, artifact_fingerprint TEXT NOT NULL, format_id TEXT NOT NULL, dimension INTEGER NOT NULL, manifest_revision INTEGER NOT NULL, claim_token TEXT, checkpoint_chunk_id INTEGER, state TEXT NOT NULL, chunk_count INTEGER NOT NULL)",
    "CREATE TABLE semantic_members(generation_id TEXT NOT NULL REFERENCES semantic_generations(generation_id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, relative_path TEXT NOT NULL, identity TEXT NOT NULL, content_sha256 TEXT NOT NULL, line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(generation_id,ordinal))",
    "CREATE INDEX semantic_members_path ON semantic_members(generation_id,relative_path,ordinal)",
)


def _validate_catalog(conn, schema, version):
    catalog = dict(
        conn.execute("SELECT name,sql FROM sqlite_schema WHERE sql IS NOT NULL")
    )
    for statement in schema:
        name = (
            statement.split("TABLE ", 1)[1].split("(", 1)[0].split(" ", 1)[0]
            if "TABLE " in statement
            else statement.split("INDEX ", 1)[1].split(" ", 1)[0]
        )
        if catalog.get(name) != statement:
            raise ValueError("project_index_schema_mismatch")
    if [row[0] for row in conn.execute("SELECT version FROM schema_version")] != [
        version
    ]:
        raise ValueError("project_index_schema_version")


def index_store_path(data_root: Path, key: ProjectIndexKey) -> Path:
    """Select an opaque complete-key namespace under an admitted private root."""
    return (
        data_root
        / "project_index"
        / key.binding_namespace
        / key.namespace
        / "workspace_project_index.db"
    )


def prepare_scan_page(page: ScanPage) -> PreparedScanPage:
    """Chunk bounded captures before acquiring registry or cache publication locks."""
    if type(page) is not ScanPage:
        raise TypeError("scan_page_required")
    chunks = _prepare_chunks(page.captures)
    digest = hashlib.sha256(
        json.dumps(asdict(page), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PreparedScanPage(page, chunks, digest)


def prepare_scan_unit(unit: ScanUnit) -> PreparedScanUnit:
    """Prepare all chunks and the complete replay digest outside publication locks."""
    if type(unit) is not ScanUnit:
        raise TypeError("scan_unit_required")
    chunks = _prepare_chunks(unit.captures)
    digest = hashlib.sha256(
        json.dumps(asdict(unit), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PreparedScanUnit(unit, chunks, digest)


def _prepare_chunks(captures):
    from tldw_chatbook.RAG_Search.chunking_service import ChunkingService

    chunker = ChunkingService()
    chunks = []
    for capture in captures:
        before = len(chunks)
        for chunk in chunker.chunk_text(
            capture.text, chunk_size=400, chunk_overlap=100, method="words"
        ):
            start, end = chunk["start_char"], chunk["end_char"]
            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end <= len(capture.text)
            ):
                raise ValueError("invalid_chunk_mapping")
            chunks.append(
                PreparedChunk(
                    capture.relative_path,
                    start,
                    end,
                    capture.text.count("\n", 0, start) + 1,
                    capture.text.count("\n", 0, max(start, end - 1)) + 1,
                )
            )
        if len(chunks) == before:
            chunks.append(PreparedChunk(capture.relative_path, 0, 0, 1, 1))
    return tuple(chunks)


def _path_allowed(path: str, exclusions) -> bool:
    """Apply the requesting run's exclusion floor before SQL rank and LIMIT."""
    value = path.casefold()
    for exclusion in exclusions:
        denied = exclusion.value.replace("\\", "/").strip("/").casefold()
        if exclusion.kind == "subtree" and (
            not denied or value == denied or value.startswith(denied + "/")
        ):
            return False
        if exclusion.kind == "file" and value == denied:
            return False
        if exclusion.kind == "name" and value.rsplit("/", 1)[-1] == denied:
            return False
        if exclusion.kind == "direct_children" and value.rpartition("/")[0] == denied:
            return False
    return True


_MUTATING_METHODS = frozenset(
    {
        "reconcile_authority",
        "suspend_job",
        "configure",
        "configure_semantic",
        "claim_job",
        "admit_job",
        "replace_retired_job",
        "retire_job",
        "_commit_prepared",
        "revoke",
        "clear_revoked",
        "clear_owned_namespace",
        "reclaim_free_pages",
    }
)


def _activated(function):
    @wraps(function)
    def run(store, *args, **kwargs):
        from tldw_chatbook.Backup_Recovery.activation import execution_scope
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        from tldw_chatbook.Backup_Recovery.participants import _core_operation

        # The registry fence may already own a different repository operation.
        # Enter this installed owner's independent provenance before acquiring its
        # activation lease; never borrow the registry's path authority.
        with store.sql_unit(write=function.__name__ in _MUTATING_METHODS):
            with _core_operation(store):
                with execution_scope((OWNER_ID,), store.db_path) as allowed:
                    if not allowed:
                        raise RecoveryRequired("project_index_activation_required")
                    return function(store, *args, **kwargs)

    return run


class WorkspaceProjectIndexDB(SQLiteStoreKernel):
    """One key per namespace. Every writer is an explicit durable job claim.

    Lock order: binding allocation gate, then caller's registry fence, then
    this store's transaction. This class never calls the registry. Prepared
    pages contain no deferred work.
    close() retires only the calling thread's actual held native connection.
    """

    _kernel_sqlite_owner = OWNER_ID
    _CURRENT_SCHEMA_VERSION = 4

    def __init__(
        self,
        path: Path,
        *,
        cleanup_only: bool = False,
        migration_context=None,
    ) -> None:
        self.db_path = Path(path).absolute()
        self.db_path_str = str(self.db_path)
        self.is_memory_db = False
        self._claims: dict[str, str] = {}
        self._owned_lifetime = None
        self.publication_cache_bytes: int | None = None
        self.status_cache_bytes: int | None = None
        self.schema_version = 0
        self._allocation_group = (
            self.db_path.parent.parent
            if re.fullmatch(r"[0-9a-f]{64}", self.db_path.parent.name)
            and re.fullmatch(r"[0-9a-f]{64}", self.db_path.parent.parent.name)
            else None
        )
        self._init_store_kernel()
        with self.sql_unit(write=not self.db_path.exists() and not cleanup_only):
            try:
                # Native ownership acquisition and disk accounting never happen in a
                # SQL transaction. Recheck the version after the migration capability.
                with self.connection() as conn:
                    exists = conn.execute(
                        "SELECT 1 FROM sqlite_schema WHERE type='table'"
                    ).fetchone()
                    versions = (
                        [
                            r[0]
                            for r in conn.execute("SELECT version FROM schema_version")
                        ]
                        if exists
                        else []
                    )
                needs_migration = (
                    bool(exists) and versions in ([1], [2], [3]) and not cleanup_only
                )
                if (
                    self._allocation_group is not None
                    and versions == [4]
                    and not (self._allocation_group / "allocation-v1.lock").exists()
                ):
                    raise ValueError("index_allocation_control_missing")
                if needs_migration and migration_context is None:
                    with self.connection() as conn:
                        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
                        pages = (MAX_CACHE_BYTES - 1024**2) // (2 * page_size)
                        if conn.execute("PRAGMA page_count").fetchone()[0] > pages:
                            raise ValueError("index_cache_limit")
                context = (
                    migration_context(self)
                    if needs_migration and migration_context is not None
                    else nullcontext()
                )
                with context:
                    try:
                        unit = (
                            self.transaction()
                            if not exists
                            or (versions in ([1], [2], [3]) and not cleanup_only)
                            else self.connection()
                        )
                        with unit as conn:
                            exists = conn.execute(
                                "SELECT 1 FROM sqlite_schema WHERE type='table'"
                            ).fetchone()
                            if not exists:
                                if cleanup_only:
                                    raise ValueError("index_cleanup_namespace_unknown")
                                for statement in _SCHEMA:
                                    conn.execute(statement)
                                conn.execute("INSERT INTO schema_version VALUES(4)")
                                self.schema_version = 4
                            else:
                                versions = [
                                    r[0]
                                    for r in conn.execute(
                                        "SELECT version FROM schema_version"
                                    )
                                ]
                                if versions == [1]:
                                    if cleanup_only:
                                        raise ValueError(
                                            "index_cleanup_rebuild_required"
                                        )
                                    self._migrate(conn, "002")
                                    versions = [2]
                                if versions == [2]:
                                    _validate_catalog(conn, _SCHEMA_V2, 2)
                                    if cleanup_only:
                                        self.schema_version = 2
                                        return
                                    self._migrate(conn, "003")
                                    versions = [3]
                                if versions == [3]:
                                    _validate_catalog(conn, _SCHEMA_V3, 3)
                                    if cleanup_only:
                                        self.schema_version = 3
                                        return
                                    self._migrate(conn, "004")
                                _validate_catalog(conn, _SCHEMA, 4)
                                self.schema_version = 4
                    except BaseException:
                        # The migration capability remains held until this positive
                        # close finishes; a strict callback retains it on failure.
                        self.close_native()
                        raise
            except ProjectIndexRetirementUnproven:
                raise
            except BaseException:
                self.close_native()
                raise

    @contextmanager
    def sql_unit(self, *, write: bool = False):
        """Own one finite native SQLite unit under the binding allocation gate."""
        group = self._allocation_group
        if group is None:
            # Historical standalone keyword fixtures have no binding owner.
            yield
            return
        depth = getattr(self._thread_local, "allocation_depth", 0)
        if depth:
            with allocation_store(group, self):
                self._thread_local.allocation_depth = depth + 1
                try:
                    yield
                finally:
                    self._thread_local.allocation_depth -= 1
            return
        gate_path = group / "allocation-v1.lock"
        if not write and not gate_path.exists():
            # Validated legacy keyword reads remain available before opt-in.
            verify_legacy_pre_open(self.db_path, group)
            try:
                yield
            finally:
                conn = getattr(self._thread_local, "conn", None)
                if conn is not None and conn.in_transaction:
                    raise ProjectIndexRetirementUnproven(
                        "index_transaction_settlement_unproven"
                    )
                self.close_native()
                verify_legacy_settlement(self.db_path, group)
            return
        with allocation_gate(group, existing_only=not write) as lease:
            with allocation_store(group, self):
                verify_pre_open(self.db_path, group)
                self._thread_local.allocation_depth = 1
                try:
                    yield
                finally:
                    try:
                        conn = getattr(self._thread_local, "conn", None)
                        if conn is not None and conn.in_transaction:
                            raise ProjectIndexRetirementUnproven(
                                "index_transaction_settlement_unproven"
                            )
                        self.close_native()
                        verify_settlement(self.db_path, group)
                    except BaseException:
                        retain_uncertain(group, lease)
                        raise
                    finally:
                        self._thread_local.allocation_depth = 0

    @contextmanager
    def connection(self):
        """A read or standalone SQL unit closes before releasing its gate."""
        if self._allocation_group is None:
            with super().connection() as conn:
                yield conn
            return
        with self.sql_unit():
            with super().connection() as conn:
                yield conn

    @contextmanager
    def transaction(self, *, immediate: bool | None = None, semantic: bool = False):
        """Settle body and COMMIT errors before native close and gate release."""
        if self._allocation_group is None:
            if semantic:
                raise ValueError("index_semantic_binding_required")
            with super().transaction(immediate=immediate) as conn:
                yield conn
            return
        with self.sql_unit(write=True):
            conn = self._held_connection()
            if semantic or self._semantic_enabled(conn):
                admit_connection(
                    conn,
                    self.db_path,
                    self._allocation_group,
                    budget=MAX_CACHE_BYTES,
                    owned_lifetime=self._owned_lifetime,
                )
            else:
                admit_keyword_connection(
                    conn, self.db_path, self._allocation_group, budget=MAX_CACHE_BYTES
                )
            if immediate is None:
                immediate = self._transaction_immediate_default
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield conn
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise
            else:
                try:
                    conn.commit()
                except BaseException:
                    if conn.in_transaction:
                        conn.rollback()
                    raise

    @staticmethod
    def _semantic_enabled(conn) -> bool:
        if not conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='semantic_control'"
        ).fetchone():
            return False
        row = conn.execute(
            "SELECT enabled FROM semantic_control WHERE singleton=1"
        ).fetchone()
        return bool(row and row[0])

    def bind_native_lifetime(self, lease) -> None:
        """Supply this operation's exact live binding capability for admission."""
        from tldw_chatbook.Workspaces.project_index.lifetime import BindingLifetime

        if (
            self._allocation_group is None
            or not isinstance(lease, BindingLifetime)
            or lease.path.parent.absolute() != self._allocation_group
            or not re.fullmatch(r"[0-9a-f]{64}\.lifetime\.lock", lease.path.name)
        ):
            raise ValueError("index_allocation_owner_refused")
        lease.assert_live()
        self._owned_lifetime = lease

    @staticmethod
    def _migrate(conn, suffix):
        path = (
            Path(__file__).with_name("migrations")
            / f"workspace_project_index_{suffix}.sql"
        )
        for statement in path.read_text(encoding="utf-8").split(";"):
            if statement.strip():
                conn.execute(statement)

    def close_native(self) -> None:
        """Confirm exact native/core retirement on the owning thread."""
        from tldw_chatbook.Backup_Recovery.participants import _core_cached_connection

        connection = _core_cached_connection(
            self, getattr(self._thread_local, "conn", None)
        )
        self.close()
        if connection is not None:
            try:
                sqlite3.Connection.in_transaction.__get__(connection)
            except sqlite3.ProgrammingError:
                if _core_cached_connection(self, connection) is None:
                    return
            raise ProjectIndexRetirementUnproven("project_index_connection_not_retired")

    @_activated
    def _open_connection(self):
        return connect_private_sqlite(OWNER_ID, self.db_path_str, timeout=0)

    def _configure_connection(self, conn):
        super()._configure_connection(conn)
        if conn.execute("PRAGMA page_count").fetchone()[0] == 0:
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA secure_delete=ON")
        conn.execute("PRAGMA busy_timeout=0")
        conn.execute("PRAGMA cache_spill=OFF")
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        # DELETE journaling can retain one old page for every database page.
        # Reserve both physical copies plus journal headers inside the 1-GiB cap.
        conn.execute(
            f"PRAGMA max_page_count={(MAX_CACHE_BYTES - 1024**2) // (2 * page_size)}"
        )
        return conn

    def _control(self, conn, key: ProjectIndexKey, *, create: bool = True):
        if type(key) is not ProjectIndexKey:
            raise TypeError("index_key_required")
        row = conn.execute("SELECT * FROM control WHERE singleton=1").fetchone()
        if row is None:
            if not create:
                return {
                    "namespace": key.namespace,
                    "epoch": 0,
                    "processing_revision": 0,
                    "policy": "[[],[]]",
                    "enabled": 0,
                    "state": "Off",
                    "job_revision": 0,
                    "manifest_revision": 0,
                    "last_scan_at": None,
                }
            conn.execute(
                "INSERT INTO control VALUES(1,?,0,0,'[[],[]]',0,'Off',0,0,NULL)",
                (key.namespace,),
            )
            row = conn.execute("SELECT * FROM control WHERE singleton=1").fetchone()
        if row["namespace"] != key.namespace:
            raise ValueError("key_mismatch")
        return row

    @staticmethod
    def _scope(key, row):
        return ProjectIndexScope(key, row["epoch"], row["processing_revision"])

    def _current(self, conn, scope, *, create: bool = False):
        if type(scope) is not ProjectIndexScope:
            raise TypeError("index_scope_required")
        row = self._control(conn, scope.key, create=create)
        return row if self._scope(scope.key, row) == scope else None

    def _cache_bytes(self) -> int:
        # The constructor receives a dedicated namespace in production.
        total = 0
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                total += Path(str(self.db_path) + suffix).stat().st_size
            except FileNotFoundError:
                pass
        return total

    def load_status(self, key: ProjectIndexKey) -> ProjectIndexStatus:
        """Read control state; a fresh key is Off and starts no background work."""
        with self.connection() as conn:
            row = self._control(conn, key, create=False)
            job = conn.execute("SELECT job_id FROM active_job").fetchone()
            counts = [
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("entries", "files")
            ]
            result = (
                self._scope(key, row),
                row["state"],
                *counts,
                row["last_scan_at"],
                job[0] if job else None,
                row["job_revision"],
                row["manifest_revision"],
            )
        (
            scope,
            state,
            discovered,
            indexed,
            scanned,
            job_id,
            job_revision,
            manifest_revision,
        ) = result
        return ProjectIndexStatus(
            scope,
            state,
            discovered,
            indexed,
            self._cache_bytes()
            if self.status_cache_bytes is None
            else self.status_cache_bytes,
            scanned,
            job_id,
            job_revision,
            manifest_revision,
        )

    def load_semantic_state(self, scope: ProjectIndexScope) -> ProjectSemanticState:
        """Read consent and readiness without constructing an embedding backend."""
        with self.connection() as conn:
            if self._current(conn, scope) is None:
                raise ValueError("stale_scope")
            if self.schema_version < 4:
                return ProjectSemanticState(scope, 0, False, None, "Off")
            row = conn.execute(
                "SELECT * FROM semantic_control WHERE singleton=1"
            ).fetchone()
            if row is None:
                return ProjectSemanticState(scope, 0, False, None, "Off")
            if row["namespace"] != scope.key.namespace:
                raise ValueError("key_mismatch")
            settings = (
                ProjectSemanticSettings(**json.loads(row["settings_json"]))
                if row["settings_json"] is not None
                else None
            )
            state = row["state"] if row["epoch"] == scope.epoch else "Paused"
            reason = (
                row["reason"]
                if row["epoch"] == scope.epoch
                else "keyword_scope_changed"
            )
            return ProjectSemanticState(
                scope,
                row["semantic_revision"],
                bool(row["enabled"]),
                settings,
                state,
                reason,
                row["ready_generation_id"] if row["epoch"] == scope.epoch else None,
            )

    @_activated
    def configure_semantic(
        self,
        scope: ProjectIndexScope,
        settings: ProjectSemanticSettings | None,
        *,
        enabled: bool,
        expected_semantic_revision: int,
    ) -> ProjectSemanticState:
        """CAS only the binding's semantic consent and setting revision."""
        if (
            type(scope) is not ProjectIndexScope
            or (settings is not None and type(settings) is not ProjectSemanticSettings)
            or type(enabled) is not bool  # noqa: E721 - reject bool/int subclasses
            or type(expected_semantic_revision) is not int  # noqa: E721 - reject bool
            or expected_semantic_revision < 0
            or (enabled and settings is None)
        ):
            raise ValueError("invalid_semantic_configuration")
        with self.transaction(semantic=True) as conn:
            if self._current(conn, scope, create=True) is None:
                raise ValueError("stale_scope")
            prior = conn.execute(
                "SELECT namespace,semantic_revision FROM semantic_control WHERE singleton=1"
            ).fetchone()
            if prior is not None and prior["namespace"] != scope.key.namespace:
                raise ValueError("key_mismatch")
            revision = prior["semantic_revision"] if prior is not None else 0
            if revision != expected_semantic_revision:
                raise ValueError("stale_semantic_revision")
            approved = (
                settings is not None
                and settings.destination_fingerprint
                == settings.approved_destination_fingerprint
            )
            state = "Off" if not enabled else "Pending" if approved else "Paused"
            reason = (
                "destination_approval_required" if enabled and not approved else None
            )
            encoded = (
                json.dumps(asdict(settings), sort_keys=True, separators=(",", ":"))
                if settings is not None
                else None
            )
            conn.execute(
                "INSERT INTO semantic_control(singleton,namespace,epoch,semantic_revision,enabled,settings_json,state,reason,ready_generation_id) VALUES(1,?,?,?,?,?,?,?,NULL) ON CONFLICT(singleton) DO UPDATE SET epoch=excluded.epoch,semantic_revision=excluded.semantic_revision,enabled=excluded.enabled,settings_json=excluded.settings_json,state=excluded.state,reason=excluded.reason,ready_generation_id=NULL",
                (
                    scope.key.namespace,
                    scope.epoch,
                    revision + 1,
                    int(enabled),
                    encoded,
                    state,
                    reason,
                ),
            )
        return self.load_semantic_state(scope)

    def assert_native_lifetime(self) -> None:
        """Refuse unknown historical owners even after durable revocation."""
        with self.connection() as conn:
            if conn.execute("SELECT 1 FROM legacy_retirement").fetchone():
                raise ValueError("legacy_retirement_unproven")

    @staticmethod
    def _remember_legacy(conn) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO legacy_retirement SELECT 1 FROM active_job WHERE owner_token NOT LIKE 'native-v1:%'"
        )

    def policy(self, scope: ProjectIndexScope) -> ProjectIndexPolicy:
        """Read the immutable exact-scope processing policy."""
        with self.connection() as conn:
            row = self._current(conn, scope)
            if row is None:
                raise ValueError("stale_scope")
            return ProjectIndexPolicy(*json.loads(row["policy"]))

    @_activated
    def reconcile_authority(self, key: ProjectIndexKey, revision: int) -> None:
        """Invalidate stale readiness/claims on the durable registry witness."""
        with self.transaction() as conn:
            self._control(conn, key)
            old = conn.execute("SELECT revision FROM authority_witness").fetchone()
            if old is not None and old[0] != revision:
                conn.execute(
                    "UPDATE control SET epoch=epoch+1,job_revision=job_revision+1,state=CASE WHEN enabled=1 THEN 'Paused' ELSE 'Off' END"
                )
                self._remember_legacy(conn)
                conn.execute("DELETE FROM active_job")
            conn.execute(
                "INSERT OR REPLACE INTO authority_witness VALUES(1,?)", (revision,)
            )

    @_activated
    def suspend_job(
        self, claim: ProjectIndexJobClaim, *, state: str, advance: bool = True
    ) -> None:
        """Fence further publications without losing the explicit Resume handle."""
        if state not in {"Paused", "Interrupted", "Partial", "Error"}:
            raise ValueError("invalid_job_state")
        with self.transaction() as conn:
            row = self._current(conn, claim.scope)
            if row is None:
                raise ValueError("stale_scope")
            old = conn.execute("SELECT * FROM active_job").fetchone()
            if old is None or (
                old["job_id"],
                old["job_revision"],
                old["owner_token"],
            ) != (claim.job_id, claim.job_revision, claim.owner_token):
                raise ValueError("stale_job")
            conn.execute(
                "UPDATE active_job SET job_revision=job_revision+?", (int(advance),)
            )
            conn.execute(
                "UPDATE control SET state=?,job_revision=job_revision+?",
                (state, int(advance)),
            )

    @_activated
    def configure(
        self,
        key: ProjectIndexKey,
        policy: ProjectIndexPolicy,
        *,
        enabled: bool,
        expected_scope: ProjectIndexScope,
    ) -> ProjectIndexScope:
        """Change enablement/policy only at the exact scope observed by the caller."""
        if (
            expected_scope.key != key
            or type(policy) is not ProjectIndexPolicy
            or type(enabled) is not bool  # noqa: E721 - reject non-boolean configuration
        ):
            raise ValueError("invalid_configuration")
        encoded = json.dumps((policy.include_prefixes, policy.text_suffixes))
        with self.transaction() as conn:
            row = self._current(conn, expected_scope, create=True)
            if row is None:
                raise ValueError("stale_scope")
            changed = encoded != row["policy"]
            conn.execute(
                "UPDATE control SET epoch=epoch+1,processing_revision=processing_revision+?,policy=?,enabled=?,state=?,job_revision=job_revision+1 WHERE singleton=1",
                (int(changed), encoded, enabled, "Paused" if enabled else "Off"),
            )
            self._remember_legacy(conn)
            conn.execute("DELETE FROM active_job")
            return self._scope(key, self._control(conn, key))

    @_activated
    def claim_job(
        self,
        scope: ProjectIndexScope,
        *,
        owner_token: str,
        authority_revision: int,
        expected_job_revision: int,
    ) -> ProjectIndexJobClaim:
        """Reserve one writer before worker admission; no expiry-based takeover."""
        if not owner_token or len(owner_token) > 256:
            raise ValueError("invalid_owner_token")
        with self.transaction() as conn:
            row = self._current(conn, scope)
            if row is None or not row["enabled"]:
                raise ValueError("stale_scope")
            if conn.execute("SELECT 1 FROM active_job").fetchone():
                raise ValueError("job_active")
            if row["job_revision"] != expected_job_revision:
                raise ValueError("stale_job")
            conn.execute("DELETE FROM scan_seen")
            conn.execute("DELETE FROM scan_pages")
            conn.execute("DELETE FROM directories")
            job_id = uuid4().hex
            revision = row["job_revision"] + 1
            conn.execute(
                "INSERT INTO active_job VALUES(1,?,?,?,?)",
                (job_id, owner_token, revision, authority_revision),
            )
            conn.execute(
                "INSERT INTO directories(path,identity,complete) VALUES('',?,0) ON CONFLICT(path) DO UPDATE SET complete=0",
                (scope.key.root_identity_fingerprint,),
            )
            conn.execute(
                "UPDATE control SET job_revision=?,state='Building'", (revision,)
            )
        self._claims[job_id] = owner_token
        return ProjectIndexJobClaim(
            scope, job_id, owner_token, revision, authority_revision
        )

    def pending_directories(
        self, scope: ProjectIndexScope, *, limit: int = 256, after: str | None = None
    ) -> tuple[str, ...]:
        """Project a bounded durable frontier; complete pages never requeue on Resume."""
        if type(limit) is not int or not 1 <= limit <= 256:  # noqa: E721 - bool is not a limit
            raise ValueError("invalid_frontier_limit")
        with self.connection() as conn:
            if self._current(conn, scope) is None:
                return ()
            return tuple(
                row[0]
                for row in conn.execute(
                    "SELECT path FROM directories WHERE complete=0 AND (? IS NULL OR path>?) ORDER BY path LIMIT ?",
                    (after, after, limit),
                )
            )

    def eligible_directories(self, claim: ProjectIndexJobClaim) -> tuple[str, ...]:
        """Read one indexed bounded page; terminal failures retry on a later run."""
        with self.connection() as conn:
            row = self._current(conn, claim.scope)
            if row is None or row["job_revision"] != claim.job_revision:
                return ()
            return tuple(
                row[0]
                for row in conn.execute(
                    "SELECT path FROM directories WHERE complete=0 AND attempt_revision<? ORDER BY attempt_revision,path LIMIT 256",
                    (claim.job_revision,),
                )
            )

    @_activated
    def admit_job(self, claim: ProjectIndexJobClaim) -> bool:
        """Admit one scanner unit at the exact live claim under a registry fence."""
        with self.connection() as conn:
            row = self._current(conn, claim.scope)
            if (
                row is None
                or not row["enabled"]
                or row["state"] != "Building"
                or row["job_revision"] != claim.job_revision
            ):
                return False
            return (
                conn.execute(
                    "SELECT 1 FROM active_job WHERE job_id=? AND owner_token=? AND job_revision=? AND authority_revision=?",
                    (
                        claim.job_id,
                        claim.owner_token,
                        claim.job_revision,
                        claim.authority_revision,
                    ),
                ).fetchone()
                is not None
            )

    def read_claim(self, scope: ProjectIndexScope) -> ProjectIndexJobClaim | None:
        """Read the exact internal claim needed for explicit restart recovery."""
        with self.connection() as conn:
            if self._current(conn, scope) is None:
                return None
            row = conn.execute("SELECT * FROM active_job").fetchone()
            return (
                ProjectIndexJobClaim(
                    scope,
                    row["job_id"],
                    row["owner_token"],
                    row["job_revision"],
                    row["authority_revision"],
                )
                if row
                else None
            )

    @_activated
    def replace_retired_job(
        self,
        expected_claim: ProjectIndexJobClaim,
        *,
        owner_token: str,
        authority_revision: int,
        retirement_verifier: IndexRetirementVerifier,
        restart: bool = False,
    ) -> ProjectIndexJobClaim:
        """CAS-replace an orphan only after code proves that exact owner retired.

        K3 observes retirement before entering any registry/cache fence. The
        verifier is only a bounded check of that already-established evidence;
        it must not perform process waits, joins or I/O. No elapsed-time or
        user-supplied boolean grants takeover. Recheck the old claim in the CAS.
        """
        if type(expected_claim) is not ProjectIndexJobClaim or not callable(
            retirement_verifier
        ):
            raise ValueError("retirement_unproven")
        if not owner_token or len(owner_token) > 256:
            raise ValueError("invalid_owner_token")
        if retirement_verifier(expected_claim) is not True:
            raise ValueError("retirement_unproven")
        with self.transaction() as conn:
            row = self._current(conn, expected_claim.scope)
            if row is None or not row["enabled"]:
                raise ValueError("stale_scope")
            old = conn.execute("SELECT * FROM active_job").fetchone()
            if old is None or (
                old["job_id"],
                old["owner_token"],
                old["job_revision"],
                old["authority_revision"],
            ) != (
                expected_claim.job_id,
                expected_claim.owner_token,
                expected_claim.job_revision,
                expected_claim.authority_revision,
            ):
                raise ValueError("stale_job")
            if restart:
                conn.execute("DELETE FROM scan_seen")
                conn.execute("DELETE FROM scan_pages")
                conn.execute("DELETE FROM directories")
                conn.execute(
                    "INSERT INTO directories(path,identity,complete) VALUES('',?,0)",
                    (expected_claim.scope.key.root_identity_fingerprint,),
                )
            revision = row["job_revision"] + 1
            job_id = uuid4().hex
            conn.execute(
                "UPDATE active_job SET job_id=?,owner_token=?,job_revision=?,authority_revision=?",
                (job_id, owner_token, revision, authority_revision),
            )
            conn.execute(
                "UPDATE control SET job_revision=?,state='Building'", (revision,)
            )
        self._claims[job_id] = owner_token
        return ProjectIndexJobClaim(
            expected_claim.scope, job_id, owner_token, revision, authority_revision
        )

    @_activated
    def retire_job(
        self,
        scope: ProjectIndexScope,
        job_id: str,
        *,
        owner_token: str,
        expected_job_revision: int,
        state: str,
    ) -> bool:
        """Release only an owned claim after its worker has actually retired.

        The service must establish physical retirement before calling this seam;
        unknown/dead-process claims remain Interrupted until explicit recovery.
        """
        if state not in {"Paused", "Interrupted", "Ready", "Partial", "Error"}:
            raise ValueError("invalid_job_state")
        if self._claims.get(job_id) != owner_token:
            return False
        with self.transaction() as conn:
            if self._current(conn, scope) is None:
                return False
            removed = conn.execute(
                "DELETE FROM active_job WHERE job_id=? AND owner_token=? AND job_revision=?",
                (job_id, owner_token, expected_job_revision),
            ).rowcount
            if not removed:
                return False
            conn.execute(
                "UPDATE control SET job_revision=job_revision+1,state=?", (state,)
            )
        self._claims.pop(job_id, None)
        return True

    def commit_scan_page(
        self,
        scope: ProjectIndexScope,
        job_id: str,
        page: PreparedScanPage,
        *,
        expected_job_revision: int,
        authority_revision: int,
    ) -> bool:
        """Singleton compatibility adapter retaining the original replay digest."""
        if type(page) is not PreparedScanPage:
            raise TypeError("prepared_scan_page_required")
        try:
            return self._commit_prepared(
                scope,
                job_id,
                page,
                expected_job_revision=expected_job_revision,
                authority_revision=authority_revision,
            )
        except sqlite3.OperationalError as error:
            if (
                self._allocation_group is not None
                or getattr(error, "sqlite_errorcode", None) != sqlite3.SQLITE_FULL
            ):
                raise
            # Retain the historical standalone keyword fixture contract only.
            with self.transaction() as conn:
                if self._current(conn, scope) is not None:
                    conn.execute(
                        "UPDATE control SET state='Partial' WHERE EXISTS("
                        "SELECT 1 FROM active_job WHERE job_id=? AND job_revision=?)",
                        (job_id, expected_job_revision),
                    )
            return False

    def commit_scan_unit(
        self,
        scope: ProjectIndexScope,
        job_id: str,
        page: PreparedScanUnit,
        *,
        expected_job_revision: int,
        authority_revision: int,
    ) -> bool:
        """Atomically publish a prepared bounded unit under the registry fence."""
        if type(page) is not PreparedScanUnit:
            raise TypeError("prepared_scan_unit_required")
        return self._commit_prepared(
            scope,
            job_id,
            page,
            expected_job_revision=expected_job_revision,
            authority_revision=authority_revision,
        )

    @_activated
    def _commit_prepared(
        self, scope, job_id, page, *, expected_job_revision, authority_revision
    ):
        return self._publish_page(
            scope, job_id, page, expected_job_revision, authority_revision
        )

    def _publish_page(
        self, scope, job_id, page, expected_job_revision, authority_revision
    ):
        scanned = page.page if type(page) is PreparedScanPage else page.unit
        receipts = (scanned,) if type(page) is PreparedScanPage else scanned.directories
        owner = self._claims.get(job_id)
        if owner is None or scanned.authority_revision != authority_revision:
            return False
        with self.transaction() as conn:
            row = self._current(conn, scope)
            if row is None or not row["enabled"] or row["state"] != "Building":
                return False
            job = conn.execute(
                "SELECT 1 FROM active_job WHERE job_id=? AND owner_token=? AND job_revision=? AND authority_revision=?",
                (job_id, owner, expected_job_revision, authority_revision),
            ).fetchone()
            if job is None:
                return False
            if conn.execute(
                "SELECT 1 FROM scan_pages WHERE job_id=? AND digest=?",
                (job_id, page.digest),
            ).fetchone():
                return True
            new_entries = sum(
                not conn.execute(
                    "SELECT 1 FROM entries WHERE path=?", (e.relative_path,)
                ).fetchone()
                for e in scanned.entries
            )
            count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            if (
                count + new_entries > MAX_ENTRIES
                or (
                    self._cache_bytes()
                    if self.publication_cache_bytes is None
                    else self.publication_cache_bytes
                )
                + sum(c.size_bytes for c in scanned.captures) * 6
                > MAX_CACHE_BYTES
            ):
                conn.execute("UPDATE control SET state='Partial'")
                return False
            for entry in scanned.entries:
                conn.execute(
                    "INSERT OR IGNORE INTO scan_seen VALUES(?)", (entry.relative_path,)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO entries VALUES(?,?,?,?,?)",
                    (
                        entry.relative_path,
                        entry.kind,
                        entry.identity,
                        entry.mtime_ns,
                        entry.size_bytes,
                    ),
                )
                if entry.kind == "directory":
                    conn.execute(
                        "INSERT INTO directories(path,identity,complete) VALUES(?,?,0) ON CONFLICT(path) DO UPDATE SET identity=excluded.identity,complete=CASE WHEN directories.identity=excluded.identity THEN directories.complete ELSE 0 END,attempt_revision=CASE WHEN directories.identity=excluded.identity THEN directories.attempt_revision ELSE 0 END",
                        (entry.relative_path, entry.identity),
                    )
            unchanged = set()
            for capture in scanned.captures:
                old = conn.execute(
                    "SELECT content_sha256 FROM files WHERE path=?",
                    (capture.relative_path,),
                ).fetchone()
                if old is not None and old[0] == capture.content_sha256:
                    conn.execute(
                        "UPDATE files SET identity=?,mtime_ns=?,size_bytes=? WHERE path=?",
                        (
                            capture.identity,
                            capture.mtime_ns,
                            capture.size_bytes,
                            capture.relative_path,
                        ),
                    )
                    unchanged.add(capture.relative_path)
                    continue
                conn.execute(
                    "DELETE FROM chunk_fts WHERE rowid IN (SELECT id FROM chunks WHERE path=?)",
                    (capture.relative_path,),
                )
                conn.execute("DELETE FROM files WHERE path=?", (capture.relative_path,))
                conn.execute(
                    "INSERT INTO files VALUES(?,?,?,?,?,?)",
                    (
                        capture.relative_path,
                        capture.identity,
                        capture.mtime_ns,
                        capture.size_bytes,
                        capture.content_sha256,
                        capture.text,
                    ),
                )
            texts = {
                capture.relative_path: capture.text for capture in scanned.captures
            }
            for chunk in page.chunks:
                if chunk.relative_path in unchanged:
                    continue
                text = texts[chunk.relative_path][chunk.start_char : chunk.end_char]
                rowid = conn.execute(
                    "INSERT INTO chunks(path,text,start_char,end_char,line_start,line_end) VALUES(?,?,?,?,?,?)",
                    (
                        chunk.relative_path,
                        text,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.line_start,
                        chunk.line_end,
                    ),
                ).lastrowid
                conn.execute(
                    "INSERT INTO chunk_fts(rowid,path,text) VALUES(?,?,?)",
                    (rowid, chunk.relative_path, text),
                )
            # Entries first, receipts second: publishing a parent cannot reset
            # the completed child receipt produced in this same atomic unit.
            for receipt in receipts:
                conn.execute(
                    "INSERT INTO directories VALUES(?,?,?,?) ON CONFLICT(path) DO UPDATE SET identity=excluded.identity,complete=excluded.complete,attempt_revision=excluded.attempt_revision",
                    (
                        receipt.directory,
                        receipt.directory_identity,
                        receipt.directory_complete,
                        0 if getattr(receipt, "more", False) else expected_job_revision,
                    ),
                )
                if receipt.directory_complete:
                    # A finished parent may retire missing children and their entire
                    # subtrees. Unfinished/unreadable directories retain old members.
                    parent = receipt.directory + "/" if receipt.directory else ""
                    while True:
                        stale = conn.execute(
                            "SELECT path FROM entries WHERE substr(path,1,?)=? AND instr(substr(path,?),'/')=0 AND path NOT IN (SELECT path FROM scan_seen) LIMIT 256",
                            (len(parent), parent, len(parent) + 1),
                        ).fetchall()
                        if not stale:
                            break
                        for item in stale:
                            path = item[0]
                            prefix = path + "/"
                            bounds = (path, prefix, prefix + chr(0x10FFFF))
                            conn.execute(
                                "DELETE FROM chunk_fts WHERE rowid IN (SELECT id FROM chunks WHERE path=? OR (path>=? AND path<?))",
                                bounds,
                            )
                            for table in (
                                "files",
                                "entries",
                                "directories",
                                "scan_seen",
                            ):
                                conn.execute(
                                    f"DELETE FROM {table} WHERE path=? OR (path>=? AND path<?)",
                                    bounds,
                                )
            conn.execute("INSERT INTO scan_pages VALUES(?,?)", (job_id, page.digest))
            conn.execute(
                "UPDATE control SET manifest_revision=manifest_revision+1,last_scan_at=?",
                (utc_now_iso(),),
            )
            return True

    @staticmethod
    def _eligibility_predicate(policy, floor):
        """Share exact policy/floor semantics between coverage and ranking."""

        prefixes = tuple(prefix.casefold() for prefix in policy.include_prefixes)
        all_prefixes = not prefixes or "" in prefixes
        prefix_heads = tuple(prefix + "/" for prefix in prefixes)
        # Normalize once with the running stdlib's suffix semantics (including
        # trailing-dot behavior), without constructing a Path for every hit.
        suffixes = frozenset(
            suffix
            for suffix in policy.text_suffixes
            if Path("x" + suffix).suffix.casefold() == suffix
        )

        # Freeze this query's current run floor before SQLite evaluates rows.
        # Keep the same normalization as the live-validation predicate.
        exclusions = tuple(
            (item.kind, item.value.replace("\\", "/").strip("/").casefold())
            for item in floor
        )
        subtrees = frozenset(value for kind, value in exclusions if kind == "subtree")
        denied_paths = frozenset(
            value for kind, value in exclusions if kind in {"file", "subtree"}
        )
        denied_names = frozenset(value for kind, value in exclusions if kind == "name")
        denied_parents = frozenset(
            value for kind, value in exclusions if kind == "direct_children"
        )
        subtree_heads = tuple(value + "/" for value in subtrees)
        deny_all = "" in subtrees

        def allowed(path):
            value = path.casefold()
            if (
                deny_all
                or value in denied_paths
                or value.startswith(subtree_heads)
                or value.rsplit("/", 1)[-1] in denied_names
                or value.rpartition("/")[0] in denied_parents
            ):
                return 0
            if (
                not all_prefixes
                and value not in prefixes
                and not value.startswith(prefix_heads)
            ):
                return 0
            if not policy.text_suffixes:
                return 1
            name = path.rsplit("/", 1)[-1]
            stem, dot, tail = name.rpartition(".")
            return int(bool(stem) and (dot + tail).casefold() in suffixes)

        return allowed

    @_activated
    def has_compatible_files(self, scope, exclusions) -> bool:
        """Read only the existence of metadata under this exact enabled scope."""
        with self.connection() as conn:
            row = self._current(conn, scope)
            if row is None or not row["enabled"]:
                return False
            policy = ProjectIndexPolicy(*json.loads(row["policy"]))
            conn.create_function(
                "index_path_allowed", 1, self._eligibility_predicate(policy, exclusions)
            )
            try:
                return (
                    conn.execute(
                        "SELECT 1 FROM files WHERE index_path_allowed(path)=1 LIMIT 1"
                    ).fetchone()
                    is not None
                )
            finally:
                conn.create_function("index_path_allowed", 1, None)

    @_activated
    def lexical_candidates(
        self, authority: ProjectIndexSearchAuthority, query: str, *, limit: int
    ) -> tuple[IndexCandidate, ...]:
        """Return metadata only, filtering the run exclusion floor before top-k."""
        if (
            type(authority) is not ProjectIndexSearchAuthority
            or type(limit) is not int  # noqa: E721 - bool is not a limit
            or not 1 <= limit <= 100
        ):
            raise ValueError("invalid_search_authority_or_limit")
        if not isinstance(query, str) or not query.strip() or len(query) > 2048:
            raise ValueError("invalid_query")
        if not authority.guard():
            return ()
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return ()
        match = " AND ".join('"' + term + '"' for term in terms)
        with self.connection() as conn:
            row = self._current(conn, authority.scope)
            if row is None or not row["enabled"]:
                return ()
            policy = ProjectIndexPolicy(*json.loads(row["policy"]))

            allowed = self._eligibility_predicate(policy, authority.exclusions)

            conn.create_function(
                "index_path_allowed",
                1,
                allowed,
            )
            try:
                rows = conn.execute(
                    "SELECT f.path,f.identity,f.content_sha256,c.line_start,c.line_end FROM chunk_fts JOIN chunks c ON c.id=chunk_fts.rowid JOIN files f ON f.path=c.path WHERE chunk_fts MATCH ? AND index_path_allowed(f.path)=1 ORDER BY bm25(chunk_fts),f.path,c.id LIMIT ?",
                    (match, limit),
                ).fetchall()
            finally:
                conn.create_function("index_path_allowed", 1, None)
            return tuple(
                IndexCandidate(*tuple(result), rank) for rank, result in enumerate(rows)
            )

    @_activated
    def revoke(
        self,
        key: ProjectIndexKey,
        *,
        clear: bool,
        expected_scope: ProjectIndexScope | None = None,
    ) -> ProjectIndexScope:
        """Commit Off before cleanup, retaining durable epoch history.

        Use clear=True only outside an enclosing registry fence. Under a fence,
        use clear=False, exit the fence, then call clear_revoked(returned_scope).
        """
        with self.transaction() as conn:
            row = self._control(conn, key)
            if expected_scope is not None and self._scope(key, row) != expected_scope:
                raise ValueError("stale_scope")
            conn.execute(
                "UPDATE control SET epoch=epoch+1,enabled=0,state='Off',job_revision=job_revision+1"
            )
            self._remember_legacy(conn)
            conn.execute("DELETE FROM active_job")
            if self.schema_version >= 4:
                conn.execute(
                    "UPDATE semantic_control SET enabled=0,state='Off',reason=NULL,"
                    "ready_generation_id=NULL,epoch=epoch+1,semantic_revision=semantic_revision+1"
                )
            revoked = self._scope(key, self._control(conn, key))
        if clear:
            self.clear_revoked(revoked)
        return revoked

    @_activated
    def clear_revoked(self, scope: ProjectIndexScope) -> bool:
        """Clear derived rows only while the exact revoked scope remains Off.

        Run outside the registry fence. A concurrent Enable invalidates cleanup;
        retries for an already completed cleanup are harmless.
        """
        with self.transaction() as conn:
            row = self._current(conn, scope)
            if row is None or row["enabled"] or row["state"] != "Off":
                return False
            if conn.execute(
                "SELECT 1 FROM cleanup_receipts WHERE epoch=?", (scope.epoch,)
            ).fetchone():
                return True
            tables = (
                ("semantic_members", "semantic_generations")
                if self.schema_version >= 4
                else ()
            ) + (
                "chunks",
                "files",
                "chunk_fts",
                "entries",
                "directories",
                "scan_pages",
                "scan_seen",
            )
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
            conn.execute("UPDATE control SET manifest_revision=manifest_revision+1")
            conn.execute(
                "INSERT INTO cleanup_receipts VALUES(?,?)",
                (scope.epoch, utc_now_iso()),
            )
            return True

    def cleanup_eligible(self, namespace: str, binding_namespace: str) -> None:
        """Validate only owned cleanup metadata; this never grants query authority."""
        with self.connection() as conn:
            if (
                conn.execute("PRAGMA auto_vacuum").fetchone()[0] != 2
                or conn.execute("PRAGMA journal_mode").fetchone()[0] != "delete"
            ):
                raise ValueError("index_cleanup_rebuild_required")
            row = conn.execute("SELECT namespace FROM control").fetchone()
            if row is None or row[0] != namespace:
                raise ValueError("index_cleanup_namespace_unknown")
            if conn.execute("SELECT 1 FROM legacy_retirement").fetchone():
                raise ValueError("legacy_retirement_unproven")
            claim = conn.execute("SELECT owner_token FROM active_job").fetchone()
            if claim and not claim[0].startswith(
                "native-v1:" + binding_namespace + ":"
            ):
                raise ValueError("legacy_retirement_unproven")

    @_activated
    def clear_owned_namespace(self, namespace: str, binding_namespace: str) -> None:
        """Revoke retired v2 derived data while preserving control/CAS history.

        Service retains every native capability in the exact binding group and
        the admitted data pin. Call outside registry fences after current-scope
        revocation and bounded whole-binding journal-headroom admission.
        """
        self.cleanup_eligible(namespace, binding_namespace)
        with self.transaction() as conn:
            conn.execute(
                "UPDATE control SET epoch=epoch+1,enabled=0,state='Off',job_revision=job_revision+1,manifest_revision=manifest_revision+1"
            )
            conn.execute("DELETE FROM active_job")
            tables = (
                ("semantic_members", "semantic_generations")
                if self.schema_version >= 4
                else ()
            ) + (
                "chunks",
                "files",
                "chunk_fts",
                "entries",
                "directories",
                "scan_pages",
                "scan_seen",
            )
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
            conn.execute(
                "INSERT INTO cleanup_receipts SELECT epoch,? FROM control",
                (utc_now_iso(),),
            )

    @_activated
    def reclaim_free_pages(self) -> None:
        """Incrementally shrink this same file; never VACUUM or replace the DB."""
        with self.connection() as conn:
            if conn.execute("PRAGMA auto_vacuum").fetchone()[0] != 2:
                raise ValueError("index_cleanup_rebuild_required")
            while True:
                before = conn.execute("PRAGMA freelist_count").fetchone()[0]
                if not before:
                    break
                conn.execute("PRAGMA incremental_vacuum(256)").fetchall()
                after = conn.execute("PRAGMA freelist_count").fetchone()[0]
                if after >= before:
                    raise RuntimeError("index_cleanup_reclamation_incomplete")
