"""Private per-binding keyword cache with atomic publication and durable CAS."""

from __future__ import annotations

from dataclasses import asdict
from functools import wraps
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from tldw_chatbook.DB.store_kernel import SQLiteStoreKernel
from tldw_chatbook.Workspaces.project_index.models import (
    MAX_CACHE_BYTES,
    MAX_ENTRIES,
    OWNER_ID,
    IndexCandidate,
    IndexRetirementVerifier,
    PreparedChunk,
    PreparedScanPage,
    ProjectIndexJobClaim,
    ProjectIndexKey,
    ProjectIndexPolicy,
    ProjectIndexScope,
    ProjectIndexSearchAuthority,
    ProjectIndexStatus,
    ScanPage,
)

_SCHEMA = (
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
    from tldw_chatbook.RAG_Search.chunking_service import ChunkingService

    chunker = ChunkingService()
    chunks = []
    for capture in page.captures:
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
    digest = hashlib.sha256(
        json.dumps(asdict(page), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PreparedScanPage(page, tuple(chunks), digest)


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


def _activated(function):
    @wraps(function)
    def run(store, *args, **kwargs):
        from tldw_chatbook.Backup_Recovery.activation import execution_scope
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        from tldw_chatbook.Backup_Recovery.participants import _core_operation

        # The registry fence may already own a different repository operation.
        # Enter this installed owner's independent provenance before acquiring its
        # activation lease; never borrow the registry's path authority.
        with _core_operation(store):
            with execution_scope((OWNER_ID,), store.db_path) as allowed:
                if not allowed:
                    raise RecoveryRequired("project_index_activation_required")
                return function(store, *args, **kwargs)

    return run


class WorkspaceProjectIndexDB(SQLiteStoreKernel):
    """One key per namespace. Every writer is an explicit durable job claim.

    Lock order: caller's registry fence, then this store's transaction. This
    class never calls the registry. Prepared pages contain no deferred work.
    close() retires only the calling thread's actual held native connection.
    """

    _kernel_sqlite_owner = OWNER_ID
    _CURRENT_SCHEMA_VERSION = 2

    def __init__(self, path: Path, *, cleanup_only: bool = False) -> None:
        self.db_path = Path(path).absolute()
        self.db_path_str = str(self.db_path)
        self.is_memory_db = False
        self._claims: dict[str, str] = {}
        self.publication_cache_bytes: int | None = None
        self.status_cache_bytes: int | None = None
        self._init_store_kernel()
        try:
            with self.transaction() as conn:
                existing = conn.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table'"
                ).fetchall()
                if not existing:
                    if cleanup_only:
                        raise ValueError("index_cleanup_namespace_unknown")
                    for statement in _SCHEMA:
                        conn.execute(statement)
                    conn.execute("INSERT INTO schema_version VALUES(2)")
                else:
                    versions = [
                        r[0] for r in conn.execute("SELECT version FROM schema_version")
                    ]
                    if versions == [1]:
                        if cleanup_only:
                            raise ValueError("index_cleanup_rebuild_required")
                        migration = (
                            Path(__file__).with_name("migrations")
                            / "workspace_project_index_002.sql"
                        )
                        for statement in migration.read_text(encoding="utf-8").split(
                            ";"
                        ):
                            if statement.strip():
                                conn.execute(statement)
                    catalog = {
                        row[0]: row[1]
                        for row in conn.execute(
                            "SELECT name,sql FROM sqlite_schema WHERE sql IS NOT NULL"
                        )
                    }
                    for statement in _SCHEMA:
                        name = (
                            statement.split("TABLE ", 1)[1]
                            .split("(", 1)[0]
                            .split(" ", 1)[0]
                        )
                        if catalog.get(name) != statement:
                            raise ValueError("project_index_schema_mismatch")
                    if [
                        r[0] for r in conn.execute("SELECT version FROM schema_version")
                    ] != [2]:
                        raise ValueError("project_index_schema_version")
        except BaseException:
            self.close()
            raise

    def close_native(self) -> None:
        """Confirm native closure on the calling owner thread, including failures."""
        connection = getattr(self._thread_local, "conn", None)
        self.close()
        if connection is not None:
            try:
                sqlite3.Connection.in_transaction.__get__(connection)
            except sqlite3.ProgrammingError:
                return
            raise RuntimeError("project_index_connection_not_retired")

    @_activated
    def _open_connection(self):
        return super()._open_connection()

    def _configure_connection(self, conn):
        super()._configure_connection(conn)
        if conn.execute("PRAGMA page_count").fetchone()[0] == 0:
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA secure_delete=ON")
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        # DELETE journaling can retain one old page for every database page.
        # Reserve both physical copies plus journal headers inside the 1-GiB cap.
        conn.execute(
            f"PRAGMA max_page_count={(MAX_CACHE_BYTES - 1024**2) // (2 * page_size)}"
        )
        return conn

    def _control(self, conn, key: ProjectIndexKey):
        if type(key) is not ProjectIndexKey:
            raise TypeError("index_key_required")
        row = conn.execute("SELECT * FROM control WHERE singleton=1").fetchone()
        if row is None:
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

    def _current(self, conn, scope):
        if type(scope) is not ProjectIndexScope:
            raise TypeError("index_scope_required")
        row = self._control(conn, scope.key)
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
        with self.transaction() as conn:
            row = self._control(conn, key)
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

    def assert_native_lifetime(self) -> None:
        """Refuse unknown historical owners even after durable revocation."""
        with self.transaction() as conn:
            if conn.execute("SELECT 1 FROM legacy_retirement").fetchone():
                raise ValueError("legacy_retirement_unproven")

    @staticmethod
    def _remember_legacy(conn) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO legacy_retirement SELECT 1 FROM active_job WHERE owner_token NOT LIKE 'native-v1:%'"
        )

    def policy(self, scope: ProjectIndexScope) -> ProjectIndexPolicy:
        """Read the immutable exact-scope processing policy."""
        with self.transaction() as conn:
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
            row = self._current(conn, expected_scope)
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
                "INSERT INTO directories VALUES('',?,0) ON CONFLICT(path) DO UPDATE SET complete=0",
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
        with self.transaction() as conn:
            if self._current(conn, scope) is None:
                return ()
            return tuple(
                row[0]
                for row in conn.execute(
                    "SELECT path FROM directories WHERE complete=0 AND (? IS NULL OR path>?) ORDER BY path LIMIT ?",
                    (after, after, limit),
                )
            )

    @_activated
    def admit_job(self, claim: ProjectIndexJobClaim) -> bool:
        """Admit one scanner unit at the exact live claim under a registry fence."""
        with self.transaction() as conn:
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
        with self.transaction() as conn:
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
                    "INSERT INTO directories VALUES('',?,0)",
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

    @_activated
    def commit_scan_page(
        self,
        scope: ProjectIndexScope,
        job_id: str,
        page: PreparedScanPage,
        *,
        expected_job_revision: int,
        authority_revision: int,
    ) -> bool:
        """Publish a pre-chunked bounded page under the caller's registry fence."""
        if type(page) is not PreparedScanPage:
            raise TypeError("prepared_scan_page_required")
        try:
            return self._publish_page(
                scope, job_id, page, expected_job_revision, authority_revision
            )
        except sqlite3.OperationalError as error:
            if getattr(error, "sqlite_errorcode", None) != sqlite3.SQLITE_FULL:
                raise
            # SQLite has rolled back the entire page. Preserve the exact claim
            # while making the exhausted coverage explicit.
            with self.transaction() as conn:
                if self._current(conn, scope) is not None:
                    conn.execute(
                        "UPDATE control SET state='Partial' WHERE EXISTS(SELECT 1 FROM active_job WHERE job_id=? AND job_revision=?)",
                        (job_id, expected_job_revision),
                    )
            return False

    def _publish_page(
        self, scope, job_id, page, expected_job_revision, authority_revision
    ):
        scanned = page.page
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
                        "INSERT INTO directories VALUES(?,?,0) ON CONFLICT(path) DO UPDATE SET identity=excluded.identity,complete=0",
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
                    "INSERT INTO chunk_fts(rowid,text) VALUES(?,?)", (rowid, text)
                )
            conn.execute(
                "INSERT INTO directories VALUES(?,?,?) ON CONFLICT(path) DO UPDATE SET identity=excluded.identity,complete=excluded.complete",
                (
                    scanned.directory,
                    scanned.directory_identity,
                    scanned.directory_complete,
                ),
            )
            if scanned.directory_complete:
                # A finished parent may retire missing children and their entire
                # subtrees. Unfinished/unreadable directories retain old members.
                parent = scanned.directory + "/" if scanned.directory else ""
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
                        for table in ("files", "entries", "directories", "scan_seen"):
                            conn.execute(
                                f"DELETE FROM {table} WHERE path=? OR (path>=? AND path<?)",
                                bounds,
                            )
            conn.execute("INSERT INTO scan_pages VALUES(?,?)", (job_id, page.digest))
            conn.execute(
                "UPDATE control SET manifest_revision=manifest_revision+1,last_scan_at=?",
                (datetime.now(timezone.utc).isoformat(),),
            )
            return True

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
        with self.transaction() as conn:
            row = self._current(conn, authority.scope)
            if row is None or not row["enabled"]:
                return ()
            policy = ProjectIndexPolicy(*json.loads(row["policy"]))

            def allowed(path):
                prefixes = policy.include_prefixes
                return int(
                    _path_allowed(path, authority.exclusions)
                    and (
                        not prefixes
                        or any(
                            not prefix
                            or path == prefix
                            or path.startswith(prefix + "/")
                            for prefix in prefixes
                        )
                    )
                    and (
                        not policy.text_suffixes
                        or Path(path).suffix.casefold() in policy.text_suffixes
                    )
                )

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
            for table in (
                "chunks",
                "files",
                "chunk_fts",
                "entries",
                "directories",
                "scan_pages",
                "scan_seen",
            ):
                conn.execute(f"DELETE FROM {table}")
            conn.execute("UPDATE control SET manifest_revision=manifest_revision+1")
            conn.execute(
                "INSERT INTO cleanup_receipts VALUES(?,?)",
                (scope.epoch, datetime.now(timezone.utc).isoformat()),
            )
            return True

    def cleanup_eligible(self, namespace: str, binding_namespace: str) -> None:
        """Validate only owned cleanup metadata; this never grants query authority."""
        with self.transaction() as conn:
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
            for table in (
                "chunks",
                "files",
                "chunk_fts",
                "entries",
                "directories",
                "scan_pages",
                "scan_seen",
            ):
                conn.execute(f"DELETE FROM {table}")
            conn.execute(
                "INSERT INTO cleanup_receipts SELECT epoch,? FROM control",
                (datetime.now(timezone.utc).isoformat(),),
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
