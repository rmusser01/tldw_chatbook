"""Private, transactional Chunking Lab recovery checkpoints (ADR-118).

One store belongs to one profile and one calling thread. Construction performs
no I/O. The async writer owns its connection on a dedicated worker thread.
Clear is logical removal, not secure erasure. No template catalog writes occur.
"""

# ruff: noqa: N999 - Existing DB owner naming convention and published module API.

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from tldw_chatbook.Chunking.lab_models import (
    LabSession,
    RunRequest,
    RunResult,
    SampleSnapshot,
    canonical_json,
    validate_session_references,
)
from tldw_chatbook.Chunking.lab_recovery import (
    export_recovery,
    parse_recovery,
    rebase_recovery,
    validate_active,
)
from tldw_chatbook.Chunking.lab_state import accept_result, new_session
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

SCHEMA_VERSION = 1
_MAX_SAMPLE_BYTES = 2 * 1024 * 1024
_MAX_RESULT_BYTES = 32 * 1024 * 1024
_MAX_CHUNKS = 10_000


class CheckpointConflict(RuntimeError):
    """Another writer or epoch owns the durable checkpoint."""


class RecoverySchemaError(RuntimeError):
    """Recovery data cannot be safely interpreted; preserve the database."""


@dataclass(frozen=True)
class CheckpointToken:
    """Durable CAS identity, distinct from a UI revision."""

    profile_key: str
    epoch: str
    revision: int
    generation: int


@dataclass
class _CapturedBlob:
    # Holding the original prevents id reuse. Only the detached value is used
    # after first capture; subsequent mutation of the public dict cannot change
    # stored bytes. A new snapshot must have a new identity, as lab_state does.
    original: dict
    kind: str
    digest: str
    value: dict


class CheckpointStore:
    """Store checkpoints beneath an existing trusted profile data directory.

    The private SQLite seam validates the selected path and sidecar namespace.
    WAL + synchronous FULL commits publication durably. Schema creation and all
    writes use explicit transactions; no executescript implicit commits.
    ``recovery_warning`` explains fallback without changing the disk on load.
    """

    def __init__(self, path: Path, profile_key: str):
        if not profile_key:
            raise ValueError("Profile key must be nonempty")
        self.path = Path(path)
        self.profile_key = profile_key
        self.recovery_warning: str | None = None
        self._conn: sqlite3.Connection | None = None
        self._captured: dict[int, _CapturedBlob] = {}
        self._fallback: tuple[int, int] | None = None
        self._measurements: dict = {}

    def _connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = connect_private_sqlite(
            "db.chunking_lab", self.path, isolation_level=None, timeout=5
        )
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=FULL")
            # Inspect version and initialize under one lock. Otherwise another
            # opener could publish tables between our version/table reads.
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise RecoverySchemaError("Unsupported recovery database schema")
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if version == 0 and tables:
                raise RecoverySchemaError("Unrecognized recovery database schema")
            if version == 0:
                for statement in (
                    "CREATE TABLE lab_blobs (digest TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)",
                    "CREATE TABLE lab_checkpoints (id INTEGER PRIMARY KEY, revision INTEGER NOT NULL, document TEXT NOT NULL)",
                    "CREATE TABLE lab_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), profile_key TEXT NOT NULL, epoch TEXT NOT NULL, generation INTEGER NOT NULL, current_checkpoint INTEGER REFERENCES lab_checkpoints(id), previous_checkpoint INTEGER REFERENCES lab_checkpoints(id), restore_undo_checkpoint INTEGER REFERENCES lab_checkpoints(id))",
                    "PRAGMA user_version=1",
                ):
                    conn.execute(statement)
            conn.execute("COMMIT")
            conn.execute("PRAGMA journal_mode=WAL")
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RecoverySchemaError("Recovery database integrity check failed")
            self._conn = conn
            return conn
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()
            raise

    def _state(self, conn: sqlite3.Connection) -> tuple | None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise RecoverySchemaError("Unsupported recovery database schema")
        try:
            row = conn.execute(
                "SELECT profile_key, epoch, generation, current_checkpoint, previous_checkpoint, restore_undo_checkpoint FROM lab_state WHERE singleton=1"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise RecoverySchemaError("Malformed recovery database schema") from exc
        if row is not None and row[0] != self.profile_key:
            raise RecoverySchemaError("Recovery database belongs to another profile")
        if (row is None or row[3] is None) and (
            conn.execute("SELECT 1 FROM lab_checkpoints LIMIT 1").fetchone()
            or conn.execute("SELECT 1 FROM lab_blobs LIMIT 1").fetchone()
            or (row is not None and any(value is not None for value in row[4:]))
        ):
            raise RecoverySchemaError("Recovery content has lost its state reference")
        if row is not None and (
            not isinstance(row[1], str)
            or not row[1]
            or type(row[2]) is not int
            or row[2] < 1
        ):
            raise RecoverySchemaError("Malformed recovery state identity")
        return row

    def _capture(
        self, value: dict, kind: str, pending: dict, used: set
    ) -> _CapturedBlob:
        identity = id(value)
        used.add(identity)
        cached = self._captured.get(identity)
        if cached is not None and cached.original is value and cached.kind == kind:
            return cached
        model = {"sample": SampleSnapshot, "result": RunResult, "request": RunRequest}[
            kind
        ]
        detached = model.model_validate(value).model_dump(mode="json")
        sample = (
            detached
            if kind == "sample"
            else detached.get("sample", detached.get("request", {}).get("sample"))
        )
        if (
            sample is not None
            and len(sample["text"].encode("utf-8")) > _MAX_SAMPLE_BYTES
        ):
            raise ValueError("Sample exceeds the 2 MiB recovery limit")
        encoded = canonical_json(detached)
        if kind == "result":
            if len(encoded.encode("utf-8")) > _MAX_RESULT_BYTES:
                raise ValueError("Result exceeds the 32 MiB recovery limit")
            if detached["report"] and len(detached["report"]["chunks"]) > _MAX_CHUNKS:
                raise ValueError("Result exceeds the 10,000 chunk recovery limit")
        digest = hashlib.sha256((kind + "\n" + encoded).encode("utf-8")).hexdigest()
        captured = _CapturedBlob(value, kind, digest, detached)
        self._captured[identity] = captured
        pending[digest] = (kind, encoded)
        return captured

    def _pack(self, session: LabSession) -> tuple[str, dict, set]:
        pending: dict[str, tuple[str, str]] = {}
        used: set[int] = set()
        # Never call LabSession.model_dump here: it would copy/validate every
        # retained report for each draft edit. Only small mutable state is copied.
        small = json.loads(
            canonical_json(
                {
                    "profile_key": session.profile_key,
                    "epoch": session.epoch,
                    "revision": session.revision,
                    "content_revision": session.content_revision,
                    "candidates": session.candidates,
                    "view": session.view,
                }
            )
        )
        captured_samples = {
            key: self._capture(value, "sample", pending, used)
            for key, value in session.samples.items()
        }
        captured_results = {
            key: self._capture(value, "result", pending, used)
            for key, value in session.results.items()
        }
        small["samples"] = {
            key: value.digest for key, value in captured_samples.items()
        }
        small["results"] = {
            key: value.digest for key, value in captured_results.items()
        }
        small["undo"] = []
        for entry in session.undo:
            entry = dict(entry)
            if entry.get("kind") == "sample":
                entry["sample"] = self._capture(
                    entry["sample"], "sample", pending, used
                ).digest
            small["undo"].append(json.loads(canonical_json(entry)))
        batch = session.batch
        small["batch"] = None
        if batch is not None:
            requests = {
                key: self._capture(value, "request", pending, used)
                for key, value in batch["requests"].items()
            }
            small["batch"] = json.loads(
                canonical_json(
                    {
                        **batch,
                        "requests": {
                            key: value.digest for key, value in requests.items()
                        },
                    }
                )
            )
            batch = {
                **small["batch"],
                "requests": {key: value.value for key, value in requests.items()},
            }
        graph = LabSession.model_construct(
            **{
                **small,
                "batch": batch,
                "samples": {
                    key: value.value for key, value in captured_samples.items()
                },
                "results": {
                    key: value.value for key, value in captured_results.items()
                },
            }
        )
        validate_session_references(graph)
        object.__setattr__(graph, "_recovery_measurements", self._measurements)
        validate_active(graph, reuse=True)
        self._measurements = graph._recovery_measurements
        return (
            canonical_json({"schema_version": SCHEMA_VERSION, "session": small}),
            pending,
            used,
        )

    def _unpack(self, conn: sqlite3.Connection, checkpoint_id: int) -> LabSession:
        row = conn.execute(
            "SELECT revision, document FROM lab_checkpoints WHERE id=?",
            (checkpoint_id,),
        ).fetchone()
        if row is None:
            raise RecoverySchemaError("Missing recovery checkpoint")
        envelope = json.loads(row[1])
        if (
            not isinstance(envelope, dict)
            or envelope.get("schema_version") != SCHEMA_VERSION
        ):
            raise RecoverySchemaError("Unsupported checkpoint schema")
        document = envelope["session"]
        if not isinstance(document, dict):
            raise RecoverySchemaError("Malformed checkpoint document")
        if (
            not isinstance(document.get("samples"), dict)
            or not isinstance(document.get("results"), dict)
            or not isinstance(document.get("undo"), list)
            or any(not isinstance(entry, dict) for entry in document["undo"])
            or (
                document.get("batch") is not None
                and (
                    not isinstance(document["batch"], dict)
                    or not isinstance(document["batch"].get("requests"), dict)
                )
            )
        ):
            raise RecoverySchemaError("Malformed checkpoint references")

        def blob(digest: str, kind: str) -> dict:
            row = conn.execute(
                "SELECT kind, payload FROM lab_blobs WHERE digest=?", (digest,)
            ).fetchone()
            if row is None or row[0] != kind:
                raise RecoverySchemaError("Missing recovery snapshot")
            if (
                hashlib.sha256((kind + "\n" + row[1]).encode("utf-8")).hexdigest()
                != digest
            ):
                raise RecoverySchemaError("Recovery snapshot integrity check failed")
            return json.loads(row[1])

        document["samples"] = {
            key: blob(value, "sample") for key, value in document["samples"].items()
        }
        document["results"] = {
            key: blob(value, "result") for key, value in document["results"].items()
        }
        for entry in document["undo"]:
            if entry.get("kind") == "sample":
                entry["sample"] = blob(entry["sample"], "sample")
        if document["batch"] is not None:
            document["batch"]["requests"] = {
                key: blob(value, "request")
                for key, value in document["batch"]["requests"].items()
            }
        session = LabSession.model_validate(document)
        if session.revision != row[0] or session.profile_key != self.profile_key:
            raise RecoverySchemaError("Checkpoint identity mismatch")
        return session

    def load(self) -> tuple[LabSession, CheckpointToken] | None:
        """Read a consistent checkpoint, normalizing unfinished work in memory.

        Fallback returns the *current generation* and recovered revision. The
        warning explains the rollback; no write or replacement authority is
        silently granted after an unrecoverable load error.
        """
        conn = self._connection()
        # Reload acquires fresh durable authority. Another instance may have
        # cleared/pruned payloads since this store last published them.
        self._captured.clear()
        self._measurements.clear()
        conn.execute("BEGIN")
        try:
            state = self._state(conn)
            self.recovery_warning = None
            self._fallback = None
            if state is None:
                return None
            profile, epoch, generation, current, previous, _ = state
            if current is None:
                session = new_session(profile).model_copy(update={"epoch": epoch})
            else:
                session = None
                for checkpoint_id in (current, previous):
                    if checkpoint_id is None:
                        continue
                    try:
                        session = self._unpack(conn, checkpoint_id)
                        if session.epoch != epoch:
                            raise RecoverySchemaError("Checkpoint epoch mismatch")
                        if checkpoint_id != current:
                            self.recovery_warning = "Recovered the previous valid checkpoint; newer edits were unavailable."
                            self._fallback = (generation, checkpoint_id)
                        break
                    except (
                        ValueError,
                        TypeError,
                        KeyError,
                        sqlite3.Error,
                        RecoverySchemaError,
                    ):
                        session = None
                if session is None:
                    raise RecoverySchemaError(
                        "No valid recovery checkpoint; preserve the database"
                    )
            token = CheckpointToken(profile, epoch, session.revision, generation)
            if session.batch is not None:
                for run_id, request in session.batch["requests"].items():
                    if run_id not in session.batch.get("outcomes", {}):
                        session = accept_result(
                            session,
                            RunResult(
                                request=RunRequest.model_validate(request),
                                status="interrupted",
                                report=None,
                                started_at="",
                                finished_at="",
                                elapsed_ms=0,
                                error={
                                    "message": "Preview interrupted before recovery"
                                },
                            ),
                        )
            return session, token
        finally:
            conn.execute("ROLLBACK")

    def _check_expected(
        self, state: tuple | None, expected: CheckpointToken | None
    ) -> None:
        if state is None and expected is None:
            return
        if (
            state is None
            or expected is None
            or (expected.profile_key, expected.epoch, expected.generation) != state[:3]
        ):
            raise CheckpointConflict(
                "Recovery changed in another writer; reload or export the in-memory draft"
            )

    def save(
        self, session: LabSession, *, expected: CheckpointToken | None
    ) -> CheckpointToken:
        """Publish new immutable payloads and their checkpoint in one transaction."""
        conn = self._connection()
        if session.profile_key != self.profile_key:
            raise CheckpointConflict("Session belongs to another profile")
        try:
            document, pending, used = self._pack(session)
            conn.execute("BEGIN IMMEDIATE")
            state = self._state(conn)
            self._check_expected(state, expected)
            if state is not None and session.epoch != state[1]:
                raise CheckpointConflict("Session epoch has been replaced or cleared")
            if expected is not None and session.revision < expected.revision:
                raise CheckpointConflict("Cannot overwrite a newer checkpoint revision")
            for digest, (kind, payload) in pending.items():
                conn.execute(
                    "INSERT OR IGNORE INTO lab_blobs(digest,kind,payload) VALUES(?,?,?)",
                    (digest, kind, payload),
                )
            checkpoint_id = conn.execute(
                "INSERT INTO lab_checkpoints(revision,document) VALUES(?,?)",
                (session.revision, document),
            ).lastrowid
            if state is None:
                conn.execute(
                    "INSERT INTO lab_state VALUES(1,?,?,1,?,NULL,NULL)",
                    (self.profile_key, session.epoch, checkpoint_id),
                )
                generation = 1
            else:
                previous = state[3]
                if self._fallback is not None and self._fallback[0] == state[2]:
                    previous = self._fallback[1]
                restore_undo = state[5]
                if restore_undo is not None:
                    old_document = conn.execute(
                        "SELECT document FROM lab_checkpoints WHERE id=?", (previous,)
                    ).fetchone()[0]
                    if self._content_document(old_document) != self._content_document(
                        document
                    ):
                        restore_undo = None
                conn.execute(
                    "UPDATE lab_state SET current_checkpoint=?, previous_checkpoint=?, restore_undo_checkpoint=?, epoch=?, generation=generation+1 WHERE singleton=1 AND epoch=? AND generation=?",
                    (
                        checkpoint_id,
                        previous,
                        restore_undo,
                        session.epoch,
                        expected.epoch,
                        expected.generation,
                    ),
                )
                generation = state[2] + 1
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            # Failed insertions cannot leave cache entries pretending the blob
            # exists durably. A retry captures the latest identities afresh.
            self._captured.clear()
            self._measurements.clear()
            raise
        self._captured = {
            key: value for key, value in self._captured.items() if key in used
        }
        self._fallback = None
        # GC has its own transaction AFTER publication. A crash or GC failure
        # can leave harmless garbage, never invalidate the committed checkpoint.
        try:
            self._collect(conn)
        except (sqlite3.Error, ValueError, KeyError, TypeError, RecoverySchemaError):
            pass
        return CheckpointToken(
            self.profile_key, session.epoch, session.revision, generation
        )

    @staticmethod
    def _content_document(document: str) -> dict:
        """View navigation preserves restore undo; changing the sample does not."""
        value = json.loads(document)["session"]
        value.setdefault("content_revision", 0)
        value.pop("revision")
        value["view"] = {"sample_hash": value["view"]["sample_hash"]}
        return value

    def replace(
        self, imported: LabSession, displaced: LabSession, *, expected: CheckpointToken
    ) -> tuple[LabSession, CheckpointToken]:
        """Atomically preserve in-memory content and install new recovery authority.

        The coordinator must first quiesce execution. The caller must serialize
        this through AutosaveWriter; epochs are returned only after real COMMIT.
        """
        if (displaced.profile_key, displaced.epoch) != (
            self.profile_key,
            expected.epoch,
        ) or displaced.revision < expected.revision:
            raise CheckpointConflict("Displaced session does not own current authority")
        imported = parse_recovery(export_recovery(imported))
        return self._replace_transaction(imported, displaced, expected=expected)

    def undo_restore(
        self, *, expected: CheckpointToken
    ) -> tuple[LabSession, CheckpointToken]:
        """Consume the explicit displaced checkpoint using a fresh epoch."""
        return self._replace_transaction(None, None, expected=expected)

    def _replace_transaction(
        self,
        imported: LabSession | None,
        displaced: LabSession | None,
        *,
        expected: CheckpointToken,
    ) -> tuple[LabSession, CheckpointToken]:
        conn = self._connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            state = self._state(conn)
            self._check_expected(state, expected)
            undoing = imported is None
            if undoing:
                if state[5] is None:
                    raise ValueError("There is no recovery restore to undo")
                imported = self._unpack(conn, state[5])
                displaced = self._unpack(conn, state[3])
            epoch = str(uuid.uuid4())
            replacement = rebase_recovery(imported, self.profile_key, epoch)
            fallback = rebase_recovery(displaced, self.profile_key, epoch)
            # Three bounded checkpoints: exact displaced undo, same-epoch previous
            # fallback, and current. All referenced new blobs share this COMMIT.
            sessions = (
                [fallback, replacement]
                if undoing
                else [displaced, fallback, replacement]
            )
            checkpoint_ids, all_used = [], set()
            for session in sessions:
                document, pending, used = self._pack(session)
                all_used.update(used)
                for digest, (kind, payload) in pending.items():
                    conn.execute(
                        "INSERT OR IGNORE INTO lab_blobs(digest,kind,payload) VALUES(?,?,?)",
                        (digest, kind, payload),
                    )
                checkpoint_ids.append(
                    conn.execute(
                        "INSERT INTO lab_checkpoints(revision,document) VALUES(?,?)",
                        (session.revision, document),
                    ).lastrowid
                )
            conn.execute(
                "UPDATE lab_state SET epoch=?, generation=generation+1, current_checkpoint=?, previous_checkpoint=?, restore_undo_checkpoint=? WHERE singleton=1",
                (
                    epoch,
                    checkpoint_ids[-1],
                    checkpoint_ids[-2],
                    None if undoing else checkpoint_ids[0],
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            self._captured.clear()
            self._measurements.clear()
            raise
        self._captured = {
            key: value for key, value in self._captured.items() if key in all_used
        }
        self._fallback = None
        try:
            self._collect(conn)
        except (sqlite3.Error, ValueError, KeyError, TypeError, RecoverySchemaError):
            pass
        return replacement, CheckpointToken(
            self.profile_key, epoch, replacement.revision, state[2] + 1
        )

    def _collect(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        try:
            state = self._state(conn)
            retained = {value for value in state[3:] if value is not None}
            references: set[str] = set()
            for checkpoint_id in retained:
                row = conn.execute(
                    "SELECT document FROM lab_checkpoints WHERE id=?", (checkpoint_id,)
                ).fetchone()
                document = json.loads(row[0])["session"]
                references.update(document["samples"].values())
                references.update(document["results"].values())
                references.update(
                    entry["sample"]
                    for entry in document["undo"]
                    if entry.get("kind") == "sample"
                )
                if document["batch"] is not None:
                    references.update(document["batch"]["requests"].values())
            for (checkpoint_id,) in conn.execute(
                "SELECT id FROM lab_checkpoints"
            ).fetchall():
                if checkpoint_id not in retained:
                    conn.execute(
                        "DELETE FROM lab_checkpoints WHERE id=?", (checkpoint_id,)
                    )
            for (digest,) in conn.execute("SELECT digest FROM lab_blobs").fetchall():
                if digest not in references:
                    conn.execute("DELETE FROM lab_blobs WHERE digest=?", (digest,))
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def clear(self, *, expected: CheckpointToken) -> CheckpointToken:
        """Publish a new-epoch tombstone and delete every content reference."""
        conn = self._connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            state = self._state(conn)
            self._check_expected(state, expected)
            epoch = str(uuid.uuid4())
            generation = state[2] + 1
            conn.execute(
                "UPDATE lab_state SET epoch=?, generation=?, current_checkpoint=NULL, previous_checkpoint=NULL, restore_undo_checkpoint=NULL WHERE singleton=1",
                (epoch, generation),
            )
            conn.execute("DELETE FROM lab_checkpoints")
            conn.execute("DELETE FROM lab_blobs")
            conn.execute("COMMIT")
            self._captured.clear()
            self._measurements.clear()
            return CheckpointToken(self.profile_key, epoch, 0, generation)
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        """Close on the owning thread and release captured immutable payloads."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._captured.clear()
        self._measurements.clear()
