"""Dedicated encrypted SQLite repository for Personal Context objects."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from pydantic import BaseModel, ValidationError
from tldw_profile_core import (
    ActorType,
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    ProposalState,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
    canonical_bytes,
)
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

from .crypto import EncryptedEnvelope, EnvelopeCipher
from .key_protector import (
    KeyringProfileKeyProtector,
    ProfileKeyMaterial,
    ProfileKeyProtector,
    ProfileLockedError,
)
from .repository_models import QuarantineEntry
from .runtime_policy import GLOBAL_POLICY_ID


SCHEMA_VERSION = 4
_ENVELOPE_SCHEMA_VERSION = 1
_WRAP_NONCE_BYTES = 12
_PEER_ENVELOPE = "chatbook-local-v1"
MAX_UNRESOLVED_PROPOSALS = 200
_COLLECTION_PAGE_SIZE = 128


class RepositorySchemaError(RuntimeError):
    """Report an unversioned, foreign, or unsupported repository schema."""


class ProfileAlreadyExistsError(RuntimeError):
    """Report an attempt to create a second profile in one installation."""


class ConcurrentProfileUpdateError(RuntimeError):
    """Report a compare-and-set head mismatch."""


class ProfileIntegrityError(RuntimeError):
    """Report an authenticated-envelope or canonical-integrity failure."""


class ProfileDestroyedError(ProfileLockedError):
    """Report an attempt to mutate a durably destroyed profile."""


class ProposalLimitExceededError(RuntimeError):
    """Report that the durable unresolved proposal ceiling was reached."""


class RecordCollisionError(RuntimeError):
    """Report an active semantic-key collision found inside a write transaction."""

    def __init__(self, record_id: str) -> None:
        self.record_id = record_id
        super().__init__("record_collision")


@dataclass(frozen=True, slots=True)
class AgentAuthorityFence:
    """Peer-local authority versions that must still match at commit time."""

    scope_id: str
    global_policy_version: str | None
    scope_policy_version: str | None
    binding_version: str | None
    binding_required: bool


def profile_presence_hint(db_path: str | os.PathLike[str]) -> bool:
    """Read only content-free profile presence without creating or migrating."""

    candidate = Path(db_path)
    try:
        if not candidate.is_file():
            return False
        with connect_private_sqlite(
            "personal_context.repository",
            candidate,
            read_only=True,
            must_exist=True,
        ) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'profile_meta'"
            ).fetchone()
            if table is None:
                return False
            row = connection.execute(
                "SELECT destroyed FROM profile_meta WHERE singleton = 1"
            ).fetchone()
        return row is not None and row[0] == 0
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return False


_LOCAL_UNDO_SCHEMA = """
    CREATE TABLE local_undo (
        undo_id TEXT PRIMARY KEY,
        record_id TEXT NOT NULL,
        encrypted_undo_version TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
_LOCAL_RECORD_LINK_SCHEMA = """
    CREATE TABLE local_record_links (
        record_id TEXT PRIMARY KEY,
        encrypted_link_version TEXT NOT NULL
    )
    """


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE personal_context_schema (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE profile_meta (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        profile_id TEXT NOT NULL,
        purge_generation INTEGER NOT NULL,
        destroyed INTEGER NOT NULL DEFAULT 0,
        current_manifest_version TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE encrypted_objects (
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        scope_id TEXT,
        is_tombstone INTEGER NOT NULL DEFAULT 0,
        algorithm TEXT NOT NULL,
        nonce BLOB NOT NULL,
        ciphertext BLOB NOT NULL,
        wrapped_dek BLOB NOT NULL,
        key_version INTEGER NOT NULL,
        integrity_tag TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (object_type, object_id, version_id)
    )
    """,
    """
    CREATE TABLE object_heads (
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        PRIMARY KEY (object_type, object_id)
    )
    """,
    """
    CREATE TABLE local_runtime_policy (
        scope_id TEXT PRIMARY KEY,
        encrypted_policy_version TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE local_scope_bindings (
        scope_id TEXT PRIMARY KEY,
        encrypted_binding_version TEXT NOT NULL
    )
    """,
    _LOCAL_UNDO_SCHEMA,
    _LOCAL_RECORD_LINK_SCHEMA,
    """
    CREATE TABLE encrypted_outbox (
        outbox_id TEXT PRIMARY KEY,
        sequence INTEGER NOT NULL UNIQUE,
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        envelope_version TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        destination_envelope_id TEXT,
        quarantine_reason TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE quarantine (
        quarantine_id TEXT PRIMARY KEY,
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        version_id TEXT,
        reason_code TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX quarantine_object_idx ON quarantine(object_type, object_id, version_id)",
)


def _now() -> datetime:
    now = datetime.now(UTC)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


def _now_text() -> str:
    return _now().isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _uuid(label: str) -> str:
    return f"{label}-{uuid.uuid4()}"


class PersonalContextRepository:
    """Persist canonical profile objects as independently encrypted versions."""

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        *,
        key_protector: ProfileKeyProtector | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._profile_ref = (
            "personal-context:"
            + hashlib.sha256(os.fsencode(str(self.db_path.absolute()))).hexdigest()
        )
        self._protector = key_protector or KeyringProfileKeyProtector()
        self._keys: ProfileKeyMaterial | None = None

        with self._transaction() as connection:
            is_new = self._inspect_schema(connection)
            if is_new:
                keys = self._protector.load_or_create(self._profile_ref)
                self._initialize_schema(connection)
            else:
                meta = connection.execute(
                    "SELECT destroyed FROM profile_meta WHERE singleton = 1"
                ).fetchone()
                keys = (
                    None
                    if meta is not None and bool(meta["destroyed"])
                    else self._protector.load(self._profile_ref)
                )
        self._keys = keys

    def _connect(self) -> sqlite3.Connection:
        """Open a new owner-checked autocommit connection for this operation."""

        connection = connect_private_sqlite(
            "personal_context.repository",
            self.db_path,
            isolation_level=None,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            secure_delete = connection.execute("PRAGMA secure_delete = ON").fetchone()
            if secure_delete is None or int(secure_delete[0]) != 1:
                raise ProfileIntegrityError(
                    "Personal Context requires SQLite secure deletion."
                )
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except BaseException:
            connection.close()
            raise

    def _iter_head_rows(self, object_type: str) -> Iterator[sqlite3.Row]:
        """Yield one complete head set through bounded, stable keyset pages."""

        if object_type not in {"record", "scope", "proposal"}:
            raise ValueError("Unsupported collection object type.")
        after_object_id = ""
        while True:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT encrypted_objects.* FROM object_heads "
                    "JOIN encrypted_objects USING (object_type, object_id, version_id) "
                    "WHERE object_type = ? AND object_id > ? "
                    "ORDER BY object_id LIMIT ?",
                    (object_type, after_object_id, _COLLECTION_PAGE_SIZE),
                ).fetchall()
            if not rows:
                return
            yield from rows
            after_object_id = rows[-1]["object_id"]

    def close(self) -> None:
        """Close the repository; operations own no persistent connection."""

    def is_destroyed(self) -> bool:
        """Return the content-free durable local-removal marker."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT destroyed FROM profile_meta WHERE singleton = 1"
            ).fetchone()
        return row is not None and bool(row["destroyed"])

    def _inspect_schema(self, connection: sqlite3.Connection) -> bool:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
        if not tables:
            return True
        if "personal_context_schema" not in tables:
            raise RepositorySchemaError(
                "Refusing to open a nonempty unversioned or foreign database."
            )
        rows = connection.execute(
            "SELECT singleton, version FROM personal_context_schema"
        ).fetchall()
        if len(rows) != 1 or rows[0]["singleton"] != 1:
            raise RepositorySchemaError("Personal Context schema marker is invalid.")
        version = rows[0]["version"]
        if version not in {1, 2, 3, SCHEMA_VERSION}:
            raise RepositorySchemaError(
                f"Unsupported Personal Context schema version: {version!r}."
            )
        v1_expected = {
            "personal_context_schema",
            "profile_meta",
            "encrypted_objects",
            "object_heads",
            "local_runtime_policy",
            "local_scope_bindings",
            "encrypted_outbox",
            "quarantine",
        }
        if not v1_expected.issubset(tables):
            raise RepositorySchemaError("Personal Context schema is incomplete.")
        if version == 1:
            connection.execute(_LOCAL_UNDO_SCHEMA)
            connection.execute(_LOCAL_RECORD_LINK_SCHEMA)
            version = 2
            tables.add("local_undo")
            tables.add("local_record_links")
        if version == 2:
            outbox_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(encrypted_outbox)")
            }
            for column_name, definition in (
                ("destination_envelope_id", "destination_envelope_id TEXT"),
                ("quarantine_reason", "quarantine_reason TEXT"),
                ("updated_at", "updated_at TEXT"),
            ):
                if column_name not in outbox_columns:
                    connection.execute(
                        f"ALTER TABLE encrypted_outbox ADD COLUMN {definition}"
                    )
            connection.execute(
                "UPDATE encrypted_outbox SET updated_at = created_at "
                "WHERE updated_at IS NULL"
            )
            version = 3
        if version == 3:
            outbox_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(encrypted_outbox)")
            }
            if "sequence" not in outbox_columns:
                connection.execute(
                    "ALTER TABLE encrypted_outbox ADD COLUMN sequence INTEGER"
                )
            rows = connection.execute(
                "SELECT outbox_id FROM encrypted_outbox "
                "ORDER BY created_at, rowid"
            ).fetchall()
            for sequence, row in enumerate(rows, start=1):
                connection.execute(
                    "UPDATE encrypted_outbox SET sequence = ? WHERE outbox_id = ?",
                    (sequence, row["outbox_id"]),
                )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS encrypted_outbox_sequence_idx "
                "ON encrypted_outbox(sequence)"
            )
            if connection.execute(
                "SELECT 1 FROM encrypted_outbox WHERE sequence IS NULL LIMIT 1"
            ).fetchone() is not None:
                raise RepositorySchemaError("Personal Context outbox order is invalid.")
            version = SCHEMA_VERSION
        connection.execute(
            "UPDATE personal_context_schema SET version = ? WHERE singleton = 1",
            (version,),
        )
        if not {"local_undo", "local_record_links"}.issubset(tables):
            raise RepositorySchemaError("Personal Context schema is incomplete.")
        profile_meta_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(profile_meta)")
        }
        if "destroyed" not in profile_meta_columns:
            raise RepositorySchemaError("Personal Context schema is incomplete.")
        return False

    def _initialize_schema(self, connection: sqlite3.Connection) -> None:
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO personal_context_schema(singleton, version) VALUES (1, ?)",
            (SCHEMA_VERSION,),
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
            self._truncate_wal_if_possible(connection)
        except BaseException:
            try:
                connection.rollback()
            except BaseException:
                pass
            raise
        finally:
            connection.close()

    @staticmethod
    def _truncate_wal_if_possible(connection: sqlite3.Connection) -> bool:
        """Scrub checkpointed WAL frames without waiting on active readers."""

        try:
            row = connection.execute("PRAGMA journal_mode").fetchone()
            if row is None or str(row[0]).lower() != "wal":
                return True
            prior_timeout = connection.execute("PRAGMA busy_timeout").fetchone()
            connection.execute("PRAGMA busy_timeout = 0")
            try:
                checkpoint = connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
            finally:
                timeout = 5000 if prior_timeout is None else int(prior_timeout[0])
                connection.execute(f"PRAGMA busy_timeout = {timeout}")
            return checkpoint is not None and int(checkpoint[0]) == 0
        except (sqlite3.Error, TypeError, ValueError):
            return False

    def _checkpoint_after_snapshot(self) -> None:
        """Release WAL history that a completed repository snapshot had pinned."""

        try:
            with closing(self._connect()) as connection:
                self._truncate_wal_if_possible(connection)
        except (OSError, sqlite3.Error, ProfileIntegrityError):
            return

    @contextmanager
    def _mutation(
        self,
        *,
        profile_id: str | None = None,
        allow_empty: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        """Open a write transaction and reject absent or destroyed profile state."""

        with self._transaction() as connection:
            meta = connection.execute(
                "SELECT profile_id, destroyed FROM profile_meta WHERE singleton = 1"
            ).fetchone()
            if meta is None:
                if not allow_empty:
                    raise ProfileDestroyedError(
                        "No active Personal Context profile exists."
                    )
            elif meta["destroyed"]:
                raise ProfileDestroyedError(
                    "The Personal Context profile was destroyed."
                )
            elif profile_id is not None and meta["profile_id"] != profile_id:
                raise ValueError("Object does not belong to the local profile.")
            yield connection

    @staticmethod
    def _require_authority_fence(
        connection: sqlite3.Connection,
        fence: AgentAuthorityFence | None,
    ) -> None:
        if fence is None:
            return

        def version(table: str, column: str, scope_id: str) -> str | None:
            row = connection.execute(
                f"SELECT {column} FROM {table} WHERE scope_id = ?",
                (scope_id,),
            ).fetchone()
            return None if row is None else row[0]

        current = (
            version(
                "local_runtime_policy",
                "encrypted_policy_version",
                GLOBAL_POLICY_ID,
            ),
            version(
                "local_runtime_policy",
                "encrypted_policy_version",
                fence.scope_id,
            ),
            version(
                "local_scope_bindings",
                "encrypted_binding_version",
                fence.scope_id,
            ),
        )
        expected = (
            fence.global_policy_version,
            fence.scope_policy_version,
            fence.binding_version,
        )
        if current != expected or (fence.binding_required and current[2] is None):
            raise ConcurrentProfileUpdateError(
                "Personal Context authority changed concurrently."
            )

    def _require_keys(self) -> ProfileKeyMaterial:
        if self._keys is None:
            raise ProfileLockedError("Profile key material is unavailable.")
        return self._keys

    @staticmethod
    def _canonical_payload(value: BaseModel | Mapping[str, Any]) -> bytes:
        if isinstance(value, BaseModel):
            return canonical_bytes(value)
        return canonical_json_bytes(value)

    @staticmethod
    def _aad(object_type: str, object_id: str, version_id: str) -> bytes:
        return canonical_json_bytes(
            {
                "peer_envelope": _PEER_ENVELOPE,
                "object_type": object_type,
                "object_id": object_id,
                "version_id": version_id,
                "schema_version": _ENVELOPE_SCHEMA_VERSION,
            }
        )

    @staticmethod
    def _integrity_tag(key: bytes, aad: bytes, plaintext: bytes) -> str:
        digest = hmac.new(key, aad + b"\x00" + plaintext, hashlib.sha256).hexdigest()
        return f"hmac-sha256-v1:{digest}"

    def _insert_encrypted(
        self,
        connection: sqlite3.Connection,
        *,
        object_type: str,
        object_id: str,
        version_id: str,
        value: BaseModel | Mapping[str, Any],
        scope_id: str | None = None,
        is_tombstone: bool = False,
    ) -> None:
        keys = self._require_keys()
        plaintext = self._canonical_payload(value)
        aad = self._aad(object_type, object_id, version_id)
        envelope = EnvelopeCipher(
            keys.encryption_key, key_version=keys.key_version
        ).encrypt(plaintext, aad)
        connection.execute(
            """
            INSERT INTO encrypted_objects(
                object_type, object_id, version_id, scope_id, is_tombstone,
                algorithm, nonce, ciphertext, wrapped_dek, key_version,
                integrity_tag, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                object_type,
                object_id,
                version_id,
                scope_id,
                int(is_tombstone),
                envelope.algorithm,
                envelope.nonce,
                envelope.ciphertext,
                envelope.wrap_nonce + envelope.wrapped_dek,
                envelope.key_version,
                self._integrity_tag(keys.integrity_key, aad, plaintext),
                _now_text(),
            ),
        )

    def _decrypt_row(self, row: sqlite3.Row) -> bytes:
        keys = self._require_keys()
        for column in (
            "object_type",
            "object_id",
            "version_id",
            "algorithm",
            "integrity_tag",
        ):
            if not isinstance(row[column], str):
                raise ProfileIntegrityError(
                    f"Encrypted object {column} has an invalid type."
                )
        for column in ("nonce", "ciphertext", "wrapped_dek"):
            if not isinstance(row[column], (bytes, bytearray, memoryview)):
                raise ProfileIntegrityError(
                    f"Encrypted object {column} has an invalid type."
                )
        key_version = row["key_version"]
        if (
            type(key_version) is not int
            or key_version < 1
            or key_version != keys.key_version
        ):
            raise ProfileIntegrityError("Encrypted object key version is untrusted.")
        wrapped = bytes(row["wrapped_dek"])
        if len(wrapped) <= _WRAP_NONCE_BYTES:
            raise ProfileIntegrityError("Encrypted object wrapper is invalid.")
        envelope = EncryptedEnvelope(
            algorithm=row["algorithm"],
            nonce=bytes(row["nonce"]),
            wrap_nonce=wrapped[:_WRAP_NONCE_BYTES],
            ciphertext=bytes(row["ciphertext"]),
            wrapped_dek=wrapped[_WRAP_NONCE_BYTES:],
            key_version=key_version,
        )
        aad = self._aad(row["object_type"], row["object_id"], row["version_id"])
        try:
            plaintext = EnvelopeCipher(
                keys.encryption_key, key_version=keys.key_version
            ).decrypt(envelope, aad)
        except (InvalidTag, TypeError, ValueError) as exc:
            raise ProfileIntegrityError(
                "Encrypted object authentication failed."
            ) from exc
        expected = self._integrity_tag(keys.integrity_key, aad, plaintext)
        if not hmac.compare_digest(expected, row["integrity_tag"]):
            raise ProfileIntegrityError("Canonical object integrity failed.")
        return plaintext

    def _head_row(self, object_type: str, object_id: str) -> sqlite3.Row | None:
        with closing(self._connect()) as connection:
            return connection.execute(
                """
                SELECT encrypted_objects.*
                FROM object_heads
                JOIN encrypted_objects USING (object_type, object_id, version_id)
                WHERE object_type = ? AND object_id = ?
                """,
                (object_type, object_id),
            ).fetchone()

    @staticmethod
    def _set_head(
        connection: sqlite3.Connection,
        *,
        object_type: str,
        object_id: str,
        version_id: str,
        expected_version_id: str | None,
    ) -> None:
        current = connection.execute(
            "SELECT version_id FROM object_heads WHERE object_type = ? AND object_id = ?",
            (object_type, object_id),
        ).fetchone()
        if current is None:
            if expected_version_id is not None:
                raise ConcurrentProfileUpdateError("Object head does not exist.")
            connection.execute(
                "INSERT INTO object_heads VALUES (?, ?, ?)",
                (object_type, object_id, version_id),
            )
            return
        if current["version_id"] != expected_version_id:
            raise ConcurrentProfileUpdateError("Object head changed concurrently.")
        updated = connection.execute(
            """
            UPDATE object_heads SET version_id = ?
            WHERE object_type = ? AND object_id = ? AND version_id = ?
            """,
            (version_id, object_type, object_id, expected_version_id),
        )
        if updated.rowcount != 1:
            raise ConcurrentProfileUpdateError("Object head changed concurrently.")

    def create_provisional_profile(self) -> ProfileManifest:
        """Create the installation's only local profile manifest."""

        self._require_keys()
        now = _now()
        profile_id = _uuid("profile-local")
        version_id = _uuid("manifest-version")
        manifest = ProfileManifest(
            profile_id=profile_id,
            revision=0,
            purge_generation=0,
            created_at=now,
            updated_at=now,
            current_version_id=version_id,
        )
        with self._mutation(allow_empty=True) as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM profile_meta WHERE singleton = 1"
                ).fetchone()
                is not None
            ):
                raise ProfileAlreadyExistsError(
                    "One Personal Context profile already exists for this installation."
                )
            self._insert_encrypted(
                connection,
                object_type="manifest",
                object_id=profile_id,
                version_id=version_id,
                value=manifest,
            )
            self._set_head(
                connection,
                object_type="manifest",
                object_id=profile_id,
                version_id=version_id,
                expected_version_id=None,
            )
            connection.execute(
                """
                INSERT INTO profile_meta(
                    singleton, profile_id, purge_generation, destroyed,
                    current_manifest_version, created_at
                ) VALUES (1, ?, 0, 0, ?, ?)
                """,
                (profile_id, version_id, _now_text()),
            )
            self._insert_outbox(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                body={"version": 1, "manifest": manifest.model_dump(mode="json")},
            )
        return manifest

    def create_profile_with_global_scope(
        self, manifest: ProfileManifest, global_scope: ProfileScope
    ) -> None:
        """Atomically persist one manifest and its required global scope."""

        if (
            global_scope.profile_id != manifest.profile_id
            or global_scope.kind is not ScopeKind.GLOBAL
        ):
            raise ValueError("Global scope must belong to the new profile.")
        with self._mutation(allow_empty=True) as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM profile_meta WHERE singleton = 1"
                ).fetchone()
                is not None
            ):
                raise ProfileAlreadyExistsError(
                    "One Personal Context profile already exists for this installation."
                )
            self._insert_encrypted(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                value=manifest,
            )
            self._set_head(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                expected_version_id=None,
            )
            self._insert_scope(connection, global_scope, expected_version_id=None)
            connection.execute(
                """
                INSERT INTO profile_meta(
                    singleton, profile_id, purge_generation, destroyed,
                    current_manifest_version, created_at
                ) VALUES (1, ?, ?, 0, ?, ?)
                """,
                (
                    manifest.profile_id,
                    manifest.purge_generation,
                    manifest.current_version_id,
                    _now_text(),
                ),
            )
            self._insert_outbox(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                body={"version": 1, "manifest": manifest.model_dump(mode="json")},
            )
            self._insert_outbox(
                connection,
                object_type="scope",
                object_id=global_scope.scope_id,
                version_id=global_scope.version_id,
                body={"version": 1, "scope": global_scope.model_dump(mode="json")},
            )

    def reinitialize_destroyed_profile(
        self, manifest: ProfileManifest, global_scope: ProfileScope
    ) -> None:
        """Explicitly replace one destroyed local generation with a fresh profile."""

        if (
            global_scope.profile_id != manifest.profile_id
            or global_scope.kind is not ScopeKind.GLOBAL
        ):
            raise ValueError("Global scope must belong to the new profile.")
        prepared_keys = False
        try:
            with self._transaction() as connection:
                meta = connection.execute(
                    "SELECT destroyed FROM profile_meta WHERE singleton = 1"
                ).fetchone()
                if meta is None or not bool(meta["destroyed"]):
                    raise ProfileDestroyedError(
                        "A fresh profile can only replace a removed local copy."
                    )
                keys = self._protector.load_or_create(self._profile_ref)
                prepared_keys = True
                self._keys = keys
                connection.execute("DELETE FROM profile_meta WHERE singleton = 1")
                self._insert_encrypted(
                    connection,
                    object_type="manifest",
                    object_id=manifest.profile_id,
                    version_id=manifest.current_version_id,
                    value=manifest,
                )
                self._set_head(
                    connection,
                    object_type="manifest",
                    object_id=manifest.profile_id,
                    version_id=manifest.current_version_id,
                    expected_version_id=None,
                )
                self._insert_scope(connection, global_scope, expected_version_id=None)
                connection.execute(
                    """
                    INSERT INTO profile_meta(
                        singleton, profile_id, purge_generation, destroyed,
                        current_manifest_version, created_at
                    ) VALUES (1, ?, ?, 0, ?, ?)
                    """,
                    (
                        manifest.profile_id,
                        manifest.purge_generation,
                        manifest.current_version_id,
                        _now_text(),
                    ),
                )
                self._insert_outbox(
                    connection,
                    object_type="manifest",
                    object_id=manifest.profile_id,
                    version_id=manifest.current_version_id,
                    body={"version": 1, "manifest": manifest.model_dump(mode="json")},
                )
                self._insert_outbox(
                    connection,
                    object_type="scope",
                    object_id=global_scope.scope_id,
                    version_id=global_scope.version_id,
                    body={"version": 1, "scope": global_scope.model_dump(mode="json")},
                )
        except BaseException:
            self._keys = None
            if prepared_keys:
                try:
                    self._protector.delete(self._profile_ref)
                except Exception:
                    pass
            raise

    def get_manifest(self) -> ProfileManifest | None:
        """Return the current authenticated manifest, if one exists."""

        self._require_keys()
        with closing(self._connect()) as connection:
            meta = connection.execute(
                "SELECT profile_id, current_manifest_version FROM profile_meta WHERE singleton = 1"
            ).fetchone()
            if meta is None:
                return None
            row = connection.execute(
                """
                SELECT * FROM encrypted_objects
                WHERE object_type = 'manifest' AND object_id = ? AND version_id = ?
                """,
                (meta["profile_id"], meta["current_manifest_version"]),
            ).fetchone()
        if row is None:
            raise ProfileLockedError("The profile manifest is unavailable.")
        try:
            return ProfileManifest.model_validate_json(self._decrypt_row(row))
        except (ProfileIntegrityError, ValidationError) as exc:
            raise ProfileLockedError(
                "The profile manifest could not be authenticated."
            ) from exc

    def commit_manifest_version(
        self,
        manifest: ProfileManifest,
        *,
        expected_version_id: str,
    ) -> None:
        """Commit one exact synced manifest revision without an outbound echo."""

        with self._mutation(profile_id=manifest.profile_id) as connection:
            meta = connection.execute(
                "SELECT current_manifest_version FROM profile_meta WHERE singleton = 1"
            ).fetchone()
            if meta is None or meta["current_manifest_version"] != expected_version_id:
                raise ConcurrentProfileUpdateError(
                    "Profile manifest changed concurrently."
                )
            row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = 'manifest' "
                "AND object_id = ? AND version_id = ?",
                (manifest.profile_id, expected_version_id),
            ).fetchone()
            if row is None:
                raise ProfileIntegrityError("Current manifest is unavailable.")
            current = ProfileManifest.model_validate_json(self._decrypt_row(row))
            if (
                manifest.revision != current.revision + 1
                or manifest.purge_generation != current.purge_generation
                or manifest.created_at != current.created_at
                or manifest.current_version_id == expected_version_id
                or manifest.updated_at < current.updated_at
            ):
                raise ConcurrentProfileUpdateError("Manifest lineage changed.")
            self._insert_encrypted(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                value=manifest,
            )
            self._set_head(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                expected_version_id=expected_version_id,
            )
            connection.execute(
                "UPDATE profile_meta SET current_manifest_version = ?, "
                "purge_generation = ? WHERE singleton = 1",
                (manifest.current_version_id, manifest.purge_generation),
            )

    def commit_record_version(
        self,
        record: ProfileRecord,
        *,
        expected_version_id: str | None,
        outbox_body: Mapping[str, Any] | None = None,
        allow_orphan_tombstone: bool = False,
    ) -> None:
        """Atomically insert an immutable record, CAS its head, and queue sync."""

        orphan_tombstone = (
            allow_orphan_tombstone
            and expected_version_id is None
            and record.parent_version_id is not None
            and record.state is RecordState.DELETED
            and record.payload is None
        )
        with self._mutation(profile_id=record.profile_id) as connection:
            if record.parent_version_id != expected_version_id and not orphan_tombstone:
                raise ConcurrentProfileUpdateError(
                    "Record parent does not match the expected head."
                )
            self._insert_encrypted(
                connection,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                scope_id=record.scope_id,
                is_tombstone=record.state.value == "deleted",
                value=record,
            )
            self._set_head(
                connection,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                expected_version_id=expected_version_id,
            )
            if (
                outbox_body is not None
                and record.controls.sync_mode is SyncMode.SYNCABLE
            ):
                self._insert_outbox(
                    connection,
                    object_type="record",
                    object_id=record.record_id,
                    version_id=record.version_id,
                    body=outbox_body,
                )
    def commit_record_and_manifest(
        self,
        record: ProfileRecord,
        manifest: ProfileManifest,
        *,
        expected_record_version: str | None,
        expected_manifest_version: str,
        undo_id: str | None = None,
        undo_body: Mapping[str, Any] | None = None,
        undo_expires_at: str | None = None,
        consume_undo_id: str | None = None,
        outbox_body: Mapping[str, Any] | None = None,
        authority_fence: AgentAuthorityFence | None = None,
    ) -> None:
        """Atomically CAS a record, next manifest, outbox, and optional Undo."""

        if record.profile_id != manifest.profile_id:
            raise ValueError("Record and manifest profile identities differ.")
        if record.parent_version_id != expected_record_version:
            raise ConcurrentProfileUpdateError(
                "Record parent does not match the expected head."
            )
        if (undo_body is None) != (undo_id is None) or (undo_body is None) != (
            undo_expires_at is None
        ):
            raise ValueError("Undo metadata must be supplied together.")

        with self._mutation(profile_id=record.profile_id) as connection:
            self._require_authority_fence(connection, authority_fence)
            meta = connection.execute(
                "SELECT current_manifest_version FROM profile_meta WHERE singleton = 1"
            ).fetchone()
            if (
                meta is None
                or meta["current_manifest_version"] != expected_manifest_version
            ):
                raise ConcurrentProfileUpdateError(
                    "Profile manifest changed concurrently."
                )
            current_manifest_row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = 'manifest' "
                "AND object_id = ? AND version_id = ?",
                (manifest.profile_id, expected_manifest_version),
            ).fetchone()
            if current_manifest_row is None:
                raise ProfileIntegrityError("Current manifest is unavailable.")
            current_manifest = ProfileManifest.model_validate_json(
                self._decrypt_row(current_manifest_row)
            )
            if (
                manifest.revision != current_manifest.revision + 1
                or manifest.purge_generation != current_manifest.purge_generation
                or manifest.created_at != current_manifest.created_at
            ):
                raise ValueError("Next manifest is not a valid revision successor.")

            self._insert_encrypted(
                connection,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                scope_id=record.scope_id,
                is_tombstone=record.state.value == "deleted",
                value=record,
            )
            self._set_head(
                connection,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                expected_version_id=expected_record_version,
            )
            if record.state is RecordState.DELETED:
                self._retire_record_content(
                    connection,
                    record.record_id,
                    keep_version_id=record.version_id,
                )
            self._insert_encrypted(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                value=manifest,
            )
            self._set_head(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                expected_version_id=expected_manifest_version,
            )
            connection.execute(
                "UPDATE profile_meta SET current_manifest_version = ?, "
                "purge_generation = ? WHERE singleton = 1",
                (manifest.current_version_id, manifest.purge_generation),
            )
            self._insert_outbox(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                body={"version": 1, "manifest": manifest.model_dump(mode="json")},
            )
            if undo_body is not None:
                assert undo_id is not None and undo_expires_at is not None
                prior_undo = connection.execute(
                    "SELECT undo_id FROM local_undo WHERE record_id = ?",
                    (record.record_id,),
                ).fetchall()
                for row in prior_undo:
                    self._delete_undo(connection, row["undo_id"])
                undo_version = _uuid("undo-version")
                self._insert_encrypted(
                    connection,
                    object_type="undo",
                    object_id=undo_id,
                    version_id=undo_version,
                    scope_id=record.scope_id,
                    value=undo_body,
                )
                connection.execute(
                    "INSERT INTO local_undo VALUES (?, ?, ?, ?, ?)",
                    (
                        undo_id,
                        record.record_id,
                        undo_version,
                        undo_expires_at,
                        _now_text(),
                    ),
                )
            if consume_undo_id is not None:
                self._delete_undo(connection, consume_undo_id)
            if (
                outbox_body is not None
                and record.controls.sync_mode is SyncMode.SYNCABLE
            ):
                self._insert_outbox(
                    connection,
                    object_type="record",
                    object_id=record.record_id,
                    version_id=record.version_id,
                    body=outbox_body,
                )

    def commit_interview_batch(
        self,
        records: tuple[ProfileRecord, ...],
        manifest: ProfileManifest,
        *,
        expected_record_versions: Mapping[str, str | None],
        expected_manifest_version: str,
    ) -> None:
        """Atomically commit selected interview records and one manifest revision."""

        record_ids = tuple(record.record_id for record in records)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("Interview batch contains duplicate record IDs.")
        if set(expected_record_versions) != set(record_ids):
            raise ValueError("Interview batch record fences are incomplete.")
        if any(record.profile_id != manifest.profile_id for record in records):
            raise ValueError("Interview batch crosses profile identities.")
        if any(
            record.parent_version_id != expected_record_versions[record.record_id]
            for record in records
        ):
            raise ConcurrentProfileUpdateError(
                "Interview record parent does not match its expected head."
            )

        with self._mutation(profile_id=manifest.profile_id) as connection:
            meta = connection.execute(
                "SELECT current_manifest_version FROM profile_meta WHERE singleton = 1"
            ).fetchone()
            if (
                meta is None
                or meta["current_manifest_version"] != expected_manifest_version
            ):
                raise ConcurrentProfileUpdateError(
                    "Profile manifest changed concurrently."
                )
            current_manifest_row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = 'manifest' "
                "AND object_id = ? AND version_id = ?",
                (manifest.profile_id, expected_manifest_version),
            ).fetchone()
            if current_manifest_row is None:
                raise ProfileIntegrityError("Current manifest is unavailable.")
            current_manifest = ProfileManifest.model_validate_json(
                self._decrypt_row(current_manifest_row)
            )
            if (
                manifest.revision != current_manifest.revision + 1
                or manifest.purge_generation != current_manifest.purge_generation
                or manifest.created_at != current_manifest.created_at
                or manifest.current_version_id == expected_manifest_version
            ):
                raise ValueError("Next manifest is not a valid revision successor.")
            for record in records:
                current = connection.execute(
                    "SELECT version_id FROM object_heads "
                    "WHERE object_type = 'record' AND object_id = ?",
                    (record.record_id,),
                ).fetchone()
                expected = expected_record_versions[record.record_id]
                if (current is None and expected is not None) or (
                    current is not None and current["version_id"] != expected
                ):
                    raise ConcurrentProfileUpdateError(
                        "Interview record changed concurrently."
                    )

            for record in records:
                expected = expected_record_versions[record.record_id]
                self._insert_encrypted(
                    connection,
                    object_type="record",
                    object_id=record.record_id,
                    version_id=record.version_id,
                    scope_id=record.scope_id,
                    is_tombstone=record.state is RecordState.DELETED,
                    value=record,
                )
                self._set_head(
                    connection,
                    object_type="record",
                    object_id=record.record_id,
                    version_id=record.version_id,
                    expected_version_id=expected,
                )
                if record.state is RecordState.DELETED:
                    self._retire_record_content(
                        connection,
                        record.record_id,
                        keep_version_id=record.version_id,
                    )
                if record.controls.sync_mode is SyncMode.SYNCABLE:
                    self._insert_outbox(
                        connection,
                        object_type="record",
                        object_id=record.record_id,
                        version_id=record.version_id,
                        body={"version": 1, "record": record.model_dump(mode="json")},
                    )

            self._insert_encrypted(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                value=manifest,
            )
            self._set_head(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                expected_version_id=expected_manifest_version,
            )
            connection.execute(
                "UPDATE profile_meta SET current_manifest_version = ?, "
                "purge_generation = ? WHERE singleton = 1",
                (manifest.current_version_id, manifest.purge_generation),
            )
            self._insert_outbox(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                body={"version": 1, "manifest": manifest.model_dump(mode="json")},
            )

    def commit_device_only_split(
        self,
        tombstone: ProfileRecord,
        private_record: ProfileRecord,
        manifest: ProfileManifest,
        *,
        expected_record_version: str,
        expected_manifest_version: str,
    ) -> None:
        """Atomically tombstone a shared identity and create its private successor."""

        if (
            tombstone.profile_id != manifest.profile_id
            or private_record.profile_id != manifest.profile_id
            or tombstone.record_id == private_record.record_id
            or tombstone.parent_version_id != expected_record_version
            or private_record.parent_version_id is not None
            or tombstone.state is not RecordState.DELETED
            or tombstone.controls.sync_mode is not SyncMode.SYNCABLE
            or private_record.state is not RecordState.ACTIVE
            or private_record.controls.sync_mode is not SyncMode.DEVICE_ONLY
        ):
            raise ValueError("Device-only split records are invalid.")
        with self._mutation(profile_id=manifest.profile_id) as connection:
            meta = connection.execute(
                "SELECT current_manifest_version FROM profile_meta WHERE singleton = 1"
            ).fetchone()
            if (
                meta is None
                or meta["current_manifest_version"] != expected_manifest_version
            ):
                raise ConcurrentProfileUpdateError(
                    "Profile manifest changed concurrently."
                )
            current_manifest_row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = 'manifest' "
                "AND object_id = ? AND version_id = ?",
                (manifest.profile_id, expected_manifest_version),
            ).fetchone()
            if current_manifest_row is None:
                raise ProfileIntegrityError("Current manifest is unavailable.")
            current_manifest = ProfileManifest.model_validate_json(
                self._decrypt_row(current_manifest_row)
            )
            if (
                manifest.revision != current_manifest.revision + 1
                or manifest.purge_generation != current_manifest.purge_generation
                or manifest.created_at != current_manifest.created_at
            ):
                raise ValueError("Next manifest is not a valid revision successor.")

            for record, expected in (
                (tombstone, expected_record_version),
                (private_record, None),
            ):
                self._insert_encrypted(
                    connection,
                    object_type="record",
                    object_id=record.record_id,
                    version_id=record.version_id,
                    scope_id=record.scope_id,
                    is_tombstone=record.state is RecordState.DELETED,
                    value=record,
                )
                self._set_head(
                    connection,
                    object_type="record",
                    object_id=record.record_id,
                    version_id=record.version_id,
                    expected_version_id=expected,
                )

            self._retire_record_content(
                connection,
                tombstone.record_id,
                keep_version_id=tombstone.version_id,
            )

            self._insert_encrypted(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                value=manifest,
            )
            self._set_head(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                expected_version_id=expected_manifest_version,
            )
            connection.execute(
                "UPDATE profile_meta SET current_manifest_version = ?, "
                "purge_generation = ? WHERE singleton = 1",
                (manifest.current_version_id, manifest.purge_generation),
            )
            self._insert_outbox(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                body={"version": 1, "manifest": manifest.model_dump(mode="json")},
            )

            link_version = _uuid("record-link-version")
            self._insert_encrypted(
                connection,
                object_type="record_link",
                object_id=private_record.record_id,
                version_id=link_version,
                scope_id=private_record.scope_id,
                value={"version": 1, "source_record_id": tombstone.record_id},
            )
            connection.execute(
                "INSERT INTO local_record_links VALUES (?, ?)",
                (private_record.record_id, link_version),
            )
            self._insert_outbox(
                connection,
                object_type="record",
                object_id=tombstone.record_id,
                version_id=tombstone.version_id,
                body={"version": 1, "record": tombstone.model_dump(mode="json")},
            )

    @staticmethod
    def _retire_record_content(
        connection: sqlite3.Connection,
        record_id: str,
        *,
        keep_version_id: str,
    ) -> None:
        """Remove prior record versions and unsent outbox bodies on tombstone."""

        pending = connection.execute(
            "SELECT outbox_id, envelope_version FROM encrypted_outbox "
            "WHERE object_type = 'record' AND object_id = ? AND status = 'pending'",
            (record_id,),
        ).fetchall()
        connection.execute(
            "DELETE FROM encrypted_outbox WHERE object_type = 'record' "
            "AND object_id = ? AND status = 'pending'",
            (record_id,),
        )
        for row in pending:
            connection.execute(
                "DELETE FROM object_heads WHERE object_type = 'outbox' "
                "AND object_id = ? AND version_id = ?",
                (row["outbox_id"], row["envelope_version"]),
            )
            connection.execute(
                "DELETE FROM encrypted_objects WHERE object_type = 'outbox' "
                "AND object_id = ? AND version_id = ?",
                (row["outbox_id"], row["envelope_version"]),
            )
        connection.execute(
            "DELETE FROM encrypted_objects WHERE object_type = 'record' "
            "AND object_id = ? AND version_id != ?",
            (record_id, keep_version_id),
        )
        stale_undo = connection.execute(
            "SELECT undo_id FROM local_undo WHERE record_id = ?",
            (record_id,),
        ).fetchall()
        for row in stale_undo:
            PersonalContextRepository._delete_undo(connection, row["undo_id"])

    @staticmethod
    def _retire_pending_outbox_content(
        connection: sqlite3.Connection,
        *,
        object_type: str,
        object_id: str,
    ) -> None:
        """Remove superseded pending snapshots while preserving durable receipts."""

        pending = connection.execute(
            "SELECT outbox_id, envelope_version FROM encrypted_outbox "
            "WHERE object_type = ? AND object_id = ? AND status = 'pending'",
            (object_type, object_id),
        ).fetchall()
        for row in pending:
            PersonalContextRepository._shred_outbox_body(
                connection,
                row["outbox_id"],
                row["envelope_version"],
            )
            connection.execute(
                "DELETE FROM encrypted_outbox WHERE outbox_id = ?",
                (row["outbox_id"],),
            )

    def get_record(self, record_id: str) -> ProfileRecord | None:
        """Return one current record, quarantining corrupt content."""

        row = self._head_row("record", record_id)
        if row is None or self._is_quarantined("record", record_id, row["version_id"]):
            return None
        try:
            return ProfileRecord.model_validate_json(self._decrypt_row(row))
        except (ProfileIntegrityError, ValidationError):
            self.quarantine_object(
                "record", record_id, row["version_id"], "integrity_failure"
            )
            return None

    def get_record_derivation(self, record_id: str) -> str | None:
        """Return the encrypted peer-local source identity for a private record."""

        with closing(self._connect()) as connection:
            metadata = connection.execute(
                "SELECT encrypted_link_version FROM local_record_links "
                "WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if metadata is None:
                return None
            row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = 'record_link' "
                "AND object_id = ? AND version_id = ?",
                (record_id, metadata["encrypted_link_version"]),
            ).fetchone()
        if row is None:
            raise ProfileIntegrityError("Encrypted record link is unavailable.")
        body = json.loads(self._decrypt_row(row))
        if (
            not isinstance(body, dict)
            or set(body) != {"version", "source_record_id"}
            or body["version"] != 1
            or not isinstance(body["source_record_id"], str)
            or not body["source_record_id"]
        ):
            raise ProfileIntegrityError("Encrypted record link is invalid.")
        return body["source_record_id"]

    def list_records(self) -> list[ProfileRecord]:
        """Return authenticated current records, omitting quarantined objects."""

        records: list[ProfileRecord] = []
        for row in self._iter_head_rows("record"):
            record = self.get_record(row["object_id"])
            if record is not None:
                records.append(record)
        return records

    def _insert_scope(
        self,
        connection: sqlite3.Connection,
        scope: ProfileScope,
        *,
        expected_version_id: str | None,
    ) -> None:
        self._insert_encrypted(
            connection,
            object_type="scope",
            object_id=scope.scope_id,
            version_id=scope.version_id,
            scope_id=scope.scope_id,
            value=scope,
        )
        self._set_head(
            connection,
            object_type="scope",
            object_id=scope.scope_id,
            version_id=scope.version_id,
            expected_version_id=expected_version_id,
        )

    def commit_scope(
        self, scope: ProfileScope, *, expected_version_id: str | None = None
    ) -> None:
        """Commit one encrypted canonical scope revision."""

        with self._mutation(profile_id=scope.profile_id) as connection:
            self._insert_scope(
                connection,
                scope,
                expected_version_id=expected_version_id,
            )

    def commit_scope_with_binding(
        self, scope: ProfileScope, binding: Mapping[str, Any]
    ) -> None:
        """Atomically create a canonical scope and its encrypted local binding."""

        binding_version = _uuid("scope-binding-version")
        with self._mutation(profile_id=scope.profile_id) as connection:
            self._require_unique_workspace_binding(connection, binding)
            self._insert_scope(
                connection,
                scope,
                expected_version_id=None,
            )
            self._insert_encrypted(
                connection,
                object_type="scope_binding",
                object_id=scope.scope_id,
                version_id=binding_version,
                scope_id=scope.scope_id,
                value=binding,
            )
            self._set_head(
                connection,
                object_type="scope_binding",
                object_id=scope.scope_id,
                version_id=binding_version,
                expected_version_id=None,
            )
            connection.execute(
                "INSERT INTO local_scope_bindings VALUES (?, ?)",
                (scope.scope_id, binding_version),
            )
            self._insert_outbox(
                connection,
                object_type="scope",
                object_id=scope.scope_id,
                version_id=scope.version_id,
                body={"version": 1, "scope": scope.model_dump(mode="json")},
            )

    def get_scope(self, scope_id: str) -> ProfileScope | None:
        row = self._head_row("scope", scope_id)
        if row is None or self._is_quarantined("scope", scope_id, row["version_id"]):
            return None
        try:
            return ProfileScope.model_validate_json(self._decrypt_row(row))
        except (ProfileIntegrityError, ValidationError):
            self.quarantine_object(
                "scope", scope_id, row["version_id"], "integrity_failure"
            )
            return None

    def list_scopes(self) -> list[ProfileScope]:
        return [
            scope
            for row in self._iter_head_rows("scope")
            if (scope := self.get_scope(row["object_id"]))
        ]

    def commit_proposal(
        self,
        proposal: ProfileProposal,
        *,
        unresolved_limit: int = MAX_UNRESOLVED_PROPOSALS,
        expire_before: datetime | None = None,
        authority_fence: AgentAuthorityFence | None = None,
        enqueue_outbox: bool = True,
    ) -> None:
        """Commit a new immutable proposal head."""

        if proposal.state is not ProposalState.PENDING:
            raise ValueError("Only pending proposals may be committed.")
        version_id = _uuid("proposal-version")
        with self._mutation(profile_id=proposal.profile_id) as connection:
            self._require_authority_fence(connection, authority_fence)
            if expire_before is not None:
                self._expire_due_proposals_in_connection(connection, expire_before)
            pending = 0
            rows = connection.execute(
                "SELECT encrypted_objects.* FROM object_heads "
                "JOIN encrypted_objects USING (object_type, object_id, version_id) "
                "WHERE object_type = 'proposal'"
            ).fetchall()
            for row in rows:
                current = ProfileProposal.model_validate_json(self._decrypt_row(row))
                pending += current.state is ProposalState.PENDING
            if pending >= unresolved_limit:
                raise ProposalLimitExceededError("unresolved_proposal_limit")
            self._insert_encrypted(
                connection,
                object_type="proposal",
                object_id=proposal.proposal_id,
                version_id=version_id,
                scope_id=proposal.scope_id,
                value=proposal,
            )
            self._set_head(
                connection,
                object_type="proposal",
                object_id=proposal.proposal_id,
                version_id=version_id,
                expected_version_id=None,
            )
            proposed_record = proposal.proposed_record
            syncable = (
                proposed_record is None
                or proposed_record.controls.sync_mode is not SyncMode.DEVICE_ONLY
            )
            if enqueue_outbox and syncable:
                self._insert_outbox(
                    connection,
                    object_type="proposal",
                    object_id=proposal.proposal_id,
                    version_id=version_id,
                    body={
                        "version": 1,
                        "proposal": proposal.model_dump(mode="json"),
                    },
                )

    def commit_synced_proposal(self, proposal: ProfileProposal) -> None:
        """Commit one exact inbound proposal revision without an outbound echo."""

        with self._mutation(profile_id=proposal.profile_id) as connection:
            row = connection.execute(
                "SELECT encrypted_objects.* FROM object_heads "
                "JOIN encrypted_objects USING (object_type, object_id, version_id) "
                "WHERE object_type = 'proposal' AND object_id = ?",
                (proposal.proposal_id,),
            ).fetchone()
            if row is None:
                version_id = _uuid("proposal-version")
                self._insert_encrypted(
                    connection,
                    object_type="proposal",
                    object_id=proposal.proposal_id,
                    version_id=version_id,
                    scope_id=proposal.scope_id,
                    value=proposal,
                )
                self._set_head(
                    connection,
                    object_type="proposal",
                    object_id=proposal.proposal_id,
                    version_id=version_id,
                    expected_version_id=None,
                )
                return

            current = ProfileProposal.model_validate_json(self._decrypt_row(row))
            if current == proposal:
                return
            if (
                current.state is ProposalState.PENDING
                and proposal.state is not ProposalState.PENDING
            ):
                resolved = self._resolve_proposal_in_connection(
                    connection,
                    row,
                    proposal.state,
                    version_id=_uuid("proposal-version"),
                    enqueue_outbox=False,
                )
                if resolved != proposal:
                    raise ConcurrentProfileUpdateError(
                        "Synced proposal receipt differs from the pending proposal."
                    )
                return
            raise ConcurrentProfileUpdateError("Proposal changed concurrently.")

    def expire_due_proposals(self, expire_before: datetime) -> int:
        """Transactionally replace every due pending proposal with a receipt."""

        with self._mutation() as connection:
            return self._expire_due_proposals_in_connection(connection, expire_before)

    def _expire_due_proposals_in_connection(
        self, connection: sqlite3.Connection, expire_before: datetime
    ) -> int:
        rows = connection.execute(
            "SELECT encrypted_objects.* FROM object_heads "
            "JOIN encrypted_objects USING (object_type, object_id, version_id) "
            "WHERE object_type = 'proposal'"
        ).fetchall()
        expired = 0
        for row in rows:
            proposal = ProfileProposal.model_validate_json(self._decrypt_row(row))
            if (
                proposal.state is ProposalState.PENDING
                and proposal.expires_at <= expire_before
            ):
                self._resolve_proposal_in_connection(
                    connection,
                    row,
                    ProposalState.EXPIRED,
                    version_id=_uuid("proposal-version"),
                )
                expired += 1
        return expired

    def get_proposal(self, proposal_id: str) -> ProfileProposal | None:
        """Return one authenticated current proposal head."""

        row = self._head_row("proposal", proposal_id)
        if row is None or self._is_quarantined(
            "proposal", proposal_id, row["version_id"]
        ):
            return None
        try:
            return ProfileProposal.model_validate_json(self._decrypt_row(row))
        except (ProfileIntegrityError, ValidationError):
            self.quarantine_object(
                "proposal", proposal_id, row["version_id"], "integrity_failure"
            )
            return None

    def list_proposals(self) -> list[ProfileProposal]:
        """Return all authenticated current proposal states."""

        proposals: list[ProfileProposal] = []
        for row in self._iter_head_rows("proposal"):
            if self._is_quarantined("proposal", row["object_id"], row["version_id"]):
                continue
            try:
                proposals.append(
                    ProfileProposal.model_validate_json(self._decrypt_row(row))
                )
            except (ProfileIntegrityError, ValidationError):
                self.quarantine_object(
                    "proposal", row["object_id"], row["version_id"], "integrity_failure"
                )
        return proposals

    def read_export_snapshot(
        self,
    ) -> tuple[
        ProfileManifest,
        tuple[ProfileScope, ...],
        tuple[ProfileRecord, ...],
        tuple[ProfileProposal, ...],
    ]:
        """Read all canonical export heads from one SQLite snapshot."""

        connection = self._connect()
        try:
            connection.execute("BEGIN")
            meta = connection.execute(
                "SELECT profile_id, current_manifest_version, destroyed "
                "FROM profile_meta WHERE singleton = 1"
            ).fetchone()
            if meta is None or meta["destroyed"]:
                raise ProfileDestroyedError(
                    "No active Personal Context profile exists."
                )
            manifest_row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = 'manifest' "
                "AND object_id = ? AND version_id = ?",
                (meta["profile_id"], meta["current_manifest_version"]),
            ).fetchone()
            if manifest_row is None:
                raise ProfileIntegrityError("Current manifest is unavailable.")
            manifest = ProfileManifest.model_validate_json(
                self._decrypt_row(manifest_row)
            )

            def current_rows(object_type: str) -> list[sqlite3.Row]:
                return connection.execute(
                    "SELECT encrypted_objects.* FROM object_heads "
                    "JOIN encrypted_objects "
                    "USING (object_type, object_id, version_id) "
                    "WHERE object_type = ? ORDER BY object_id",
                    (object_type,),
                ).fetchall()

            scopes = tuple(
                ProfileScope.model_validate_json(self._decrypt_row(row))
                for row in current_rows("scope")
            )
            records = tuple(
                ProfileRecord.model_validate_json(self._decrypt_row(row))
                for row in current_rows("record")
            )
            proposals = tuple(
                ProfileProposal.model_validate_json(self._decrypt_row(row))
                for row in current_rows("proposal")
            )
            connection.commit()
            return manifest, scopes, records, proposals
        except (ValidationError, sqlite3.Error) as exc:
            connection.rollback()
            raise ProfileIntegrityError(
                "Canonical export snapshot is invalid."
            ) from exc
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._checkpoint_after_snapshot()

    def resolve_proposal(
        self, proposal_id: str, state: ProposalState
    ) -> ProfileProposal:
        """Replace a pending proposal head with a content-free resolution receipt."""

        if state is ProposalState.PENDING:
            raise ValueError("A proposal resolution must be terminal.")
        version_id = _uuid("proposal-version")
        with self._mutation() as connection:
            row = connection.execute(
                """
                SELECT encrypted_objects.*
                FROM object_heads
                JOIN encrypted_objects USING (object_type, object_id, version_id)
                WHERE object_type = 'proposal' AND object_id = ?
                """,
                (proposal_id,),
            ).fetchone()
            if row is None:
                raise KeyError(proposal_id)
            resolved = self._resolve_proposal_in_connection(
                connection, row, state, version_id=version_id
            )
        return resolved

    def _resolve_proposal_in_connection(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        state: ProposalState,
        *,
        version_id: str,
        enqueue_outbox: bool = True,
    ) -> ProfileProposal:
        current = ProfileProposal.model_validate_json(self._decrypt_row(row))
        if current.state is not ProposalState.PENDING:
            raise ValueError("Only a pending proposal can be resolved.")
        resolved = ProfileProposal.model_validate(
            {
                **current.model_dump(mode="python"),
                "state": state,
                "proposed_record": None,
                "confidence": None,
            }
        )
        self._insert_encrypted(
            connection,
            object_type="proposal",
            object_id=current.proposal_id,
            version_id=version_id,
            scope_id=resolved.scope_id,
            value=resolved,
        )
        self._set_head(
            connection,
            object_type="proposal",
            object_id=current.proposal_id,
            version_id=version_id,
            expected_version_id=row["version_id"],
        )
        connection.execute(
            "DELETE FROM encrypted_objects WHERE object_type = 'proposal' "
            "AND object_id = ? AND version_id != ?",
            (current.proposal_id, version_id),
        )
        self._retire_pending_outbox_content(
            connection,
            object_type="proposal",
            object_id=current.proposal_id,
        )
        if enqueue_outbox:
            self._insert_outbox(
                connection,
                object_type="proposal",
                object_id=resolved.proposal_id,
                version_id=version_id,
                body={
                    "version": 1,
                    "proposal": resolved.model_dump(mode="json"),
                },
            )
        return resolved

    def accept_proposal_and_record(
        self,
        proposal_id: str,
        record: ProfileRecord,
        manifest: ProfileManifest,
        *,
        expected_record_version: str | None,
        expected_manifest_version: str,
        outbox_body: Mapping[str, Any] | None = None,
        expire_before: datetime | None = None,
        allow_user_review_rewrite: bool = False,
    ) -> ProfileProposal:
        """Atomically accept a proposal and commit its canonical record effects."""

        if record.profile_id != manifest.profile_id:
            raise ValueError("Record and manifest profile identities differ.")
        if record.parent_version_id != expected_record_version:
            raise ConcurrentProfileUpdateError(
                "Record parent does not match the expected head."
            )
        with self._mutation(profile_id=record.profile_id) as connection:
            if expire_before is not None:
                self._expire_due_proposals_in_connection(connection, expire_before)
            proposal_row = connection.execute(
                "SELECT encrypted_objects.* FROM object_heads "
                "JOIN encrypted_objects USING (object_type, object_id, version_id) "
                "WHERE object_type = 'proposal' AND object_id = ?",
                (proposal_id,),
            ).fetchone()
            if proposal_row is None:
                raise KeyError(proposal_id)
            proposal = ProfileProposal.model_validate_json(
                self._decrypt_row(proposal_row)
            )
            if proposal.state is ProposalState.EXPIRED:
                return proposal
            if proposal.state is not ProposalState.PENDING:
                raise ValueError("Only a pending proposal can be accepted.")
            self._require_no_record_collision_in_connection(
                connection,
                record,
                at=expire_before or datetime.now(UTC),
            )
            if proposal.operation.value in {"create", "update"}:
                assert proposal.proposed_record is not None
                proposed_with_approval = ProfileRecord.model_validate(
                    {
                        **proposal.proposed_record.model_dump(mode="python"),
                        "provenance": record.provenance,
                    }
                )
                provenance_invalid = (
                    record.provenance.source != proposal.provenance.source
                    or record.provenance.actor is not ActorType.USER
                    or record.provenance.reason_code != "user_approved_agent_proposal"
                    or record.provenance.source_references
                    != proposal.provenance.source_references
                    or record.provenance.source_hashes
                    != proposal.provenance.source_hashes
                    or record.provenance.derived_from_record_id
                    != proposal.provenance.derived_from_record_id
                )
                proposed_record = proposal.proposed_record
                assert record.payload is not None
                expected_semantic_key = SemanticKey(
                    namespace=record.payload.kind,
                    subject=getattr(record.payload, "subject", record.payload.kind),
                )
                expected_expires_at = (
                    record.updated_at + timedelta(days=30)
                    if record.kind.value == "working_context"
                    and not proposed_record.no_expiry
                    else proposed_record.expires_at
                )
                user_edit_invalid = allow_user_review_rewrite and (
                    record.profile_id != proposal.proposed_record.profile_id
                    or record.record_id != proposal.proposed_record.record_id
                    or record.scope_id != proposal.proposed_record.scope_id
                    or record.kind is not proposal.proposed_record.kind
                    or record.parent_version_id
                    != proposal.proposed_record.parent_version_id
                    or record.state is not proposal.proposed_record.state
                    or record.controls != proposal.proposed_record.controls
                    or record.created_at != proposal.proposed_record.created_at
                    or record.updated_at < proposal.proposed_record.updated_at
                    or record.expires_at != expected_expires_at
                    or record.no_expiry != proposal.proposed_record.no_expiry
                    or record.semantic_key != expected_semantic_key
                    or record.version_id == proposal.proposed_record.version_id
                )
                if (
                    provenance_invalid
                    or user_edit_invalid
                    or (
                        not allow_user_review_rewrite
                        and proposed_with_approval != record
                    )
                ):
                    raise ValueError("Accepted record differs from the proposal.")
            elif proposal.operation.value == "archive":
                if (
                    record.record_id != proposal.target_record_id
                    or record.parent_version_id != proposal.base_version_id
                    or record.state is not RecordState.ARCHIVED
                    or record.provenance.actor is not ActorType.USER
                ):
                    raise ValueError("Archive acceptance differs from the proposal.")
            else:
                if (
                    record.record_id == proposal.target_record_id
                    or record.provenance.derived_from_record_id
                    != proposal.target_record_id
                    or record.provenance.actor is not ActorType.USER
                ):
                    raise ValueError("Promotion acceptance differs from the proposal.")
                source_row = connection.execute(
                    "SELECT encrypted_objects.* FROM object_heads "
                    "JOIN encrypted_objects USING (object_type, object_id, version_id) "
                    "WHERE object_type = 'record' AND object_id = ?",
                    (proposal.target_record_id,),
                ).fetchone()
                if (
                    source_row is None
                    or source_row["version_id"] != proposal.base_version_id
                ):
                    raise ConcurrentProfileUpdateError(
                        "Promotion source changed concurrently."
                    )
                source = ProfileRecord.model_validate_json(
                    self._decrypt_row(source_row)
                )
                if (
                    source.scope_id != proposal.scope_id
                    or record.kind is not source.kind
                    or record.payload != source.payload
                    or record.semantic_key != source.semantic_key
                    or record.state is not source.state
                    or record.controls != source.controls
                    or record.expires_at != source.expires_at
                    or record.no_expiry != source.no_expiry
                ):
                    raise ValueError("Promotion acceptance differs from the source.")

            meta = connection.execute(
                "SELECT current_manifest_version FROM profile_meta WHERE singleton = 1"
            ).fetchone()
            if (
                meta is None
                or meta["current_manifest_version"] != expected_manifest_version
            ):
                raise ConcurrentProfileUpdateError(
                    "Profile manifest changed concurrently."
                )
            current_manifest_row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = 'manifest' "
                "AND object_id = ? AND version_id = ?",
                (manifest.profile_id, expected_manifest_version),
            ).fetchone()
            if current_manifest_row is None:
                raise ProfileIntegrityError("Current manifest is unavailable.")
            current_manifest = ProfileManifest.model_validate_json(
                self._decrypt_row(current_manifest_row)
            )
            if (
                manifest.revision != current_manifest.revision + 1
                or manifest.purge_generation != current_manifest.purge_generation
                or manifest.created_at != current_manifest.created_at
            ):
                raise ValueError("Next manifest is not a valid revision successor.")

            self._insert_encrypted(
                connection,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                scope_id=record.scope_id,
                value=record,
            )
            self._set_head(
                connection,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                expected_version_id=expected_record_version,
            )
            self._insert_encrypted(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                value=manifest,
            )
            self._set_head(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                expected_version_id=expected_manifest_version,
            )
            connection.execute(
                "UPDATE profile_meta SET current_manifest_version = ?, "
                "purge_generation = ? WHERE singleton = 1",
                (manifest.current_version_id, manifest.purge_generation),
            )
            self._insert_outbox(
                connection,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                body={"version": 1, "manifest": manifest.model_dump(mode="json")},
            )
            if (
                outbox_body is not None
                and record.controls.sync_mode is SyncMode.SYNCABLE
            ):
                self._insert_outbox(
                    connection,
                    object_type="record",
                    object_id=record.record_id,
                    version_id=record.version_id,
                    body=outbox_body,
                )
            return self._resolve_proposal_in_connection(
                connection,
                proposal_row,
                ProposalState.ACCEPTED,
                version_id=_uuid("proposal-version"),
            )

    def _require_no_record_collision_in_connection(
        self,
        connection: sqlite3.Connection,
        record: ProfileRecord,
        *,
        at: datetime,
    ) -> None:
        """Reject active semantic-key collisions within the caller's transaction."""

        if (
            record.state is not RecordState.ACTIVE
            or record.semantic_key is None
            or (record.expires_at is not None and record.expires_at <= at)
        ):
            return
        rows = connection.execute(
            "SELECT encrypted_objects.* FROM object_heads "
            "JOIN encrypted_objects USING (object_type, object_id, version_id) "
            "WHERE object_type = 'record'"
        ).fetchall()
        for row in rows:
            existing = ProfileRecord.model_validate_json(self._decrypt_row(row))
            if existing.record_id == record.record_id:
                continue
            if (
                existing.scope_id == record.scope_id
                and existing.kind is record.kind
                and existing.semantic_key == record.semantic_key
                and existing.state is RecordState.ACTIVE
                and (existing.expires_at is None or existing.expires_at > at)
            ):
                raise RecordCollisionError(existing.record_id)

    def commit_runtime_policy(
        self,
        scope_id: str,
        body: Mapping[str, Any],
        *,
        expected_version_id: str | None = None,
    ) -> str:
        """Encrypt and compare-and-set one peer-local runtime policy."""

        return self._commit_local_body(
            table="local_runtime_policy",
            version_column="encrypted_policy_version",
            object_type="runtime_policy",
            scope_id=scope_id,
            body=body,
            expected_version_id=expected_version_id,
        )

    def get_runtime_policy(self, scope_id: str) -> dict[str, Any] | None:
        return self._get_local_body(
            "local_runtime_policy",
            "encrypted_policy_version",
            "runtime_policy",
            scope_id,
        )

    def get_runtime_policy_version(self, scope_id: str) -> str | None:
        return self._get_local_version(
            "local_runtime_policy", "encrypted_policy_version", scope_id
        )

    def commit_scope_binding(
        self,
        scope_id: str,
        body: Mapping[str, Any],
        *,
        expected_version_id: str | None = None,
        require_unique_local_workspace_id: bool = False,
    ) -> str:
        """Encrypt and compare-and-set one peer-local workspace binding."""

        return self._commit_local_body(
            table="local_scope_bindings",
            version_column="encrypted_binding_version",
            object_type="scope_binding",
            scope_id=scope_id,
            body=body,
            expected_version_id=expected_version_id,
            require_unique_local_workspace_id=require_unique_local_workspace_id,
        )

    def get_scope_binding(self, scope_id: str) -> dict[str, Any] | None:
        return self._get_local_body(
            "local_scope_bindings",
            "encrypted_binding_version",
            "scope_binding",
            scope_id,
        )

    def get_scope_binding_version(self, scope_id: str) -> str | None:
        return self._get_local_version(
            "local_scope_bindings", "encrypted_binding_version", scope_id
        )

    def list_scope_bindings(self) -> dict[str, dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT scope_id FROM local_scope_bindings ORDER BY scope_id"
            ).fetchall()
        return {
            row["scope_id"]: body
            for row in rows
            if (body := self.get_scope_binding(row["scope_id"])) is not None
        }

    def get_validated_scope_binding(self, scope_id: str) -> dict[str, Any] | None:
        """Return one exact v1 workspace mapping or quarantine it as unlinked."""

        version = self.get_scope_binding_version(scope_id)
        if version is None:
            return None
        try:
            body = self.get_scope_binding(scope_id)
            if body is None:
                return None
            local_workspace_id, label = self._validated_workspace_binding(body)
        except (ProfileIntegrityError, TypeError, ValueError):
            self.quarantine_object(
                "scope_binding",
                scope_id,
                version,
                "invalid_workspace_binding",
            )
            return None
        return {
            "version": 1,
            "local_workspace_id": local_workspace_id,
            "label": label,
        }

    def list_validated_scope_bindings(self) -> dict[str, dict[str, Any]]:
        """Return only authenticated exact-v1 workspace mappings."""

        with closing(self._connect()) as connection:
            scope_ids = [
                row["scope_id"]
                for row in connection.execute(
                    "SELECT scope_id FROM local_scope_bindings ORDER BY scope_id"
                )
            ]
        return {
            scope_id: body
            for scope_id in scope_ids
            if (body := self.get_validated_scope_binding(scope_id)) is not None
        }

    def _get_local_version(
        self, table: str, version_column: str, scope_id: str
    ) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"SELECT {version_column} FROM {table} WHERE scope_id = ?",
                (scope_id,),
            ).fetchone()
        return None if row is None else row[0]

    def _commit_local_body(
        self,
        *,
        table: str,
        version_column: str,
        object_type: str,
        scope_id: str,
        body: Mapping[str, Any],
        expected_version_id: str | None,
        require_unique_local_workspace_id: bool = False,
    ) -> str:
        version_id = _uuid(f"{object_type}-version")
        with self._mutation() as connection:
            if require_unique_local_workspace_id:
                self._require_unique_workspace_binding(
                    connection,
                    body,
                    excluding_scope_id=scope_id,
                )
            current = connection.execute(
                f"SELECT {version_column} FROM {table} WHERE scope_id = ?",
                (scope_id,),
            ).fetchone()
            current_version = None if current is None else current[0]
            if current_version != expected_version_id:
                raise ConcurrentProfileUpdateError(
                    "Local metadata head changed concurrently."
                )
            self._insert_encrypted(
                connection,
                object_type=object_type,
                object_id=scope_id,
                version_id=version_id,
                scope_id=scope_id,
                value=body,
            )
            self._set_head(
                connection,
                object_type=object_type,
                object_id=scope_id,
                version_id=version_id,
                expected_version_id=expected_version_id,
            )
            connection.execute(
                f"INSERT INTO {table}(scope_id, {version_column}) VALUES (?, ?) "
                f"ON CONFLICT(scope_id) DO UPDATE SET {version_column} = excluded.{version_column}",
                (scope_id, version_id),
            )
        return version_id

    @staticmethod
    def _validated_workspace_binding(body: Any) -> tuple[str, str]:
        if (
            not isinstance(body, dict)
            or set(body) != {"version", "local_workspace_id", "label"}
            or type(body.get("version")) is not int
            or body["version"] != 1
            or not isinstance(body.get("local_workspace_id"), str)
            or not body["local_workspace_id"].strip()
            or len(body["local_workspace_id"]) > 16_384
            or not isinstance(body.get("label"), str)
            or len(body["label"]) > 16_384
        ):
            raise ProfileIntegrityError("Encrypted workspace binding is invalid.")
        return body["local_workspace_id"], body["label"]

    def _require_unique_workspace_binding(
        self,
        connection: sqlite3.Connection,
        candidate: Mapping[str, Any],
        *,
        excluding_scope_id: str | None = None,
    ) -> None:
        candidate_id, _ = self._validated_workspace_binding(dict(candidate))
        metadata = connection.execute(
            "SELECT scope_id, encrypted_binding_version "
            "FROM local_scope_bindings ORDER BY scope_id"
        ).fetchall()
        for item in metadata:
            if item["scope_id"] == excluding_scope_id:
                continue
            row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = 'scope_binding' "
                "AND object_id = ? AND version_id = ?",
                (item["scope_id"], item["encrypted_binding_version"]),
            ).fetchone()
            if row is None:
                raise ProfileIntegrityError(
                    "Encrypted workspace binding is unavailable."
                )
            try:
                body = json.loads(self._decrypt_row(row))
            except (TypeError, ValueError) as exc:
                raise ProfileIntegrityError(
                    "Encrypted workspace binding is invalid."
                ) from exc
            existing_id, _ = self._validated_workspace_binding(body)
            if existing_id == candidate_id:
                raise ValueError("Local workspace is already mapped.")

    def _get_local_body(
        self,
        table: str,
        version_column: str,
        object_type: str,
        scope_id: str,
    ) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            version = connection.execute(
                f"SELECT {version_column} FROM {table} WHERE scope_id = ?", (scope_id,)
            ).fetchone()
            if version is None:
                return None
            row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = ? "
                "AND object_id = ? AND version_id = ?",
                (object_type, scope_id, version[0]),
            ).fetchone()
        if row is None:
            raise ProfileIntegrityError("Encrypted local metadata is unavailable.")
        return json.loads(self._decrypt_row(row))

    @staticmethod
    def _delete_undo(connection: sqlite3.Connection, undo_id: str) -> None:
        row = connection.execute(
            "SELECT encrypted_undo_version FROM local_undo WHERE undo_id = ?",
            (undo_id,),
        ).fetchone()
        if row is None:
            raise ConcurrentProfileUpdateError("Undo artifact is unavailable.")
        connection.execute("DELETE FROM local_undo WHERE undo_id = ?", (undo_id,))
        connection.execute(
            "DELETE FROM encrypted_objects WHERE object_type = 'undo' "
            "AND object_id = ? AND version_id = ?",
            (undo_id, row["encrypted_undo_version"]),
        )

    def list_undo_ids(self, *, now: str) -> list[str]:
        """Return unexpired encrypted Undo identifiers and purge expired rows."""

        with self._mutation() as connection:
            expired = connection.execute(
                "SELECT undo_id FROM local_undo WHERE expires_at <= ?", (now,)
            ).fetchall()
            for row in expired:
                self._delete_undo(connection, row["undo_id"])
            rows = connection.execute(
                "SELECT undo_id FROM local_undo ORDER BY rowid DESC"
            ).fetchall()
        return [row["undo_id"] for row in rows]

    def get_undo(self, undo_id: str, *, now: str) -> dict[str, Any] | None:
        with self._mutation() as connection:
            metadata = connection.execute(
                "SELECT * FROM local_undo WHERE undo_id = ?", (undo_id,)
            ).fetchone()
            if metadata is None:
                return None
            if metadata["expires_at"] <= now:
                self._delete_undo(connection, undo_id)
                return None
            row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = 'undo' "
                "AND object_id = ? AND version_id = ?",
                (undo_id, metadata["encrypted_undo_version"]),
            ).fetchone()
            if row is None:
                raise ProfileIntegrityError("Encrypted Undo artifact is unavailable.")
            return json.loads(self._decrypt_row(row))

    def _insert_outbox(
        self,
        connection: sqlite3.Connection,
        *,
        object_type: str,
        object_id: str,
        version_id: str,
        body: Mapping[str, Any],
    ) -> str:
        outbox_id = _uuid("outbox")
        envelope_version = _uuid("outbox-envelope")
        self._insert_encrypted(
            connection,
            object_type="outbox",
            object_id=outbox_id,
            version_id=envelope_version,
            value=body,
        )
        self._set_head(
            connection,
            object_type="outbox",
            object_id=outbox_id,
            version_id=envelope_version,
            expected_version_id=None,
        )
        now = _now_text()
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence "
            "FROM encrypted_outbox"
        ).fetchone()
        sequence = int(row["next_sequence"])
        connection.execute(
            """
            INSERT INTO encrypted_outbox (
                outbox_id, sequence, object_type, object_id, version_id,
                envelope_version, status, created_at,
                destination_envelope_id, quarantine_reason, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, NULL, NULL, ?)
            """,
            (
                outbox_id,
                sequence,
                object_type,
                object_id,
                version_id,
                envelope_version,
                now,
                now,
            ),
        )
        return outbox_id

    def commit_outbox_body(
        self,
        *,
        object_type: str,
        object_id: str,
        version_id: str,
        body: Mapping[str, Any],
    ) -> str:
        """Commit an independently encrypted exact-wire outbox body."""

        with self._mutation() as connection:
            return self._insert_outbox(
                connection,
                object_type=object_type,
                object_id=object_id,
                version_id=version_id,
                body=body,
            )

    def get_outbox_body(self, outbox_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            outbox = connection.execute(
                "SELECT envelope_version FROM encrypted_outbox WHERE outbox_id = ?",
                (outbox_id,),
            ).fetchone()
            if outbox is None:
                return None
            row = connection.execute(
                "SELECT * FROM encrypted_objects WHERE object_type = 'outbox' "
                "AND object_id = ? AND version_id = ?",
                (outbox_id, outbox["envelope_version"]),
            ).fetchone()
        if row is None:
            return None
        return json.loads(self._decrypt_row(row))

    def list_pending_outbox(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        """Return bounded content-free metadata for pending encrypted entries."""

        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("Outbox limit must be between 1 and 500.")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT outbox_id, object_type, object_id, version_id, status, "
                "created_at FROM encrypted_outbox WHERE status = 'pending' "
                "ORDER BY sequence LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def acknowledge_outbox(self, outbox_id: str, destination_envelope_id: str) -> None:
        """Record an idempotent destination receipt and shred the source body."""

        if not isinstance(destination_envelope_id, str) or not destination_envelope_id:
            raise ValueError("Destination envelope id is required.")
        with self._mutation() as connection:
            row = connection.execute(
                "SELECT envelope_version, destination_envelope_id FROM encrypted_outbox "
                "WHERE outbox_id = ?",
                (outbox_id,),
            ).fetchone()
            if row is None:
                raise KeyError(outbox_id)
            existing = row["destination_envelope_id"]
            if existing is not None and existing != destination_envelope_id:
                raise ConcurrentProfileUpdateError("Outbox receipt changed concurrently.")
            self._shred_outbox_body(connection, outbox_id, row["envelope_version"])
            connection.execute(
                "UPDATE encrypted_outbox SET status = 'dispatched', "
                "destination_envelope_id = ?, quarantine_reason = NULL, updated_at = ? "
                "WHERE outbox_id = ?",
                (destination_envelope_id, _now_text(), outbox_id),
            )

    def quarantine_outbox(
        self,
        outbox_id: str,
        reason_code: str,
        *,
        preserve_body: bool = False,
    ) -> None:
        """Quarantine an entry, optionally retaining its authenticated source body."""

        if (
            not isinstance(reason_code, str)
            or not reason_code
            or len(reason_code) > 128
        ):
            raise ValueError("Outbox quarantine reason is invalid.")
        with self._mutation() as connection:
            row = connection.execute(
                "SELECT envelope_version FROM encrypted_outbox WHERE outbox_id = ?",
                (outbox_id,),
            ).fetchone()
            if row is None:
                raise KeyError(outbox_id)
            if not preserve_body:
                self._shred_outbox_body(connection, outbox_id, row["envelope_version"])
            connection.execute(
                "UPDATE encrypted_outbox SET status = 'quarantined', "
                "quarantine_reason = ?, updated_at = ? WHERE outbox_id = ?",
                (reason_code, _now_text(), outbox_id),
            )

    def get_outbox_receipt(self, outbox_id: str) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT destination_envelope_id FROM encrypted_outbox WHERE outbox_id = ?",
                (outbox_id,),
            ).fetchone()
        return None if row is None else row["destination_envelope_id"]

    def get_outbox_quarantine_reason(self, outbox_id: str) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT quarantine_reason FROM encrypted_outbox WHERE outbox_id = ?",
                (outbox_id,),
            ).fetchone()
        return None if row is None else row["quarantine_reason"]

    @staticmethod
    def _shred_outbox_body(
        connection: sqlite3.Connection,
        outbox_id: str,
        envelope_version: str,
    ) -> None:
        connection.execute(
            "DELETE FROM object_heads WHERE object_type = 'outbox' "
            "AND object_id = ? AND version_id = ?",
            (outbox_id, envelope_version),
        )
        connection.execute(
            "DELETE FROM encrypted_objects WHERE object_type = 'outbox' "
            "AND object_id = ? AND version_id = ?",
            (outbox_id, envelope_version),
        )

    def quarantine_object(
        self,
        object_type: str,
        object_id: str,
        version_id: str | None,
        reason_code: str,
    ) -> None:
        """Record a content-free quarantine receipt once per object version."""

        with self._mutation() as connection:
            exists = connection.execute(
                "SELECT 1 FROM quarantine WHERE object_type = ? AND object_id = ? "
                "AND version_id IS ?",
                (object_type, object_id, version_id),
            ).fetchone()
            if exists is None:
                connection.execute(
                    "INSERT INTO quarantine VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        _uuid("quarantine"),
                        object_type,
                        object_id,
                        version_id,
                        reason_code,
                        _now_text(),
                    ),
                )

    def _is_quarantined(
        self, object_type: str, object_id: str, version_id: str | None
    ) -> bool:
        with closing(self._connect()) as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM quarantine WHERE object_type = ? AND object_id = ? "
                    "AND version_id IS ?",
                    (object_type, object_id, version_id),
                ).fetchone()
                is not None
            )

    def list_quarantine(self) -> list[QuarantineEntry]:
        """Return content-free quarantine metadata."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM quarantine ORDER BY created_at, quarantine_id"
            ).fetchall()
        return [QuarantineEntry(**dict(row)) for row in rows]

    def destroy_profile_content(self) -> None:
        """Durably fence writes, purge content, then delete protected keys."""

        with self._transaction() as connection:
            meta = connection.execute(
                "SELECT destroyed FROM profile_meta WHERE singleton = 1"
            ).fetchone()
            if meta is None:
                raise ProfileDestroyedError(
                    "No active Personal Context profile exists."
                )
            if not meta["destroyed"]:
                self._require_keys()
                for table in (
                    "encrypted_outbox",
                    "local_runtime_policy",
                    "local_scope_bindings",
                    "local_undo",
                    "local_record_links",
                    "object_heads",
                    "encrypted_objects",
                    "quarantine",
                ):
                    connection.execute(f"DELETE FROM {table}")
                connection.execute(
                    """
                    UPDATE profile_meta
                    SET purge_generation = purge_generation + 1, destroyed = 1
                    WHERE singleton = 1
                    """
                )
        try:
            self._protector.delete(self._profile_ref)
        finally:
            self._keys = None
