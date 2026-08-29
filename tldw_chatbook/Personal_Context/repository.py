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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from pydantic import BaseModel, ValidationError
from tldw_profile_core import (
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProposalState,
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


SCHEMA_VERSION = 1
_WRAP_NONCE_BYTES = 12
_PEER_ENVELOPE = "chatbook-local-v1"


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
    """
    CREATE TABLE encrypted_outbox (
        outbox_id TEXT PRIMARY KEY,
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        envelope_version TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
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
            if self._inspect_schema(connection):
                keys = self._protector.load_or_create(self._profile_ref)
                self._initialize_schema(connection)
            else:
                keys = self._protector.load(self._profile_ref)
        self._keys = keys

    def _connect(self) -> sqlite3.Connection:
        """Open a new owner-checked autocommit connection for this operation."""

        connection = connect_private_sqlite(
            "personal_context.repository",
            self.db_path,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def close(self) -> None:
        """Close the repository; operations own no persistent connection."""

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
        if version != SCHEMA_VERSION:
            raise RepositorySchemaError(
                f"Unsupported Personal Context schema version: {version!r}."
            )
        expected = {
            "personal_context_schema",
            "profile_meta",
            "encrypted_objects",
            "object_heads",
            "local_runtime_policy",
            "local_scope_bindings",
            "encrypted_outbox",
            "quarantine",
        }
        if not expected.issubset(tables):
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
        except BaseException:
            try:
                connection.rollback()
            except BaseException:
                pass
            raise
        finally:
            connection.close()

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
                "schema_version": SCHEMA_VERSION,
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
        return manifest

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

    def commit_record_version(
        self,
        record: ProfileRecord,
        *,
        expected_version_id: str | None,
        outbox_body: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically insert an immutable record, CAS its head, and queue sync."""

        with self._mutation(profile_id=record.profile_id) as connection:
            if record.parent_version_id != expected_version_id:
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

    def list_records(self) -> list[ProfileRecord]:
        """Return authenticated current records, omitting quarantined objects."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT encrypted_objects.*
                FROM object_heads
                JOIN encrypted_objects USING (object_type, object_id, version_id)
                WHERE object_type = 'record'
                ORDER BY object_id
                """
            ).fetchall()
        records: list[ProfileRecord] = []
        for row in rows:
            record = self.get_record(row["object_id"])
            if record is not None:
                records.append(record)
        return records

    def commit_proposal(self, proposal: ProfileProposal) -> None:
        """Commit a new immutable proposal head."""

        version_id = _uuid("proposal-version")
        with self._mutation(profile_id=proposal.profile_id) as connection:
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

    def list_proposals(self) -> list[ProfileProposal]:
        """Return all authenticated current proposal states."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT encrypted_objects.* FROM object_heads
                JOIN encrypted_objects USING (object_type, object_id, version_id)
                WHERE object_type = 'proposal' ORDER BY object_id
                """
            ).fetchall()
        proposals: list[ProfileProposal] = []
        for row in rows:
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
                object_id=proposal_id,
                version_id=version_id,
                scope_id=resolved.scope_id,
                value=resolved,
            )
            self._set_head(
                connection,
                object_type="proposal",
                object_id=proposal_id,
                version_id=version_id,
                expected_version_id=row["version_id"],
            )
        return resolved

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

    def commit_scope_binding(
        self,
        scope_id: str,
        body: Mapping[str, Any],
        *,
        expected_version_id: str | None = None,
    ) -> str:
        """Encrypt and compare-and-set one peer-local workspace binding."""

        return self._commit_local_body(
            table="local_scope_bindings",
            version_column="encrypted_binding_version",
            object_type="scope_binding",
            scope_id=scope_id,
            body=body,
            expected_version_id=expected_version_id,
        )

    def get_scope_binding(self, scope_id: str) -> dict[str, Any] | None:
        return self._get_local_body(
            "local_scope_bindings",
            "encrypted_binding_version",
            "scope_binding",
            scope_id,
        )

    def _commit_local_body(
        self,
        *,
        table: str,
        version_column: str,
        object_type: str,
        scope_id: str,
        body: Mapping[str, Any],
        expected_version_id: str | None,
    ) -> str:
        version_id = _uuid(f"{object_type}-version")
        with self._mutation() as connection:
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
        connection.execute(
            "INSERT INTO encrypted_outbox VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (
                outbox_id,
                object_type,
                object_id,
                version_id,
                envelope_version,
                _now_text(),
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
            raise ProfileIntegrityError("Encrypted outbox body is unavailable.")
        return json.loads(self._decrypt_row(row))

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
