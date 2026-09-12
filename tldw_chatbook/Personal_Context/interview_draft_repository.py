"""Encrypted, local-only persistence for expiring interview drafts."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping

from cryptography.exceptions import InvalidTag

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

from .crypto import ALGORITHM, NONCE_BYTES, EncryptedEnvelope, EnvelopeCipher
from .key_protector import ProfileKeyProtector


_MAX_RETENTION = timedelta(days=30)
_WRAP_NONCE_BYTES = 12


class InterviewDraftExpiredError(RuntimeError):
    """Report an expired draft after its protector key is destroyed."""


class InterviewDraftConflictError(RuntimeError):
    """Report that another interview flow changed the draft first."""


@dataclass(frozen=True, slots=True)
class StoredInterviewDraft:
    session_id: str
    expires_at: datetime
    revision: int
    payload: Mapping[str, Any] = field(repr=False)


def _clock() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("draft expiry must be timezone-aware")
    return value.astimezone(UTC)


def _session_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or "\x00" in value
    ):
        raise ValueError("session ID must be bounded nonblank text")
    return value


class InterviewDraftRepository:
    """Store draft ciphertext without any Sync or ordinary-backup surface."""

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None,
        *,
        key_protector: ProfileKeyProtector | None = None,
        clock: Callable[[], datetime] = _clock,
    ) -> None:
        self._path = None if db_path is None else Path(db_path)
        self._protector = key_protector
        self._clock = clock
        self._memory: dict[str, StoredInterviewDraft] = {}
        self._memory_lock = RLock()
        if self._path is not None:
            if key_protector is None:
                raise ValueError("protected drafts require a key protector")
            with closing(self._connect()) as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS interview_drafts (
                        session_id TEXT PRIMARY KEY,
                        expires_at TEXT NOT NULL,
                        algorithm TEXT NOT NULL,
                        nonce BLOB NOT NULL,
                        ciphertext BLOB NOT NULL,
                        wrapped_dek BLOB NOT NULL,
                        key_version INTEGER NOT NULL,
                        revision INTEGER NOT NULL DEFAULT 1,
                        cleanup_pending INTEGER NOT NULL DEFAULT 0
                            CHECK (cleanup_pending IN (0, 1))
                    )
                    """
                )
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(interview_drafts)"
                    ).fetchall()
                }
                if "revision" not in columns:
                    connection.execute(
                        "ALTER TABLE interview_drafts "
                        "ADD COLUMN revision INTEGER NOT NULL DEFAULT 1"
                    )
                if "cleanup_pending" not in columns:
                    connection.execute(
                        "ALTER TABLE interview_drafts "
                        "ADD COLUMN cleanup_pending INTEGER NOT NULL DEFAULT 0"
                    )

    @classmethod
    def memory_only(
        cls, *, clock: Callable[[], datetime] = _clock
    ) -> "InterviewDraftRepository":
        return cls(None, clock=clock)

    @property
    def is_memory_only(self) -> bool:
        return self._path is None

    def _connect(self) -> sqlite3.Connection:
        assert self._path is not None
        connection = connect_private_sqlite(
            "personal_context.interview_drafts",
            self._path,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _key_ref(self, session_id: str) -> str:
        session_id = _session_id(session_id)
        location = "memory" if self._path is None else str(self._path.absolute())
        digest = hashlib.sha256(
            os.fsencode(location) + b"\x00" + session_id.encode("utf-8")
        ).hexdigest()
        return f"personal-context-interview:{digest}"

    @staticmethod
    def _aad(session_id: str, expires_at: str, revision: int) -> bytes:
        return (
            b"tldw-chatbook:interview-draft:v2\x00"
            + session_id.encode("utf-8")
            + b"\x00"
            + expires_at.encode("ascii")
            + b"\x00"
            + str(revision).encode("ascii")
        )

    def save(
        self,
        session_id: str,
        payload: Mapping[str, Any],
        *,
        expires_at: datetime,
        expected_revision: int | None = None,
    ) -> StoredInterviewDraft:
        session_id = _session_id(session_id)
        now = _utc(self._clock())
        expires_at = _utc(expires_at)
        if expires_at <= now:
            raise ValueError("draft expiry must be an aware future timestamp")
        if expires_at - now > _MAX_RETENTION:
            raise ValueError("interview drafts may be retained for at most 30 days")
        try:
            safe_payload = json.loads(
                json.dumps(
                    dict(payload),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError):
            raise ValueError("interview draft payload must be a JSON object") from None
        if not isinstance(safe_payload, dict):
            raise ValueError("interview draft payload must be a JSON object")
        expires_text = expires_at.isoformat()
        if self._path is None:
            with self._memory_lock:
                current = self._memory.get(session_id)
                if current is None:
                    if expected_revision is not None:
                        raise InterviewDraftConflictError(
                            "Interview draft changed concurrently."
                        )
                    revision = 1
                else:
                    if expected_revision != current.revision:
                        raise InterviewDraftConflictError(
                            "Interview draft changed concurrently."
                        )
                    revision = current.revision + 1
                stored_payload = deepcopy(safe_payload)
                self._memory[session_id] = StoredInterviewDraft(
                    session_id, expires_at, revision, stored_payload
                )
                return StoredInterviewDraft(
                    session_id, expires_at, revision, deepcopy(stored_payload)
                )

        assert self._protector is not None
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT revision, cleanup_pending FROM interview_drafts "
                    "WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    if expected_revision is not None:
                        raise InterviewDraftConflictError(
                            "Interview draft changed concurrently."
                        )
                    revision = 1
                else:
                    if row["cleanup_pending"] != 0:
                        raise InterviewDraftConflictError(
                            "Interview draft cleanup is already pending."
                        )
                    if expected_revision != row["revision"]:
                        raise InterviewDraftConflictError(
                            "Interview draft changed concurrently."
                        )
                    revision = row["revision"] + 1
                plaintext = json.dumps(
                    {
                        "version": 1,
                        "expires_at": expires_text,
                        "revision": revision,
                        "payload": safe_payload,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                keys = self._protector.load_or_create(self._key_ref(session_id))
                envelope = EnvelopeCipher(
                    keys.encryption_key,
                    key_version=keys.key_version,
                ).encrypt(plaintext, self._aad(session_id, expires_text, revision))
                values = (
                    expires_text,
                    envelope.algorithm,
                    envelope.nonce,
                    envelope.ciphertext,
                    envelope.wrap_nonce + envelope.wrapped_dek,
                    envelope.key_version,
                    revision,
                )
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO interview_drafts(
                            session_id, expires_at, algorithm, nonce, ciphertext,
                            wrapped_dek, key_version, revision
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (session_id, *values),
                    )
                else:
                    updated = connection.execute(
                        """
                        UPDATE interview_drafts SET
                            expires_at = ?, algorithm = ?, nonce = ?,
                            ciphertext = ?, wrapped_dek = ?, key_version = ?,
                            revision = ?
                        WHERE session_id = ? AND revision = ?
                            AND cleanup_pending = 0
                        """,
                        (*values, session_id, expected_revision),
                    )
                    if updated.rowcount != 1:
                        raise InterviewDraftConflictError(
                            "Interview draft changed concurrently."
                        )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return StoredInterviewDraft(session_id, expires_at, revision, safe_payload)

    def load(self, session_id: str) -> StoredInterviewDraft | None:
        session_id = _session_id(session_id)
        if self._path is None:
            with self._memory_lock:
                draft = self._memory.get(session_id)
                if draft is not None and draft.expires_at <= _utc(self._clock()):
                    self._memory.pop(session_id, None)
                    return None
                return (
                    None
                    if draft is None
                    else StoredInterviewDraft(
                        draft.session_id,
                        draft.expires_at,
                        draft.revision,
                        deepcopy(draft.payload),
                    )
                )
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM interview_drafts WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        if row["cleanup_pending"] != 0:
            return None
        expires_text = row["expires_at"]
        revision = row["revision"]
        try:
            expires_at = _utc(datetime.fromisoformat(expires_text))
            if expires_text != expires_at.isoformat():
                raise ValueError("non-canonical expiry")
            if type(revision) is not int or revision < 1:
                raise ValueError("invalid revision")
        except (TypeError, ValueError):
            self._invalidate(session_id, "Interview draft expiry metadata is invalid.")
        assert self._protector is not None
        keys = self._protector.load(self._key_ref(session_id))
        if (
            row["algorithm"] != ALGORITHM
            or not isinstance(row["nonce"], (bytes, bytearray, memoryview))
            or len(row["nonce"]) != NONCE_BYTES
            or not isinstance(row["ciphertext"], (bytes, bytearray, memoryview))
            or not isinstance(row["wrapped_dek"], (bytes, bytearray, memoryview))
            or type(row["key_version"]) is not int
            or row["key_version"] < 1
        ):
            self._invalidate(session_id, "Interview draft envelope is invalid.")
        wrapped = bytes(row["wrapped_dek"])
        if len(wrapped) <= _WRAP_NONCE_BYTES:
            self._invalidate(session_id, "Interview draft envelope is invalid.")
        envelope = EncryptedEnvelope(
            algorithm=row["algorithm"],
            nonce=bytes(row["nonce"]),
            wrap_nonce=wrapped[:_WRAP_NONCE_BYTES],
            ciphertext=bytes(row["ciphertext"]),
            wrapped_dek=wrapped[_WRAP_NONCE_BYTES:],
            key_version=row["key_version"],
        )
        try:
            body = json.loads(
                EnvelopeCipher(
                    keys.encryption_key,
                    key_version=keys.key_version,
                )
                .decrypt(envelope, self._aad(session_id, expires_text, revision))
                .decode("utf-8")
            )
        except (InvalidTag, TypeError, ValueError, UnicodeDecodeError):
            self._invalidate(
                session_id, "Interview draft expiry authentication failed."
            )
        if (
            not isinstance(body, dict)
            or set(body) != {"version", "expires_at", "revision", "payload"}
            or body["version"] != 1
            or body["expires_at"] != expires_text
            or body["revision"] != revision
            or not isinstance(body["payload"], dict)
        ):
            self._invalidate(
                session_id, "Interview draft expiry authentication failed."
            )
        if expires_at <= _utc(self._clock()):
            self.delete(session_id)
            return None
        return StoredInterviewDraft(session_id, expires_at, revision, body["payload"])

    def _invalidate(self, session_id: str, message: str) -> None:
        try:
            self.delete(session_id)
        except Exception as exc:
            raise ValueError(message) from exc
        raise ValueError(message)

    def require_active(self, session_id: str) -> StoredInterviewDraft:
        session_id = _session_id(session_id)
        existed = False
        if self._path is not None:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT 1 FROM interview_drafts WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            existed = row is not None
        else:
            existed = session_id in self._memory
        draft = self.load(session_id)
        if draft is None:
            if existed:
                raise InterviewDraftExpiredError("interview draft expired")
            raise KeyError(session_id)
        return draft

    def delete(self, session_id: str) -> None:
        session_id = _session_id(session_id)
        if self._path is None:
            with self._memory_lock:
                self._memory.pop(session_id, None)
            return
        self._mark_cleanup_pending(session_id)
        assert self._protector is not None
        self._protector.delete(self._key_ref(session_id))
        self._delete_row(session_id)

    def _mark_cleanup_pending(self, session_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "UPDATE interview_drafts SET cleanup_pending = 1 "
                    "WHERE session_id = ?",
                    (session_id,),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _delete_row(self, session_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "DELETE FROM interview_drafts WHERE session_id = ?",
                    (session_id,),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _row_exists(self, session_id: str) -> bool:
        with closing(self._connect()) as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM interview_drafts WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                is not None
            )

    def expire(self) -> tuple[str, ...]:
        now = _utc(self._clock())
        if self._path is None:
            with self._memory_lock:
                expired = tuple(
                    session_id
                    for session_id, draft in self._memory.items()
                    if draft.expires_at <= now
                )
            for session_id in expired:
                self.delete(session_id)
            return expired
        else:
            with closing(self._connect()) as connection:
                rows = tuple(
                    (row["session_id"], row["cleanup_pending"])
                    for row in connection.execute(
                        "SELECT session_id, cleanup_pending "
                        "FROM interview_drafts ORDER BY rowid"
                    )
                )
        expired: list[str] = []
        for session_id, cleanup_pending in rows:
            try:
                if cleanup_pending != 0:
                    self.delete(session_id)
                else:
                    self.load(session_id)
            except Exception:
                pass
            if not self._row_exists(session_id):
                expired.append(session_id)
        return tuple(expired)
