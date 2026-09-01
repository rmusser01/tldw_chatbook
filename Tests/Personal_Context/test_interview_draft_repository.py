from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest

from tldw_chatbook.Personal_Context.interview_draft_repository import (
    InterviewDraftConflictError,
    InterviewDraftExpiredError,
    InterviewDraftRepository,
)
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.DB import private_sqlite


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


def test_draft_repository_has_private_file_only_no_backup_owner_policy() -> None:
    policy = private_sqlite._SQLITE_OWNER_POLICIES["personal_context.interview_drafts"]

    assert policy.allowed_target_kinds == frozenset(
        {private_sqlite.SQLiteTargetKind.PRIVATE_FILE}
    )
    assert policy.centralized_backup_allowed is False


def test_protected_draft_is_encrypted_resumable_local_only_and_key_destroyed(
    tmp_path,
    caplog,
) -> None:
    canary = "RAW_INTERVIEW_CANARY_9f90d1"
    path = tmp_path / "interviews.db"
    protector = InMemoryProfileKeyProtector()
    repository = InterviewDraftRepository(
        path,
        key_protector=protector,
        clock=lambda: NOW,
    )

    with caplog.at_level(logging.DEBUG):
        repository.save(
            "session-1",
            {"answer": canary, "sync": "NO_SYNC"},
            expires_at=NOW + timedelta(days=30),
        )
    resumed = InterviewDraftRepository(
        path,
        key_protector=protector,
        clock=lambda: NOW,
    ).load("session-1")
    with sqlite3.connect(path) as connection:
        cleanup_pending = connection.execute(
            "SELECT cleanup_pending FROM interview_drafts WHERE session_id = ?",
            ("session-1",),
        ).fetchone()[0]

    assert resumed.payload == {"answer": canary, "sync": "NO_SYNC"}
    assert cleanup_pending == 0
    assert protector.is_empty is False
    durable = b"".join(
        candidate.read_bytes()
        for candidate in path.parent.glob(f"{path.name}*")
        if candidate.is_file()
    )
    assert canary.encode() not in durable
    assert canary not in caplog.text
    assert b"encrypted_outbox" not in durable

    repository.delete("session-1")
    assert protector.is_empty is True
    assert repository.load("session-1") is None


@pytest.mark.parametrize("session_id", ["", " ", "x" * 129, "bad\x00id"])
def test_session_id_must_be_bounded_nonblank_text(tmp_path, session_id) -> None:
    repository = InterviewDraftRepository(
        tmp_path / "interviews.db",
        key_protector=InMemoryProfileKeyProtector(),
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="session ID"):
        repository.save(session_id, {"answer": "x"}, expires_at=NOW + timedelta(days=1))


def test_draft_expiry_is_normalized_to_utc_for_persistence_and_queries(
    tmp_path,
) -> None:
    path = tmp_path / "interviews.db"
    repository = InterviewDraftRepository(
        path,
        key_protector=InMemoryProfileKeyProtector(),
        clock=lambda: NOW,
    )
    same_instant = (NOW + timedelta(days=1)).astimezone(timezone(timedelta(hours=-7)))

    repository.save("session-1", {"answer": "x"}, expires_at=same_instant)

    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            "SELECT expires_at FROM interview_drafts WHERE session_id = ?",
            ("session-1",),
        ).fetchone()[0]
    assert stored == (NOW + timedelta(days=1)).isoformat()


def test_memory_only_draft_never_writes_and_cannot_resume(tmp_path) -> None:
    existing_paths = set(tmp_path.iterdir())
    repository = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    repository.save(
        "session-1",
        {"answer": "volatile"},
        expires_at=NOW + timedelta(days=30),
    )

    assert repository.is_memory_only is True
    assert repository.load("session-1").payload["answer"] == "volatile"
    assert (
        InterviewDraftRepository.memory_only(clock=lambda: NOW).load("session-1")
        is None
    )
    assert set(tmp_path.iterdir()) == existing_paths


def test_expired_draft_is_destroyed_and_maximum_retention_is_thirty_days(
    tmp_path,
) -> None:
    protector = InMemoryProfileKeyProtector()
    now = NOW
    repository = InterviewDraftRepository(
        tmp_path / "expiry.db",
        key_protector=protector,
        clock=lambda: now,
    )

    with pytest.raises(ValueError, match="30 days"):
        repository.save(
            "too-long",
            {"answer": "x"},
            expires_at=NOW + timedelta(days=30, seconds=1),
        )
    repository.save(
        "expires",
        {"answer": "x"},
        expires_at=NOW + timedelta(days=30),
    )
    now = NOW + timedelta(days=30, seconds=1)

    with pytest.raises(InterviewDraftExpiredError):
        repository.require_active("expires")
    assert protector.is_empty is True
    assert repository.load("expires") is None
    assert repository.expire() == ()


def test_load_rejects_invalid_envelope_shapes_without_exposing_plaintext(
    tmp_path,
) -> None:
    path = tmp_path / "interviews.db"
    repository = InterviewDraftRepository(
        path,
        key_protector=InMemoryProfileKeyProtector(),
        clock=lambda: NOW,
    )
    repository.save("session-1", {"answer": "x"}, expires_at=NOW + timedelta(days=1))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE interview_drafts SET algorithm = 'unsupported' WHERE session_id = ?",
            ("session-1",),
        )

    with pytest.raises(ValueError, match="draft envelope"):
        repository.load("session-1")


def test_invalid_draft_payload_is_rejected_before_any_key_is_created(tmp_path) -> None:
    protector = InMemoryProfileKeyProtector()
    repository = InterviewDraftRepository(
        tmp_path / "interviews.db", key_protector=protector, clock=lambda: NOW
    )

    with pytest.raises(ValueError, match="JSON object"):
        repository.save(
            "session-1",
            {"answer": object()},
            expires_at=NOW + timedelta(days=1),
        )

    assert protector.is_empty is True


def test_delete_still_attempts_key_destruction_when_db_delete_fails(
    tmp_path, monkeypatch
) -> None:
    class TrackingProtector(InMemoryProfileKeyProtector):
        deleted = False

        def delete(self, profile_ref: str) -> None:
            self.deleted = True
            super().delete(profile_ref)

    protector = TrackingProtector()
    repository = InterviewDraftRepository(
        tmp_path / "interviews.db", key_protector=protector, clock=lambda: NOW
    )
    repository.save("session-1", {"answer": "x"}, expires_at=NOW + timedelta(days=1))
    monkeypatch.setattr(
        repository,
        "_delete_row",
        lambda _session_id: (_ for _ in ()).throw(sqlite3.OperationalError("fail")),
    )

    with pytest.raises(sqlite3.OperationalError, match="fail"):
        repository.delete("session-1")

    assert protector.deleted is True
    assert repository.load("session-1") is None
    with sqlite3.connect(tmp_path / "interviews.db") as connection:
        assert (
            connection.execute(
                "SELECT cleanup_pending FROM interview_drafts WHERE session_id = ?",
                ("session-1",),
            ).fetchone()[0]
            == 1
        )


def test_delete_with_protector_failure_keeps_discoverable_row_for_retry(
    tmp_path,
) -> None:
    class FailOnceProtector(InMemoryProfileKeyProtector):
        fail = True

        def delete(self, profile_ref: str) -> None:
            if self.fail:
                self.fail = False
                raise RuntimeError("protector unavailable")
            super().delete(profile_ref)

    protector = FailOnceProtector()
    repository = InterviewDraftRepository(
        tmp_path / "interviews.db", key_protector=protector, clock=lambda: NOW
    )
    repository.save("session-1", {"answer": "x"}, expires_at=NOW + timedelta(days=1))

    with pytest.raises(RuntimeError, match="protector unavailable"):
        repository.delete("session-1")

    assert repository.load("session-1") is None
    with pytest.raises(InterviewDraftExpiredError):
        repository.require_active("session-1")
    with sqlite3.connect(tmp_path / "interviews.db") as connection:
        assert (
            connection.execute(
                "SELECT cleanup_pending FROM interview_drafts WHERE session_id = ?",
                ("session-1",),
            ).fetchone()[0]
            == 1
        )
    assert protector.is_empty is False
    repository.delete("session-1")
    assert protector.is_empty is True
    assert repository.load("session-1") is None


def test_expiry_cleanup_retries_after_key_deletion_failure(tmp_path) -> None:
    class FailOnceProtector(InMemoryProfileKeyProtector):
        fail = True

        def delete(self, profile_ref: str) -> None:
            if self.fail:
                self.fail = False
                raise RuntimeError("protector unavailable")
            super().delete(profile_ref)

    now = NOW
    protector = FailOnceProtector()
    repository = InterviewDraftRepository(
        tmp_path / "interviews.db", key_protector=protector, clock=lambda: now
    )
    repository.save("session-1", {"answer": "x"}, expires_at=NOW + timedelta(days=1))
    now = NOW + timedelta(days=2)

    assert repository.expire() == ()
    with sqlite3.connect(tmp_path / "interviews.db") as connection:
        row = connection.execute(
            "SELECT cleanup_pending FROM interview_drafts WHERE session_id = ?",
            ("session-1",),
        ).fetchone()
    assert row[0] == 1

    assert repository.expire() == ("session-1",)
    assert protector.is_empty is True


def test_expiry_sweep_isolates_cleanup_row_failure_and_retries_without_key(
    tmp_path, monkeypatch
) -> None:
    class TrackingProtector(InMemoryProfileKeyProtector):
        def __init__(self) -> None:
            super().__init__()
            self.deleted_refs: list[str] = []

        def delete(self, profile_ref: str) -> None:
            self.deleted_refs.append(profile_ref)
            super().delete(profile_ref)

    now = NOW
    path = tmp_path / "interviews.db"
    protector = TrackingProtector()
    repository = InterviewDraftRepository(
        path, key_protector=protector, clock=lambda: now
    )
    for session_id in ("first", "later"):
        repository.save(
            session_id,
            {"answer": f"secret-{session_id}"},
            expires_at=NOW + timedelta(days=1),
        )
    real_delete_row = repository._delete_row
    failed = False

    def fail_first_row_delete(session_id: str) -> None:
        nonlocal failed
        if session_id == "first" and not failed:
            failed = True
            raise sqlite3.OperationalError("injected row deletion failure")
        real_delete_row(session_id)

    monkeypatch.setattr(repository, "_delete_row", fail_first_row_delete)
    now = NOW + timedelta(days=2)

    assert repository.expire() == ("later",)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT session_id, cleanup_pending FROM interview_drafts ORDER BY rowid"
        ).fetchall()
    assert rows == [("first", 1)]
    assert protector.is_empty is True
    assert len(protector.deleted_refs) == 2

    assert repository.expire() == ("first",)
    assert repository.expire() == ()
    assert len(protector.deleted_refs) == 3
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM interview_drafts").fetchone()[0]
            == 0
        )


def test_indexed_expiry_extension_fails_authentication_and_destroys_draft(
    tmp_path,
) -> None:
    path = tmp_path / "interviews.db"
    protector = InMemoryProfileKeyProtector()
    repository = InterviewDraftRepository(
        path, key_protector=protector, clock=lambda: NOW
    )
    repository.save("session-1", {"answer": "x"}, expires_at=NOW + timedelta(days=1))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE interview_drafts SET expires_at = ? WHERE session_id = ?",
            ((NOW + timedelta(days=2)).isoformat(), "session-1"),
        )

    with pytest.raises(ValueError, match="expiry authentication"):
        repository.load("session-1")

    assert repository.load("session-1") is None
    assert protector.is_empty is True


def test_expiry_sweep_destroys_malformed_or_extended_metadata(tmp_path) -> None:
    path = tmp_path / "interviews.db"
    protector = InMemoryProfileKeyProtector()
    repository = InterviewDraftRepository(
        path, key_protector=protector, clock=lambda: NOW
    )
    for session_id in ("malformed", "extended"):
        repository.save(session_id, {"answer": "x"}, expires_at=NOW + timedelta(days=1))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE interview_drafts SET expires_at = 'not-a-date' "
            "WHERE session_id = 'malformed'"
        )
        connection.execute(
            "UPDATE interview_drafts SET expires_at = ? WHERE session_id = 'extended'",
            ((NOW + timedelta(days=20)).isoformat(),),
        )

    assert set(repository.expire()) == {"malformed", "extended"}
    assert protector.is_empty is True


def test_draft_save_uses_optimistic_revision_fence_in_memory() -> None:
    repository = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    created = repository.save(
        "session-1", {"answer": "initial"}, expires_at=NOW + timedelta(days=1)
    )
    first_reader = repository.load("session-1")
    stale_reader = repository.load("session-1")

    saved = repository.save(
        "session-1",
        {"answer": "winner"},
        expires_at=NOW + timedelta(days=1),
        expected_revision=first_reader.revision,
    )
    with pytest.raises(InterviewDraftConflictError):
        repository.save(
            "session-1",
            {"answer": "stale"},
            expires_at=NOW + timedelta(days=1),
            expected_revision=stale_reader.revision,
        )

    assert created.revision == 1
    assert saved.revision == 2
    assert repository.load("session-1").payload["answer"] == "winner"


def test_memory_draft_nested_payloads_are_isolated_by_revision() -> None:
    repository = InterviewDraftRepository.memory_only(clock=lambda: NOW)
    created = repository.save(
        "session-1",
        {"turns": [], "nested": {"answer": "initial"}},
        expires_at=NOW + timedelta(days=1),
    )
    created.payload["turns"].append({"answer": "escaped"})
    assert repository.load("session-1").payload["turns"] == []

    winner = repository.load("session-1")
    stale = repository.load("session-1")
    winner.payload["turns"].append({"answer": "winner"})
    assert repository.load("session-1").payload["turns"] == []
    repository.save(
        "session-1",
        winner.payload,
        expires_at=NOW + timedelta(days=1),
        expected_revision=winner.revision,
    )
    stale.payload["nested"]["answer"] = "stale"
    with pytest.raises(InterviewDraftConflictError):
        repository.save(
            "session-1",
            stale.payload,
            expires_at=NOW + timedelta(days=1),
            expected_revision=stale.revision,
        )

    stored = repository.load("session-1").payload
    assert stored["turns"] == [{"answer": "winner"}]
    assert stored["nested"]["answer"] == "initial"


def test_protected_draft_save_uses_optimistic_revision_fence(tmp_path) -> None:
    path = tmp_path / "interviews.db"
    protector = InMemoryProfileKeyProtector()
    first = InterviewDraftRepository(path, key_protector=protector, clock=lambda: NOW)
    second = InterviewDraftRepository(path, key_protector=protector, clock=lambda: NOW)
    created = first.save(
        "session-1", {"answer": "initial"}, expires_at=NOW + timedelta(days=1)
    )
    stale = second.load("session-1")

    first.save(
        "session-1",
        {"answer": "winner"},
        expires_at=NOW + timedelta(days=1),
        expected_revision=created.revision,
    )
    with pytest.raises(InterviewDraftConflictError):
        second.save(
            "session-1",
            {"answer": "stale"},
            expires_at=NOW + timedelta(days=1),
            expected_revision=stale.revision,
        )

    assert first.load("session-1").payload["answer"] == "winner"


def test_existing_draft_schema_is_migrated_with_revision_and_cleanup_fences(
    tmp_path,
) -> None:
    path = tmp_path / "interviews.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE interview_drafts (
                session_id TEXT PRIMARY KEY,
                expires_at TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                nonce BLOB NOT NULL,
                ciphertext BLOB NOT NULL,
                wrapped_dek BLOB NOT NULL,
                key_version INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO interview_drafts(
                session_id, expires_at, algorithm, nonce, ciphertext,
                wrapped_dek, key_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-session",
                (NOW + timedelta(days=1)).isoformat(),
                "legacy-algorithm",
                b"nonce",
                b"ciphertext",
                b"wrapped",
                1,
            ),
        )

    InterviewDraftRepository(
        path,
        key_protector=InMemoryProfileKeyProtector(),
        clock=lambda: NOW,
    )

    with sqlite3.connect(path) as connection:
        columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(interview_drafts)")
        }
        cleanup_pending = connection.execute(
            "SELECT cleanup_pending FROM interview_drafts WHERE session_id = ?",
            ("legacy-session",),
        ).fetchone()[0]
    assert "revision" in columns
    assert columns["cleanup_pending"][3] == 1
    assert columns["cleanup_pending"][4] == "0"
    assert cleanup_pending == 0
    assert "payload" not in columns
