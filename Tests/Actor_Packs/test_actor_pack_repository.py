"""Portable Actor identity and Persona intent repository coverage."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.repository import (
    ActorPackRepository,
    ActorPackRepositoryError,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


UUID_A = "123e4567-e89b-42d3-a456-426614174000"
UUID_B = "223e4567-e89b-42d3-a456-426614174000"
UUID_C = "323e4567-e89b-42d3-a456-426614174000"
SHA_A = "a" * 64
SHA_B = "b" * 64


@pytest.fixture
def repository(tmp_path: Path):
    database = CharactersRAGDB(tmp_path / "actor-packs.db", client_id="actor-packs")
    repo = ActorPackRepository(database, uuid_factory=lambda: uuid.UUID(UUID_A))
    yield repo
    database.close_connection()


def test_assign_identity_is_stable_and_cross_kind_unique(repository) -> None:
    first = repository.assign_identity("character", 7)
    second = repository.assign_identity("character", 7)

    assert first == second
    assert first.actor_kind == "character"
    assert first.local_actor_id == "7"
    assert first.portable_uuid == UUID_A
    assert first.source_portable_uuid is None

    with pytest.raises(
        ActorPackRepositoryError, match="actor_pack_repository_write_failed"
    ):
        repository.assign_identity("persona", "guide")
    assert repository.get_identity("persona", "guide") is None


def test_copy_gets_fresh_identity_and_exact_source_provenance(tmp_path: Path) -> None:
    database = CharactersRAGDB(tmp_path / "copy.db", client_id="actor-pack-copy")
    values = iter((uuid.UUID(UUID_A), uuid.UUID(UUID_B)))
    repository = ActorPackRepository(database, uuid_factory=lambda: next(values))
    try:
        original = repository.assign_identity("persona", "guide")
        copied = repository.assign_identity(
            "persona", "guide-copy", source_portable_uuid=original.portable_uuid
        )
        assert copied.portable_uuid == UUID_B
        assert copied.source_portable_uuid == UUID_A
        assert repository.get_identity("persona", "guide-copy") == copied
    finally:
        database.close_connection()


def test_lookup_by_portable_uuid_returns_exact_cross_kind_identity(
    repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = repository.assign_identity("character", 7)
    transaction = repository.db.transaction
    entered = False

    @contextmanager
    def observe_transaction(*args, **kwargs):
        nonlocal entered
        with transaction(*args, **kwargs) as cursor:
            entered = True
            assert getattr(repository.db._local, "transaction_depth", 0) == 1
            yield cursor

    monkeypatch.setattr(repository.db, "transaction", observe_transaction)

    assert repository.get_identity_by_portable_uuid(UUID_A) == identity
    assert repository.get_identity_by_portable_uuid(UUID_B) is None
    assert entered


@pytest.mark.parametrize(
    ("actor_kind", "actor_id", "source_uuid", "source"),
    [
        ("tool", "guide", None, "local"),
        ("character", "7", None, "local"),
        ("character", 0, None, "local"),
        ("persona", "", None, "local"),
        ("persona", "guide", "not-a-uuid", "local"),
        ("persona", "guide", UUID_A.upper(), "local"),
        ("persona", "guide", None, "server"),
    ],
)
def test_assignment_rejects_invalid_or_nonlocal_identity(
    repository, actor_kind, actor_id, source_uuid, source
) -> None:
    expected = (
        "actor_pack_source_not_local"
        if source == "server"
        else "actor_pack_identity_invalid"
    )
    with pytest.raises(ActorPackRepositoryError, match=expected):
        repository.assign_identity(
            actor_kind,
            actor_id,
            source=source,
            source_portable_uuid=source_uuid,
        )


def test_assignment_rejects_ambiguous_transaction_nesting(repository) -> None:
    with repository.db.transaction(immediate=True):
        with pytest.raises(
            ActorPackRepositoryError,
            match="actor_pack_repository_transaction_active",
        ):
            repository.assign_identity("persona", "guide")
    assert repository.get_identity("persona", "guide") is None


def test_generated_identity_requires_exact_uuid4_variant(tmp_path: Path) -> None:
    database = CharactersRAGDB(tmp_path / "uuid-version.db", client_id="uuid-version")
    repository = ActorPackRepository(
        database,
        uuid_factory=lambda: uuid.UUID("123e4567-e89b-12d3-a456-426614174000"),
    )
    try:
        with pytest.raises(
            ActorPackRepositoryError, match="actor_pack_identity_invalid"
        ):
            repository.assign_identity("persona", "guide")
        assert repository.get_identity("persona", "guide") is None
    finally:
        database.close_connection()


def test_package_owned_assignment_requires_one_reserved_transaction(repository) -> None:
    with pytest.raises(
        ActorPackRepositoryError,
        match="actor_pack_repository_transaction_active",
    ):
        repository._assign_identity_in_transaction("character", 7)

    with repository.db.transaction(immediate=True):
        identity = repository._assign_identity_in_transaction("character", 7)
    assert identity.portable_uuid == UUID_A

    with repository.db.transaction(immediate=True):
        with repository.db.transaction():
            with pytest.raises(
                ActorPackRepositoryError,
                match="actor_pack_repository_transaction_active",
            ):
                repository._assign_identity_in_transaction("character", 8)
    assert repository.get_identity("character", 8) is None


def test_concurrent_assignment_returns_one_stable_identity(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.db"
    bootstrap = CharactersRAGDB(path, client_id="bootstrap")
    bootstrap.close_connection()
    barrier = threading.Barrier(2)
    results: list[str] = []
    errors: list[BaseException] = []

    def assign(value: str, client_id: str) -> None:
        database = CharactersRAGDB(path, client_id=client_id)
        repo = ActorPackRepository(database, uuid_factory=lambda: uuid.UUID(value))
        try:
            barrier.wait(timeout=2)
            results.append(repo.assign_identity("persona", "guide").portable_uuid)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            database.close_connection()

    threads = (
        threading.Thread(target=assign, args=(UUID_A, "writer-a")),
        threading.Thread(target=assign, args=(UUID_B, "writer-b")),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert len(results) == 2
    assert len(set(results)) == 1


def test_prepare_commit_and_cleanup_persona_intent_are_durable(repository) -> None:
    intent = repository.prepare_persona_intent(
        persona_id="guide",
        operation="create",
        old_profile=None,
        new_profile={"id": "guide", "name": "Guide"},
        old_store_sha256=SHA_A,
        new_store_sha256=SHA_B,
        old_registry_uuid=None,
        new_registry_uuid=UUID_A,
        intent_id="a" * 32,
    )
    assert intent.state == "prepared"
    assert intent.old_profile_json is None
    assert intent.new_profile_json == '{"id":"guide","name":"Guide"}'
    assert repository.list_persona_intents() == (intent,)

    identity, committed = repository.commit_persona_intent(intent.intent_id)
    assert identity.portable_uuid == UUID_A
    assert committed.state == "committed"
    assert repository.get_identity("persona", "guide") == identity

    repository.cleanup_persona_intent(intent.intent_id, expected_state="committed")
    assert repository.list_persona_intents() == ()


def test_intent_commit_rolls_back_identity_when_transition_is_stale(repository) -> None:
    intent = repository.prepare_persona_intent(
        persona_id="guide",
        operation="create",
        old_profile=None,
        new_profile={"id": "guide"},
        old_store_sha256=SHA_A,
        new_store_sha256=SHA_B,
        old_registry_uuid=None,
        new_registry_uuid=UUID_A,
        intent_id="b" * 32,
    )
    repository.db.execute_query(
        """
        UPDATE actor_pack_persona_intents
           SET state = 'quarantined', quarantine_reason = 'recovery_blocked'
         WHERE intent_id = ?
        """,
        (intent.intent_id,),
    )
    repository.db.get_connection().commit()

    with pytest.raises(
        ActorPackRepositoryError, match="actor_pack_intent_state_changed"
    ):
        repository.commit_persona_intent(intent.intent_id)
    assert repository.get_identity("persona", "guide") is None


def test_intent_commit_rolls_back_identity_when_state_update_aborts(repository) -> None:
    intent = repository.prepare_persona_intent(
        persona_id="guide",
        operation="create",
        old_profile=None,
        new_profile={"id": "guide"},
        old_store_sha256=SHA_A,
        new_store_sha256=SHA_B,
        old_registry_uuid=None,
        new_registry_uuid=UUID_A,
        intent_id="e" * 32,
    )
    repository.db.execute_query(
        """
        CREATE TRIGGER actor_pack_test_abort_intent_commit
        BEFORE UPDATE OF state ON actor_pack_persona_intents
        WHEN NEW.state = 'committed'
        BEGIN
            SELECT RAISE(ABORT, 'test abort');
        END
        """
    )
    repository.db.get_connection().commit()

    with pytest.raises(
        ActorPackRepositoryError, match="actor_pack_repository_write_failed"
    ):
        repository.commit_persona_intent(intent.intent_id)
    assert repository.get_identity("persona", "guide") is None
    assert repository.list_persona_intents() == (intent,)


def test_intent_authority_guard_runs_inside_reserved_transaction(repository) -> None:
    intent = repository.prepare_persona_intent(
        persona_id="guide",
        operation="create",
        old_profile=None,
        new_profile={"id": "guide"},
        old_store_sha256=SHA_A,
        new_store_sha256=SHA_B,
        old_registry_uuid=None,
        new_registry_uuid=UUID_A,
        intent_id="f" * 32,
    )
    observed: list[tuple[bool, int]] = []

    def deny() -> bool:
        observed.append(
            (
                repository.db.get_connection().in_transaction,
                getattr(repository.db._local, "transaction_depth", 0),
            )
        )
        return False

    with pytest.raises(
        ActorPackRepositoryError, match="actor_pack_intent_state_changed"
    ):
        repository.commit_persona_intent(intent.intent_id, authority_guard=deny)
    assert observed == [(True, 1)]
    assert repository.get_identity("persona", "guide") is None
    assert repository.list_persona_intents() == (intent,)


def test_quarantine_is_idempotent_and_path_free(repository) -> None:
    intent = repository.prepare_persona_intent(
        persona_id="guide",
        operation="update",
        old_profile={"id": "guide", "name": "Old"},
        new_profile={"id": "guide", "name": "New"},
        old_store_sha256=SHA_A,
        new_store_sha256=SHA_B,
        old_registry_uuid=UUID_A,
        new_registry_uuid=UUID_A,
        intent_id="c" * 32,
    )
    quarantined = repository.quarantine_persona_intent(
        intent.intent_id, "authority_mismatch"
    )
    assert quarantined.state == "quarantined"
    assert quarantined.quarantine_reason == "authority_mismatch"
    assert (
        repository.quarantine_persona_intent(intent.intent_id, "authority_mismatch")
        == quarantined
    )


def test_corrupt_stored_identity_and_intent_use_fixed_categories(repository) -> None:
    repository.assign_identity("persona", "guide")
    repository.db.execute_query("PRAGMA ignore_check_constraints = ON")
    repository.db.execute_query(
        "UPDATE actor_portable_identities SET portable_uuid = 'private/path'"
    )
    repository.db.get_connection().commit()
    with pytest.raises(ActorPackRepositoryError) as identity_error:
        repository.get_identity("persona", "guide")
    assert str(identity_error.value) == "actor_pack_repository_corrupt"

    repository.db.execute_query(
        """
        INSERT INTO actor_pack_persona_intents(
            intent_id, persona_id, operation, state,
            old_profile_json, new_profile_json,
            old_profile_sha256, new_profile_sha256,
            old_store_sha256, new_store_sha256,
            old_registry_uuid, new_registry_uuid
        ) VALUES (?, ?, 'create', 'prepared', NULL, ?, NULL, ?, ?, ?, NULL, ?)
        """,
        ("d" * 32, "guide", "[]", SHA_A, SHA_A, SHA_B, UUID_B),
    )
    repository.db.get_connection().commit()
    with pytest.raises(ActorPackRepositoryError) as intent_error:
        repository.list_persona_intents()
    assert str(intent_error.value) == "actor_pack_repository_corrupt"


def test_sqlite_failures_do_not_expose_database_details(
    repository, monkeypatch
) -> None:
    def fail(*_args, **_kwargs):
        raise sqlite3.OperationalError("/Users/private/actor-packs.db")

    monkeypatch.setattr(repository.db, "execute_query", fail)
    with pytest.raises(ActorPackRepositoryError) as read_error:
        repository.get_identity("persona", "guide")
    assert str(read_error.value) == "actor_pack_repository_read_failed"
    with pytest.raises(ActorPackRepositoryError) as write_error:
        repository.assign_identity("persona", "guide")
    assert str(write_error.value) == "actor_pack_repository_write_failed"
    assert "private" not in repr(read_error.value)
    assert "private" not in repr(write_error.value)
