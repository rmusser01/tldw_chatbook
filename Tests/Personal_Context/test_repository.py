from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest
from tldw_profile_core import (
    ProfileManifest,
    ProfileScope,
    ProposalState,
    ScopeKind,
    SyncMode,
)

import Tests.Personal_Context.test_profile_reconciliation as reconciliation_tests
import tldw_chatbook.Personal_Context.repository as repository_module
import tldw_chatbook.Personal_Context.repository_keys as repository_keys
from Tests.Personal_Context.test_key_protector import SecureKeyring
from tldw_chatbook.Personal_Context.key_protector import (
    InMemoryProfileKeyProtector,
    KeyringProfileKeyProtector,
    ProfileKeyMaterial,
    ProfileLockedError,
)
from tldw_chatbook.Personal_Context.reconciliation import build_reconciliation_plan
from tldw_chatbook.Personal_Context.service import PersonalContextService
from tldw_chatbook.Personal_Context.repository import (
    ConcurrentProfileUpdateError,
    PersonalContextRepository,
    ProfileAlreadyExistsError,
    ProfileDestroyedError,
    RepositorySchemaError,
)


class _FailDeleteOnceProtector(InMemoryProfileKeyProtector):
    def __init__(self) -> None:
        super().__init__()
        self.delete_calls = 0

    def delete(self, profile_ref: str) -> None:
        self.delete_calls += 1
        if self.delete_calls == 1:
            raise ProfileLockedError("temporary keyring failure")
        super().delete(profile_ref)


def test_simultaneous_first_openers_create_no_key_and_one_profile_wins(
    tmp_path,
) -> None:
    db_path = tmp_path / "personal-context.db"
    protector = InMemoryProfileKeyProtector()
    repository_module.connect_private_sqlite(
        "personal_context.repository", db_path, isolation_level=None
    ).close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(PersonalContextRepository, db_path, key_protector=protector)
            for _ in range(2)
        ]
        repositories = [future.result(timeout=5) for future in futures]

    assert protector.is_empty
    assert repositories[0]._keys is None and repositories[1]._keys is None
    manifest = repositories[0].create_provisional_profile()
    with pytest.raises(ProfileAlreadyExistsError):
        repositories[1].create_provisional_profile()
    assert list(protector._materials) == [_stored_account(db_path)]
    assert repositories[0].get_manifest() == manifest
    assert repositories[1].get_manifest() == manifest
    reopened = PersonalContextRepository(db_path, key_protector=protector)
    assert reopened.get_manifest() == manifest


def test_schema_initialization_failure_creates_no_key(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "personal-context.db"
    protector = InMemoryProfileKeyProtector()

    def fail_initialization(*_args, **_kwargs):
        raise sqlite3.OperationalError("simulated schema failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            PersonalContextRepository, "_initialize_schema", fail_initialization
        )
        with pytest.raises(sqlite3.OperationalError, match="simulated schema failure"):
            PersonalContextRepository(db_path, key_protector=protector)

    assert protector.is_empty
    retried = PersonalContextRepository(db_path, key_protector=protector)
    assert retried._keys is None
    manifest = retried.create_provisional_profile()
    assert retried.get_manifest() == manifest


def test_repository_creates_one_profile_and_reopens_with_fresh_connection(
    tmp_path, memory_protector, record_factory
) -> None:
    db_path = tmp_path / "personal-context.db"
    first = PersonalContextRepository(db_path, key_protector=memory_protector)
    manifest = first.create_provisional_profile()
    record = record_factory(manifest.profile_id)
    first.commit_record_version(record, expected_version_id=None)
    first.close()

    reopened = PersonalContextRepository(db_path, key_protector=memory_protector)

    assert reopened.get_manifest() == manifest
    assert reopened.get_record(record.record_id) == record
    with pytest.raises(ProfileAlreadyExistsError):
        reopened.create_provisional_profile()


def test_repository_versions_are_immutable_and_head_updates_use_cas(
    tmp_path, memory_protector, record_factory
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    original = record_factory(manifest.profile_id)
    repo.commit_record_version(original, expected_version_id=None)
    stale = record_factory(
        manifest.profile_id,
        version_id="record-version-stale",
        parent_version_id=original.version_id,
        value="stale",
    )

    with pytest.raises(ConcurrentProfileUpdateError):
        repo.commit_record_version(stale, expected_version_id="wrong-version")
    assert repo.get_record(original.record_id) == original

    updated = record_factory(
        manifest.profile_id,
        version_id="record-version-2",
        parent_version_id=original.version_id,
        value="more detail",
    )
    repo.commit_record_version(updated, expected_version_id=original.version_id)
    assert repo.get_record(original.record_id) == updated

    duplicate_version = record_factory(
        manifest.profile_id,
        version_id=updated.version_id,
        parent_version_id=updated.version_id,
        value="duplicate version body",
    )
    with pytest.raises(sqlite3.IntegrityError):
        repo.commit_record_version(
            duplicate_version, expected_version_id=updated.version_id
        )


def test_failed_cas_rolls_back_object_and_outbox_insert(
    tmp_path, memory_protector, record_factory
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    original = record_factory(manifest.profile_id)
    repo.commit_record_version(original, expected_version_id=None)
    attempted = record_factory(
        manifest.profile_id,
        version_id="rolled-back-version",
        parent_version_id=original.version_id,
        value="ROLLBACK-CANARY",
    )

    with pytest.raises(ConcurrentProfileUpdateError):
        repo.commit_record_version(
            attempted,
            expected_version_id="stale-head",
            outbox_body={"private_snapshot": "OUTBOX-ROLLBACK-CANARY"},
        )

    with repo._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM encrypted_objects WHERE version_id = ?",
                (attempted.version_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM encrypted_outbox WHERE object_type = 'record'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("existing_head", [False, True])
def test_record_parent_must_match_expected_head_without_partial_writes(
    tmp_path, memory_protector, record_factory, existing_head
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    expected_version_id = None
    if existing_head:
        original = record_factory(manifest.profile_id)
        repo.commit_record_version(original, expected_version_id=None)
        expected_version_id = original.version_id
    attempted = record_factory(
        manifest.profile_id,
        version_id="invalid-parent-version",
        parent_version_id="ghost-version" if expected_version_id is None else None,
    )

    with pytest.raises(ConcurrentProfileUpdateError, match="parent"):
        repo.commit_record_version(
            attempted,
            expected_version_id=expected_version_id,
            outbox_body={"must_not": "commit"},
        )

    with repo._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM encrypted_objects WHERE version_id = ?",
                (attempted.version_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM encrypted_outbox WHERE object_type = 'record'"
            ).fetchone()[0]
            == 0
        )
        head = connection.execute(
            "SELECT version_id FROM object_heads "
            "WHERE object_type = 'record' AND object_id = ?",
            (attempted.record_id,),
        ).fetchone()
    assert (None if head is None else head["version_id"]) == expected_version_id


def test_proposals_resolve_to_content_free_encrypted_receipts(
    tmp_path, memory_protector, proposal_factory
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    proposal = proposal_factory(manifest.profile_id)

    repo.commit_proposal(proposal)
    assert repo.list_proposals() == [proposal]

    resolved = repo.resolve_proposal(proposal.proposal_id, ProposalState.REJECTED)
    assert resolved.state is ProposalState.REJECTED
    assert resolved.proposed_record is None
    assert resolved.confidence is None
    assert repo.list_proposals() == [resolved]


def test_terminal_proposal_cannot_be_resolved_again(
    tmp_path, memory_protector, proposal_factory
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    proposal = proposal_factory(manifest.profile_id)
    repo.commit_proposal(proposal)
    accepted = repo.resolve_proposal(proposal.proposal_id, ProposalState.ACCEPTED)
    with repo._connect() as connection:
        before_head = connection.execute(
            "SELECT version_id FROM object_heads "
            "WHERE object_type = 'proposal' AND object_id = ?",
            (proposal.proposal_id,),
        ).fetchone()[0]
        before_receipts = connection.execute(
            "SELECT COUNT(*) FROM encrypted_objects "
            "WHERE object_type = 'proposal' AND object_id = ?",
            (proposal.proposal_id,),
        ).fetchone()[0]

    with pytest.raises(ValueError, match="pending"):
        repo.resolve_proposal(proposal.proposal_id, ProposalState.REJECTED)

    with repo._connect() as connection:
        after_head = connection.execute(
            "SELECT version_id FROM object_heads "
            "WHERE object_type = 'proposal' AND object_id = ?",
            (proposal.proposal_id,),
        ).fetchone()[0]
        after_receipts = connection.execute(
            "SELECT COUNT(*) FROM encrypted_objects "
            "WHERE object_type = 'proposal' AND object_id = ?",
            (proposal.proposal_id,),
        ).fetchone()[0]
    assert repo.list_proposals() == [accepted]
    assert (after_head, after_receipts) == (before_head, before_receipts)


def test_policies_bindings_and_outbox_bodies_are_encrypted(
    tmp_path, memory_protector
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    repo.create_provisional_profile()

    policy_version = repo.commit_runtime_policy(
        "scope-global", {"agent_grant": "POLICY-CANARY"}
    )
    binding_version = repo.commit_scope_binding(
        "scope-global", {"workspace_label": "BINDING-CANARY"}
    )
    outbox_id = repo.commit_outbox_body(
        object_type="record",
        object_id="record-1",
        version_id="record-version-1",
        body={"snapshot": "OUTBOX-CANARY"},
    )

    assert repo.get_runtime_policy("scope-global") == {"agent_grant": "POLICY-CANARY"}
    assert repo.get_scope_binding("scope-global") == {
        "workspace_label": "BINDING-CANARY"
    }
    assert repo.get_outbox_body(outbox_id) == {"snapshot": "OUTBOX-CANARY"}
    assert policy_version and binding_version and outbox_id


def test_corrupt_record_is_quarantined_and_omitted(
    tmp_path, memory_protector, record_factory
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    record = record_factory(manifest.profile_id)
    repo.commit_record_version(record, expected_version_id=None)

    with repo._connect() as connection:
        connection.execute(
            "UPDATE encrypted_objects SET integrity_tag = ? "
            "WHERE object_type = 'record' AND object_id = ?",
            ("hmac-sha256-v1:" + "0" * 64, record.record_id),
        )

    assert repo.list_records() == []
    assert repo.list_quarantine()[0].reason_code == "integrity_failure"


def test_collection_heads_are_read_with_bounded_keyset_pages(
    tmp_path, memory_protector, record_factory, monkeypatch
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    record = record_factory(manifest.profile_id)
    repo.commit_record_version(record, expected_version_id=None)
    statements: list[str] = []
    original_connect = repo._connect

    def traced_connect():
        connection = original_connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(repo, "_connect", traced_connect)
    monkeypatch.setattr(repository_module, "_COLLECTION_PAGE_SIZE", 1, raising=False)

    assert repo.list_records() == [record]
    head_queries = [
        statement
        for statement in statements
        if "FROM object_heads" in statement
        and "object_type = 'record'" in statement
        and "object_id >" in statement
    ]
    assert len(head_queries) == 2
    assert all("LIMIT 1" in statement for statement in head_queries)


def test_untrusted_envelope_key_version_is_quarantined(
    tmp_path, memory_protector, record_factory
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    record = record_factory(manifest.profile_id)
    repo.commit_record_version(record, expected_version_id=None)

    with repo._connect() as connection:
        connection.execute(
            "UPDATE encrypted_objects SET key_version = 2 "
            "WHERE object_type = 'record' AND object_id = ?",
            (record.record_id,),
        )

    assert repo.list_records() == []
    assert repo.list_quarantine()[0].reason_code == "integrity_failure"


@pytest.mark.parametrize(
    ("column", "malformed_value"),
    [
        ("integrity_tag", sqlite3.Binary(b"not-text")),
        ("nonce", "not-bytes"),
        ("wrapped_dek", "not-bytes"),
        ("algorithm", sqlite3.Binary(b"aes-256-gcm-v1")),
    ],
)
def test_malformed_envelope_column_types_are_quarantined(
    tmp_path, memory_protector, record_factory, column, malformed_value
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    record = record_factory(manifest.profile_id)
    repo.commit_record_version(record, expected_version_id=None)

    with repo._connect() as connection:
        connection.execute(
            f"UPDATE encrypted_objects SET {column} = ? "
            "WHERE object_type = 'record' AND object_id = ?",
            (malformed_value, record.record_id),
        )

    assert repo.get_record(record.record_id) is None
    assert repo.list_quarantine()[0].reason_code == "integrity_failure"


def test_device_only_record_does_not_create_outbox(
    tmp_path, memory_protector, record_factory
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    record = record_factory(manifest.profile_id, sync_mode=SyncMode.DEVICE_ONLY)
    repo.commit_record_version(
        record,
        expected_version_id=None,
        outbox_body={"must_not": "leave-device"},
    )
    with repo._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM encrypted_outbox WHERE object_type = 'record'"
            ).fetchone()[0]
            == 0
        )


def test_key_destruction_locks_existing_repository_without_replacement(
    tmp_path, memory_protector, record_factory
) -> None:
    db_path = tmp_path / "personal-context.db"
    repo = PersonalContextRepository(db_path, key_protector=memory_protector)
    manifest = repo.create_provisional_profile()
    repo.commit_record_version(
        record_factory(manifest.profile_id), expected_version_id=None
    )

    repo.destroy_profile_content()
    with repo._connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM encrypted_objects").fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM encrypted_outbox").fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM local_runtime_policy").fetchone()[
                0
            ]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM local_scope_bindings").fetchone()[
                0
            ]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM object_heads").fetchone()[0] == 0
        )
    with pytest.raises(ProfileLockedError):
        repo.get_manifest()
    reopened = PersonalContextRepository(db_path, key_protector=memory_protector)
    assert reopened.is_destroyed() is True
    with pytest.raises(ProfileLockedError):
        reopened.get_manifest()


def test_key_destruction_retries_failed_custody_deletion(tmp_path) -> None:
    db_path = tmp_path / "personal-context.db"
    protector = _FailDeleteOnceProtector()
    repo = PersonalContextRepository(db_path, key_protector=protector)
    repo.create_provisional_profile()

    with pytest.raises(ProfileLockedError, match="temporary keyring failure"):
        repo.destroy_profile_content()

    with repo._connect() as connection:
        meta = connection.execute(
            "SELECT purge_generation, destroyed FROM profile_meta"
        ).fetchone()
        assert tuple(meta) == (1, 1)
        assert (
            connection.execute("SELECT COUNT(*) FROM encrypted_objects").fetchone()[0]
            == 0
        )
    with pytest.raises(ProfileDestroyedError):
        repo.commit_outbox_body(
            object_type="record",
            object_id="record-1",
            version_id="record-version-1",
            body={"must_not": "commit"},
        )

    repo.destroy_profile_content()

    assert protector.delete_calls == 2
    assert protector.is_empty


def test_fresh_generation_failure_rolls_back_to_removed_and_can_retry(
    tmp_path, memory_protector, monkeypatch
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    old_manifest = repo.create_provisional_profile()
    repo.destroy_profile_content()
    fresh_manifest = old_manifest.model_copy(
        update={
            "profile_id": "profile-fresh",
            "current_version_id": "manifest-version-fresh",
        }
    )
    fresh_scope = ProfileScope(
        scope_id="scope-fresh",
        profile_id=fresh_manifest.profile_id,
        kind=ScopeKind.GLOBAL,
        version_id="scope-version-fresh",
        created_at=fresh_manifest.created_at,
        updated_at=fresh_manifest.updated_at,
    )
    original_insert_scope = repo._insert_scope

    def fail_scope_insert(*_args, **_kwargs) -> None:
        raise sqlite3.OperationalError("injected fresh-generation failure")

    monkeypatch.setattr(repo, "_insert_scope", fail_scope_insert)
    with pytest.raises(sqlite3.OperationalError, match="fresh-generation"):
        repo.reinitialize_destroyed_profile(fresh_manifest, fresh_scope)

    assert repo.is_destroyed()
    assert memory_protector.is_empty

    monkeypatch.setattr(repo, "_insert_scope", original_insert_scope)
    repo.reinitialize_destroyed_profile(fresh_manifest, fresh_scope)
    assert repo.get_manifest() == fresh_manifest


def test_destruction_fence_rejects_every_stale_repository_mutation(
    tmp_path, memory_protector, record_factory, proposal_factory
) -> None:
    db_path = tmp_path / "personal-context.db"
    destroying_repo = PersonalContextRepository(db_path, key_protector=memory_protector)
    manifest = destroying_repo.create_provisional_profile()
    stale_repo = PersonalContextRepository(db_path, key_protector=memory_protector)
    record = record_factory(manifest.profile_id)
    proposal = proposal_factory(manifest.profile_id)

    destroying_repo.destroy_profile_content()

    stale_mutations = (
        lambda: stale_repo.commit_outbox_body(
            object_type="record",
            object_id=record.record_id,
            version_id=record.version_id,
            body={"stale": "outbox"},
        ),
        lambda: stale_repo.commit_record_version(record, expected_version_id=None),
        lambda: stale_repo.commit_proposal(proposal),
        lambda: stale_repo.commit_runtime_policy("scope-global", {"stale": "policy"}),
        lambda: stale_repo.commit_scope_binding("scope-global", {"stale": "binding"}),
        lambda: stale_repo.quarantine_object(
            "record", record.record_id, record.version_id, "stale"
        ),
        stale_repo.create_provisional_profile,
    )
    for mutate in stale_mutations:
        with pytest.raises(ProfileDestroyedError):
            mutate()

    with stale_repo._connect() as connection:
        meta = connection.execute(
            "SELECT purge_generation, destroyed FROM profile_meta"
        ).fetchone()
        assert tuple(meta) == (1, 1)
        for table in (
            "encrypted_objects",
            "encrypted_outbox",
            "local_runtime_policy",
            "local_scope_bindings",
            "object_heads",
            "quarantine",
        ):
            assert (
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
            )


@pytest.mark.parametrize("version", [0, repository_module.SCHEMA_VERSION + 1])
def test_repository_fails_closed_on_foreign_or_newer_schema(
    tmp_path, memory_protector, version
) -> None:
    db_path = tmp_path / "personal-context.db"
    connection = sqlite3.connect(db_path)
    if version == 0:
        connection.execute("CREATE TABLE foreign_owner (value TEXT)")
    else:
        connection.execute(
            "CREATE TABLE personal_context_schema (singleton INTEGER PRIMARY KEY, version INTEGER NOT NULL)"
        )
        connection.execute(
            "INSERT INTO personal_context_schema VALUES (1, ?)", (version,)
        )
    connection.commit()
    connection.close()

    with pytest.raises(RepositorySchemaError):
        PersonalContextRepository(db_path, key_protector=memory_protector)


def test_v1_repository_migrates_atomically_and_preserves_encrypted_objects(
    tmp_path, memory_protector, record_factory
) -> None:
    db_path = tmp_path / "personal-context.db"
    original = PersonalContextRepository(db_path, key_protector=memory_protector)
    manifest = original.create_provisional_profile()
    record = record_factory(manifest.profile_id)
    original.commit_record_version(record, expected_version_id=None)
    original.close()

    # Exact Task-24400 storage shape: schema v1 predates the local Undo table.
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE local_undo")
        connection.execute("DROP TABLE local_record_links")
        connection.execute(
            "UPDATE personal_context_schema SET version = 1 WHERE singleton = 1"
        )

    migrated = PersonalContextRepository(db_path, key_protector=memory_protector)

    assert migrated.get_manifest() == manifest
    assert migrated.get_record(record.record_id) == record
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT version FROM personal_context_schema WHERE singleton = 1"
        ).fetchone() == (repository_module.SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'local_undo'"
        ).fetchone() == ("local_undo",)
        assert connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'local_record_links'"
        ).fetchone() == ("local_record_links",)


def test_v5_repository_adds_explicit_unlinked_workspace_state(
    tmp_path, memory_protector
) -> None:
    db_path = tmp_path / "personal-context.db"
    original = PersonalContextRepository(db_path, key_protector=memory_protector)
    manifest = original.create_provisional_profile()
    original.close()

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE local_unlinked_scopes")
        connection.execute(
            "UPDATE personal_context_schema SET version = 5 WHERE singleton = 1"
        )

    migrated = PersonalContextRepository(db_path, key_protector=memory_protector)

    assert migrated.get_manifest() == manifest
    assert migrated.is_scope_explicitly_unlinked("scope-unknown") is False
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT version FROM personal_context_schema WHERE singleton = 1"
        ).fetchone() == (repository_module.SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'local_unlinked_scopes'"
        ).fetchone() == ("local_unlinked_scopes",)


def test_existing_repository_with_missing_protector_never_creates_replacement(
    tmp_path, memory_protector
) -> None:
    db_path = tmp_path / "personal-context.db"
    repo = PersonalContextRepository(db_path, key_protector=memory_protector)
    repo.create_provisional_profile()
    repo.close()
    memory_protector.clear_without_authorization()

    with pytest.raises(ProfileLockedError, match="key material is unavailable"):
        PersonalContextRepository(db_path, key_protector=memory_protector)
    assert memory_protector.is_empty


def test_repeated_reads_close_every_operation_connection(
    tmp_path, memory_protector, record_factory, monkeypatch
) -> None:
    db_path = tmp_path / "personal-context.db"
    repo = PersonalContextRepository(db_path, key_protector=memory_protector)
    manifest = repo.create_provisional_profile()
    record = record_factory(manifest.profile_id)
    repo.commit_record_version(record, expected_version_id=None)

    opened = []

    class TrackingConnection(sqlite3.Connection):
        closed = False

        def close(self) -> None:
            self.closed = True
            super().close()

    def tracked_connect(_owner_id, database, **kwargs):
        connection = sqlite3.connect(database, factory=TrackingConnection, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(repository_module, "connect_private_sqlite", tracked_connect)
    for _ in range(3):
        assert repo.get_manifest() == manifest
        assert repo.get_record(record.record_id) == record

    assert opened
    assert all(connection.closed for connection in opened)


def _next_manifest(manifest: ProfileManifest, version_id: str) -> ProfileManifest:
    return manifest.model_copy(
        update={
            "revision": manifest.revision + 1,
            "current_version_id": version_id,
        }
    )


def test_interview_batch_rejects_duplicate_record_ids_without_any_write(
    tmp_path, memory_protector, record_factory
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    record = record_factory(manifest.profile_id, record_id="record-duplicate")

    with pytest.raises(ValueError, match="duplicate record IDs"):
        repo.commit_interview_batch(
            (record, record),
            _next_manifest(manifest, "manifest-version-next"),
            expected_record_versions={record.record_id: None},
            expected_manifest_version=manifest.current_version_id,
        )

    assert repo.get_record(record.record_id) is None
    assert repo.get_manifest() == manifest


def test_interview_batch_stale_head_rolls_back_every_record_manifest_and_outbox(
    tmp_path, memory_protector, record_factory
) -> None:
    repo = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    current = record_factory(manifest.profile_id)
    repo.commit_record_version(current, expected_version_id=None)
    update = record_factory(
        manifest.profile_id,
        version_id="record-version-next",
        parent_version_id=current.version_id,
        value="updated",
    )
    created = record_factory(
        manifest.profile_id,
        record_id="record-created",
        version_id="record-created-version",
    )

    with pytest.raises(ConcurrentProfileUpdateError):
        repo.commit_interview_batch(
            (created, update),
            _next_manifest(manifest, "manifest-version-next"),
            expected_record_versions={
                current.record_id: "stale-version",
                created.record_id: None,
            },
            expected_manifest_version=manifest.current_version_id,
        )

    assert repo.get_record(current.record_id) == current
    assert repo.get_record(created.record_id) is None
    assert repo.get_manifest() == manifest
    with repo._connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM encrypted_outbox WHERE object_type = 'record'"
            ).fetchone()[0]
            == 0
        )


# TASK-40133: the repository names its key's keyring account (schema v9).
# Every keyring here is the in-process `SecureKeyring` dict (or the in-memory
# protector); `Tests/conftest.py` also pins PYTHON_KEYRING_BACKEND to the null
# backend, which `KeyringProfileKeyProtector` refuses, so nothing reaches the OS.

_KEYRING_SERVICE = "tldw_chatbook.personal_context"
_V8_PROFILE_META = """
    CREATE TABLE profile_meta (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        profile_id TEXT NOT NULL,
        purge_generation INTEGER NOT NULL,
        destroyed INTEGER NOT NULL DEFAULT 0,
        current_manifest_version TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """


class _FailDeleteOnceKeyring(SecureKeyring):
    __module__ = "keyring.backends.macOS"

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_delete = False

    def delete_password(self, service: str, username: str) -> None:
        if self.fail_next_delete:
            self.fail_next_delete = False
            raise RuntimeError("temporary keyring failure")
        super().delete_password(service, username)


def _path_account(db_path) -> str:
    return (
        "personal-context:"
        + hashlib.sha256(os.fsencode(str(Path(db_path).absolute()))).hexdigest()
    )


def _stored_account(db_path) -> str | None:
    """The stored reference, or None where the schema has no such column."""

    with closing(sqlite3.connect(db_path)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(profile_meta)")}
        if "key_account" not in columns:
            return None
        row = connection.execute("SELECT key_account FROM profile_meta").fetchone()
    return None if row is None else row[0]


def _set_stored_account(db_path, value: str) -> None:
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.execute("UPDATE profile_meta SET key_account = ?", (value,))


def _keyring_repo(db_path, backend) -> PersonalContextRepository:
    return PersonalContextRepository(
        db_path, key_protector=KeyringProfileKeyProtector(backend)
    )


def _opened(db_path, backend) -> PersonalContextRepository | None:
    try:
        return _keyring_repo(db_path, backend)
    except ProfileLockedError:
        return None


def _profile(db_path, backend, record_factory) -> list[str]:
    repo = _keyring_repo(db_path, backend)
    manifest = repo.create_provisional_profile()
    repo.commit_record_version(
        record_factory(manifest.profile_id), expected_version_id=None
    )
    return [record.record_id for record in repo.list_records()]


def _as_v8(db_path, backend) -> None:
    """Make a real v8 database: v8's catalog, its key under the path account."""

    account = _stored_account(db_path)
    protector = KeyringProfileKeyProtector(backend)
    if account:
        protector.replace(_path_account(db_path), protector.load(account))
        protector.delete(account)
    with closing(sqlite3.connect(db_path)) as connection, connection:
        if account is not None:
            connection.execute("ALTER TABLE profile_meta RENAME TO profile_meta_v9")
            connection.execute(_V8_PROFILE_META)
            connection.execute(
                "INSERT INTO profile_meta SELECT singleton, profile_id, "
                "purge_generation, destroyed, current_manifest_version, created_at "
                "FROM profile_meta_v9"
            )
            connection.execute("DROP TABLE profile_meta_v9")
        connection.execute("UPDATE personal_context_schema SET version = 8")


def _schema_version(db_path) -> int:
    with closing(sqlite3.connect(db_path)) as connection:
        return connection.execute(
            "SELECT version FROM personal_context_schema"
        ).fetchone()[0]


def _move(source, target) -> None:
    target.parent.mkdir(mode=0o700, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        if Path(f"{source}{suffix}").exists():
            os.rename(f"{source}{suffix}", f"{target}{suffix}")


def test_v8_database_stamps_its_path_account_once_its_key_authenticates(
    tmp_path, record_factory
) -> None:
    backend = SecureKeyring()
    db_path = tmp_path / "personal-context.db"
    records = _profile(db_path, backend, record_factory)
    _as_v8(db_path, backend)
    before = dict(backend.values)

    repo = _keyring_repo(db_path, backend)

    assert _schema_version(db_path) == repository_module.SCHEMA_VERSION == 9
    assert _stored_account(db_path) == _path_account(db_path)
    assert [record.record_id for record in repo.list_records()] == records
    assert backend.values == before


def test_v8_database_at_a_moved_path_stays_locked_until_moved_back(
    tmp_path, record_factory
) -> None:
    backend = SecureKeyring()
    original = tmp_path / "original" / "personal-context.db"
    original.parent.mkdir(mode=0o700)
    _profile(original, backend, record_factory)
    _as_v8(original, backend)
    before = dict(backend.values)
    moved = tmp_path / "moved" / "personal-context.db"
    _move(original, moved)

    assert _opened(moved, backend) is None
    # The migration commits before any key is read: current, but unstamped.
    assert _schema_version(moved) == 9
    assert _stored_account(moved) == ""

    _move(moved, original)
    reopened = _opened(original, backend)
    assert reopened is not None
    assert _stored_account(original) == _path_account(original)
    assert backend.values == before


def test_new_and_fresh_profiles_get_new_unused_accounts(tmp_path) -> None:
    backend = SecureKeyring()
    db_path = tmp_path / "personal-context.db"
    path_account = _path_account(db_path)
    other = ProfileKeyMaterial(b"o" * 32, b"p" * 32)
    KeyringProfileKeyProtector(backend).replace(path_account, other)
    planted = dict(backend.values)

    repo = _keyring_repo(db_path, backend)
    assert backend.values == planted, "a database with no profile holds no key"

    service = PersonalContextService(repo)
    service.create_profile()
    first = _stored_account(db_path)
    assert first not in {"", path_account}
    assert planted.items() <= backend.values.items()
    assert set(backend.values) == set(planted) | {(_KEYRING_SERVICE, first)}

    service.remove_local_profile(confirm_only_copy=True)
    service.start_fresh_profile()
    fresh = _stored_account(db_path)
    assert fresh not in {"", path_account, first}
    assert set(backend.values) == set(planted) | {(_KEYRING_SERVICE, fresh)}
    assert planted.items() <= backend.values.items()
    reopened = _keyring_repo(db_path, backend)
    assert reopened.get_manifest() == service.get_manifest()


def test_profile_creation_never_adopts_an_occupied_account(
    tmp_path, monkeypatch
) -> None:
    backend = SecureKeyring()
    db_path = tmp_path / "personal-context.db"
    repo = _keyring_repo(db_path, backend)
    occupied = "personal-context:profile_" + "a" * 32
    KeyringProfileKeyProtector(backend).replace(
        occupied, ProfileKeyMaterial(b"o" * 32, b"p" * 32)
    )
    planted = dict(backend.values)
    monkeypatch.setattr(repository_keys.secrets, "token_hex", lambda _n: "a" * 32)

    with pytest.raises(ProfileLockedError, match="already in use"):
        repo.create_provisional_profile()

    assert backend.values == planted
    assert repo.get_manifest() is None


@pytest.mark.parametrize("create", ["provisional", "fresh_generation"])
def test_failed_profile_creation_deletes_only_the_account_it_created(
    tmp_path, monkeypatch, create
) -> None:
    backend = SecureKeyring()
    db_path = tmp_path / "personal-context.db"
    repo = _keyring_repo(db_path, backend)
    if create == "fresh_generation":
        service = PersonalContextService(repo)
        service.create_profile()
        service.remove_local_profile(confirm_only_copy=True)
    KeyringProfileKeyProtector(backend).replace(
        _path_account(db_path), ProfileKeyMaterial(b"o" * 32, b"p" * 32)
    )
    planted = dict(backend.values)
    before = _stored_account(db_path)

    def fail(*_args, **_kwargs):
        raise sqlite3.OperationalError("injected creation failure")

    monkeypatch.setattr(repo, "_insert_outbox", fail)
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        if create == "provisional":
            repo.create_provisional_profile()
        else:
            PersonalContextService(repo).start_fresh_profile()

    assert backend.values == planted
    assert _stored_account(db_path) == before


def test_v9_database_moves_with_its_account_and_v8_does_not(
    tmp_path, record_factory
) -> None:
    backend = SecureKeyring()
    v9 = tmp_path / "v9" / "personal-context.db"
    v9.parent.mkdir(mode=0o700)
    records = _profile(v9, backend, record_factory)
    v8 = tmp_path / "v8" / "personal-context.db"
    v8.parent.mkdir(mode=0o700)
    _profile(v8, backend, record_factory)
    _as_v8(v8, backend)
    before = dict(backend.values)

    _move(v9, tmp_path / "v9-moved" / "personal-context.db")
    _move(v8, tmp_path / "v8-moved" / "personal-context.db")

    moved = _opened(tmp_path / "v9-moved" / "personal-context.db", backend)
    assert moved is not None
    assert [record.record_id for record in moved.list_records()] == records
    assert _opened(tmp_path / "v8-moved" / "personal-context.db", backend) is None
    assert backend.values == before


@pytest.mark.parametrize(
    "stored",
    [
        "recovery:setup_required:personal_context_key_not_in_backup",
        "not an account",
        "personal-context:profile_" + "f" * 32,
        "other database's account",
    ],
)
def test_a_stored_account_that_does_not_authenticate_leaves_the_profile_locked(
    tmp_path, record_factory, stored
) -> None:
    backend = SecureKeyring()
    db_path = tmp_path / "a" / "personal-context.db"
    db_path.parent.mkdir(mode=0o700)
    _profile(db_path, backend, record_factory)
    other_path = tmp_path / "b" / "personal-context.db"
    other_path.parent.mkdir(mode=0o700)
    _profile(other_path, backend, record_factory)
    protector = KeyringProfileKeyProtector(backend)
    if stored == "other database's account":
        stored = _stored_account(other_path)
        # Unrelated keys share key_version 1: only authentication tells them apart.
        assert protector.load(stored).key_version == 1
        assert protector.load(_stored_account(db_path)).key_version == 1
    elif not stored.startswith("personal-context:"):
        # Malformed names never reach the keyring, even holding the right key.
        backend.values[(_KEYRING_SERVICE, stored)] = backend.values[
            (_KEYRING_SERVICE, _stored_account(db_path))
        ]
    _set_stored_account(db_path, stored)
    before = dict(backend.values)

    with pytest.raises(ProfileLockedError):
        _keyring_repo(db_path, backend)

    assert backend.values == before
    assert _stored_account(db_path) == stored


def test_unstamped_database_whose_path_account_holds_another_key_stays_locked(
    tmp_path, record_factory
) -> None:
    backend = SecureKeyring()
    db_path = tmp_path / "a" / "personal-context.db"
    db_path.parent.mkdir(mode=0o700)
    _profile(db_path, backend, record_factory)
    _as_v8(db_path, backend)
    other_path = tmp_path / "b" / "personal-context.db"
    other_path.parent.mkdir(mode=0o700)
    _profile(other_path, backend, record_factory)
    protector = KeyringProfileKeyProtector(backend)
    other_account = _stored_account(other_path) or _path_account(other_path)
    protector.replace(_path_account(db_path), protector.load(other_account))
    before = dict(backend.values)

    with pytest.raises(ProfileLockedError):
        _keyring_repo(db_path, backend)

    assert backend.values == before
    assert _stored_account(db_path) == ""


def _link(repo: PersonalContextRepository) -> None:
    local_manifest = repo.get_manifest()
    local_scopes = tuple(repo.list_scopes())
    remote = reconciliation_tests._snapshot(
        scopes=(
            reconciliation_tests._scope(
                "profile-server", "scope-server-global", ScopeKind.GLOBAL
            ),
        ),
        records=(),
    )
    plan = build_reconciliation_plan(
        local_manifest=local_manifest,
        local_scopes=local_scopes,
        local_records=tuple(repo.list_records()),
        local_proposals=(),
        remote=remote,
        local_workspace_bindings={},
    )
    reconciliation_tests._freeze(repo, plan)
    repo.apply_reviewed_link(
        plan=plan, remote=remote, decisions={}, integrity_key=b"s" * 32
    )


def test_link_and_remove_act_only_on_the_account_this_database_names(
    tmp_path,
) -> None:
    backend = _FailDeleteOnceKeyring()
    db_path = tmp_path / "a" / "personal-context.db"
    db_path.parent.mkdir(mode=0o700)
    other_path = tmp_path / "b" / "personal-context.db"
    other_path.parent.mkdir(mode=0o700)
    service = PersonalContextService(_keyring_repo(db_path, backend))
    service.create_profile()
    PersonalContextService(_keyring_repo(other_path, backend)).create_profile()
    account = _stored_account(db_path)
    other = {
        key: value for key, value in backend.values.items() if key[1] != account
    }

    _link(service._repository)

    linked = KeyringProfileKeyProtector(backend).load(account)
    assert linked.integrity_key == b"s" * 32 and linked.key_version == 2
    assert other.items() <= backend.values.items()

    backend.fail_next_delete = True
    with pytest.raises(ProfileLockedError):
        service.remove_local_profile(confirm_only_copy=True)
    assert _stored_account(db_path) == account, "Remove records the account"
    assert (_KEYRING_SERVICE, account) in backend.values

    service.finish_secure_removal()

    assert _stored_account(db_path) == ""
    assert backend.values == other


def test_a_stale_handle_cannot_remove_an_account_it_never_authenticated(
    tmp_path,
) -> None:
    backend = SecureKeyring()
    db_path = tmp_path / "personal-context.db"
    stale = _keyring_repo(db_path, backend)
    stale.create_provisional_profile()
    current = PersonalContextService(_keyring_repo(db_path, backend))
    current.remove_local_profile(confirm_only_copy=True)
    current.start_fresh_profile()
    fresh_account = _stored_account(db_path)
    before = dict(backend.values)

    with pytest.raises(ProfileLockedError):
        stale.destroy_profile_content()

    assert backend.values == before
    assert _stored_account(db_path) == fresh_account
    assert current.get_manifest() is not None


def test_start_fresh_link_and_remove_leave_another_databases_path_entry(
    tmp_path,
) -> None:
    backend = SecureKeyring()
    db_path = tmp_path / "personal-context.db"
    service = PersonalContextService(_keyring_repo(db_path, backend))
    service.create_profile()
    service.remove_local_profile(confirm_only_copy=True)
    path_entry = (_KEYRING_SERVICE, _path_account(db_path))
    KeyringProfileKeyProtector(backend).replace(
        path_entry[1], ProfileKeyMaterial(b"o" * 32, b"p" * 32)
    )
    planted = backend.values[path_entry]

    service.start_fresh_profile()
    _link(service._repository)
    service.remove_local_profile(confirm_only_copy=True)

    assert backend.values.get(path_entry) == planted
    assert set(backend.values) == {path_entry}


# Fresh schemas the shipped repository versions built. Statements unchanged
# since are taken from the current constructor; the digests pin each set to
# the commit that shipped it (named in Personal_Context/recovery.py).
_V1_OUTBOX = """
    CREATE TABLE encrypted_outbox (
        outbox_id TEXT PRIMARY KEY,
        object_type TEXT NOT NULL,
        object_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        envelope_version TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
_V5_FREEZE = """
    CREATE TABLE first_link_freeze (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        plan_id TEXT NOT NULL,
        snapshot_token TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
_V7_REBASELINE = """
    CREATE TABLE IF NOT EXISTS first_link_rebaseline_commit (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        plan_id TEXT NOT NULL,
        target_profile_id TEXT NOT NULL,
        target_integrity_key_id TEXT NOT NULL,
        target_purge_generation INTEGER NOT NULL,
        rebaseline_version INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """
_HISTORICAL_FRESH_DIGESTS = {
    1: "02785f2d966ad81ca801f53a7e31a297af34d82b82ff27ebc17ab6db1b5e2a54",
    2: "8a8bc3e1db4e1cfab516b45253d7f08b2ac39f8a61c22ce22f8bf35b3d5aa5be",
    4: "84e772ebd12d7073aad036aad99532335de2e540a017a6fa30a53df1b2259f61",
    5: "6e053e3428be6b6f5e217f5897788ababc55c2b7abb49711475cddb802ba43d3",
    6: "bed1527022eb15de2b8c1cb8b38c86d8c371d5efb9b332e15eda2fb5a4ec66e2",
    7: "73e2d103ca41e90d923791f953a8ab9dbee3ac55a9b0a2f9223b615bd4a46783",
    8: "e657b589a255c0fd6d38e31f03f049a4faa4001edd807b35619cf27ca082c1bd",
}


def _historical_fresh_schema(version: int) -> list[str]:
    current = {
        re.search(r"CREATE \w+ (?:IF NOT EXISTS )?(\w+)", statement).group(1): statement
        for statement in repository_module._SCHEMA_STATEMENTS
    }
    current["profile_meta"] = _V8_PROFILE_META
    names = [
        "personal_context_schema",
        "profile_meta",
        "encrypted_objects",
        "object_heads",
        "local_runtime_policy",
        "local_scope_bindings",
        "encrypted_outbox",
        "quarantine",
        "quarantine_object_idx",
    ]
    if version >= 2:
        names += ["local_undo", "local_record_links"]
    if version >= 5:
        names.append("first_link_freeze")
    if version >= 6:
        names.append("local_unlinked_scopes")
    if version >= 7:
        names.append("first_link_rebaseline_commit")
    if version < 4:
        current["encrypted_outbox"] = _V1_OUTBOX
    if version < 7:
        current["first_link_freeze"] = _V5_FREEZE
    if version < 8:
        current["first_link_rebaseline_commit"] = _V7_REBASELINE
    return [current[name] for name in names]


def _catalog(db_path) -> tuple[str, ...]:
    with closing(sqlite3.connect(db_path)) as connection:
        return tuple(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL "
                "ORDER BY type, name"
            )
        )


def test_every_reachable_v9_catalog_is_listed_and_built_by_the_repository(
    tmp_path,
) -> None:
    from tldw_chatbook.Personal_Context.recovery import REPOSITORY_SCHEMAS

    fresh = tmp_path / "fresh.db"
    PersonalContextRepository(fresh, key_protector=InMemoryProfileKeyProtector())
    built = {"fresh": _catalog(fresh)}
    for version, digest in _HISTORICAL_FRESH_DIGESTS.items():
        statements = _historical_fresh_schema(version)
        assert hashlib.sha256(
            "\x00".join(sorted(statements)).encode()
        ).hexdigest() == digest, version
        db_path = tmp_path / f"v{version}.db"
        with closing(sqlite3.connect(db_path)) as connection, connection:
            for statement in statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO personal_context_schema(singleton, version) "
                "VALUES (1, ?)",
                (version,),
            )
        os.chmod(db_path, 0o600)
        PersonalContextRepository(db_path, key_protector=InMemoryProfileKeyProtector())
        assert _schema_version(db_path) == 9
        built[f"v{version}"] = _catalog(db_path)

    listed = [schema for version, schema in REPOSITORY_SCHEMAS]
    assert {version for version, _ in REPOSITORY_SCHEMAS} == {9}
    assert len(listed) == len(set(listed))
    assert set(built.values()) == set(listed)
