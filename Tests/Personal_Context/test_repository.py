from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from tldw_profile_core import (
    ProfileManifest,
    ProfileScope,
    ProposalState,
    ScopeKind,
    SyncMode,
)

import tldw_chatbook.Personal_Context.repository as repository_module
from tldw_chatbook.Personal_Context.key_protector import (
    InMemoryProfileKeyProtector,
    ProfileKeyMaterial,
    ProfileLockedError,
)
from tldw_chatbook.Personal_Context.repository import (
    ConcurrentProfileUpdateError,
    PersonalContextRepository,
    ProfileAlreadyExistsError,
    ProfileDestroyedError,
    RepositorySchemaError,
)


class _DeliberatelyRacyProtector:
    def __init__(self) -> None:
        self.material: ProfileKeyMaterial | None = None
        self.load_or_create_calls = 0
        self._lock = threading.Lock()
        self._race = threading.Barrier(2)

    def load_or_create(self, _profile_ref: str) -> ProfileKeyMaterial:
        if self.material is not None:
            return self.material
        with self._lock:
            self.load_or_create_calls += 1
            call = self.load_or_create_calls
        candidate = ProfileKeyMaterial(bytes([call]) * 32, bytes([call + 16]) * 32)
        try:
            self._race.wait(timeout=1)
        except threading.BrokenBarrierError:
            pass
        self.material = candidate
        return candidate

    def load(self, _profile_ref: str) -> ProfileKeyMaterial:
        if self.material is None:
            raise ProfileLockedError("Profile key material is unavailable.")
        return self.material

    def delete(self, _profile_ref: str) -> None:
        self.material = None


class _TrackingMemoryProtector(InMemoryProfileKeyProtector):
    def __init__(self) -> None:
        super().__init__()
        self.first_material: ProfileKeyMaterial | None = None

    def load_or_create(self, profile_ref: str) -> ProfileKeyMaterial:
        material = super().load_or_create(profile_ref)
        if self.first_material is None:
            self.first_material = material
        return material


class _FailDeleteOnceProtector(InMemoryProfileKeyProtector):
    def __init__(self) -> None:
        super().__init__()
        self.delete_calls = 0

    def delete(self, profile_ref: str) -> None:
        self.delete_calls += 1
        if self.delete_calls == 1:
            raise ProfileLockedError("temporary keyring failure")
        super().delete(profile_ref)


def test_simultaneous_first_openers_share_the_winning_key(tmp_path) -> None:
    db_path = tmp_path / "personal-context.db"
    protector = _DeliberatelyRacyProtector()
    repository_module.connect_private_sqlite(
        "personal_context.repository", db_path, isolation_level=None
    ).close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(PersonalContextRepository, db_path, key_protector=protector)
            for _ in range(2)
        ]
        repositories = [future.result(timeout=5) for future in futures]

    assert protector.load_or_create_calls == 1
    assert repositories[0]._keys == repositories[1]._keys == protector.material
    manifest = repositories[0].create_provisional_profile()
    assert repositories[0].get_manifest() == manifest
    assert repositories[1].get_manifest() == manifest


def test_schema_initialization_failure_keeps_winning_key_for_retry(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "personal-context.db"
    protector = _TrackingMemoryProtector()

    def fail_initialization(*_args, **_kwargs):
        raise sqlite3.OperationalError("simulated schema failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            PersonalContextRepository, "_initialize_schema", fail_initialization
        )
        with pytest.raises(sqlite3.OperationalError, match="simulated schema failure"):
            PersonalContextRepository(db_path, key_protector=protector)

    assert not protector.is_empty
    retried = PersonalContextRepository(db_path, key_protector=protector)
    assert retried._keys == protector.first_material


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
