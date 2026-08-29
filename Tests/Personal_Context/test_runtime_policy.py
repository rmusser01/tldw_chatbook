from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from tldw_chatbook.Personal_Context.bootstrap import bootstrap_personal_context_service
from tldw_chatbook.Personal_Context.key_protector import ProfileLockedError
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.repository import (
    ProfileIntegrityError,
    RepositorySchemaError,
)
from tldw_chatbook.Personal_Context.runtime_policy import (
    AgentAuthority,
    PersonalContextAuthorityError,
)
from tldw_chatbook.Personal_Context.service import PersonalContextService


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def test_runtime_defaults_disabled_then_known_scope_authority_defaults_propose(
    tmp_path, memory_protector
) -> None:
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    service = PersonalContextService(repository, clock=lambda: NOW)
    assert service.status().state == "absent"
    service.create_profile()
    scope = service.list_scopes()[0]

    assert service.status().runtime_enabled is False
    assert service.status().state == "disabled"
    with pytest.raises(PersonalContextAuthorityError) as caught:
        service.require_agent_authority(scope.scope_id, AgentAuthority.READ_ONLY)
    assert caught.value.reason_code == "personal_context_disabled"

    service.set_runtime_enabled(True)
    assert (
        service.require_agent_authority(scope.scope_id, AgentAuthority.READ_ONLY)
        is None
    )
    assert (
        service.require_agent_authority(scope.scope_id, AgentAuthority.PROPOSE) is None
    )
    with pytest.raises(PersonalContextAuthorityError) as caught:
        service.require_agent_authority(scope.scope_id, AgentAuthority.DIRECT_WRITE)
    assert caught.value.reason_code == "agent_authority_denied"


def test_enablement_and_per_scope_authority_are_encrypted_and_fail_closed(
    tmp_path, memory_protector
) -> None:
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    service = PersonalContextService(repository, clock=lambda: NOW)
    service.create_profile()
    scope = service.list_scopes()[0]

    service.set_runtime_enabled(True)
    service.set_scope_authority(scope.scope_id, AgentAuthority.PROPOSE)
    assert service.status().runtime_enabled is True
    assert service.status().state == "ready"
    assert (
        service.require_agent_authority(scope.scope_id, AgentAuthority.READ_ONLY)
        is None
    )
    assert (
        service.require_agent_authority(scope.scope_id, AgentAuthority.PROPOSE) is None
    )
    with pytest.raises(PersonalContextAuthorityError) as caught:
        service.require_agent_authority(scope.scope_id, AgentAuthority.DIRECT_WRITE)
    assert caught.value.reason_code == "agent_authority_denied"

    durable = repository.db_path.read_bytes()
    assert b"direct_write" not in durable
    assert b"propose" not in durable


def test_disabling_preserves_profile_and_user_crud(
    tmp_path, memory_protector, record_factory
) -> None:
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    service = PersonalContextService(repository, clock=lambda: NOW)
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    record = record_factory(manifest.profile_id)
    record = type(record).model_validate(
        {**record.model_dump(mode="python"), "scope_id": scope.scope_id}
    )
    service.create_record(record)
    service.set_runtime_enabled(False)

    assert service.list_records(scope_ids=(scope.scope_id,)) == (record,)


def test_unknown_and_unmapped_workspace_authority_fail_closed(
    tmp_path, memory_protector
) -> None:
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    service = PersonalContextService(repository, clock=lambda: NOW)
    manifest = service.create_profile()
    service.set_runtime_enabled(True)
    with pytest.raises(ValueError, match="scope"):
        service.set_scope_authority("missing-scope", AgentAuthority.READ_ONLY)

    from tldw_profile_core import ProfileScope, ScopeKind

    unmapped = ProfileScope(
        scope_id="unmapped-scope",
        profile_id=manifest.profile_id,
        kind=ScopeKind.WORKSPACE,
        version_id="unmapped-version",
        created_at=NOW,
        updated_at=NOW,
    )
    repository.commit_scope(unmapped)
    with pytest.raises(PersonalContextAuthorityError) as caught:
        service.require_agent_authority(unmapped.scope_id, AgentAuthority.READ_ONLY)
    assert caught.value.reason_code == "scope_unmapped"


class MissingProtector:
    def load_or_create(self, _profile_ref):
        raise AssertionError("locked bootstrap must not create replacement keys")

    def load(self, _profile_ref):
        raise ProfileLockedError("unavailable")

    def delete(self, _profile_ref):
        raise AssertionError


def test_locked_bootstrap_returns_fail_closed_facade_without_key_replacement(
    tmp_path, memory_protector
) -> None:
    db_path = tmp_path / "personal-context.db"
    first = PersonalContextRepository(db_path, key_protector=memory_protector)
    first.create_provisional_profile()

    locked = bootstrap_personal_context_service(
        db_path=db_path, key_protector=MissingProtector()
    )

    assert locked.status().locked is True
    assert locked.status().profile_present is True
    assert locked.status().reason_code == "profile_locked"
    with pytest.raises(ProfileLockedError):
        locked.list_scopes()


class FailingCreationProtector:
    def load_or_create(self, _profile_ref):
        raise ProfileLockedError("unavailable")

    def load(self, _profile_ref):
        raise AssertionError

    def delete(self, _profile_ref):
        raise AssertionError


def test_locked_bootstrap_does_not_claim_an_empty_profile_is_present(tmp_path) -> None:
    locked = bootstrap_personal_context_service(
        db_path=tmp_path / "personal-context.db",
        key_protector=FailingCreationProtector(),
    )

    status = locked.status()
    assert status.locked is True
    assert status.profile_present is False


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        (RepositorySchemaError("unsupported"), "repository_schema_invalid"),
        (ProfileIntegrityError("corrupt"), "profile_integrity_invalid"),
        (sqlite3.IntegrityError("invalid"), "repository_unavailable"),
        (OSError("unavailable"), "repository_unavailable"),
        (ValueError("invalid path"), "repository_unavailable"),
    ],
)
def test_bootstrap_maps_expected_repository_failures_to_locked_status(
    tmp_path, monkeypatch, failure, reason_code
) -> None:
    def fail_repository(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.bootstrap.PersonalContextRepository",
        fail_repository,
    )

    service = bootstrap_personal_context_service(db_path=tmp_path / "profile.db")

    status = service.status()
    assert status.locked is True
    assert status.profile_present is False
    assert status.reason_code == reason_code


def test_bootstrap_rejects_unsupported_and_corrupt_databases_without_mutation(
    tmp_path, memory_protector
) -> None:
    foreign = tmp_path / "foreign.db"
    with sqlite3.connect(foreign) as connection:
        connection.execute("CREATE TABLE foreign_data(value TEXT)")
    foreign_before = foreign.read_bytes()

    unsupported = bootstrap_personal_context_service(
        db_path=foreign, key_protector=memory_protector
    )

    assert unsupported.status().reason_code == "repository_schema_invalid"
    assert foreign.read_bytes() == foreign_before

    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not-a-sqlite-database")
    corrupt_before = corrupt.read_bytes()

    unavailable = bootstrap_personal_context_service(
        db_path=corrupt, key_protector=memory_protector
    )

    assert unavailable.status().reason_code == "repository_unavailable"
    assert corrupt.read_bytes() == corrupt_before


def test_corrupt_encrypted_runtime_policies_fail_closed(
    tmp_path, memory_protector
) -> None:
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    service = PersonalContextService(repository, clock=lambda: NOW)
    service.create_profile()
    scope = service.list_scopes()[0]
    service.set_runtime_enabled(True)

    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE encrypted_objects SET ciphertext = ? "
            "WHERE object_type = 'runtime_policy' AND object_id = ?",
            (b"corrupt", "personal-context-global-policy"),
        )

    status = service.status()
    assert status.runtime_enabled is False
    assert status.reason_code == "runtime_policy_invalid"
    with pytest.raises(PersonalContextAuthorityError) as caught:
        service.require_agent_authority(scope.scope_id, AgentAuthority.READ_ONLY)
    assert caught.value.reason_code == "runtime_policy_invalid"


def test_corrupt_scope_authority_fails_closed(tmp_path, memory_protector) -> None:
    repository = PersonalContextRepository(
        tmp_path / "personal-context.db", key_protector=memory_protector
    )
    service = PersonalContextService(repository, clock=lambda: NOW)
    service.create_profile()
    scope = service.list_scopes()[0]
    service.set_runtime_enabled(True)
    service.set_scope_authority(scope.scope_id, AgentAuthority.PROPOSE)

    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE encrypted_objects SET ciphertext = ? "
            "WHERE object_type = 'runtime_policy' AND object_id = ?",
            (b"corrupt", scope.scope_id),
        )

    with pytest.raises(PersonalContextAuthorityError) as caught:
        service.require_agent_authority(scope.scope_id, AgentAuthority.READ_ONLY)
    assert caught.value.reason_code == "agent_authority_denied"
