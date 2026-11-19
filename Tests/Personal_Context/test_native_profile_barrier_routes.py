"""Barrier integration through genuine native profile owners."""

import pytest

from Tests.Personal_Context.native_barrier_helpers import install_v2_manifest, sql_state


@pytest.mark.parametrize("bootstrap", [False, True])
def test_current_service_and_bootstrap_report_present_generic_unavailable(
    tmp_path, memory_protector, bootstrap
):
    from tldw_chatbook.Personal_Context.bootstrap import (
        bootstrap_personal_context_service,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
    from tldw_chatbook.Personal_Context.service import (
        PersonalContextService,
        ProfileOperationalState,
    )

    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    manifest = repo.create_provisional_profile()
    service = PersonalContextService(repo)
    service.set_runtime_enabled(True)
    assert service.status().state is ProfileOperationalState.READY
    if bootstrap:
        assert (
            bootstrap_personal_context_service(
                db_path=repo.db_path, key_protector=memory_protector
            )
            .status()
            .state
            is ProfileOperationalState.READY
        )
    install_v2_manifest(repo)
    before = sql_state(repo)
    if bootstrap:
        service = bootstrap_personal_context_service(
            db_path=repo.db_path, key_protector=memory_protector
        )
        assert service._repository is None
    status = service.status()
    assert status.state is ProfileOperationalState.LOCKED
    assert status.profile_present and status.locked and not status.runtime_enabled
    assert status.reason_code == "personal_context_compatibility_unavailable"
    assert manifest.profile_id not in repr(status)
    assert sql_state(repo) == before
    if not bootstrap:
        # Synthetic repair demonstrates fresh status; it is not a production V2 cutover.
        with repo._transaction() as connection:
            connection.execute(
                "UPDATE profile_meta SET current_manifest_version=?",
                (manifest.current_version_id,),
            )
            connection.execute(
                "UPDATE object_heads SET version_id=? WHERE object_type='manifest'",
                (manifest.current_version_id,),
            )
        assert service.status().state is ProfileOperationalState.READY


@pytest.fixture
def healthy_native_service(tmp_path, memory_protector):
    from datetime import UTC, datetime

    from tldw_profile_core import PreferencePayload

    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
    from tldw_chatbook.Personal_Context.runtime_policy import AgentAuthority
    from tldw_chatbook.Personal_Context.service import PersonalContextService

    repo = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=memory_protector
    )
    service = PersonalContextService(
        repo, clock=lambda: datetime(2026, 8, 30, 12, tzinfo=UTC)
    )
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    service.set_runtime_enabled(True)
    service.set_scope_authority(scope.scope_id, AgentAuthority.PROPOSE)
    record = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="native-route-canary"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    return repo, service, manifest, scope, record


def test_prepared_context_and_profile_tool_catalog_deny_current_v2(
    healthy_native_service,
):
    from Tests.Agents.test_profile_tool_provider import _bind_provider
    from tldw_chatbook.Personal_Context.context_service import (
        ProfileContextRequest,
        ProfileContextService,
    )
    from tldw_chatbook.Personal_Context.runtime_policy import AgentAuthority

    repo, service, manifest, scope, record = healthy_native_service
    builder = ProfileContextService(service, clock=service.clock)
    request = ProfileContextRequest(
        current_user_text="response detail", available_input_tokens=20000
    )
    result = builder.build_explained_snapshot(request)
    assert "native-route-canary" in result.snapshot.serialized_block
    assert builder.explanation_is_current(result.explanation, request)
    provider = _bind_provider(service, manifest, scope, AgentAuthority.PROPOSE)
    assert provider.list_catalog()
    positive = provider.invoke("profile_get", {"record_id": record.record_id})
    assert positive.ok and "native-route-canary" in positive.content
    install_v2_manifest(repo)
    before = sql_state(repo)
    assert not builder.explanation_is_current(result.explanation, request)
    blocked = builder.build_explained_snapshot(request)
    assert not blocked.snapshot.serialized_block
    assert not blocked.explanation.rows
    assert provider.list_catalog() == []
    denied = provider.invoke("profile_get", {"record_id": record.record_id})
    assert not denied.ok
    for canary in ("native-route-canary", manifest.profile_id, record.record_id):
        assert canary not in denied.content
    assert sql_state(repo) == before


def test_settings_and_proposal_owners_deny_current_v2(healthy_native_service):
    from tldw_profile_core import ActorType, PreferencePayload, ProfileProposeRequest

    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.runtime_policy import (
        PersonalContextAuthorityError,
    )
    from tldw_chatbook.Personal_Context.service import ProfileOperationalState

    repo, service, manifest, scope, _record = healthy_native_service
    proposals = service.proposal_service()
    request = ProfileProposeRequest(
        operation="create",
        proposed_payload=PreferencePayload(
            subject="separate.preference", polarity="like", value="brief"
        ),
    )

    def create():
        return proposals.create(
            request,
            profile_id=manifest.profile_id,
            scope_id=scope.scope_id,
            turn_id="turn-1",
            session_id="session-1",
        )

    proposal = create()
    assert proposal.proposed_record is not None
    assert service.settings_snapshot().records
    install_v2_manifest(repo)
    before = sql_state(repo)
    settings = service.settings_snapshot()
    assert settings.status.state is ProfileOperationalState.LOCKED
    assert settings.records == settings.scopes == settings.proposals == ()
    for action in (
        create,
        lambda: proposals.accept(proposal.proposal_id, user_actor=ActorType.USER),
    ):
        with pytest.raises((ProfileCompatibilityError, PersonalContextAuthorityError)):
            action()
    assert sql_state(repo) == before


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_interview_admission_denies_before_configured_provider(
    healthy_native_service, mode
):
    from Tests.Personal_Context.test_interview_coordinator import (
        _coordinator,
        _ProviderSpy,
    )
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )

    repo, service, _manifest, scope, _record = healthy_native_service
    spy = _ProviderSpy()
    coordinator = _coordinator(adaptive=spy, service=service)
    coordinator.start(kind="personal", scope_id=scope.scope_id, mode=mode)
    assert len(spy.calls) == (1 if mode == "adaptive" else 0)
    install_v2_manifest(repo)
    before = sql_state(repo)
    prior_calls = len(spy.calls)
    with pytest.raises(ProfileCompatibilityError):
        coordinator.start(kind="personal", scope_id=scope.scope_id, mode=mode)
    assert len(spy.calls) == prior_calls
    assert sql_state(repo) == before


@pytest.mark.parametrize("recovery", [False, True])
def test_exports_deny_before_destination_write(
    healthy_native_service, tmp_path, recovery
):
    from tldw_chatbook.Personal_Context.export_service import (
        ExportRequest,
        RecoveryExportRequest,
    )
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )

    repo, service, _manifest, _scope, _record = healthy_native_service
    positive = tmp_path / "positive-profile.json"
    blocked = tmp_path / "blocked-profile.json"

    def export(destination):
        if recovery:
            return service.export_recovery(
                RecoveryExportRequest(
                    destination=destination, passphrase="synthetic passphrase"
                )
            )
        return service.export_plaintext(
            ExportRequest(destination=destination, confirm_plaintext=True)
        )

    export(positive)
    assert positive.exists()
    install_v2_manifest(repo)
    before = sql_state(repo)
    with pytest.raises(ProfileCompatibilityError):
        export(blocked)
    assert not blocked.exists()
    assert sql_state(repo) == before


def test_sync_dispatcher_does_not_stage_or_acknowledge_blocked_profile(tmp_path):
    from tldw_profile_core import PreferencePayload

    from Tests.Sync_Interop.test_personal_context_dispatcher import (
        SCOPE,
        STORAGE_KEY,
        _dependencies,
        _dispatcher,
    )
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )

    outbox, sync, adapter, service = _dependencies(tmp_path)
    dispatcher = _dispatcher(outbox, sync, adapter)
    positive = dispatcher.dispatch_pending(
        device_id="device-1", storage_key=STORAGE_KEY, **SCOPE
    )
    assert positive["dispatched"] > 0
    scope = service.list_scopes()[0]
    service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="canary"
        ),
        semantic_key=None,
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    repo = service._repository
    assert outbox.list_pending()
    install_v2_manifest(repo)
    before = sql_state(repo)
    staged = sync.list_pending_sync_v2_outbox_envelopes(**SCOPE)
    with pytest.raises(ProfileCompatibilityError):
        dispatcher.dispatch_pending(
            device_id="device-1", storage_key=STORAGE_KEY, **SCOPE
        )
    assert sync.list_pending_sync_v2_outbox_envelopes(**SCOPE) == staged
    assert sql_state(repo) == before


_WRITE_ROUTES = [
    "manifest",
    "record",
    "record_manifest",
    "interview",
    "split",
    "scope",
    "scope_binding",
    "proposal",
    "synced_proposal",
    "expiry",
    "resolution",
    "acceptance",
    "runtime",
    "binding",
    "outbox",
    "ack",
    "quarantine_outbox",
    "quarantine_object",
]


@pytest.mark.parametrize("route", _WRITE_ROUTES)
def test_real_v1_write_routes_then_current_v2_deny_without_side_effects(
    healthy_native_service, proposal_factory, route
):
    from tldw_profile_core import (
        ActorType,
        ProfileProposal,
        ProfileScope,
        ProposalState,
        RecordState,
        ScopeKind,
        SyncMode,
    )

    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )

    repo, service, _original_manifest, scope, record = healthy_native_service
    manifest = repo.get_manifest()
    successor = manifest.model_copy(
        update={
            "revision": manifest.revision + 1,
            "current_version_id": "next-manifest",
            "updated_at": service.clock(),
        }
    )
    updated = record.model_copy(
        update={"version_id": "next-record", "parent_version_id": record.version_id}
    )
    pending = proposal_factory(manifest.profile_id)
    nested = pending.proposed_record.model_copy(
        update={
            "scope_id": scope.scope_id,
            "semantic_key": None,
            "payload": pending.proposed_record.payload.model_copy(
                update={"subject": "separate.preference"}
            ),
        }
    )
    pending = ProfileProposal.model_validate(
        {
            **pending.model_dump(mode="python"),
            "scope_id": scope.scope_id,
            "proposed_record": nested,
        }
    )
    if route in {"resolution", "acceptance"}:
        repo.commit_proposal(pending)
    outbox_id = repo.list_pending_outbox()[0]["outbox_id"]
    workspace = ProfileScope(
        scope_id="new-workspace",
        profile_id=manifest.profile_id,
        kind=ScopeKind.WORKSPACE,
        version_id="workspace-v1",
        created_at=service.clock(),
        updated_at=service.clock(),
    )
    binding = {
        "version": 1,
        "local_workspace_id": "synthetic-workspace",
        "label": "Synthetic",
    }
    tombstone = updated.model_copy(
        update={"state": RecordState.DELETED, "payload": None, "semantic_key": None}
    )
    private = record.model_copy(
        update={
            "record_id": "private-record",
            "version_id": "private-v1",
            "parent_version_id": None,
            "controls": record.controls.model_copy(
                update={"sync_mode": SyncMode.DEVICE_ONLY}
            ),
        }
    )
    approved = nested.model_copy(
        update={
            "provenance": pending.provenance.model_copy(
                update={
                    "actor": ActorType.USER,
                    "reason_code": "user_approved_agent_proposal",
                }
            )
        }
    )

    def action():
        if route == "manifest":
            return repo.commit_manifest_version(
                successor, expected_version_id=manifest.current_version_id
            )
        if route == "record":
            return repo.commit_record_version(
                updated, expected_version_id=record.version_id
            )
        if route == "record_manifest":
            return repo.commit_record_and_manifest(
                updated,
                successor,
                expected_record_version=record.version_id,
                expected_manifest_version=manifest.current_version_id,
            )
        if route == "interview":
            return repo.commit_interview_batch(
                (updated,),
                successor,
                expected_record_versions={record.record_id: record.version_id},
                expected_manifest_version=manifest.current_version_id,
            )
        if route == "split":
            return repo.commit_device_only_split(
                tombstone,
                private,
                successor,
                expected_record_version=record.version_id,
                expected_manifest_version=manifest.current_version_id,
            )
        if route == "scope":
            return repo.commit_scope(workspace)
        if route == "scope_binding":
            return repo.commit_scope_with_binding(workspace, binding)
        if route == "proposal":
            return repo.commit_proposal(pending)
        if route == "synced_proposal":
            return repo.commit_synced_proposal(pending)
        if route == "expiry":
            return repo.expire_due_proposals(service.clock())
        if route == "resolution":
            return repo.resolve_proposal(pending.proposal_id, ProposalState.REJECTED)
        if route == "acceptance":
            return repo.accept_proposal_and_record(
                pending.proposal_id,
                approved,
                successor,
                expected_record_version=None,
                expected_manifest_version=manifest.current_version_id,
            )
        if route == "runtime":
            return repo.commit_runtime_policy(
                "synthetic-policy", {"version": 1, "enabled": False}
            )
        if route == "binding":
            return repo.commit_scope_binding(workspace.scope_id, binding)
        if route == "outbox":
            return repo.commit_outbox_body(
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                body={"synthetic": "opaque-v1"},
            )
        if route == "ack":
            return repo.acknowledge_outbox(outbox_id, "synthetic-receipt")
        if route == "quarantine_outbox":
            return repo.quarantine_outbox(outbox_id, "synthetic-failure")
        return repo.quarantine_object(
            "record", record.record_id, record.version_id, "synthetic-failure"
        )

    action()  # Same real public native path succeeds before the whole-profile barrier.
    install_v2_manifest(repo)
    before = sql_state(repo)
    with pytest.raises(ProfileCompatibilityError):
        action()
    assert sql_state(repo) == before


@pytest.mark.parametrize("fail_after", [1, 2])
def test_multi_object_write_failure_rolls_back_all_sql_rows(
    healthy_native_service, monkeypatch, fail_after
):
    repo, _service, _, _, record = healthy_native_service
    manifest = repo.get_manifest()
    successor = manifest.model_copy(
        update={
            "revision": manifest.revision + 1,
            "current_version_id": "rollback-next-manifest",
        }
    )
    updated = record.model_copy(
        update={
            "parent_version_id": record.version_id,
            "version_id": "rollback-next-record",
        }
    )
    original = repo._insert_encrypted
    inserts = []

    def fail(connection, **kwargs):
        original(connection, **kwargs)
        inserts.append(kwargs["object_type"])
        if len(inserts) == fail_after:
            raise RuntimeError("synthetic participating insert failure")

    before = sql_state(repo)
    monkeypatch.setattr(repo, "_insert_encrypted", fail)
    with pytest.raises(RuntimeError, match="synthetic participating insert failure"):
        repo.commit_record_and_manifest(
            updated,
            successor,
            expected_record_version=record.version_id,
            expected_manifest_version=manifest.current_version_id,
        )
    assert len(inserts) == fail_after
    assert sql_state(repo) == before
    monkeypatch.setattr(repo, "_insert_encrypted", original)
    repo.commit_record_and_manifest(
        updated,
        successor,
        expected_record_version=record.version_id,
        expected_manifest_version=manifest.current_version_id,
    )
    assert repo.get_record(record.record_id) == updated
    assert repo.get_manifest() == successor


def test_two_open_repositories_recheck_manifest_when_writer_enters_transaction(
    healthy_native_service, memory_protector, monkeypatch
):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager

    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    repo, _service, _, _, record = healthy_native_service
    peer = PersonalContextRepository(repo.db_path, key_protector=memory_protector)
    assert peer.get_record(record.record_id) == record
    prepared = record.model_copy(
        update={
            "parent_version_id": record.version_id,
            "version_id": "stale-prepared-write",
        }
    )
    entered, release = threading.Event(), threading.Event()
    original = repo._transaction

    @contextmanager
    def delayed():
        entered.set()
        assert release.wait(5)
        with original() as connection:
            yield connection

    monkeypatch.setattr(repo, "_transaction", delayed)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            repo.commit_record_version, prepared, expected_version_id=record.version_id
        )
        assert entered.wait(5)
        try:
            install_v2_manifest(peer)
            before = sql_state(peer)
        finally:
            release.set()
        with pytest.raises(ProfileCompatibilityError):
            future.result(timeout=5)
    assert sql_state(peer) == before


@pytest.mark.parametrize("operation", ["freeze", "apply"])
def test_first_link_current_guard_preserves_v1_and_blocks_v2_before_custody(
    tmp_path, memory_protector, operation
):
    from tldw_profile_core import ScopeKind

    from Tests.Personal_Context.test_profile_reconciliation import (
        _freeze,
        _manifest,
        _scope,
        _snapshot,
    )
    from tldw_chatbook.Personal_Context.native_compatibility import (
        ProfileCompatibilityError,
    )
    from tldw_chatbook.Personal_Context.reconciliation import build_reconciliation_plan
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository

    def prepare(name):
        root = tmp_path / name
        root.mkdir()
        repository = PersonalContextRepository(
            root / "profile.db", key_protector=memory_protector
        )
        manifest = _manifest("profile-local", "manifest-local")
        scope = _scope(manifest.profile_id, "scope-local", ScopeKind.GLOBAL)
        repository.create_profile_with_global_scope(manifest, scope)
        remote = _snapshot(
            scopes=(_scope("profile-server", "scope-server", ScopeKind.GLOBAL),),
            records=(),
        )
        plan = build_reconciliation_plan(
            local_manifest=manifest,
            local_scopes=(scope,),
            local_records=(),
            local_proposals=(),
            remote=remote,
            local_workspace_bindings={},
        )
        if operation == "apply":
            _freeze(repository, plan)
        return repository, remote, plan

    def action(repository, remote, plan):
        if operation == "freeze":
            return _freeze(repository, plan)
        return repository.apply_reviewed_link(
            plan=plan, remote=remote, decisions={}, integrity_key=b"s" * 32
        )

    healthy, remote, plan = prepare("healthy")
    action(healthy, remote, plan)
    blocked, remote, plan = prepare("blocked")
    install_v2_manifest(blocked)
    before = sql_state(blocked)
    keys_before = blocked._require_keys()
    with pytest.raises(ProfileCompatibilityError):
        action(blocked, remote, plan)
    assert sql_state(blocked) == before
    assert blocked._require_keys() == keys_before
    assert memory_protector.load(blocked._profile_ref) == keys_before
