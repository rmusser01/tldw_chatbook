from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from tldw_profile_core import PreferencePayload

from tldw_chatbook.Agents.profile_tool_provider import (
    ProfileToolProvider,
    ProfileToolRunScope,
)
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.proposal_service import ProfileProposalQuota
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.runtime_policy import AgentAuthority
from tldw_chatbook.Personal_Context.service import PersonalContextService


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


class Ids:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, label: str) -> str:
        self.value += 1
        return f"{label}-{self.value}"


def _service(tmp_path):
    service = PersonalContextService(
        PersonalContextRepository(
            tmp_path / "personal-context.db",
            key_protector=InMemoryProfileKeyProtector(),
        ),
        clock=lambda: NOW,
        id_factory=Ids(),
    )
    manifest = service.create_profile()
    scope = service.list_scopes()[0]
    service.set_runtime_enabled(True)
    return service, manifest, scope


def _bind_provider(
    service,
    manifest,
    scope,
    authority: AgentAuthority,
    *,
    message_id: str | None = "message-1",
    message_text: str | None = "I prefer concise replies.",
    kill_switch=lambda: False,
    quota: ProfileProposalQuota | None = None,
    run_id: str = "turn-1",
    session_id: str = "session-1",
):
    if service.get_scope_authority(scope.scope_id) is not authority:
        service.set_scope_authority(scope.scope_id, authority)
    view = service.authorized_context_view(
        active_workspace_scope_id=(
            scope.scope_id if scope.kind.value == "workspace" else None
        )
    )
    run_scope = ProfileToolRunScope(
        run_id=run_id,
        session_id=session_id,
        profile_id=manifest.profile_id,
        scope_id=scope.scope_id,
        authority=authority,
        generation=view.generation,
        authority_revision=view.authority_revision,
        current_user_message_id=message_id,
        current_user_text=message_text,
    )
    kwargs = {} if quota is None else {"quota": quota}
    return ProfileToolProvider(
        service, run_scope=run_scope, kill_switch=kill_switch, **kwargs
    )


def _provider(tmp_path, authority: AgentAuthority):
    service, manifest, scope = _service(tmp_path)
    return _bind_provider(service, manifest, scope, authority)


def test_propose_catalog_is_derived_from_captured_and_live_authority(
    tmp_path,
) -> None:
    provider = _provider(tmp_path, AgentAuthority.PROPOSE)

    assert {entry.name for entry in provider.list_catalog()} == {
        "profile_search",
        "profile_get",
        "profile_propose",
    }


def test_workspace_propose_catalog_includes_promotion(tmp_path) -> None:
    service, manifest, _global_scope = _service(tmp_path)
    workspace = service.create_workspace_scope("workspace-1", "Project")
    provider = _bind_provider(service, manifest, workspace, AgentAuthority.PROPOSE)

    assert {entry.name for entry in provider.list_catalog()} == {
        "profile_search",
        "profile_get",
        "profile_propose",
        "profile_promote",
    }


def test_profile_tool_schemas_are_the_shared_core_request_schemas(
    tmp_path,
) -> None:
    provider = _provider(tmp_path, AgentAuthority.PROPOSE)

    schema = provider.load_schema("personal-context:profile_propose")

    assert schema.name == "profile_propose"
    assert schema.parameters["additionalProperties"] is False
    assert "$defs" in schema.parameters


def test_direct_update_schema_pins_the_exact_trusted_message_id(tmp_path) -> None:
    provider = _provider(tmp_path, AgentAuthority.DIRECT_WRITE)

    schema = provider.load_schema("personal-context:profile_update")

    assert schema.parameters["properties"]["current_user_message_id"]["const"] == (
        "message-1"
    )


def test_read_only_and_direct_write_catalogs_follow_live_policy(
    tmp_path,
) -> None:
    service, manifest, scope = _service(tmp_path)
    read_only = _bind_provider(service, manifest, scope, AgentAuthority.READ_ONLY)
    assert {entry.name for entry in read_only.list_catalog()} == {
        "profile_search",
        "profile_get",
    }
    direct_without_message = _bind_provider(
        service,
        manifest,
        scope,
        AgentAuthority.DIRECT_WRITE,
        message_id=None,
        message_text=None,
    )

    assert {entry.name for entry in direct_without_message.list_catalog()} == {
        "profile_search",
        "profile_get",
        "profile_propose",
    }


def test_search_returns_only_canonical_agent_visible_active_records(
    tmp_path,
) -> None:
    service, manifest, scope = _service(tmp_path)
    visible = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="private.detail", polarity="like", value="PRIVATE-CANARY"
        ),
        semantic_key={"namespace": "preference", "subject": "private.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "user_only"},
    )
    archived = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="old.detail", polarity="like", value="ARCHIVED-CANARY"
        ),
        semantic_key={"namespace": "preference", "subject": "old.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    service.archive_record(archived.record_id, expected_version_id=archived.version_id)
    provider = _bind_provider(service, manifest, scope, AgentAuthority.READ_ONLY)

    result = provider.invoke("profile_search", {"query": "detail", "limit": 20})

    payload = json.loads(result.content)
    assert result.ok is True
    assert payload["operation"] == "search"
    assert payload["status"] == "applied"
    assert payload["data"]["records"] == [visible.model_dump(mode="json")]
    assert "CANARY" not in result.content


def test_get_refuses_other_workspace_without_disclosing_the_record(
    tmp_path,
) -> None:
    service, manifest, global_scope = _service(tmp_path)
    workspace = service.create_workspace_scope("workspace-1", "Project")
    other = service.create_workspace_scope("workspace-2", "Other")
    service.set_scope_authority(workspace.scope_id, "read_only")
    hidden = service.create_manual_record(
        scope_id=other.scope_id,
        payload=PreferencePayload(
            subject="private.detail", polarity="like", value="WORKSPACE-CANARY"
        ),
        semantic_key={"namespace": "preference", "subject": "private.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    provider = _bind_provider(service, manifest, workspace, AgentAuthority.READ_ONLY)

    result = provider.invoke("profile_get", {"record_id": hidden.record_id})

    assert result.ok is False
    assert result.error == "permission_denied"
    assert hidden.record_id not in result.error


def test_propose_returns_canonical_receipt_without_changing_context(
    tmp_path,
) -> None:
    service, manifest, scope = _service(tmp_path)
    provider = _bind_provider(service, manifest, scope, AgentAuthority.PROPOSE)

    result = provider.invoke(
        "profile_propose",
        {
            "operation": "create",
            "evidence_span": "I prefer concise replies.",
            "proposed_payload": {
                "kind": "preference",
                "subject": "response.detail",
                "polarity": "like",
                "value": "concise",
            },
        },
    )

    payload = json.loads(result.content)
    assert result.ok is True
    assert payload["operation"] == "propose"
    assert payload["status"] == "proposal_created"
    assert "data" not in payload
    assert "response.detail" not in result.content
    assert "concise" not in result.content
    assert "message-1" not in result.content
    assert service.list_records(scope_ids=(scope.scope_id,)) == ()


def test_direct_update_requires_exact_message_and_case_sensitive_span(
    tmp_path,
) -> None:
    service, manifest, scope = _service(tmp_path)
    current = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="long"
        ),
        semantic_key={"namespace": "preference", "subject": "stable-key"},
        controls={"sync_mode": "device_only", "agent_visibility": "agent_visible"},
    )
    provider = _bind_provider(service, manifest, scope, AgentAuthority.DIRECT_WRITE)
    arguments = {
        "record_id": current.record_id,
        "base_version_id": current.version_id,
        "current_user_message_id": "message-1",
        "evidence_span": "I prefer concise replies.",
        "proposed_payload": {
            "kind": "preference",
            "subject": "changed-by-agent",
            "polarity": "like",
            "value": "concise",
        },
    }

    assert (
        provider.invoke(
            "profile_update", {**arguments, "current_user_message_id": "other"}
        ).error
        == "review_required"
    )
    assert (
        provider.invoke(
            "profile_update",
            {**arguments, "evidence_span": "i prefer concise replies."},
        ).error
        == "review_required"
    )

    result = provider.invoke("profile_update", arguments)
    updated = service.get_record(current.record_id)

    assert result.ok is True
    assert json.loads(result.content)["status"] == "applied"
    assert updated is not None
    assert updated.controls == current.controls
    assert updated.semantic_key == current.semantic_key
    assert updated.provenance.source_references == ("message-1",)
    assert updated.provenance.source_hashes == (
        hashlib.sha256("I prefer concise replies.".encode("utf-8")).hexdigest(),
    )
    assert "I prefer concise replies." not in str(updated.provenance)


def test_private_duplicate_is_generic_review_required(
    tmp_path,
) -> None:
    service, manifest, scope = _service(tmp_path)
    private = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="PRIVATE-CANARY"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "user_only"},
    )
    provider = _bind_provider(service, manifest, scope, AgentAuthority.PROPOSE)

    result = provider.invoke(
        "profile_propose",
        {
            "operation": "create",
            "proposed_payload": {
                "kind": "preference",
                "subject": "response.detail",
                "polarity": "like",
                "value": "agent guess",
            },
        },
    )

    assert result.ok is False
    assert result.error == "review_required"
    assert private.record_id not in result.error
    assert "PRIVATE-CANARY" not in result.error


def test_private_and_missing_mutation_targets_share_one_generic_result(
    tmp_path,
) -> None:
    service, manifest, scope = _service(tmp_path)
    private = service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="PRIVATE-CANARY"
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "user_only"},
    )
    provider = _bind_provider(service, manifest, scope, AgentAuthority.DIRECT_WRITE)
    payload = {
        "kind": "preference",
        "subject": "response.detail",
        "polarity": "like",
        "value": "concise",
    }

    private_proposal = provider.invoke(
        "profile_propose",
        {
            "operation": "update",
            "target_record_id": private.record_id,
            "base_version_id": private.version_id,
            "proposed_payload": payload,
        },
    )
    missing_proposal = provider.invoke(
        "profile_propose",
        {
            "operation": "update",
            "target_record_id": "missing-record",
            "base_version_id": "missing-version",
            "proposed_payload": payload,
        },
    )
    private_direct = provider.invoke(
        "profile_update",
        {
            "record_id": private.record_id,
            "base_version_id": private.version_id,
            "current_user_message_id": "message-1",
            "evidence_span": "I prefer concise replies.",
            "proposed_payload": payload,
        },
    )
    missing_direct = provider.invoke(
        "profile_update",
        {
            "record_id": "missing-record",
            "base_version_id": "missing-version",
            "current_user_message_id": "message-1",
            "evidence_span": "I prefer concise replies.",
            "proposed_payload": payload,
        },
    )

    assert private_proposal.error == missing_proposal.error == "review_required"
    assert private_direct.error == missing_direct.error == "review_required"
    assert "PRIVATE-CANARY" not in private_proposal.error
    assert private.record_id not in private_direct.error


def test_invoke_rechecks_generation_authority_revision_and_kill_switch(
    tmp_path,
) -> None:
    service, manifest, scope = _service(tmp_path)
    killed = False
    provider = _bind_provider(
        service,
        manifest,
        scope,
        AgentAuthority.PROPOSE,
        kill_switch=lambda: killed,
    )
    service.set_scope_authority(scope.scope_id, AgentAuthority.READ_ONLY)

    stale = provider.invoke("profile_search", {"query": "anything"})
    killed = True
    switched_off = provider.invoke("profile_search", {"query": "anything"})

    assert stale.error == "permission_denied"
    assert switched_off.error == "permission_denied"


def test_raising_kill_switch_fails_closed_without_escaping(tmp_path) -> None:
    service, manifest, scope = _service(tmp_path)

    def unreadable_switch() -> bool:
        raise RuntimeError("KILL-SWITCH-CANARY")

    provider = _bind_provider(
        service,
        manifest,
        scope,
        AgentAuthority.PROPOSE,
        kill_switch=unreadable_switch,
    )

    assert provider.list_catalog() == []
    result = provider.invoke("profile_search", {"query": "anything"})
    assert result.ok is False
    assert result.error == "permission_denied"


def test_locked_profile_returns_fixed_profile_locked_status(tmp_path) -> None:
    locked = PersonalContextService.locked(profile_present=True)
    scope = ProfileToolRunScope(
        run_id="turn-1",
        session_id="session-1",
        profile_id="profile-1",
        scope_id="scope-1",
        authority=AgentAuthority.READ_ONLY,
        generation=0,
        authority_revision="revision-1",
    )
    provider = ProfileToolProvider(locked, run_scope=scope)

    result = provider.invoke("profile_search", {"query": "anything"})

    assert result.ok is False
    assert result.error == "profile_locked"


def test_secret_refusal_never_returns_exception_or_input_text(tmp_path) -> None:
    service, manifest, scope = _service(tmp_path)
    provider = _bind_provider(service, manifest, scope, AgentAuthority.PROPOSE)
    canary = "api_key=abcdefghijklmnopqrstuv"

    result = provider.invoke(
        "profile_propose",
        {
            "operation": "create",
            "proposed_payload": {
                "kind": "preference",
                "subject": "secret",
                "polarity": "like",
                "value": canary,
            },
        },
    )

    assert result.ok is False
    assert result.error == "review_required"
    assert canary not in result.error


def test_quota_is_shared_across_fresh_providers_for_one_root_turn(tmp_path) -> None:
    service, manifest, scope = _service(tmp_path)
    quota = ProfileProposalQuota(per_turn=2, per_session=3)
    first = _bind_provider(
        service, manifest, scope, AgentAuthority.PROPOSE, quota=quota
    )
    second = _bind_provider(
        service, manifest, scope, AgentAuthority.PROPOSE, quota=quota
    )

    def propose(provider, subject: str):
        return provider.invoke(
            "profile_propose",
            {
                "operation": "create",
                "proposed_payload": {
                    "kind": "preference",
                    "subject": subject,
                    "polarity": "like",
                    "value": "concise",
                },
            },
        )

    assert propose(first, "response.one").ok is True
    assert propose(second, "response.two").ok is True
    assert propose(first, "response.three").error == "quota_exceeded"


def test_failed_proposal_commit_releases_provider_quota_reservation(
    tmp_path, monkeypatch
) -> None:
    service, manifest, scope = _service(tmp_path)
    quota = ProfileProposalQuota(per_turn=1, per_session=1)
    provider = _bind_provider(
        service, manifest, scope, AgentAuthority.PROPOSE, quota=quota
    )
    original = service._commit_profile_proposal

    def fail_once(_proposal):
        raise RuntimeError("PROPOSAL-COMMIT-CANARY")

    monkeypatch.setattr(service, "_commit_profile_proposal", fail_once)
    arguments = {
        "operation": "create",
        "proposed_payload": {
            "kind": "preference",
            "subject": "response.detail",
            "polarity": "like",
            "value": "concise",
        },
    }

    assert provider.invoke("profile_propose", arguments).error == "review_required"
    monkeypatch.setattr(service, "_commit_profile_proposal", original)
    assert provider.invoke("profile_propose", arguments).ok is True
