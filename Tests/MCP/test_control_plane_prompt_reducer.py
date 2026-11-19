from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.MCP.execution_log import ExecutionRecord, MCPExecutionLog
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.permission_prompt_reducer import build_permission_prompt_report
from tldw_chatbook.MCP.permission_store import (
    EffectiveToolState,
    definition_hash,
    profile_policy_digest,
    profile_policy_chain_digest,
    ProfileMutationError,
)
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)


class _LocalService:
    def __init__(self, store: LocalMCPStore, records: list[dict]) -> None:
        self.store = store
        self._records = records

    def get_external_servers(self) -> list[dict]:
        return list(self._records)

    def get_inventory(self) -> dict:
        return {"tools": []}


class _ChangingCatalogLocalService(_LocalService):
    def __init__(
        self,
        store: LocalMCPStore,
        first_records: list[dict],
        later_records: list[dict],
    ) -> None:
        super().__init__(store, first_records)
        self._later_records = later_records
        self.catalog_calls = 0

    def get_external_servers(self) -> list[dict]:
        self.catalog_calls += 1
        records = self._records if self.catalog_calls == 1 else self._later_records
        return list(records)


def _profile_record(
    *,
    profile_id: str = "docs",
    tool_name: str = "search",
    description: str = "Search docs",
    is_connected: bool = True,
) -> dict:
    return {
        "profile_id": profile_id,
        "name": profile_id,
        "is_connected": is_connected,
        "discovery_snapshot": {
            "tools": [
                {
                    "name": tool_name,
                    "description": description,
                    "inputSchema": {"type": "object"},
                }
            ]
        },
    }


def _service(
    tmp_path: Path, records: list[dict]
) -> tuple[UnifiedMCPControlPlaneService, LocalMCPStore]:
    store = LocalMCPStore(tmp_path / "store.json")
    service = UnifiedMCPControlPlaneService(
        local_service=_LocalService(store, records),
        server_service=None,
        target_store=None,
        context_store=None,
    )
    return service, store


def _append_approved(
    store: LocalMCPStore, *, ts: str, description: str = "Search docs"
) -> None:
    log = MCPExecutionLog(Path(store.path).with_name("mcp_execution_log.jsonl"))
    log.append(
        ExecutionRecord(
            ts=ts,
            server_key="local:docs",
            tool_name="search",
            initiator="agent",
            decision="approved",
            ok=True,
            status="success",
            duration_ms=12,
            error_category=None,
            exception_type=None,
            status_code=None,
            argument_names=(),
            unknown_argument_count=0,
            result_type="dict",
            result_size=1,
            definition_hash=definition_hash(description, {"type": "object"}),
        )
    )


@pytest.mark.asyncio
async def test_permission_prompt_recommendations_uses_log_catalog_and_effective_state(
    tmp_path,
):
    """Catches reports that ignore either approval history or current permission state."""
    service, store = _service(tmp_path, [_profile_record()])
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")

    report = await service.permission_prompt_recommendations()

    assert [
        (r.server_key, r.tool_name, r.approved_count) for r in report.recommendations
    ] == [("local:docs", "search", 2)]
    assert report.recommendations[0].last_seen == "2026-08-01T20:05:00+00:00"


@pytest.mark.asyncio
async def test_apply_permission_prompt_recommendation_persists_hash_safe_allow(
    tmp_path,
):
    """Catches applying a recommendation without using the existing tool-state API."""
    service, store = _service(tmp_path, [_profile_record(description="Search docs")])
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")

    recommendation = await service.apply_permission_prompt_recommendation(
        "local:docs", "search", **_reviewed_acceptance(service)
    )

    assert recommendation.server_key == "local:docs"
    assert recommendation.tool_name == "search"
    tool_entry = service.permission_store.load()["profiles"]["default"]["servers"][
        "local:docs"
    ]["tools"]["search"]
    assert tool_entry["state"] == "allow"
    assert tool_entry["definition_hash"] == definition_hash(
        "Search docs", {"type": "object"}
    )


@pytest.mark.asyncio
async def test_apply_permission_prompt_recommendation_uses_one_catalog_snapshot(
    tmp_path,
):
    """Prevents a changed second catalog read from inheriting older approvals."""
    store = LocalMCPStore(tmp_path / "store.json")
    local_service = _ChangingCatalogLocalService(
        store,
        [_profile_record(description="Search docs")],
        [_profile_record(description="Delete matching docs")],
    )
    service = UnifiedMCPControlPlaneService(
        local_service=local_service,
        server_service=None,
        target_store=None,
        context_store=None,
    )
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")

    await service.apply_permission_prompt_recommendation(
        "local:docs", "search", **_reviewed_acceptance(service)
    )

    tool_entry = service.permission_store.load()["profiles"]["default"]["servers"][
        "local:docs"
    ]["tools"]["search"]
    assert local_service.catalog_calls == 1
    assert tool_entry["definition_hash"] == definition_hash(
        "Search docs", {"type": "object"}
    )


@pytest.mark.asyncio
async def test_apply_permission_prompt_recommendation_rejects_non_recommended_tool(
    tmp_path,
):
    """Catches persisting allows for tools that do not meet recommendation criteria."""
    service, _store = _service(tmp_path, [_profile_record()])

    with pytest.raises(KeyError, match="No prompt-reduction recommendation"):
        await service.apply_permission_prompt_recommendation(
            "local:docs", "search", **_reviewed_acceptance(service)
        )


@pytest.mark.asyncio
async def test_apply_permission_prompt_recommendation_requires_permission_store(
    tmp_path,
):
    """Catches reporting success when the recommended allow cannot persist."""
    service, store = _service(tmp_path, [_profile_record()])
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")
    service._execution_log = MCPExecutionLog(
        Path(store.path).with_name("mcp_execution_log.jsonl")
    )
    service.local_service.store = None

    with pytest.raises(RuntimeError, match="MCP permission store unavailable"):
        await service.apply_permission_prompt_recommendation("local:docs", "search")


@pytest.mark.asyncio
async def test_dismissed_suggestions_are_not_reported_nor_appliable(tmp_path):
    """TASK-18928 AC#3: a dismissed suggestion is not re-proposed on any
    surface (report AND apply both consult the same snapshot)."""
    service, store = _service(tmp_path, [_profile_record()])
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")

    service.dismiss_permission_suggestion("local:docs", "search")

    report = await service.permission_prompt_recommendations()
    assert report.recommendations == []
    with pytest.raises(KeyError):
        await service.apply_permission_prompt_recommendation("local:docs", "search")


@pytest.mark.asyncio
async def test_restored_suggestion_is_reported_again(tmp_path):
    service, store = _service(tmp_path, [_profile_record()])
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")
    service.dismiss_permission_suggestion("local:docs", "search")

    service.restore_permission_suggestion("local:docs", "search")

    report = await service.permission_prompt_recommendations()
    assert [
        (r.server_key, r.tool_name) for r in report.recommendations
    ] == [("local:docs", "search")]


def test_suggestion_dismissals_round_trip_through_the_service(tmp_path):
    service, _store = _service(tmp_path, [_profile_record()])

    service.dismiss_permission_suggestion("local:docs", "search")
    service.dismiss_permission_suggestion("local:docs", "purge")
    dismissals = service.list_permission_suggestion_dismissals()

    assert sorted(dismissals) == ["local:docs::purge", "local:docs::search"]

    service.clear_permission_suggestion_dismissals()
    assert service.list_permission_suggestion_dismissals() == {}


@pytest.mark.asyncio
async def test_existing_suggestions_do_not_change_runtime_resolution(tmp_path):
    """TASK-18928 AC#5: suggestions are proposal-only -- while one merely
    exists, effective resolution and standing policy are unchanged."""
    service, store = _service(tmp_path, [_profile_record()])
    _append_approved(store, ts="2026-08-01T20:00:00+00:00")
    _append_approved(store, ts="2026-08-01T20:05:00+00:00")

    report = await service.permission_prompt_recommendations()

    assert [
        (r.server_key, r.tool_name) for r in report.recommendations
    ] == [("local:docs", "search")]
    tools = await service._local_prompt_reduction_tools()
    states = service.effective_tool_states(tools)
    assert states[("local:docs", "search")].state == "ask"
    servers = service.permission_store.load()["profiles"]["default"]["servers"]
    assert not servers.get("local:docs", {}).get("tools", {})


def test_high_risk_exclusion_is_shared_with_the_approval_card_badge():
    """TASK-18928 AC#4 drift guard: the reducer's exclusion consumes the
    SAME constants and the SAME resolved flag the permission store's
    resolution (and thereby the approval card's "(high risk)" badge)
    uses -- no parallel list anywhere."""
    import tldw_chatbook.MCP.permission_prompt_reducer as reducer_module
    import tldw_chatbook.MCP.permission_store as store_module
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        format_approval_reason,
    )

    # The reducer imports the class constants BY REFERENCE from the store:
    # redefining either there cannot leave the reducer behind.
    assert reducer_module.HIGH_RISK_TAGS is store_module.HIGH_RISK_TAGS
    assert reducer_module.ALWAYS_ASK_TOOLS is store_module.ALWAYS_ASK_TOOLS

    state = EffectiveToolState(
        state="ask", origin="global_default", risk_floored=True
    )

    # The card renders the badge from the resolved state's risk_floored
    # flag; the reducer excludes on the same flag -- one source, no drift.
    assert "high risk" in format_approval_reason({"reason": "risk_floored"}).lower()

    # And the flag is not the only guard: a high-risk-tagged tool at plain
    # ask (risk_floored=False) is still refused by the class check.
    tool = HubTool(
        server_key="local:docs",
        server_label="docs",
        source="local",
        name="process",
        description="process description",
        input_schema={"type": "object"},
        tags=("process",),
        stale=False,
        executable=True,
    )
    records = [
        {
            "ts": "2026-08-01T20:00:00+00:00",
            "server_key": "local:docs",
            "tool_name": "process",
            "initiator": "agent",
            "decision": "approved",
            "ok": True,
        },
        {
            "ts": "2026-08-01T20:01:00+00:00",
            "server_key": "local:docs",
            "tool_name": "process",
            "initiator": "agent",
            "decision": "approved",
            "ok": True,
        },
    ]

    report = build_permission_prompt_report(records, [tool], {("local:docs", "process"): state})

    assert report.recommendations == []
    assert report.excluded[("local:docs", "process")] == "high-risk-floor"

    tool = HubTool(
        server_key="local:docs",
        server_label="docs",
        source="local",
        name="process",
        description="process description",
        input_schema={"type": "object"},
        tags=(),
        stale=False,
        executable=True,
    )
    records = [
        {
            "ts": "2026-08-01T20:00:00+00:00",
            "server_key": "local:docs",
            "tool_name": "process",
            "initiator": "agent",
            "decision": "approved",
            "ok": True,
        },
        {
            "ts": "2026-08-01T20:01:00+00:00",
            "server_key": "local:docs",
            "tool_name": "process",
            "initiator": "agent",
            "decision": "approved",
            "ok": True,
        },
    ]

    report = build_permission_prompt_report(records, [tool], {("local:docs", "process"): state})

    assert report.recommendations == []
    assert report.excluded[("local:docs", "process")] == "high-risk-floor"


def _reviewed_acceptance(service, *, profile_id="default", description="Search docs"):
    payload = service.permission_store.load()
    profile = payload["profiles"][profile_id]
    return {
        "profile_id": profile_id,
        "expected_policy_chain_digest": profile_policy_chain_digest(
            payload, profile_id
        ),
        "expected_profile_digest": profile_policy_digest(profile),
        "expected_definition_hash": definition_hash(description, {"type": "object"}),
    }


@pytest.mark.asyncio
async def test_suggestion_acceptance_writes_only_reviewed_profile(tmp_path):
    service, store = _service(tmp_path, [_profile_record()])
    service.permission_store.set_global_default("ask", profile_id="research")
    for ts in ("2026-08-01T20:00:00Z", "2026-08-01T20:01:00Z"):
        _append_approved(store, ts=ts)
    reviewed = _reviewed_acceptance(service, profile_id="research")
    service.permission_store.set_global_default("deny", profile_id="unrelated")
    service.dismiss_permission_suggestion("local:docs", "other")
    await service.apply_permission_prompt_recommendation(
        "local:docs", "search", **reviewed
    )
    profiles = service.permission_store.load()["profiles"]
    assert profiles["default"]["servers"] == {}
    assert (
        profiles["research"]["servers"]["local:docs"]["tools"]["search"]["state"]
        == "allow"
    )


@pytest.mark.asyncio
async def test_suggestion_acceptance_rejects_policy_changed_after_review(tmp_path):
    service, store = _service(tmp_path, [_profile_record()])
    for ts in ("2026-08-01T20:00:00Z", "2026-08-01T20:01:00Z"):
        _append_approved(store, ts=ts)
    reviewed = _reviewed_acceptance(service)
    service.permission_store.set_tool_state("local:docs", "other", "deny")
    with pytest.raises(ProfileMutationError):
        await service.apply_permission_prompt_recommendation(
            "local:docs", "search", **reviewed
        )
    assert (
        "search"
        not in service.permission_store.load()["profiles"]["default"]["servers"][
            "local:docs"
        ]["tools"]
    )


@pytest.mark.asyncio
async def test_suggestion_acceptance_rejects_definition_changed_after_display(tmp_path):
    service, store = _service(tmp_path, [_profile_record()])
    for ts in ("2026-08-01T20:00:00Z", "2026-08-01T20:01:00Z"):
        _append_approved(store, ts=ts)
    reviewed = _reviewed_acceptance(service)
    service.local_service._records = [
        _profile_record(description="Upload private docs")
    ]
    # Even fresh approvals for the successor cannot authorize a stale displayed row.
    for ts in ("2026-08-02T20:00:00Z", "2026-08-02T20:01:00Z"):
        _append_approved(store, ts=ts, description="Upload private docs")
    with pytest.raises(KeyError):
        await service.apply_permission_prompt_recommendation(
            "local:docs", "search", **reviewed
        )
    assert service.permission_store.load()["profiles"]["default"]["servers"] == {}


@pytest.mark.asyncio
async def test_legacy_apply_without_reviewed_authority_fails_closed(tmp_path):
    service, store = _service(tmp_path, [_profile_record()])
    for ts in ("2026-08-01T20:00:00Z", "2026-08-01T20:01:00Z"):
        _append_approved(store, ts=ts)
    with pytest.raises(KeyError):
        await service.apply_permission_prompt_recommendation("local:docs", "search")
    assert service.permission_store.load()["profiles"]["default"]["servers"] == {}


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("decision", ["approved", "approved-session", "allowed"])
async def test_execution_audit_retains_fingerprint_only_for_explicit_approval(
    tmp_path, monkeypatch, decision
):
    service, _store = _service(tmp_path, [_profile_record()])

    async def execute_external_tool(_profile, _tool, _arguments):
        return {"content": []}

    monkeypatch.setattr(
        service.local_service,
        "execute_external_tool",
        execute_external_tool,
        raising=False,
    )
    await service.execute_hub_tool_result(
        "local:docs",
        "search",
        {},
        initiator="agent",
        decision=decision,
        approval_definition_hash="a" * 64,
    )
    row = service.execution_log.read_recent()[0]
    assert row.get("definition_hash") == ("a" * 64 if decision == "approved" else None)


@pytest.mark.asyncio
async def test_suggestion_acceptance_forwards_expected_profile_revision(tmp_path):
    service, store = _service(tmp_path, [_profile_record()])
    for ts in ("2026-08-01T20:00:00Z", "2026-08-01T20:01:00Z"):
        _append_approved(store, ts=ts)
    with pytest.raises(ProfileMutationError):
        await service.apply_permission_prompt_recommendation(
            "local:docs", "search", expected_revision=7, **_reviewed_acceptance(service)
        )
    assert service.permission_store.load()["profiles"]["default"]["servers"] == {}


@pytest.mark.asyncio
async def test_suggestion_acceptance_rejects_inherited_deny_at_final_writer(
    tmp_path, monkeypatch
):
    service, history = _service(tmp_path, [_profile_record()])
    store = service.permission_store
    store.ensure_profile("research")
    for ts in ("2026-08-01T20:00:00Z", "2026-08-01T20:01:00Z"):
        _append_approved(history, ts=ts)
    reviewed = _reviewed_acceptance(service, profile_id="research")
    original_set = service.set_tool_state

    def change_inherited_policy(*args, **kwargs):
        store.set_tool_state("local:docs", "search", "deny", profile_id="default")
        return original_set(*args, **kwargs)

    monkeypatch.setattr(service, "set_tool_state", change_inherited_policy)
    with pytest.raises(ProfileMutationError):
        await service.apply_permission_prompt_recommendation(
            "local:docs", "search", **reviewed
        )
    profiles = store.load()["profiles"]
    assert profiles["research"]["servers"] == {}
    assert (
        profiles["default"]["servers"]["local:docs"]["tools"]["search"]["state"]
        == "deny"
    )


@pytest.mark.asyncio
async def test_suggestion_acceptance_rejects_changed_inherited_review(tmp_path):
    service, history = _service(tmp_path, [_profile_record()])
    store = service.permission_store
    store.ensure_profile("research")
    for ts in ("2026-08-01T20:00:00Z", "2026-08-01T20:01:00Z"):
        _append_approved(history, ts=ts)
    reviewed = _reviewed_acceptance(service, profile_id="research")
    # The proposed tool remains Ask, but the reviewed fallback policy has changed.
    store.set_tool_state("local:docs", "other", "deny", profile_id="default")
    with pytest.raises(ProfileMutationError):
        await service.apply_permission_prompt_recommendation(
            "local:docs", "search", **reviewed
        )
    assert store.load()["profiles"]["research"]["servers"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("profile_id", ["default", "research"])
async def test_suggestion_legacy_authority_is_sufficient_only_for_default(
    tmp_path, profile_id
):
    service, history = _service(tmp_path, [_profile_record()])
    store = service.permission_store
    store.ensure_profile(profile_id)
    for ts in ("2026-08-01T20:00:00Z", "2026-08-01T20:01:00Z"):
        _append_approved(history, ts=ts)
    reviewed = _reviewed_acceptance(service, profile_id=profile_id)
    reviewed.pop("expected_policy_chain_digest")
    if profile_id == "default":
        await service.apply_permission_prompt_recommendation(
            "local:docs", "search", **reviewed
        )
        assert store.get_tool_entry("local:docs", "search")["state"] == "allow"
    else:
        with pytest.raises(KeyError):
            await service.apply_permission_prompt_recommendation(
                "local:docs", "search", **reviewed
            )
        assert store.load()["profiles"]["research"]["servers"] == {}
