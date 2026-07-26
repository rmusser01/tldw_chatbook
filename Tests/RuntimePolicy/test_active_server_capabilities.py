from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
from tldw_chatbook.runtime_policy.bootstrap import RuntimePolicyContext
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.tldw_api.exceptions import APIConnectionError, APIResponseError


class FakeServerRuntimeScope:
    def __init__(self, *, fail_on: str | None = None, error: Exception | None = None):
        self.calls = []
        self.fail_on = fail_on
        self.error = error

    async def get_health(self, *, mode="server"):
        self.calls.append(("get_health", mode))
        if self.fail_on == "health":
            raise self.error
        return {"status": "ok", "auth_mode": "multi_user"}

    async def get_readiness(self, *, mode="server"):
        self.calls.append(("get_readiness", mode))
        if self.fail_on == "readiness":
            raise self.error
        return {"status": "ready", "ready": True}

    async def get_docs_info(self, *, mode="server"):
        self.calls.append(("get_docs_info", mode))
        if self.fail_on == "docs_info":
            raise self.error
        return {
            "configured": True,
            "auth_mode": "multi_user",
            "capabilities": {"sync": True, "audio": True},
            "supported_features": {"read_it_later": True},
        }


class FakeProbeServerRuntimeService:
    def __init__(self):
        self.calls = []

    async def probe_health(self):
        self.calls.append(("probe_health",))
        return {"status": "ok"}

    async def probe_readiness(self):
        self.calls.append(("probe_readiness",))
        return {"ready": True}

    async def probe_docs_info(self):
        self.calls.append(("probe_docs_info",))
        return {"capabilities": {"sync": True}, "supported_features": {}}


class FakeDenyingServerRuntimeScope:
    def __init__(self):
        self.server_service = FakeProbeServerRuntimeService()

    async def get_health(self, *, mode="server"):
        raise AssertionError("capability refresh should bypass policy-gated health")

    async def get_readiness(self, *, mode="server"):
        raise AssertionError("capability refresh should bypass policy-gated readiness")

    async def get_docs_info(self, *, mode="server"):
        raise AssertionError("capability refresh should bypass policy-gated docs_info")


class RecordingRuntimeStore:
    def __init__(self) -> None:
        self.saved_states: list[RuntimeSourceState] = []

    def save(self, state: RuntimeSourceState) -> None:
        self.saved_states.append(state)


def _context(
    state: RuntimeSourceState,
    *,
    publish=None,
) -> RuntimePolicyContext:
    return RuntimePolicyContext(
        state,
        RecordingRuntimeStore(),
        publish=publish,
    )


def _commit(
    context: RuntimePolicyContext,
    state: RuntimeSourceState,
) -> None:
    _, revision = context.snapshot()
    assert context.commit_state(state, expected_revision=revision)


class BarrierProbeServerRuntimeService(FakeProbeServerRuntimeService):
    def __init__(self) -> None:
        super().__init__()
        self.health_started = asyncio.Event()
        self.release_health = asyncio.Event()

    async def probe_health(self):
        self.calls.append(("probe_health",))
        self.health_started.set()
        await self.release_health.wait()
        return {"status": "stale-health"}


class BarrierProbeRuntimeScope:
    def __init__(self) -> None:
        self.server_service = BarrierProbeServerRuntimeService()


class RecordingTargetStore:
    def __init__(self) -> None:
        self.updates: list[tuple[tuple, dict]] = []

    def update_target_status(self, *args, **kwargs) -> None:
        self.updates.append((args, kwargs))


class SupersedingNoServerContext:
    def __init__(
        self,
        captured: RuntimeSourceState,
        fresh: RuntimeSourceState,
    ) -> None:
        self.captured = captured
        self.fresh = fresh
        self.commit_calls: list[tuple[RuntimeSourceState, int]] = []
        self.snapshot_calls = 0

    @property
    def state(self) -> RuntimeSourceState:
        return self.fresh if self.snapshot_calls > 1 else self.captured

    def snapshot(self) -> tuple[RuntimeSourceState, int]:
        self.snapshot_calls += 1
        if self.snapshot_calls == 1:
            return self.captured, 7
        return self.fresh, 8

    def commit_state(
        self,
        candidate: RuntimeSourceState,
        *,
        expected_revision: int,
    ) -> bool:
        self.commit_calls.append((candidate, expected_revision))
        return False


@pytest.mark.asyncio
async def test_active_server_capabilities_refreshes_snapshot_and_policy_state():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    context = _context(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://server.example.com/api",
            server_configured=True,
            last_known_server_label="server.example.com",
        )
    )
    runtime_scope = FakeServerRuntimeScope()
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
    )

    snapshot = await service.refresh()

    assert (
        snapshot["record_id"]
        == "server:capability_snapshot:https://server.example.com/api"
    )
    assert snapshot["active_server_id"] == "https://server.example.com/api"
    assert snapshot["server_configured"] is True
    assert snapshot["reachability"] == "reachable"
    assert snapshot["auth_state"] == "authenticated"
    assert snapshot["capabilities"] == {"sync": True, "audio": True}
    assert snapshot["supported_features"] == {"read_it_later": True}
    assert snapshot["health"]["status"] == "ok"
    assert snapshot["readiness"]["ready"] is True
    assert snapshot["errors"] == []
    assert context.state.server_reachability == "reachable"
    assert context.state.server_auth_state == "authenticated"
    assert context.state.server_reachability_checked_at is not None
    assert context.state.server_auth_checked_at is not None
    assert context.snapshot()[1] == 1
    assert runtime_scope.calls == [
        ("get_health", "server"),
        ("get_readiness", "server"),
        ("get_docs_info", "server"),
    ]


@pytest.mark.asyncio
async def test_active_server_capabilities_refresh_uses_current_runtime_policy_server_identity():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    context = _context(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://old.example.com/api",
            server_configured=True,
            server_reachability="reachable",
            server_auth_state="authenticated",
        )
    )
    runtime_scope = FakeServerRuntimeScope()
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
    )
    _commit(
        context,
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://new.example.com/v1",
            server_configured=True,
            server_reachability="unknown",
            server_auth_state="unknown",
            last_known_server_label="new.example.com",
        ),
    )

    snapshot = await service.refresh()

    assert (
        snapshot["record_id"] == "server:capability_snapshot:https://new.example.com/v1"
    )
    assert snapshot["active_server_id"] == "https://new.example.com/v1"
    assert snapshot["reachability"] == "reachable"
    assert snapshot["auth_state"] == "authenticated"
    assert context.state.active_server_id == "https://new.example.com/v1"
    assert context.state.last_known_server_label == "new.example.com"


@pytest.mark.asyncio
async def test_active_server_capabilities_uses_ungated_probes_to_recover_stale_auth_state():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    context = _context(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://server.example.com/api",
            server_configured=True,
            server_auth_state="auth_required",
        )
    )
    runtime_scope = FakeDenyingServerRuntimeScope()
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
    )

    snapshot = await service.refresh()

    assert snapshot["auth_state"] == "authenticated"
    assert context.state.server_auth_state == "authenticated"
    assert runtime_scope.server_service.calls == [
        ("probe_health",),
        ("probe_readiness",),
        ("probe_docs_info",),
    ]


@pytest.mark.asyncio
async def test_active_server_capabilities_updates_target_store_status(tmp_path):
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [
            ConfiguredServerTarget(
                server_id="https://server.example.com/api",
                label="Primary",
                base_url="https://server.example.com/api",
                auth_mode="api_key",
                is_default=True,
            )
        ]
    )
    context = _context(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://server.example.com/api",
            server_configured=True,
            last_known_server_label="server.example.com",
        )
    )
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=FakeServerRuntimeScope(),
        target_store=target_store,
    )

    snapshot = await service.refresh()

    target = target_store.get_target("https://server.example.com/api")
    assert target is not None
    assert target.last_known_server_label == "server.example.com"
    assert target.last_known_reachability == "reachable"
    assert target.last_known_auth_state == "authenticated"
    assert target.updated_at is not None
    assert snapshot["errors"] == []


@pytest.mark.asyncio
async def test_active_server_capabilities_ignores_missing_target_profile(tmp_path):
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    context = _context(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://missing.example.com/api",
            server_configured=True,
        )
    )
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=FakeServerRuntimeScope(),
        target_store=target_store,
    )

    snapshot = await service.refresh()

    assert snapshot["reachability"] == "reachable"
    assert snapshot["auth_state"] == "authenticated"
    assert snapshot["errors"] == [
        {
            "reason_code": "target_profile_missing",
            "message": "Active server target profile was not found.",
        }
    ]


@pytest.mark.asyncio
async def test_active_server_capabilities_marks_unreachable_without_losing_server_identity():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    context = _context(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://server.example.com/api",
            server_configured=True,
        )
    )
    runtime_scope = FakeServerRuntimeScope(
        fail_on="health",
        error=APIConnectionError("cannot connect"),
    )
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
    )

    snapshot = await service.refresh()

    assert snapshot["active_server_id"] == "https://server.example.com/api"
    assert snapshot["reachability"] == "unreachable"
    assert snapshot["auth_state"] == "unknown"
    assert snapshot["errors"][0]["reason_code"] == "server_unreachable"
    assert context.state.active_server_id == "https://server.example.com/api"
    assert context.state.server_reachability == "unreachable"
    assert context.state.server_auth_state == "unknown"
    assert context.snapshot()[1] == 1
    assert runtime_scope.calls == [("get_health", "server")]


@pytest.mark.asyncio
async def test_active_server_capabilities_marks_auth_required_as_reachable_server():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    context = _context(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://server.example.com/api",
            server_configured=True,
        )
    )
    runtime_scope = FakeServerRuntimeScope(
        fail_on="docs_info",
        error=APIResponseError(401, "auth required", {"detail": "missing token"}),
    )
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
    )

    snapshot = await service.refresh()

    assert snapshot["reachability"] == "reachable"
    assert snapshot["auth_state"] == "auth_required"
    assert snapshot["errors"][0]["reason_code"] == "server_auth_required"
    assert context.state.server_reachability == "reachable"
    assert context.state.server_auth_state == "auth_required"
    assert runtime_scope.calls == [
        ("get_health", "server"),
        ("get_readiness", "server"),
        ("get_docs_info", "server"),
    ]


@pytest.mark.asyncio
async def test_active_server_capabilities_does_not_call_server_when_unconfigured():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    context = _context(
        RuntimeSourceState(active_source="local", server_configured=False)
    )
    runtime_scope = FakeServerRuntimeScope()
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
    )

    snapshot = await service.refresh()

    assert snapshot["record_id"] == "server:capability_snapshot:unconfigured"
    assert snapshot["active_server_id"] is None
    assert snapshot["server_configured"] is False
    assert snapshot["reachability"] == "unknown"
    assert snapshot["auth_state"] == "unknown"
    assert snapshot["errors"][0]["reason_code"] == "server_not_configured"
    assert runtime_scope.calls == []
    assert context.snapshot()[1] == 1


@pytest.mark.asyncio
async def test_active_server_capabilities_invalidates_persisted_probe_state_when_server_is_cleared():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    context = _context(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://server.example.com/api",
            server_configured=True,
            server_reachability="reachable",
            server_reachability_checked_at=datetime(
                2026, 4, 28, 12, 0, tzinfo=timezone.utc
            ),
            server_auth_state="authenticated",
            server_auth_checked_at=datetime(2026, 4, 28, 12, 1, tzinfo=timezone.utc),
            last_known_server_label="server.example.com",
        )
    )
    runtime_scope = FakeServerRuntimeScope()
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
    )
    _commit(
        context,
        RuntimeSourceState(
            active_source="local",
            active_server_id=None,
            server_configured=False,
            server_reachability="reachable",
            server_reachability_checked_at=datetime(
                2026, 4, 28, 12, 0, tzinfo=timezone.utc
            ),
            server_auth_state="authenticated",
            server_auth_checked_at=datetime(2026, 4, 28, 12, 1, tzinfo=timezone.utc),
            last_known_server_label="server.example.com",
        ),
    )

    snapshot = await service.refresh()

    assert snapshot["active_server_id"] is None
    assert snapshot["reachability"] == "unknown"
    assert snapshot["auth_state"] == "unknown"
    assert context.state.server_reachability == "unknown"
    assert context.state.server_reachability_checked_at is None
    assert context.state.server_auth_state == "unknown"
    assert context.state.server_auth_checked_at is None
    assert context.snapshot()[1] == 2
    assert runtime_scope.calls == []


@pytest.mark.asyncio
async def test_stale_capability_probe_after_source_change_returns_superseded_snapshot():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    initial = RuntimeSourceState(
        active_source="server",
        active_server_id="https://server-a.example.test/api",
        server_configured=True,
        last_known_server_label="Server A",
    )
    published: list[RuntimeSourceState] = []
    context = _context(initial, publish=published.append)
    runtime_scope = BarrierProbeRuntimeScope()
    target_store = RecordingTargetStore()
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
        target_store=target_store,
    )

    refresh_task = asyncio.create_task(service.refresh())
    await runtime_scope.server_service.health_started.wait()
    newer = replace(initial, active_source="local")
    _commit(context, newer)
    published.clear()
    runtime_scope.server_service.release_health.set()

    snapshot = await refresh_task

    assert snapshot["errors"] == [
        {
            "reason_code": "capability_result_superseded",
            "message": (
                "Capability refresh was superseded by a newer runtime selection."
            ),
        }
    ]
    assert snapshot["health"] == {}
    assert snapshot["readiness"] == {}
    assert snapshot["docs_info"] == {}
    assert snapshot["capabilities"] == {}
    assert snapshot["supported_features"] == {}
    assert snapshot["active_server_id"] == newer.active_server_id
    assert context.state == newer
    assert target_store.updates == []
    assert published == []


@pytest.mark.asyncio
async def test_stale_capability_probe_after_active_server_change_returns_superseded_snapshot():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    initial = RuntimeSourceState(
        active_source="server",
        active_server_id="https://server-a.example.test/api",
        server_configured=True,
        last_known_server_label="Server A",
    )
    published: list[RuntimeSourceState] = []
    context = _context(initial, publish=published.append)
    runtime_scope = BarrierProbeRuntimeScope()
    target_store = RecordingTargetStore()
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
        target_store=target_store,
    )

    refresh_task = asyncio.create_task(service.refresh())
    await runtime_scope.server_service.health_started.wait()
    newer = replace(
        initial,
        active_server_id="https://server-b.example.test/api",
        last_known_server_label="Server B",
    )
    _commit(context, newer)
    published.clear()
    runtime_scope.server_service.release_health.set()

    snapshot = await refresh_task

    assert snapshot["errors"][0]["reason_code"] == "capability_result_superseded"
    assert snapshot["health"] == {}
    assert snapshot["readiness"] == {}
    assert snapshot["docs_info"] == {}
    assert snapshot["active_server_id"] == newer.active_server_id
    assert snapshot["last_known_server_label"] == "Server B"
    assert context.state == newer
    assert target_store.updates == []
    assert published == []


@pytest.mark.asyncio
async def test_no_server_configured_branch_returns_superseded_fresh_authority():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    captured = RuntimeSourceState(
        active_source="local",
        server_reachability="reachable",
        server_reachability_checked_at=datetime(
            2026, 4, 28, 12, 0, tzinfo=timezone.utc
        ),
        server_auth_state="authenticated",
        server_auth_checked_at=datetime(2026, 4, 28, 12, 1, tzinfo=timezone.utc),
    )
    fresh = RuntimeSourceState(
        active_source="server",
        active_server_id="https://fresh.example.test/api",
        server_configured=True,
        last_known_server_label="Fresh",
    )
    context = SupersedingNoServerContext(captured, fresh)
    runtime_scope = FakeServerRuntimeScope()
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
    )

    snapshot = await service.refresh()

    assert len(context.commit_calls) == 1
    candidate, expected_revision = context.commit_calls[0]
    assert expected_revision == 7
    assert candidate.server_reachability == "unknown"
    assert candidate.server_auth_state == "unknown"
    assert snapshot["errors"][0]["reason_code"] == "capability_result_superseded"
    assert snapshot["active_server_id"] == fresh.active_server_id
    assert snapshot["health"] == {}
    assert snapshot["readiness"] == {}
    assert snapshot["docs_info"] == {}
    assert runtime_scope.calls == []


@pytest.mark.asyncio
async def test_capability_error_diagnostics_omit_exception_payload_sentinels():
    from tldw_chatbook.runtime_policy.server_capabilities import (
        ActiveServerCapabilityService,
    )

    endpoint_sentinel = "https://CAPABILITY-ENDPOINT-SENTINEL.example/api"
    credential_sentinel = "CAPABILITY-CREDENTIAL-SENTINEL"
    body_sentinel = "CAPABILITY-BODY-SENTINEL"
    context = _context(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://server.example.com/api",
            server_configured=True,
        )
    )
    runtime_scope = FakeServerRuntimeScope(
        fail_on="docs_info",
        error=APIResponseError(
            401,
            f"{endpoint_sentinel} {credential_sentinel}",
            {"detail": body_sentinel},
        ),
    )
    service = ActiveServerCapabilityService(
        runtime_context=context,
        server_runtime_scope_service=runtime_scope,
    )

    snapshot = await service.refresh()

    rendered = repr(snapshot["errors"])
    assert snapshot["errors"][0]["reason_code"] == "server_auth_required"
    assert endpoint_sentinel not in rendered
    assert credential_sentinel not in rendered
    assert body_sentinel not in rendered
