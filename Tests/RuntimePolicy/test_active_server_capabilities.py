from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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


def _context(state: RuntimeSourceState):
    return SimpleNamespace(state=state, persist=Mock())


@pytest.mark.asyncio
async def test_active_server_capabilities_refreshes_snapshot_and_policy_state():
    from tldw_chatbook.runtime_policy.server_capabilities import ActiveServerCapabilityService

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

    assert snapshot["record_id"] == "server:capability_snapshot:https://server.example.com/api"
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
    context.persist.assert_called_once()
    assert runtime_scope.calls == [
        ("get_health", "server"),
        ("get_readiness", "server"),
        ("get_docs_info", "server"),
    ]


@pytest.mark.asyncio
async def test_active_server_capabilities_refresh_uses_current_runtime_policy_server_identity():
    from tldw_chatbook.runtime_policy.server_capabilities import ActiveServerCapabilityService

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
    context.state = RuntimeSourceState(
        active_source="server",
        active_server_id="https://new.example.com/v1",
        server_configured=True,
        server_reachability="unknown",
        server_auth_state="unknown",
        last_known_server_label="new.example.com",
    )

    snapshot = await service.refresh()

    assert snapshot["record_id"] == "server:capability_snapshot:https://new.example.com/v1"
    assert snapshot["active_server_id"] == "https://new.example.com/v1"
    assert snapshot["reachability"] == "reachable"
    assert snapshot["auth_state"] == "authenticated"
    assert context.state.active_server_id == "https://new.example.com/v1"
    assert context.state.last_known_server_label == "new.example.com"


@pytest.mark.asyncio
async def test_active_server_capabilities_uses_ungated_probes_to_recover_stale_auth_state():
    from tldw_chatbook.runtime_policy.server_capabilities import ActiveServerCapabilityService

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
async def test_active_server_capabilities_marks_unreachable_without_losing_server_identity():
    from tldw_chatbook.runtime_policy.server_capabilities import ActiveServerCapabilityService

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
    context.persist.assert_called_once()
    assert runtime_scope.calls == [("get_health", "server")]


@pytest.mark.asyncio
async def test_active_server_capabilities_marks_auth_required_as_reachable_server():
    from tldw_chatbook.runtime_policy.server_capabilities import ActiveServerCapabilityService

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
    from tldw_chatbook.runtime_policy.server_capabilities import ActiveServerCapabilityService

    context = _context(RuntimeSourceState(active_source="local", server_configured=False))
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
    context.persist.assert_not_called()


@pytest.mark.asyncio
async def test_active_server_capabilities_invalidates_persisted_probe_state_when_server_is_cleared():
    from tldw_chatbook.runtime_policy.server_capabilities import ActiveServerCapabilityService

    context = _context(
        RuntimeSourceState(
            active_source="server",
            active_server_id="https://server.example.com/api",
            server_configured=True,
            server_reachability="reachable",
            server_reachability_checked_at=datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc),
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
    context.state = RuntimeSourceState(
        active_source="local",
        active_server_id=None,
        server_configured=False,
        server_reachability="reachable",
        server_reachability_checked_at=datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc),
        server_auth_state="authenticated",
        server_auth_checked_at=datetime(2026, 4, 28, 12, 1, tzinfo=timezone.utc),
        last_known_server_label="server.example.com",
    )

    snapshot = await service.refresh()

    assert snapshot["active_server_id"] is None
    assert snapshot["reachability"] == "unknown"
    assert snapshot["auth_state"] == "unknown"
    assert context.state.server_reachability == "unknown"
    assert context.state.server_reachability_checked_at is None
    assert context.state.server_auth_state == "unknown"
    assert context.state.server_auth_checked_at is None
    context.persist.assert_called_once()
    assert runtime_scope.calls == []
