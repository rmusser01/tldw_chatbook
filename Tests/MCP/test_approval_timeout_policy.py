"""TASK-32564: one human-approval timeout policy across existing consumers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.MCP import live_server_request_wiring as wiring
from tldw_chatbook.MCP import unified_control_plane_service as service_module
from tldw_chatbook.MCP.local_store import LocalMCPStore


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("missing", 0.0), (None, 0.0), ("bad", 0.0), (0, 0.0), (-1, -1.0), (30, 30.0)],
)
def test_console_and_service_share_approval_timeout_policy(
    monkeypatch, configured, expected
):
    def setting(section, key, default=None):
        assert (section, key) == ("mcp", "approval_timeout_seconds")
        return default if configured == "missing" else configured

    monkeypatch.setattr(controller_module, "get_cli_setting", setting)
    monkeypatch.setattr(service_module, "get_cli_setting", setting)
    controller = controller_module.ConsoleChatController.__new__(
        controller_module.ConsoleChatController
    )
    controller.mcp_approval_timeout_seconds = None
    service = service_module.UnifiedMCPControlPlaneService(
        target_store=None, context_store=None, local_service=None, server_service=None
    )
    assert controller._resolve_mcp_approval_timeout_seconds() == expected
    assert service.approval_timeout_seconds() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", ["missing", None, "bad", 0, -1])
@pytest.mark.parametrize("decision", ["approved", "denied", "cancelled"])
async def test_live_confirmation_without_deadline_remains_answerable(
    monkeypatch, tmp_path, configured, decision
):
    """Advance only the approval clock past the old default, then answer/stop."""
    monkeypatch.setattr(
        wiring,
        "get_cli_setting",
        lambda section, key, default=None: (
            default if configured == "missing" else configured
        ),
    )
    clock = SimpleNamespace(now=0.0)
    loop = SimpleNamespace(time=lambda: clock.now)
    # Keep asyncio's scheduler/outer test deadlines real; control this consumer's
    # monotonic approval clock alone so a two-minute regression stays fast.
    monkeypatch.setattr(
        wiring,
        "asyncio",
        SimpleNamespace(
            get_event_loop=lambda: loop,
            get_running_loop=lambda: loop,
            sleep=asyncio.sleep,
        ),
    )
    store = LocalMCPStore(tmp_path / "mcp.json")
    elicit = wiring.build_live_elicit_fn(store, poll_seconds=0.001)
    task = asyncio.create_task(elicit("Proceed?", {}))
    try:
        await asyncio.sleep(0)
        clock.now = 1000.0
        await asyncio.sleep(0.01)
        assert not task.done(), (
            "an unanswered confirmation expired without an opt-in ceiling"
        )
        pending = store.list_approval_requests()
        assert len(pending) == 1 and pending[0].status == "pending"
        request_id = pending[0].request_id
        if decision == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert store.list_approval_requests()[0].status == "expired"
            assert store.resolve_approval_request(request_id, "approved") is None
        else:
            store.resolve_approval_request(request_id, decision)
            result = await asyncio.wait_for(task, timeout=1)
            expected = (
                {"action": "accept", "content": {}} if decision == "approved" else None
            )
            assert result == expected
            assert store.list_approval_requests()[0].status == decision
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
