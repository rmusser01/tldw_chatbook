"""ServedRelayPublisher — child-side approval/run relay (ADR-236).

Task 2 of the served-mobile Phase 2c plan. The publisher forwards a served
child's Console approval rounds and run-status transitions to the parent
broker through the child's ``CanvasControlClient``. Every test here uses a
fake client recording frames; the protocol-level loopback round-trip for
``send_relay_event`` lives in ``Tests/Canvas/test_control_protocol.py``.

The wrapper contract (call-through-first) is asserted against the
``wrap_pending_approval_hook`` factory the app routes the Console approval
seam through: the original callback must run before any publish, must keep
its return value, and must keep running when publishing fails.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.Web_Server.served_relay_publisher import (
    ServedRelayPublisher,
    wrap_pending_approval_hook,
)


class FakeRelayClient:
    """Records relay frames; optionally blocks or raises inside the send."""

    def __init__(
        self, *, fail: bool = False, order_markers: list[str] | None = None
    ) -> None:
        self.frames: list[tuple[str, dict[str, Any]]] = []
        self.attempts: list[str] = []
        self.order_markers = order_markers if order_markers is not None else []
        self.fail = fail
        self.gate: asyncio.Event | None = None
        self.in_flight = asyncio.Event()

    async def send_relay_event(
        self, message_type: str, payload: dict[str, Any]
    ) -> None:
        self.attempts.append(message_type)
        if self.gate is not None:
            self.in_flight.set()
            await self.gate.wait()
        self.frames.append((message_type, dict(payload)))
        self.order_markers.append(f"publish:{message_type}")
        if self.fail:
            raise RuntimeError("relay send failed")


CONSOLE_APPROVAL_PAYLOAD = {
    "round_id": "round-42",
    "session_id": "session-7",
    "run_id": "run-9",
    "calls": [
        {
            "llm_name": "fs_write",
            "server_key": "agent:builtin",
            "tool_name": "fs_write",
            "server_label": "Local tools",
            "arguments": {"path": "notes/scratch.md", "content": "hi"},
            "reason": "writes a file",
            "options": ["approve_once", "deny"],
            "effects": ["fs.write"],
            "execution_policy": "bounded_abandonable",
            "path_precheck_failed": False,
            "call_id": "call-1",
            "full_command": None,
            "warning": None,
            "scope_notice": None,
            "rationale": "user asked for a note",
            "description": "Write a file",
        }
    ],
    "timeout_seconds": 30.0,
    "deadline_monotonic": 123.5,
    "summary": None,
}


async def _drain(publisher: ServedRelayPublisher) -> None:
    await publisher.wait_idle(timeout=2.0)


@pytest.mark.asyncio
async def test_approval_raised_maps_only_protocol_fields() -> None:
    client = FakeRelayClient()
    publisher = ServedRelayPublisher(client)
    publisher.attach()

    publisher.approval_raised(CONSOLE_APPROVAL_PAYLOAD)
    await _drain(publisher)

    assert client.frames == [
        (
            "relay.approval.raised",
            {
                "round_id": "round-42",
                "session_id": "session-7",
                "calls": [
                    {
                        "llm_name": "fs_write",
                        "call_id": "call-1",
                        "options": ["approve_once", "deny"],
                        "arguments": {"path": "notes/scratch.md", "content": "hi"},
                    }
                ],
                # The console payload carries a float; the protocol demands int.
                "timeout_seconds": 30,
            },
        )
    ]
    await publisher.aclose()


@pytest.mark.asyncio
async def test_approval_raised_without_round_id_is_dropped() -> None:
    client = FakeRelayClient()
    publisher = ServedRelayPublisher(client)
    publisher.attach()

    publisher.approval_raised({"session_id": "session-7", "calls": []})
    await _drain(publisher)

    assert client.frames == []
    await publisher.aclose()


@pytest.mark.asyncio
async def test_approval_cleared_none_pops_newest_mounted_round() -> None:
    client = FakeRelayClient()
    publisher = ServedRelayPublisher(client)
    publisher.attach()

    publisher.approval_raised(CONSOLE_APPROVAL_PAYLOAD)
    publisher.approval_raised({**CONSOLE_APPROVAL_PAYLOAD, "round_id": "round-43"})
    publisher.approval_cleared(None)
    await _drain(publisher)

    assert client.frames[-1] == (
        "relay.approval.cleared",
        {"round_id": "round-43", "resolution": "cleared"},
    )
    await publisher.aclose()


@pytest.mark.asyncio
async def test_approval_cleared_explicit_round_and_resolution() -> None:
    client = FakeRelayClient()
    publisher = ServedRelayPublisher(client)
    publisher.attach()

    publisher.approval_cleared("round-42", "denied")
    await _drain(publisher)

    assert client.frames == [
        ("relay.approval.cleared", {"round_id": "round-42", "resolution": "denied"})
    ]
    await publisher.aclose()


@pytest.mark.asyncio
async def test_run_status_publishes_session_and_status() -> None:
    client = FakeRelayClient()
    publisher = ServedRelayPublisher(client)
    publisher.attach()

    publisher.run_status("session-7", "completed")
    await _drain(publisher)

    assert client.frames == [
        ("relay.run.status", {"session_id": "session-7", "status": "completed"})
    ]
    await publisher.aclose()


@pytest.mark.asyncio
async def test_none_client_is_a_silent_noop() -> None:
    publisher = ServedRelayPublisher(None)
    publisher.attach()

    publisher.approval_raised(CONSOLE_APPROVAL_PAYLOAD)
    publisher.approval_cleared(None)
    publisher.run_status("session-7", "completed")
    await _drain(publisher)
    await publisher.aclose()


@pytest.mark.asyncio
async def test_publishes_before_attach_are_dropped_without_raising() -> None:
    publisher = ServedRelayPublisher(None)

    publisher.approval_raised(CONSOLE_APPROVAL_PAYLOAD)
    publisher.approval_cleared(None)
    publisher.run_status("session-7", "failed")
    await publisher.aclose()


@pytest.mark.asyncio
async def test_send_failures_are_swallowed_and_drainer_survives() -> None:
    client = FakeRelayClient(fail=True)
    publisher = ServedRelayPublisher(client)
    publisher.attach()

    publisher.run_status("session-7", "streaming")
    await _drain(publisher)
    # The send was fully attempted (the fake records the frame, then raises);
    # the failure never escaped the publisher.
    assert client.attempts == ["relay.run.status"]

    publisher.run_status("session-7", "completed")
    await _drain(publisher)

    # The drainer survived the first failure and attempted the second send.
    assert client.attempts == ["relay.run.status", "relay.run.status"]
    await publisher.aclose()


@pytest.mark.asyncio
async def test_bounded_queue_drops_when_the_drainer_stalls() -> None:
    client = FakeRelayClient()
    client.gate = asyncio.Event()
    publisher = ServedRelayPublisher(client, max_pending=2)
    publisher.attach()

    publisher.run_status("session-7", "streaming")
    while not client.in_flight.is_set():
        await asyncio.sleep(0)
    for status in ("validating", "completed", "failed", "stopped", "blocked"):
        publisher.run_status("session-7", status)
    client.gate.set()
    await _drain(publisher)

    # One in-flight + two queued; the remaining three were dropped, not raised.
    assert len(client.frames) == 3
    await publisher.aclose()


@pytest.mark.asyncio
async def test_run_state_polling_publishes_only_deltas() -> None:
    client = FakeRelayClient()
    states: dict[str, Any] = {}
    publisher = ServedRelayPublisher(client, poll_interval=0.01)
    publisher.attach(
        run_state_source=lambda: {
            session_id: SimpleNamespace(status=status)
            for session_id, status in states.items()
        }
    )

    # First observation baselines silently (no publish for a pre-existing
    # terminal status), then a transition publishes.
    states["session-7"] = ConsoleRunStatus.COMPLETED
    await asyncio.sleep(0.05)
    assert client.frames == []

    states["session-7"] = ConsoleRunStatus.FAILED
    await asyncio.sleep(0.05)
    assert client.frames == [
        ("relay.run.status", {"session_id": "session-7", "status": "failed"})
    ]
    await publisher.aclose()


@pytest.mark.asyncio
async def test_run_state_polling_tolerates_missing_controller() -> None:
    client = FakeRelayClient()
    source: list[Any] = [None]
    publisher = ServedRelayPublisher(client, poll_interval=0.01)
    publisher.attach(run_state_source=lambda: source[0])

    await asyncio.sleep(0.05)
    assert client.frames == []
    await publisher.aclose()


# --------------------------------------------------------------------- #
# Wrapper contract: the Console approval seam (call-through-first)       #
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_wrapper_runs_original_before_publish_and_returns_its_value() -> None:
    order: list[str] = []
    client = FakeRelayClient(order_markers=order)

    def original(payload):
        order.append("original")
        return "card-mounted"

    publisher = ServedRelayPublisher(client)
    publisher.attach()
    wrapped = wrap_pending_approval_hook(publisher, original)

    result = wrapped(CONSOLE_APPROVAL_PAYLOAD)
    assert result == "card-mounted"
    assert order == ["original"]
    # The publish rides the drainer; by the time the frame lands the
    # original has already run.
    await _drain(publisher)
    assert order == ["original", "publish:relay.approval.raised"]
    await publisher.aclose()


@pytest.mark.asyncio
async def test_wrapper_clear_payload_publishes_cleared_for_mounted_round() -> None:
    client = FakeRelayClient()

    def original(payload):
        return None

    publisher = ServedRelayPublisher(client)
    publisher.attach()
    wrapped = wrap_pending_approval_hook(publisher, original)

    wrapped(CONSOLE_APPROVAL_PAYLOAD)
    wrapped(None)
    await _drain(publisher)

    assert [frame[0] for frame in client.frames] == [
        "relay.approval.raised",
        "relay.approval.cleared",
    ]
    assert client.frames[1][1] == {
        "round_id": "round-42",
        "resolution": "cleared",
    }
    await publisher.aclose()


@pytest.mark.asyncio
async def test_original_runs_even_when_publishing_raises() -> None:
    client = FakeRelayClient(fail=True)
    calls: list[Any] = []

    def original(payload):
        calls.append(payload)
        return True

    publisher = ServedRelayPublisher(client)
    publisher.attach()
    wrapped = wrap_pending_approval_hook(publisher, original)

    assert wrapped(CONSOLE_APPROVAL_PAYLOAD) is True
    assert calls == [CONSOLE_APPROVAL_PAYLOAD]
    await _drain(publisher)
    assert client.attempts == ["relay.approval.raised"]
    await publisher.aclose()


@pytest.mark.asyncio
async def test_wrapper_without_publisher_returns_the_original_untouched() -> None:
    def original(payload):
        return "unwrapped"

    wrapped = wrap_pending_approval_hook(None, original)
    assert wrapped is original
    assert wrapped(None) == "unwrapped"
