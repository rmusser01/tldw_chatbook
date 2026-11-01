"""Bounded served projection history, independent of subscriber lifetime."""

import asyncio

import pytest
from pydantic import ValidationError

from Tests.Web_Server import test_mobile_console_ui as mobile_controls

mobile_browser_case = mobile_controls.mobile_browser_case
mobile_surface_gate = mobile_controls.mobile_surface_gate


def test_unread_projection_overflow_requires_snapshot_not_disconnect():
    from tldw_chatbook.Web_Server.served_console_events import ConsoleEventBuffer

    buffer = ConsoleEventBuffer(limit=2)
    for revision in range(7):
        buffer.publish("chat_changed", {"session_id": "chat-a", "revision": revision})
    batch = buffer.read(after_sequence=0)
    assert batch.reset_required
    assert len(batch.events) <= 2
    buffer.acknowledge_snapshot(batch.sequence - 1)
    assert buffer.read(after_sequence=batch.sequence).reset_required
    buffer.acknowledge_snapshot(batch.sequence)
    assert not buffer.read(after_sequence=batch.sequence).reset_required
    assert buffer.read(
        after_sequence=0
    ).reset_required  # An old subscriber still has a gap.
    buffer.publish("chat_changed", {"revision": 8})
    recovered = buffer.read(after_sequence=batch.sequence)
    assert not recovered.reset_required
    assert len(recovered.events) == 1


@pytest.mark.asyncio
async def test_wait_changed_has_no_lost_wake_and_wakes_all_observers():
    from tldw_chatbook.Web_Server.served_console_events import ConsoleEventBuffer

    buffer = ConsoleEventBuffer()
    waiters = [asyncio.create_task(buffer.wait_changed(0)) for _ in range(3)]
    await asyncio.sleep(0)
    sequence = buffer.publish("changed", {})
    buffer.note_changed(sequence)
    await asyncio.wait_for(asyncio.gather(*waiters), 1)
    await asyncio.wait_for(buffer.wait_changed(0), 1)
    pending = asyncio.create_task(buffer.wait_changed(sequence))
    await asyncio.sleep(0)
    assert not pending.done()
    buffer.publish("changed", {})
    buffer.note_changed(sequence + 1)
    await asyncio.wait_for(pending, 1)


def test_identity_models_are_frozen_closed_and_reject_paths_labels():
    from tldw_chatbook.Chat.console_remote_models import (
        ConsoleHostIdentity,
        ConsoleTarget,
    )

    host = ConsoleHostIdentity(server_epoch="epoch-a", console_id="console-a")
    target = ConsoleTarget(
        **host.model_dump(), session_id="chat-a", incarnation_id="inc-a"
    )
    assert target.turn_id is None
    with pytest.raises(ValidationError):
        host.console_id = "other"
    with pytest.raises(ValidationError):
        ConsoleHostIdentity(**host.model_dump(), secret="do-not-project")
    # Native session/incarnation/turn/round UUIDs, fleet hex IDs and direct run keys.
    for value in (
        "2b1e334c-6339-4dbd-9a70-ff6e643d673e",
        "b" * 32,
        "direct:message-a:generation-a",
        "message-a:abc123",
    ):
        assert ConsoleTarget(**{**target.model_dump(), "run_id": value}).run_id == value
    for field in ConsoleTarget.model_fields:
        for invalid in ("/private/data", "Chat title", "../run", 42):
            with pytest.raises(ValidationError):
                ConsoleTarget(**{**target.model_dump(), field: invalid})


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_overflow_without_mobile_keeps_connection_and_correlated_responses():
    from tldw_chatbook.Canvas.control_protocol import (
        CanvasControlBroker,
        CanvasControlClient,
    )

    broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
    await broker.start()
    launch = broker.issue_child("child-a")
    client = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
    try:
        events = broker.console_events("child-a")  # Exists before child publication.
        await client.start()
        for _ in range(130):
            await client.send_relay_event(
                "relay.run.status", {"session_id": "chat-a", "status": "running"}
            )
        response = await broker.request("child-a", "health.request", {}, timeout=2)
        assert response.payload["status"] == "ok"
        batch = events.read(0)
        assert batch.sequence == 130
        assert batch.reset_required
        assert len(batch.events) == 64
        assert broker.console_events("child-a") is events
        await broker.revoke_child("child-a")
        await client.wait_disconnected(timeout=1)
    finally:
        await client.aclose()
        await broker.aclose()


@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_control_lives_with_canvas_and_mobile_disabled(tmp_path, monkeypatch):
    from tldw_chatbook.Canvas.control_protocol import CanvasControlClient
    from tldw_chatbook.Canvas.profiles import runtime_snapshot_id
    from tldw_chatbook.Web_Server import serve

    monkeypatch.setattr(serve, "get_canvas_execution_enabled", lambda: False)

    class BaseServer:
        def __init__(self):
            self.host = "127.0.0.1"
            self.port = 8000

        async def on_startup(self, app):
            pass

        async def on_shutdown(self, app):
            pass

    server = serve.build_chatbook_web_server_class(BaseServer)()
    await server.on_startup(None)
    client = None
    try:
        broker = server._canvas_control_broker
        assert broker is not None
        assert server._mobile_service is None
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(
            launch.environment,
            runtime_snapshot_id=runtime_snapshot_id(server._canvas_profile_snapshot),
        )
        await client.start()
        await server._disable_canvas_runtime()
        assert server._canvas_control_broker is broker
        response = await broker.request("child-a", "health.request", {}, timeout=1)
        assert response.payload["status"] == "ok"
        assert not server._canvas_enabled()
    finally:
        if client is not None:
            await client.aclose()
        await server.on_shutdown(None)


@pytest.mark.asyncio
async def test_app_console_transport_refuses_unsupported_without_canvas_or_engine():
    from types import SimpleNamespace

    from tldw_chatbook import app_served
    from tldw_chatbook.Canvas.control_protocol import (
        CONTROL_PROTOCOL_VERSION,
        ControlMessage,
        ControlProtocolError,
    )

    app = SimpleNamespace()
    request = ControlMessage(
        CONTROL_PROTOCOL_VERSION,
        "console.snapshot.request",
        "request-a",
        None,
        {"request": {}},
    )
    with pytest.raises(ControlProtocolError, match="authority_unavailable"):
        await app_served._handle_served_control_request(app, request)


@pytest.mark.asyncio
async def test_app_publisher_starts_after_control_authentication():
    from types import SimpleNamespace

    from tldw_chatbook import app_served

    gate = asyncio.Event()
    attached = []

    async def start():
        await gate.wait()

    app = SimpleNamespace(
        served_canvas_control=SimpleNamespace(start=start),
        served_relay_publisher=SimpleNamespace(
            attach=lambda **kwargs: attached.append(kwargs)
        ),
        _served_canvas_control_start_task=None,
        _served_relay_run_states=lambda: None,
    )
    app_served._start_served_canvas_control(app, app_globals={"asyncio": asyncio})
    await asyncio.sleep(0)
    assert not attached
    gate.set()
    await app._served_canvas_control_start_task
    await asyncio.sleep(0)
    assert len(attached) == 1


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_console_host_identity_is_stable_and_replacement_changes_console():
    from tldw_chatbook.Canvas.control_protocol import CanvasControlBroker

    broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
    await broker.start()
    try:
        broker.issue_child("child-a")
        original = broker.console_identity("child-a")
        assert broker.console_identity("child-a") == original
        broker.issue_child("child-a")
        replacement = broker.console_identity("child-a")
        assert replacement.server_epoch == original.server_epoch
        assert replacement.console_id != original.console_id
    finally:
        await broker.aclose()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [False, True])
async def test_phone_reset_clears_stale_approval_and_shows_unavailable(
    mobile_browser_case, unavailable
):
    from playwright.async_api import expect

    from tldw_chatbook.Web_Server.mobile_service import MobileEvent

    case = mobile_browser_case
    service = case.service
    service.body = "Current native transcript"
    service.decisions = [
        dict(
            service.row("chat-a"),
            kind="approval",
            round_id="approval-before-reset",
            supported=True,
            calls=[
                {
                    "call_id": "call-before-reset",
                    "llm_name": "fs_read",
                    "options": ["approve_once", "deny"],
                    "arguments_summary": "/notes/readme",
                }
            ],
        )
    ]
    await case.open_chat("chat-a")
    page = case.page
    decision = page.get_by_role("button", name="Submit decision", exact=True)
    await expect(decision).to_be_enabled()
    await page.locator("#chat-draft").fill("Keep this unsent draft")
    send = page.get_by_role("button", name="Send", exact=True)
    stop = page.get_by_role("button", name="Stop", exact=True)
    await expect(send).to_be_enabled()
    await expect(stop).to_be_enabled()

    async def answerability(value, after=0):
        async with asyncio.timeout(5):
            while not any(
                name == "answerable"
                and args[-1] is value
                and args[1].round_id == "approval-before-reset"
                and args[1].session_id == "chat-a"
                and args[1].server_epoch == "epoch"
                for name, args in service.calls[after:]
            ):
                await asyncio.sleep(0.01)

    await answerability(True)
    before_reset = len(service.calls)
    service.metadata_gate = asyncio.Event()
    service.decisions = []
    try:
        await service.events_queue.put(
            MobileEvent("reset", "", {"reset_required": True}, 2)
        )
        await expect(page.locator("#connection-state")).to_contain_text("uncertain")
        await expect(decision).to_be_disabled()
        await expect(send).to_be_disabled()
        await expect(stop).to_be_disabled()
        await answerability(False, before_reset)
        assert not service.tickets and not service.commands
        assert not service.metadata_gate.is_set()

        service.unavailable = unavailable
        if unavailable:
            async with page.expect_response(
                lambda response: (
                    "/console/snapshot" in response.url and response.status == 503
                )
            ) as failed_snapshot:
                service.metadata_gate.set()
            await (await failed_snapshot.value).finished()
            await expect(decision).to_be_disabled()
            await expect(send).to_be_disabled()
            assert not service.tickets and not service.commands
            service.unavailable = False
            await service.events_queue.put(MobileEvent("console.changed", "", {}, 3))
        else:
            service.metadata_gate.set()
        await expect(decision).to_have_count(0)
        await expect(page.locator("#connection-state")).to_contain_text("Connected")
        await expect(send).to_be_enabled()
        await expect(stop).to_be_enabled()
        await expect(page.locator("#chat-draft")).to_have_value(
            "Keep this unsent draft"
        )
        assert not service.tickets and not service.commands
    finally:
        service.metadata_gate.set()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize("forged", ["old_version", "authority_field"])
async def test_overflow_does_not_hide_genuine_malformed_control_failure(forged):
    import json
    import struct

    from tldw_chatbook.Canvas.control_protocol import (
        CONTROL_PROTOCOL_VERSION,
        CanvasControlBroker,
        CanvasControlClient,
        ControlProtocolError,
    )

    broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
    await broker.start()
    launch = broker.issue_child("child-a")
    client = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
    try:
        await client.start()
        for _ in range(130):
            await client.send_relay_event(
                "relay.run.status", {"session_id": "chat-a", "status": "running"}
            )
        await broker.request("child-a", "health.request", {}, timeout=1)
        payload = {"sequence": 131}
        if forged == "authority_field":
            payload["authority"] = "approve"
        body = json.dumps(
            {
                "version": 2 if forged == "old_version" else CONTROL_PROTOCOL_VERSION,
                "type": "console.changed",
                "request_id": "forged-a",
                "deadline_ms": None,
                "payload": payload,
            }
        ).encode()
        client._writer.write(struct.pack(">I", len(body)) + body)
        await client._writer.drain()
        await client.wait_disconnected(timeout=1)
        with pytest.raises(ControlProtocolError, match="child_not_connected"):
            await broker.request("child-a", "health.request", {}, timeout=1)
    finally:
        await client.aclose()
        await broker.aclose()
