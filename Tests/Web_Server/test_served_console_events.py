"""Bounded served projection history, independent of subscriber lifetime."""

import asyncio

import pytest
from pydantic import ValidationError


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


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_mobile_observes_without_consuming_and_drops_stale_approval_on_gap(
    tmp_path,
):
    from tldw_chatbook.Canvas.control_protocol import (
        CanvasControlBroker,
        CanvasControlClient,
    )
    from tldw_chatbook.Web_Server.mobile_service import MobileChatService

    broker = CanvasControlBroker(runtime_snapshot_id="a" * 64, max_queued_events=2)
    await broker.start()
    launch = broker.issue_child("child-a")
    client = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
    service = MobileChatService(data_dir=tmp_path / "mobile", provider_gateway=None)
    try:
        await client.start()
        await client.send_relay_event(
            "relay.approval.raised",
            {
                "round_id": "round-a",
                "session_id": "chat-a",
                "calls": [],
                "timeout_seconds": 0,
            },
        )
        await broker.request("child-a", "health.request", {}, timeout=1)
        service.attach_relay(broker, child_ids=lambda: ["child-a"])
        for _ in range(100):
            if service._desktop_approvals:
                break
            await asyncio.sleep(0.01)
        assert ("child-a", "round-a") in service._desktop_approvals
        assert len(broker.console_events("child-a").read(0).events) == 1
        for _ in range(7):
            await client.send_relay_event(
                "relay.run.status", {"session_id": "chat-a", "status": "running"}
            )
        await broker.request("child-a", "health.request", {}, timeout=1)
        for _ in range(100):
            if not service._desktop_approvals:
                break
            await asyncio.sleep(0.01)
        assert not service._desktop_approvals
        assert not await service._decide_desktop_round(("child-a", "round-a"), {})
    finally:
        await service.shutdown()
        await client.aclose()
        await broker.aclose()


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
async def test_paused_mobile_observer_cannot_decide_against_replacement_child(tmp_path):
    from tldw_chatbook.Canvas.control_protocol import (
        CONTROL_PROTOCOL_VERSION,
        CanvasControlBroker,
        CanvasControlClient,
        ControlMessage,
    )
    from tldw_chatbook.Web_Server.mobile_service import MobileChatService

    broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
    await broker.start()
    launch = broker.issue_child("child-a")
    client = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
    service = MobileChatService(data_dir=tmp_path / "mobile", provider_gateway=None)
    service._relay_broker = broker
    observer = None
    replacement = None
    decisions = []

    async def handler(request):
        decisions.append(request)
        return ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            "relay.decide.response",
            request.request_id,
            None,
            {"round_id": "round-a", "delivered": True},
        )

    try:
        await client.start()
        await client.send_relay_event(
            "relay.approval.raised",
            {
                "round_id": "round-a",
                "session_id": "chat-a",
                "calls": [],
                "timeout_seconds": 0,
            },
        )
        await broker.request("child-a", "health.request", {}, timeout=1)
        observer = asyncio.create_task(service._drain_relay_child(broker, "child-a"))
        for _ in range(100):
            if service._desktop_approvals:
                break
            await asyncio.sleep(0.01)
        assert service._desktop_approvals
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        launch = broker.issue_child("child-a")
        replacement = CanvasControlClient(
            launch.environment, runtime_snapshot_id="a" * 64, handler=handler
        )
        await replacement.start()
        assert not await service._decide_desktop_round(("child-a", "round-a"), {})
        assert not decisions
    finally:
        if observer is not None:
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
        await service.shutdown()
        await client.aclose()
        if replacement is not None:
            await replacement.aclose()
        await broker.aclose()


@pytest.mark.asyncio
async def test_paused_mobile_subscription_retains_one_sticky_reset_and_bounded_memory(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Web_Server.mobile_service import MobileChatService

    service = MobileChatService(data_dir=tmp_path / "mobile", provider_gateway=None)

    async def already_started():
        pass

    monkeypatch.setattr(service, "_ensure_started", already_started)
    events = service.events()
    first = asyncio.create_task(anext(events))
    await asyncio.sleep(0)
    service._emit("run_status", "chat-a", {"status": "running"})
    assert (await first).type == "run_status"
    for revision in range(1000):
        service._emit("delta", "chat-a", {"text": str(revision)})
    queue = next(iter(service._subscribers))
    assert queue.qsize() <= 64
    reset = await anext(events)
    assert reset.type == "reset"
    assert reset.payload["reset_required"]
    for revision in range(1000):
        service._emit("delta", "chat-a", {"text": str(revision)})
    assert queue.empty()  # No stale projections resume before a fresh subscription.
    await events.aclose()
    reopened = service.events()
    first = asyncio.create_task(anext(reopened))
    await asyncio.sleep(0)
    service._emit("run_status", "chat-a", {"status": "current"})
    assert (await asyncio.wait_for(first, 1)).type == "run_status"
    await reopened.aclose()
    await service.shutdown()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize("child_reset", [False, True])
async def test_phone_reset_clears_stale_approval_and_shows_unavailable(
    monkeypatch, child_reset
):
    from pathlib import Path

    from playwright.async_api import async_playwright

    from Tests.Web_Server.test_served_shell_browser import (
        _real_playwright_browsers_path,
    )

    cache = _real_playwright_browsers_path()
    if cache:
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", cache)
    assets = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "Web_Server"
        / "static"
        / "mobile"
    )
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_content(assets.joinpath("index.html").read_text())
            await page.evaluate("""() => {
              window.fetch = async (path) => ({ok: true, json: async () => path.endsWith('/approvals') ? {approvals: [{round_id: 'round-a', surface: 'desktop', child_id: 'child-a', calls: [{call_id: 'call-a', llm_name: 'fs_write', options: ['deny']}]}]} : {items: []}});
              window.WebSocket = class extends EventTarget { constructor() { super(); window.mobileSocket = this; } };
            }""")
            await page.clock.install()
            await page.add_script_tag(content=assets.joinpath("app.js").read_text())
            await page.locator("#approvals-list button").wait_for(state="attached")
            reset_payload = {"reset_required": True}
            if child_reset:
                reset_payload["child_id"] = "child-a"
            await page.evaluate(
                """payload => window.mobileSocket.dispatchEvent(new MessageEvent('message', {data: JSON.stringify({type:'reset', payload, session_id:''})}))""",
                reset_payload,
            )
            notice = await page.locator("#status-line").inner_text()
            assert ("Refresh this page" in notice) is not child_reset
            assert await page.locator("#approvals-list button").count() == 0
            assert (
                "unavailable"
                in (await page.locator("#status-line").inner_text()).lower()
            )
            assert await page.locator("#status-line").is_visible()
            await page.locator("#approvals-button").click()
            await page.clock.run_for(20)
            assert await page.locator("#approvals-list button").count() == 0
            await page.evaluate("""() => {
                window.fetch = async () => { throw new Error('offline'); };
                document.dispatchEvent(new Event('visibilitychange'));
            }""")
            await page.clock.run_for(6100)
            assert await page.locator("#status-line").is_visible()
            assert (
                "unavailable"
                in (await page.locator("#status-line").inner_text()).lower()
            )
            if child_reset:
                page = await browser.new_page()
                await page.set_content(assets.joinpath("index.html").read_text())
                await page.evaluate(
                    """payload => {
                    window.fetch = async () => ({ok: true, json: async () => ({items: [], approvals: []})});
                    window.WebSocket = class extends EventTarget {
                        constructor() {
                            super();
                            queueMicrotask(() => this.dispatchEvent(new MessageEvent('message', {data: JSON.stringify({type: 'reset', payload, session_id: ''})})));
                        }
                    };
                }""",
                    reset_payload,
                )
                await page.add_script_tag(content=assets.joinpath("app.js").read_text())
                await page.locator("#status-line").wait_for(state="visible")
                assert (
                    "unavailable"
                    in (await page.locator("#status-line").inner_text()).lower()
                )
                assert (
                    "Refresh this page"
                    not in await page.locator("#status-line").inner_text()
                )
                assert await page.locator("#approvals-list button").count() == 0
        finally:
            await browser.close()


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


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_late_subscriptions_inherit_idle_child_projection_loss(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Canvas.control_protocol import (
        CanvasControlBroker,
        CanvasControlClient,
    )
    from tldw_chatbook.Web_Server.mobile_service import MobileChatService

    broker = CanvasControlBroker(runtime_snapshot_id="a" * 64, max_queued_events=2)
    await broker.start()
    launch = broker.issue_child("child-a")
    client = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
    service = MobileChatService(data_dir=tmp_path / "mobile", provider_gateway=None)
    observed_reset = asyncio.Event()
    emit = service._emit

    def observe(type_, session_id, payload):
        emit(type_, session_id, payload)
        if type_ == "reset":
            observed_reset.set()

    async def already_started():
        pass

    monkeypatch.setattr(service, "_emit", observe)
    monkeypatch.setattr(service, "_ensure_started", already_started)
    subscriptions = []
    try:
        await client.start()
        for _ in range(7):
            await client.send_relay_event(
                "relay.run.status", {"session_id": "chat-a", "status": "running"}
            )
        await broker.request("child-a", "health.request", {}, timeout=1)
        service.attach_relay(broker, child_ids=lambda: ["child-a"])
        await asyncio.wait_for(observed_reset.wait(), 1)
        sequence = broker.console_events("child-a").read(0).sequence
        # Both pages subscribe after the observer handled loss, with no new child event.
        for _ in range(2):
            events = service.events()
            subscriptions.append(events)
            reset = await asyncio.wait_for(anext(events), 0.2)
            assert reset.type == "reset"
            assert reset.payload == {"child_id": "child-a", "reset_required": True}
            assert broker.console_events("child-a").read(0).sequence == sequence
    finally:
        for events in subscriptions:
            await events.aclose()
        await service.shutdown()
        await client.aclose()
        await broker.aclose()
