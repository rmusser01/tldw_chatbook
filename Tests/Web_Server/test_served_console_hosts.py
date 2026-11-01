"""Retained child ownership and replaceable terminal attachment contracts."""

import asyncio
import json

import pytest

from tldw_chatbook.Web_Server.served_console_hosts import (
    ServedConsoleHost,
    ServedConsoleHosts,
    StaleTerminalAttachment,
)


class RecordingAppService:
    def __init__(self, *, write_bytes, write_str, close):
        self.write_bytes = write_bytes
        self.write_str = write_str
        self.close = close
        self.input_bytes = []
        self.operations = []
        self.stop_calls = 0
        self.start_calls = 0
        self.app_service_id = "recording-child"

    async def start(self, width, height):
        self.start_calls += 1

    async def stop(self):
        self.stop_calls += 1

    async def send_bytes(self, data):
        self.input_bytes.append(data)

    async def send_meta(self, data):
        self.operations.append(data)
        if data["type"] == "console_reattach":
            for kind in ("console_attachment_begin", "console_attachment_ready"):
                await self.write_str(
                    json.dumps([kind, {"generation": data["generation"]}])
                )

    async def set_terminal_size(self, width, height):
        self.operations.append((width, height))


@pytest.fixture
async def recording_host():
    host = ServedConsoleHost(RecordingAppService)
    await host.start()
    yield host, host.service
    await host.aclose()


async def test_old_attachment_cannot_resize_or_detach_successor(recording_host):
    host, service = recording_host
    first = await host.attach_terminal("browser-a", 80, 24)
    second = await host.attach_terminal("browser-b", 100, 30)
    with pytest.raises(StaleTerminalAttachment):
        await first.resize(10, 10)
    await first.detach()
    await second.send_bytes(b"x")
    assert service.input_bytes == [b"x"]
    assert service.stop_calls == 0
    assert host.current_attachment is second


async def test_concurrent_start_survives_cancelled_browser():
    entered, release = asyncio.Event(), asyncio.Event()
    hosts = []

    class SlowService(RecordingAppService):
        async def start(self, width, height):
            entered.set()
            await release.wait()
            await super().start(width, height)

    def factory(identity):
        host = ServedConsoleHost(SlowService)
        hosts.append(host)
        return host

    registry = ServedConsoleHosts(factory)
    first = asyncio.create_task(registry.get_or_start("admitted-profile"))
    await entered.wait()
    second = asyncio.create_task(registry.get_or_start("admitted-profile"))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert await second is hosts[0]
    assert await registry.get_or_start("admitted-profile") is hosts[0]
    assert len(hosts) == hosts[0].service.start_calls == 1
    await registry.aclose()
    assert hosts[0].service.stop_calls == 1
    with pytest.raises(RuntimeError):
        await registry.get_or_start("admitted-profile")


async def test_output_pressure_detaches_browser_without_stopping_child(recording_host):
    host, service = recording_host
    attachment = await host.attach_terminal("slow-browser", 80, 24)
    for _ in range(1000):
        await service.write_bytes(b"paint")
    assert host.current_attachment is None
    assert attachment.closed.is_set()
    assert attachment.output.qsize() <= 64
    assert service.stop_calls == 0
    successor = await host.attach_terminal("new-browser", 120, 40)
    await successor.send_bytes(b"still alive")
    assert service.input_bytes == [b"still alive"]


async def test_output_and_input_wait_for_current_render_boundary(recording_host):
    host, service = recording_host

    async def capture(data):
        service.operations.append(data)

    service.send_meta = capture
    attachment = await host.attach_terminal("browser", 80, 24)
    await service.write_bytes(b"stale paint")
    with pytest.raises(StaleTerminalAttachment):
        await attachment.send_bytes(b"premature")
    await service.write_str(
        json.dumps(["console_attachment_begin", {"generation": "old"}])
    )
    assert attachment.output.empty()
    await service.write_str(
        json.dumps(["console_attachment_begin", {"generation": attachment.generation}])
    )
    await service.write_bytes(b"full paint")
    await service.write_str(
        json.dumps(["console_attachment_ready", {"generation": attachment.generation}])
    )
    await attachment.send_bytes(b"ready")
    assert service.input_bytes == [b"ready"]
    assert (await attachment.output.get())[0] == "str"
    assert await attachment.output.get() == ("bytes", b"full paint")


async def test_saved_hydration_is_reserved_once_and_save_registers_identity(
    recording_host,
):
    host, _ = recording_host
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def hydrate():
        calls.append("hydrate")
        entered.set()
        await release.wait()
        return "native-session"

    first = asyncio.create_task(host.reserve_saved_conversation("saved-chat", hydrate))
    await entered.wait()
    second = asyncio.create_task(host.reserve_saved_conversation("saved-chat", hydrate))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    assert await second == "native-session"
    assert calls == ["hydrate"]
    host.register_saved_conversation("newly-saved-chat", "another-session")
    assert (
        await host.reserve_saved_conversation("newly-saved-chat", hydrate)
        == "another-session"
    )
    with pytest.raises(RuntimeError):
        host.register_saved_conversation("newly-saved-chat", "competing-session")


async def test_failed_retirement_keeps_registry_closed_to_replacement():
    class HangingService(RecordingAppService):
        async def stop(self):
            self.stop_calls += 1
            raise TimeoutError("physical child still alive")

    registry = ServedConsoleHosts(lambda identity: ServedConsoleHost(HangingService))
    host = await registry.get_or_start("admitted-profile")
    with pytest.raises(TimeoutError):
        await registry.aclose()
    with pytest.raises(RuntimeError):
        await registry.get_or_start("admitted-profile")
    assert host.service.start_calls == 1


async def test_detached_host_stays_available_to_phone_projection_observers():
    registry = ServedConsoleHosts(
        lambda identity: ServedConsoleHost(RecordingAppService)
    )
    host = await registry.get_or_start("admitted-profile")
    attachment = await host.attach_terminal("browser", 80, 24)
    await attachment.detach()
    assert registry.child_ids() == (host.service.app_service_id,)
    await registry.aclose()
    assert registry.child_ids() == ()


async def test_closed_child_replacement_requires_observed_process_exit():
    from types import SimpleNamespace

    created = []

    def factory(identity):
        host = ServedConsoleHost(RecordingAppService)
        host.service._process = SimpleNamespace(returncode=None)
        created.append(host)
        return host

    registry = ServedConsoleHosts(factory)
    host = await registry.get_or_start("profile")
    await host._child_closed()
    assert await registry.get_or_start("profile") is host
    with pytest.raises(RuntimeError):
        await host.attach_terminal("replacement", 80, 24)
    host.service._process.returncode = 0
    replacement = await registry.get_or_start("profile")
    assert replacement is not host
    assert len(created) == 2
    await registry.aclose()


async def test_cancelled_shutdown_observer_does_not_cancel_retirement():
    entered, release, retired = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class SlowRetirement(RecordingAppService):
        async def stop(self):
            self.stop_calls += 1
            entered.set()
            await release.wait()
            retired.set()

    registry = ServedConsoleHosts(lambda identity: ServedConsoleHost(SlowRetirement))
    host = await registry.get_or_start("admitted-profile")
    observer = asyncio.create_task(registry.aclose())
    await entered.wait()
    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert retired.is_set()
    await registry.aclose()
    assert host.service.stop_calls == 1


async def test_service_does_not_signal_an_already_reaped_child(monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.Web_Server import serve

    class ReapedService:
        def __init__(self):
            self._process = SimpleNamespace(returncode=0, pid=123456)

        async def stop(self):
            await asyncio.Event().wait()

    def unexpected_signal(*args):
        pytest.fail("an already reaped child must never be signalled")

    monkeypatch.setattr(serve.os, "killpg", unexpected_signal)
    service = serve.build_chatbook_app_service_class(ReapedService)(
        canvas_control_broker=None
    )
    await service.stop()
