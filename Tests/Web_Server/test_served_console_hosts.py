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


async def test_buffered_output_yields_to_already_waiting_receiver(recording_host):
    host, service = recording_host
    attachment = await host.attach_terminal("receiving-browser", 80, 24)
    for _ in range(2):
        assert (await attachment.receive())[0] == "str"
    assert attachment.begun and attachment.ready
    entered = asyncio.Event()
    frames = [f"paint-{index:03d}".encode() for index in range(100)]

    async def receive_frames():
        entered.set()
        return [await attachment.receive() for _ in frames]

    receiver = asyncio.create_task(receive_frames())
    try:
        await entered.wait()
        # Buffered native output can call this callback repeatedly without I/O.
        # The already-waiting consumer must get a chance to drain its queue.
        for frame in frames:
            await service.write_bytes(frame)
        assert not attachment.closed.is_set()
        assert host.current_attachment is attachment
        assert await asyncio.wait_for(receiver, 5) == [
            ("bytes", frame) for frame in frames
        ]
        assert attachment._queued_bytes == 0
        assert service.stop_calls == 0
    finally:
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)


@pytest.mark.parametrize("limit", ["frames", "bytes"])
async def test_stalled_receiver_keeps_exact_output_limits(recording_host, limit):
    host, service = recording_host
    attachment = await host.attach_terminal("stalled-browser", 80, 24)
    for _ in range(2):
        await attachment.receive()
    frames = [b"x"] * 64 if limit == "frames" else [b"x" * (1024 * 1024)]
    for frame in frames:
        await service.write_bytes(frame)
    assert not attachment.closed.is_set()
    assert attachment.output.qsize() == len(frames)
    assert attachment._queued_bytes == sum(map(len, frames))
    await service.write_bytes(b"overflow")
    assert attachment.closed.is_set()
    assert host.current_attachment is None
    assert attachment.output.empty() and attachment._queued_bytes == 0
    assert service.stop_calls == 0


@pytest.mark.parametrize("interruption", ["cancel", "replace"])
async def test_output_yield_cannot_retire_successor(recording_host, interruption):
    host, service = recording_host
    previous = await host.attach_terminal("previous-browser", 80, 24)
    for _ in range(2):
        await previous.receive()
    publisher = asyncio.create_task(service.write_bytes(b"old paint"))
    try:
        await asyncio.sleep(0)
        assert previous.output.qsize() == 1
        assert not publisher.done()
        if interruption == "cancel":
            publisher.cancel()
            with pytest.raises(asyncio.CancelledError):
                await publisher
        successor = await host.attach_terminal("successor-browser", 80, 24)
        if interruption == "replace":
            await publisher
        await previous.detach()
        assert host.current_attachment is successor
        assert not successor.closed.is_set()
        assert successor.begun and successor.ready
        await successor.send_bytes(b"current input")
        assert service.input_bytes == [b"current input"]
        assert service.stop_calls == 0
    finally:
        publisher.cancel()
        await asyncio.gather(publisher, return_exceptions=True)


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

    monkeypatch.setattr(serve.os, "killpg", unexpected_signal, raising=False)
    service = serve.build_chatbook_app_service_class(ReapedService)(
        canvas_control_broker=None
    )
    await service.stop()


async def test_served_spawn_uses_exact_argv_and_interactive_stdin(tmp_path):
    """The retained process is the app, even when command metadata uses a shell."""
    import os
    import shlex
    import subprocess
    import sys

    from textual_serve.app_service import AppService

    from tldw_chatbook.Web_Server import serve

    child = r"""
import json, os, sys
print(json.dumps([os.getpid(), sys.argv[1:]]), flush=True)
while header := sys.stdin.buffer.read(5):
    if len(header) != 5:
        break
    payload = sys.stdin.buffer.read(int.from_bytes(header[1:], 'big'))
    print(json.dumps([header[:1].decode(), payload.decode()]), flush=True)
"""
    argv = (sys.executable, "-I", "-S", "-u", "-c", child, "two words", "literal '$;&")
    command = (
        subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv) + "; :"
    )

    async def ignore(*args, **kwargs):
        pass

    service = serve.build_chatbook_app_service_class(AppService)(
        command,
        write_bytes=ignore,
        write_str=ignore,
        close=ignore,
        download_manager=None,
        canvas_control_broker=None,
    )
    # This lets the test-only RED packet exercise the old implementation too.
    service._app_argv = argv
    process = await service._open_app_process()
    try:
        async with asyncio.timeout(2):
            pid, arguments = json.loads(await process.stdout.readline())
            assert pid == process.pid
            assert arguments == list(argv[6:])
            for payload in (b"first", b"second"):
                assert await service.send_bytes(payload)
                assert json.loads(await process.stdout.readline()) == [
                    "D",
                    payload.decode(),
                ]
    finally:
        if process.returncode is None:
            if os.name == "posix":
                os.killpg(process.pid, __import__("signal").SIGKILL)
            else:
                process.kill()
        await asyncio.wait_for(process.wait(), 2)


async def test_served_retirement_windows_never_evaluates_sigkill_or_killpg(monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.Web_Server import serve

    operations = []

    class Process:
        pid = 123456
        returncode = None

        async def wait(self):
            operations.append("wait")
            if self.returncode is None:
                raise TimeoutError
            return self.returncode

        def terminate(self):
            operations.append("terminate")

        def kill(self):
            operations.append("kill")
            self.returncode = 1

    class Base:
        def __init__(self):
            self._process = Process()

        async def stop(self):
            pass

    monkeypatch.setattr(serve, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(serve, "signal", SimpleNamespace(SIGTERM=15))
    service = serve.build_chatbook_app_service_class(Base)(canvas_control_broker=None)
    await service.stop()
    assert operations == ["wait", "terminate", "wait", "kill", "wait"]
    assert service._process.returncode == 1


@pytest.mark.skipif(
    __import__("os").name != "posix", reason="POSIX shell/group witness"
)
async def test_served_retirement_reaps_actual_app_before_revoking_broker():
    import os
    import shlex
    import signal
    import sys

    from textual_serve.app_service import AppService

    from tldw_chatbook.Web_Server import serve

    child = r"""
import os, signal, time
signal.signal(signal.SIGTERM, lambda *_: None)
print(os.getpid(), flush=True)
time.sleep(20)
"""
    argv = (sys.executable, "-I", "-S", "-u", "-c", child)
    pid = None
    revoked = []

    async def ignore(*args, **kwargs):
        pass

    class Broker:
        async def revoke_child(self, child_id):
            assert process.returncode is not None
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            revoked.append(child_id)

    service = serve.build_chatbook_app_service_class(AppService)(
        shlex.join(argv) + "; :",
        write_bytes=ignore,
        write_str=ignore,
        close=ignore,
        download_manager=None,
        canvas_control_broker=Broker(),
    )
    service._app_argv = argv
    process = await service._open_app_process()
    try:
        pid = int(await asyncio.wait_for(process.stdout.readline(), 2))
        service._canvas_control_started = True
        # AppService.stop has no pipe observer here: its completion cannot prove exit.
        await service.stop()
        assert revoked == [service.app_service_id]
        assert process.returncode is not None
    finally:
        if process.returncode is None or (pid is not None and pid != process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await asyncio.wait_for(process.wait(), 2)
        if pid is not None:
            async with asyncio.timeout(2):
                while True:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    await asyncio.sleep(0.01)


async def test_served_spawn_without_argv_refuses_before_creating_process(monkeypatch):
    from tldw_chatbook.Web_Server import serve

    class Base:
        pass

    async def unexpected_spawn(*args, **kwargs):
        pytest.fail("missing argv must not create a process")

    monkeypatch.setattr(serve.asyncio, "create_subprocess_exec", unexpected_spawn)
    service = serve.build_chatbook_app_service_class(Base)(canvas_control_broker=None)
    with pytest.raises(RuntimeError, match="launch argv is unavailable"):
        await service._open_app_process()
