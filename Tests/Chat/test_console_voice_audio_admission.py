"""Duplex takes app audio only after preflight and retains an unsettled launch."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_voice_process import bootstrap, ready_true
from tldw_chatbook.Chat import console_voice_process as voice
from tldw_chatbook.UI.audio_output_admission import acquire_audio_output
from tldw_chatbook.TTS.audio_output_ownership import AudioOutputOwnership, OutputBlocked


def admit(host):
    from functools import partial

    return partial(
        acquire_audio_output,
        SimpleNamespace(audio_output=host),
        host.begin_intent(),
        "Hands-free voice",
    )


@pytest.mark.asyncio
async def test_invalid_preflight_preserves_active_output():
    host = AudioOutputOwnership()
    previous = await host.acquire("speech", lambda: asyncio.sleep(0, result=True))
    owner = voice.ConsoleVoiceProcess(
        voice.DeviceLease(),
        admit_audio=admit(host),
        popen=lambda *a, **k: pytest.fail("spawned"),
    )
    with pytest.raises(voice.VoiceProcessError):
        await owner.enter(
            bootstrap(voice),
            preflight=lambda: asyncio.sleep(0, result=False),
            current=lambda: True,
        )
    assert (await owner.close()).confirmed
    assert previous.is_current()


@pytest.mark.asyncio
async def test_close_during_held_spawn_retains_slot_and_quarantines_unknown_result():
    host = AudioOutputOwnership()
    entered, finish = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()

    def held(*a, **kw):
        assert threading.get_ident() != loop_thread
        loop.call_soon_threadsafe(entered.set)
        assert finish.wait(3)
        raise RuntimeError("opaque launch failure")

    owner = voice.ConsoleVoiceProcess(
        voice.DeviceLease(),
        popen=held,
        admit_audio=admit(host),
    )
    entry = asyncio.create_task(
        owner.enter(bootstrap(voice), preflight=ready_true, current=lambda: True)
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert host.owner == "Hands-free voice"
        takeover = asyncio.create_task(
            host.acquire("media", lambda: asyncio.sleep(0, result=True))
        )
        await asyncio.sleep(0)
        assert not takeover.done()
        assert owner.process is None
        finish.set()
        with pytest.raises(voice.VoiceProcessError):
            await entry
        with pytest.raises(OutputBlocked):
            await takeover
        assert not (await owner.close()).confirmed
        assert owner.device_lease.quarantined
    finally:
        finish.set()
        await asyncio.gather(entry, return_exceptions=True)


@pytest.mark.asyncio
async def test_listening_session_holds_output_until_checked_close(monkeypatch):
    import os
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    host = AudioOutputOwnership()
    supervisor = voice.VoiceProcessSupervisor()
    process = SimpleNamespace(
        pid=8,
        stdin=await asyncio.to_thread(open, os.devnull, "wb"),
        stdout=await asyncio.to_thread(open, os.devnull, "rb"),
        returncode=None,
    )
    process.poll = lambda: process.returncode
    tree_threads = []
    loop = asyncio.get_running_loop()
    attaching, finish_attach = asyncio.Event(), threading.Event()

    def attach():
        tree_threads.append(threading.get_ident())
        loop.call_soon_threadsafe(attaching.set)
        assert finish_attach.wait(3)

    def tree_factory(_pid):
        tree_threads.append(threading.get_ident())
        return SimpleNamespace(
            attached=True,
            attach=attach,
            observe_absence=lambda: process.returncode is not None,
        )

    session = voice.ConsoleVoiceProcess(
        supervisor.device_lease,
        admit_audio=admit(host),
        popen=lambda *a, **k: process,
        tree_factory=tree_factory,
    )
    supervisor.retain(session, session_id="live")
    commands = []

    class Pipe:
        def __init__(self, read_fd, _write_fd, **kwargs):
            self.read_fd = read_fd
            self.writer = SimpleNamespace(
                wait_written=lambda *_: True, failure=None, close=lambda: None
            )
            self.reader = SimpleNamespace(failure=None)
            self.input = SimpleNamespace(count=0)

        def send(self, op, **kw):
            commands.append(op)
            if op == "bootstrap":
                session._gate.hello(session._gate.identity)
                session._gate.stt_ready()
            elif op == "start":
                assert session._gate.permitted
                session.ready = True
            elif op == "close":
                session.native_closed = True
                session.resources_outcome = AttemptCleanupOutcome.CLEAN
                process.returncode = 0

        async def join(self):
            return True

        def stop(self):
            os.close(self.read_fd)

    monkeypatch.setattr(voice, "LifecyclePipe", Pipe)
    try:
        entering = asyncio.create_task(
            session.enter(bootstrap(voice), preflight=ready_true, current=lambda: True)
        )
        await asyncio.wait_for(attaching.wait(), 2)
        assert not entering.done() and host.owner == "Hands-free voice"
        finish_attach.set()
        await entering
        assert session.ready and host.owner == "Hands-free voice"
        next_owner = await host.acquire(
            "preview", lambda: asyncio.sleep(0, result=True)
        )
        assert next_owner.is_current()
        assert (await session.close()).confirmed
        assert commands == ["bootstrap", "start", "close"]
        assert not supervisor.device_lease.quarantined
        assert supervisor.close_task is None
        assert all(ident != threading.get_ident() for ident in tree_threads)
    finally:
        finish_attach.set()
        await session.close()
        process.stdin.close()
        process.stdout.close()
