"""Initialization ownership with gated fake subprocesses and native loading."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends import chatterbox


class Process:
    def __init__(self):
        self.returncode = None
        self.stdout = asyncio.StreamReader()
        self.commands = []
        self.waited = False
        self.stdin = SimpleNamespace(write=self.write, drain=self.drain)

    def write(self, data):
        self.commands.append(json.loads(data))

    async def drain(self):
        pass

    def terminate(self):
        self.returncode = -15
        self.stdout.feed_eof()

    def kill(self):
        self.returncode = -9

    async def wait(self):
        self.waited = True
        return self.returncode


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    monkeypatch.setattr(chatterbox, "get_audio_service", lambda: None)
    monkeypatch.setattr(chatterbox, "check_dependency", lambda *args: True)
    backend = chatterbox.ChatterboxTTSBackend(
        {"CHATTERBOX_DEVICE": "cpu", "CHATTERBOX_VOICE_DIR": str(tmp_path)}
    )
    tasks = []
    create_task = asyncio.create_task

    def track(coro, **kwargs):
        task = create_task(coro, **kwargs)
        tasks.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", track)
    yield backend
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)


async def turn():
    # Give queued tasks a chance to reach the explicit gates used below.
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_close_before_startup_runs_prevents_child_launch(owner, monkeypatch):
    calls = []

    async def spawn(*args, **kwargs):
        calls.append(Process())
        return calls[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    await owner.initialize()
    assert not owner._initialized
    await owner.close()
    await turn()
    assert calls == []
    assert owner.process is owner.model is None
    assert not owner._initializing
    await owner.initialize()
    await turn()
    assert calls == []


@pytest.mark.asyncio
async def test_close_waits_for_inflight_spawn_and_reaps_its_child(owner, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    process = Process()

    async def spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    await owner.initialize()
    await asyncio.wait_for(entered.wait(), 1)
    closing = asyncio.create_task(owner.close())
    try:
        await turn()
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing, 2)
    assert process.waited
    assert process.returncode is not None
    assert owner.process is owner.model is None
    assert not owner._initialized
    assert not owner._initializing


@pytest.mark.asyncio
async def test_close_cancels_readiness_before_it_can_publish_success(
    owner, monkeypatch
):
    entered, release = asyncio.Event(), asyncio.Event()
    process = Process()

    async def spawn(*args, **kwargs):
        return process

    async def response(**kwargs):
        entered.set()
        await release.wait()
        return {"type": "success"}

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(owner, "_read_response", response)
    await owner.initialize()
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.wait_for(owner.close(), 2)
    release.set()
    await turn()
    assert process.waited
    assert owner.process is owner.model is None
    assert not owner._initialized
    assert not owner._initializing


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["missing_wrapper", "spawn_error", "send_error"])
async def test_close_joins_native_fallback_before_releasing_model(
    owner, monkeypatch, reason
):
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    load_calls = []
    process = Process()

    def load():
        load_calls.append(True)
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(3)
        owner.model = object()
        owner._initialized = True

    async def spawn(*args, **kwargs):
        if reason == "send_error":
            return process
        raise OSError("synthetic spawn failure")

    def failed_write(data):
        raise BrokenPipeError("synthetic initialization send failure")

    if reason == "missing_wrapper":
        monkeypatch.setattr(chatterbox.Path, "exists", lambda self: False)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(owner, "_initialize_sync", load)
    if reason == "send_error":
        process.stdin.write = failed_write
    await owner.initialize()
    await asyncio.wait_for(entered.wait(), 1)
    closing = None
    try:
        initialization = owner._initialization_task
        await owner.initialize()
        assert owner._initialization_task is initialization
        assert load_calls == [True]
        closing = asyncio.create_task(owner.close())
        await turn()
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing or owner.close(), 2)
    await turn()
    assert owner.model is None
    assert not owner._initialized
    assert not owner._initializing
    if reason == "send_error":
        assert process.waited


@pytest.mark.asyncio
async def test_repeated_cancelled_close_retains_spawn_cleanup(owner, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    process = Process()

    async def spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    await owner.initialize()
    await asyncio.wait_for(entered.wait(), 1)
    first = asyncio.create_task(owner.close())
    second = asyncio.create_task(owner.close())
    try:
        await turn()
        first.cancel()
        await turn()
        first.cancel()
        await turn()
        assert not first.done()
        assert not second.done()
    finally:
        release.set()
        results = await asyncio.wait_for(
            asyncio.gather(first, second, return_exceptions=True), 2
        )
    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1] is None
    await owner.close()
    assert process.waited
    assert owner.process is owner.model is None


@pytest.mark.asyncio
async def test_nonblocking_initialization_is_shared_until_ready(owner, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    process = Process()
    calls = 0

    async def spawn(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    await owner.initialize()
    await asyncio.wait_for(entered.wait(), 1)
    await owner.initialize()
    assert calls == 1
    assert not owner._initialized
    release.set()
    process.stdout.feed_data(b'{"type": "success"}\n')
    for _ in range(30):
        if owner._initialized:
            break
        await asyncio.sleep(0)
    assert owner._initialized
    assert not owner._initializing
    assert owner.model == "process"
    await owner.close()


@pytest.mark.asyncio
async def test_close_releases_generation_waiting_for_initialization(owner, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    process = Process()

    async def spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    request = OpenAISpeechRequest(
        input="Hello.", voice="default", model="chatterbox", response_format="wav"
    )
    stream = owner.generate_speech_stream(request)
    generation = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), 1)
    closing = asyncio.create_task(owner.close())
    try:
        await turn()
    finally:
        release.set()
        results = await asyncio.wait_for(
            asyncio.gather(generation, closing, return_exceptions=True), 2
        )
        await stream.aclose()
    assert isinstance(results[0], RuntimeError)
    assert str(results[0]) == "Chatterbox backend is closing"
    assert results[1] is None
    assert process.waited
    assert owner.model is owner.process is None


@pytest.mark.asyncio
async def test_queued_generation_cannot_start_after_close_begins(owner, monkeypatch):
    owner.model = object()
    owner._initialized = True
    native_calls = []

    async def generate(*args, **kwargs):
        native_calls.append(True)
        return b"synthetic audio"

    monkeypatch.setattr(owner, "_generate_single", generate)
    request = OpenAISpeechRequest(
        input="Hello.", voice="default", model="chatterbox", response_format="wav"
    )
    stream = owner.generate_speech_stream(request)
    await owner._generation_lock.acquire()
    generation = asyncio.create_task(anext(stream))
    await turn()
    closing = asyncio.create_task(owner.close())
    try:
        await turn()
        assert owner._closing
        assert not closing.done()
    finally:
        owner._generation_lock.release()
        results = await asyncio.wait_for(
            asyncio.gather(generation, return_exceptions=True), 2
        )
        await stream.aclose()
        await asyncio.wait_for(closing, 2)
    assert native_calls == []
    assert isinstance(results[0], RuntimeError)
    assert str(results[0]) == "Chatterbox backend is closing"


@pytest.mark.asyncio
async def test_waiting_generation_rejects_fallback_ready_during_close(
    owner, monkeypatch
):
    loading, polling, published, release_poll = [asyncio.Event() for _ in range(4)]
    publish, finish_loading = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    native_calls = []
    original_sleep = asyncio.sleep

    def load():
        loop.call_soon_threadsafe(loading.set)
        assert publish.wait(3)
        owner.model = object()
        owner._initialized = True
        loop.call_soon_threadsafe(published.set)
        assert finish_loading.wait(3)

    async def spawn(*args, **kwargs):
        raise OSError("synthetic spawn failure")

    async def sleep(delay):
        if delay == 0.1:
            polling.set()
            await release_poll.wait()
        else:
            await original_sleep(delay)

    async def generate(*args, **kwargs):
        native_calls.append(True)
        return b"synthetic audio"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(owner, "_initialize_sync", load)
    monkeypatch.setattr(owner, "_generate_single", generate)
    request = OpenAISpeechRequest(
        input="Hello.", voice="default", model="chatterbox", response_format="wav"
    )
    stream = owner.generate_speech_stream(request)
    generation = asyncio.create_task(anext(stream))
    closing = None
    try:
        await asyncio.wait_for(loading.wait(), 1)
        await asyncio.wait_for(polling.wait(), 1)
        closing = asyncio.create_task(owner.close())
        await turn()
        assert owner._closing
        publish.set()
        await asyncio.wait_for(published.wait(), 1)
        assert owner._initialized
        assert not closing.done()
        release_poll.set()
        results = await asyncio.wait_for(
            asyncio.gather(generation, return_exceptions=True), 1
        )
        assert native_calls == []
        assert isinstance(results[0], RuntimeError)
        assert str(results[0]) == "Chatterbox backend is closing"
        assert not closing.done()
    finally:
        publish.set()
        finish_loading.set()
        release_poll.set()
        await stream.aclose()
        await asyncio.wait_for(closing or owner.close(), 2)
    assert owner.model is None
