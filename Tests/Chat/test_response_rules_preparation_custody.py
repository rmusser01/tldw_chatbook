"""Review regression: cancelled preprocessing must retain native helper capacity."""

import asyncio
import time
from threading import Event
from types import SimpleNamespace

import pytest

from Tests.Chat.response_rules_fixtures import source
from Tests.Chat.test_console_provider_gateway import _auxiliary_request
from Tests.Chat.test_response_rules_resources import pool_with_usage
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway


@pytest.mark.asyncio
async def test_cancelled_preparation_retains_capacity_until_thread_settles(monkeypatch):
    pool, usage = pool_with_usage()
    origin = source()
    started, release, finished = Event(), Event(), Event()
    calls = []

    def prepare(*_args, **_kwargs):
        started.set()
        try:
            assert release.wait(3)
            return SimpleNamespace(known_overflow=True)
        finally:
            finished.set()

    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **kw: calls.append(kw))
    monkeypatch.setattr(gateway, "prepare_chat_request", prepare)
    lease = pool.try_acquire(origin, purpose="checking", deadline=time.monotonic() + 30)
    task = asyncio.create_task(
        gateway.complete_auxiliary(_auxiliary_request(), rule_lease=lease)
    )
    replacement = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not finished.is_set()
        replacement = pool.try_acquire(
            origin, purpose="checking", deadline=time.monotonic() + 30
        )
        assert replacement is None, (
            "A second helper was admitted while cancelled request preparation still runs"
        )
    finally:
        if replacement is not None:
            replacement.release_unused()
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        pool.close_admission()
    assert calls == [] and usage == []


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stop", "profile", "deadline", "shutdown"])
@pytest.mark.parametrize("provider", ["openai", "llama_cpp"])
async def test_retired_valid_preparation_never_enters_transport(
    monkeypatch, reason, provider
):
    from Tests.Chat.test_console_provider_gateway import _auxiliary_resolution
    from Tests.Chat.test_response_rules_resources import await_settlement

    now, current = [time.monotonic()], [True]
    pool, usage = pool_with_usage(current=lambda _: current[0], clock=lambda: now[0])
    origin = source()
    started, release, finished = Event(), Event(), Event()
    calls = []
    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **kw: calls.append(kw))
    prepare_request = gateway.prepare_chat_request

    def prepare(*args, **kwargs):
        started.set()
        try:
            assert release.wait(3)
            return prepare_request(*args, **kwargs)
        finally:
            finished.set()

    async def local(**kwargs):
        calls.append(kwargs)
        return "late"

    monkeypatch.setattr(gateway, "prepare_chat_request", prepare)
    monkeypatch.setattr(gateway, "complete_llamacpp_chat", local)
    lease = pool.try_acquire(origin, purpose="checking", deadline=now[0] + 30)
    task = asyncio.create_task(
        gateway.complete_auxiliary(
            _auxiliary_request(
                resolution=_auxiliary_resolution(
                    provider=provider, execution_key=provider
                )
            ),
            rule_lease=lease,
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        if reason == "profile":
            current[0] = False
        elif reason == "deadline":
            now[0] += 31
        elif reason == "shutdown":
            pool.close_admission()
        else:
            pool.cancel_session(origin.session_id, "stop")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not finished.is_set()
        assert pool.unsettled_count == 1 and not lease.physical_settled
        assert (
            pool.try_acquire(origin, purpose="checking", deadline=now[0] + 30) is None
        )
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        await await_settlement(pool)
        await gateway.aclose()
    assert calls == [] and usage == []
    assert lease.physical_settled


@pytest.mark.asyncio
async def test_retired_transport_waiting_in_executor_never_calls_adapter(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from Tests.Chat.test_response_rules_resources import await_settlement

    pool, usage = pool_with_usage()
    origin = source()
    occupied, release = Event(), Event()
    calls = []
    loop = asyncio.get_running_loop()
    run_in_executor = loop.run_in_executor
    with ThreadPoolExecutor(max_workers=1) as executor:

        def occupy():
            occupied.set()
            assert release.wait(3)

        blocker = run_in_executor(executor, occupy)
        while not occupied.is_set():
            await asyncio.sleep(0)
        monkeypatch.setattr(
            loop,
            "run_in_executor",
            lambda selected, function, *args: run_in_executor(
                executor if selected is None else selected, function, *args
            ),
        )
        lease = pool.try_acquire(
            origin, purpose="checking", deadline=time.monotonic() + 30
        )
        task = asyncio.create_task(
            lease.run_sync(lambda: calls.append("adapter"), lambda _: None)
        )
        try:
            while not pool._tasks:
                await asyncio.sleep(0)
            pool.cancel_session(origin.session_id, "stop")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert pool.unsettled_count == 1 and calls == [] and usage == []
            assert (
                pool.try_acquire(
                    origin, purpose="checking", deadline=time.monotonic() + 30
                )
                is None
            )
        finally:
            release.set()
            await blocker
            await await_settlement(pool)
    assert lease.physical_settled and calls == [] and usage == []
