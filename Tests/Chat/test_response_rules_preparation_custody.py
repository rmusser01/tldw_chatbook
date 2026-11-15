"""Review regression: cancelled preprocessing must retain native helper capacity.

Also TASK-40325: app exit abandons a helper that never answers, after a bound.
"""

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
        # Queue the helper behind the blocker, whichever executor it names
        # (its own daemon threads since TASK-40325).
        monkeypatch.setattr(
            loop,
            "run_in_executor",
            lambda _selected, function, *args: run_in_executor(
                executor, function, *args
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


_HUNG_HELPER_EXIT_SCRIPT = r"""
import asyncio, os, threading, time
from loguru import logger
from Tests import network_guard, real_profile_guard
network_guard.install()
real_profile_guard.install()
warnings = []
logger.add(warnings.append, level="WARNING", format="{message}")


async def main():
    from Tests.Chat.response_rules_fixtures import source
    from Tests.Chat.test_console_provider_gateway import _auxiliary_request
    from tldw_chatbook.Chat import console_runtime
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.response_rules.resources import RuleHelperPool

    runtime = console_runtime.ConsoleRuntime(app=None)
    pool = RuleHelperPool(
        usage_sink=lambda *_args: None, current=lambda _origin: True, clock=time.monotonic
    )
    runtime.set_rule_helper_pool(pool)
    started = threading.Event()

    def adapter(**_kwargs):
        started.set()
        threading.Event().wait()  # The provider never answers.

    lease = pool.try_acquire(source(), purpose="checking", deadline=time.monotonic() + 30)
    asyncio.create_task(
        ConsoleProviderGateway(chat_api_call_fn=adapter).complete_auxiliary(
            _auxiliary_request(), rule_lease=lease
        )
    )
    assert await asyncio.to_thread(started.wait, 30)
    bound = getattr(console_runtime, "RULE_HELPER_EXIT_WAIT_SECONDS", 5.0)
    # The whole exit, interpreter included, gets the bound plus 15 s.
    watchdog = threading.Timer(
        bound + 15, lambda: (print("exit_still_waiting", flush=True), os._exit(4))
    )
    watchdog.daemon = True
    watchdog.start()
    began = time.monotonic()
    await runtime.dispose(timeout_seconds=0)
    waited = time.monotonic() - began
    assert bound <= waited < bound + 10, waited
    assert pool.unsettled_count == 1 and not lease.physical_settled
    assert [w for w in warnings if "response_rule_helpers_abandoned_at_exit count=1" in w]
    print("retired and reopened", flush=True)


asyncio.run(main())
"""


def test_exit_abandons_a_helper_that_never_answers_after_its_bound(tmp_path):
    """TASK-40325: quit is not held hostage by a provider that never replies.

    Before, dispose polled the pool with no bound, and asyncio.run then joined
    the helper's executor thread, so the process never exited.
    """
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    launch = (
        "from Tests.Chat.test_response_rules_preparation_custody import "
        "_HUNG_HELPER_EXIT_SCRIPT; "
        "exec(compile(_HUNG_HELPER_EXIT_SCRIPT, '<hung-rule-helper-exit>', 'exec'))"
    )
    _run(tmp_path, "rule-helper", "hung", script=launch, timeout=90)
