"""TASK-26031: outbound signed webhooks for run lifecycle events."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.Agents.run_webhooks import (
    WEBHOOK_SIGNATURE_HEADER,
    WebhookConfig,
    _WebhookDelivery,
    _WebhookDeliveryWorker,
    build_webhook_payload,
    deliver_webhook,
    sign_payload,
    webhook_config_from_settings,
)


def _delivery(
    run_id: str,
    *,
    extra_ids=None,
    url="https://hook.example/x",
    timeout_seconds=5.0,
) -> _WebhookDelivery:
    return _WebhookDelivery(
        config=WebhookConfig(
            enabled=True,
            url=url,
            secret="secret",
            events=("completed",),
            timeout_seconds=timeout_seconds,
        ),
        event="completed",
        run_id=run_id,
        agent_id=None,
        timestamp=None,
        extra_ids=extra_ids,
    )


def _owned_worker_thread(worker: _WebhookDeliveryWorker) -> threading.Thread:
    with worker._state_lock:
        thread = worker._thread
    assert thread is not None
    return thread


def _join_owned_thread(thread: threading.Thread, timeout: float = 2.0) -> None:
    thread.join(timeout)
    assert not thread.is_alive(), "owned webhook worker did not exit"


# --- bounded reusable delivery worker (TASK-31511) ---


@pytest.mark.parametrize("stalled_stage", ("egress", "post"))
def test_delivery_deadline_advances_fifo_before_stalled_stage_is_released(
    monkeypatch, stalled_stage
):
    """A transport-only timeout leaves the next notification behind a lookup."""
    from tldw_chatbook.Agents import run_webhooks
    from tldw_chatbook.Utils import egress

    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()
    second_posted = threading.Event()
    posts = []
    contexts = []
    warnings = []
    metrics = []

    async def stall():
        entered.set()
        try:
            while not release.is_set():
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def resolve(host):
        contexts.append((threading.current_thread(), asyncio.get_running_loop()))
        if stalled_stage == "egress" and host == "secret.example":
            await stall()
        return ["93.184.216.34"]

    async def post(url, body, headers, timeout):
        run_id = json.loads(body)["run_id"]
        if stalled_stage == "post" and run_id == "run-sensitive":
            await stall()
        posts.append(run_id)
        if run_id == "second":
            second_posted.set()

    monkeypatch.setattr(egress, "_resolve_async", resolve)
    monkeypatch.setattr(
        egress, "get_cli_setting", lambda section, key, default: default
    )
    monkeypatch.setattr(run_webhooks, "_default_post", post)
    monkeypatch.setattr(
        run_webhooks.logger,
        "warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    monkeypatch.setattr(
        run_webhooks,
        "log_counter",
        lambda name, **kwargs: metrics.append((name, kwargs)),
    )
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    owned_thread = None
    try:
        assert worker.submit(
            _delivery(
                "run-sensitive",
                url="https://secret.example/token",
                timeout_seconds=0.1,
            )
        )
        owned_thread = _owned_worker_thread(worker)
        assert entered.wait(1.0)
        assert worker.submit(_delivery("second", timeout_seconds=0.1))
        assert second_posted.wait(1.0), "next webhook remained behind stalled delivery"
        assert not release.is_set()
        assert cancelled.is_set()
        assert posts == ["second"]
        assert contexts[0] == contexts[1]
        assert contexts[0][0] is owned_thread
        assert (
            metrics.count(("run_webhook_failed", {"labels": {"event": "completed"}}))
            == 1
        )
        assert "TimeoutError" in repr(warnings)
        assert all(
            canary not in repr((warnings, metrics))
            for canary in ("secret.example", "token", "secret", "run-sensitive")
        )
    finally:
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)
    assert worker._queue.unfinished_tasks == 0


@pytest.mark.parametrize(
    ("raw_timeout", "expected_timeout"),
    (
        (None, 5.0),
        ("invalid", 5.0),
        (float("nan"), 5.0),
        (float("inf"), 5.0),
        (-5, 0.1),
        (1e12, 120.0),
        ("0.2", 0.2),
    ),
)
def test_admission_normalizes_direct_config_timeout_before_transport(
    monkeypatch, raw_timeout, expected_timeout
):
    """Direct dataclass callers must not bypass the finite delivery timeout."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    observed = []

    async def post(url, body, headers, timeout):
        observed.append(timeout)
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)

    monkeypatch.setattr(run_webhooks, "_default_post", post)
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    owned_thread = None
    try:
        assert worker.submit(
            _delivery(
                "normalized", url="https://93.184.216.34/x", timeout_seconds=raw_timeout
            )
        )
        owned_thread = _owned_worker_thread(worker)
        assert entered.wait(1.0)
        assert observed == [expected_timeout]
    finally:
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)


def test_timed_out_dns_keeps_retiring_generation_owned_until_resolver_finishes(
    monkeypatch,
):
    """Runner's finite executor join must not retire still-running DNS owners."""
    from asyncio import constants

    from tldw_chatbook.Agents import run_webhooks
    from tldw_chatbook.Utils import egress

    entered = threading.Event()
    release = threading.Event()
    draining = threading.Event()
    second_posted = threading.Event()
    posts = []
    resolver_threads = []
    auxiliary_threads = []
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    real_start = threading.Thread.start
    real_shutdown = asyncio.BaseEventLoop.shutdown_default_executor
    runner_join_timeout = constants.THREAD_JOIN_TIMEOUT

    def record_owned_start(thread):
        # Track only children started by this exact delivery generation,
        # including the executor's real shutdown helper, before they start.
        if threading.current_thread() is worker._thread:
            auxiliary_threads.append(thread)
        return real_start(thread)

    def resolve(host, port, family=0, type=0, proto=0, flags=0):
        resolver_threads.append(threading.current_thread())
        if host == "stalled.example":
            entered.set()
            release.wait()
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", 0),
            )
        ]

    async def observe_shutdown(loop, timeout=None):
        if threading.current_thread() is worker._thread:
            draining.set()
        await real_shutdown(loop, timeout=timeout)

    async def post(url, body, headers, timeout):
        posts.append(json.loads(body)["run_id"])
        second_posted.set()

    monkeypatch.setattr(threading.Thread, "start", record_owned_start)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(
        egress, "get_cli_setting", lambda section, key, default: default
    )
    monkeypatch.setattr(run_webhooks, "_default_post", post)
    monkeypatch.setattr(
        asyncio.BaseEventLoop, "shutdown_default_executor", observe_shutdown
    )
    # Accelerate only Runner's existing abandonment budget; the product must
    # retain its generation beyond it while the real executor is gated.
    monkeypatch.setattr(constants, "THREAD_JOIN_TIMEOUT", 0.02)
    owned_thread = None
    try:
        assert worker.submit(
            _delivery("stalled", url="https://stalled.example/x", timeout_seconds=0.1)
        )
        owned_thread = _owned_worker_thread(worker)
        assert entered.wait(1.0)
        assert worker.submit(_delivery("second", timeout_seconds=0.1))
        assert second_posted.wait(1.0), "real DNS await still blocked the FIFO"
        assert posts == ["second"]
        assert not release.is_set()
        assert resolver_threads[0].is_alive()
        assert draining.wait(1.0)
        owned_thread.join(0.2)
        assert owned_thread.is_alive(), (
            "generation retired while its resolver still ran"
        )
        assert _owned_worker_thread(worker) is owned_thread
        started = time.monotonic()
        assert worker.submit(_delivery("during-drain")) is False
        assert time.monotonic() - started < 0.1
        assert worker._generation == 1
    finally:
        # Restore the ordinary Runner budget before its healthy final close.
        monkeypatch.setattr(constants, "THREAD_JOIN_TIMEOUT", runner_join_timeout)
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)
        for thread in auxiliary_threads:
            _join_owned_thread(thread)
    assert worker._queue.unfinished_tasks == 0
    assert worker._thread is None
    assert all(not thread.is_alive() for thread in resolver_threads)


@pytest.mark.parametrize("fail_direct_join", (False, True))
def test_shutdown_helper_start_failure_keeps_native_resolver_owned(
    monkeypatch, fail_direct_join
):
    """Failure to start the drain helper must not release a live generation."""
    from tldw_chatbook.Agents import run_webhooks
    from tldw_chatbook.Utils import egress

    entered = threading.Event()
    release = threading.Event()
    draining = threading.Event()
    rejected = threading.Event()
    allow_helpers = threading.Event()
    auxiliary_threads = []
    warnings = []
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    real_start = threading.Thread.start
    real_shutdown = asyncio.BaseEventLoop.shutdown_default_executor
    real_executor_shutdown = run_webhooks._WebhookResolverExecutor.shutdown

    def shutdown(executor, wait=True, *, cancel_futures=False):
        if fail_direct_join and wait and threading.current_thread() is worker._thread:
            raise RuntimeError("direct-join signing-secret canary")
        return real_executor_shutdown(executor, wait, cancel_futures=cancel_futures)

    def start(thread):
        if threading.current_thread() is worker._thread:
            if draining.is_set() and not allow_helpers.is_set():
                rejected.set()
                raise RuntimeError("shutdown-helper signing-secret canary")
            auxiliary_threads.append(thread)
        return real_start(thread)

    def resolve(host, port, family=0, type=0, proto=0, flags=0):
        entered.set()
        release.wait()
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", 0),
            )
        ]

    async def observe_shutdown(loop, timeout=None):
        if threading.current_thread() is worker._thread:
            draining.set()
        await real_shutdown(loop, timeout=timeout)

    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(run_webhooks._WebhookResolverExecutor, "shutdown", shutdown)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(
        egress, "get_cli_setting", lambda section, key, default: default
    )
    monkeypatch.setattr(
        asyncio.BaseEventLoop, "shutdown_default_executor", observe_shutdown
    )
    monkeypatch.setattr(
        run_webhooks.logger,
        "warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    owned_thread = None
    try:
        assert worker.submit(_delivery("held", timeout_seconds=0.1))
        owned_thread = _owned_worker_thread(worker)
        assert entered.wait(1.0)
        assert rejected.wait(1.0)
        owned_thread.join(0.2)
        assert owned_thread.is_alive() is not fail_direct_join
        assert _owned_worker_thread(worker) is owned_thread
        assert worker.submit(_delivery("during-failed-drain")) is False
        assert not release.is_set()
        assert auxiliary_threads[0].is_alive()
    finally:
        allow_helpers.set()
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)
        for thread in auxiliary_threads:
            _join_owned_thread(thread)
    assert worker._thread is (owned_thread if fail_direct_join else None)
    assert worker._queue.unfinished_tasks == 0
    assert "signing-secret" not in repr(warnings)


@pytest.mark.parametrize("stall_at", ("resolver", "worker_entry"))
def test_stalled_native_lookups_bound_repeated_arrivals_and_recover_after_completion(
    monkeypatch, stall_at
):
    """Cancelled DNS waiters must not enqueue unbounded native resolver work."""
    from tldw_chatbook.Agents import run_webhooks
    from tldw_chatbook.Utils import egress

    release = threading.Event()
    settled = [threading.Event(), threading.Event()]
    outcomes_changed = threading.Condition()
    outcomes = []
    calls = []
    posts = []
    native_futures = []
    auxiliary_threads = []
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.5)
    real_submit = ThreadPoolExecutor.submit
    real_start = threading.Thread.start

    def record_owned_start(thread):
        if threading.current_thread() is worker._thread:
            if stall_at == "worker_entry" and len(auxiliary_threads) < 2:
                original_run = thread.run

                def enter_after_release():
                    release.wait()
                    original_run()

                thread.run = enter_after_release
            auxiliary_threads.append(thread)
        return real_start(thread)

    def capture_native_future(executor, fn, /, *args, **kwargs):
        future = real_submit(executor, fn, *args, **kwargs)
        native_futures.append(future)
        return future

    def resolve(host, port, family=0, type=0, proto=0, flags=0):
        calls.append((host, threading.current_thread()))
        if host != "healthy.example":
            release.wait()
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", 0),
            )
        ]

    async def post(url, body, headers, timeout):
        posts.append(json.loads(body)["run_id"])

    def record_outcome(name, **kwargs):
        if name in (
            "run_webhook_failed",
            "run_webhook_blocked",
            "run_webhook_delivered",
        ):
            with outcomes_changed:
                outcomes.append(name)
                outcomes_changed.notify_all()

    monkeypatch.setattr(threading.Thread, "start", record_owned_start)
    monkeypatch.setattr(ThreadPoolExecutor, "submit", capture_native_future)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(
        egress, "get_cli_setting", lambda section, key, default: default
    )
    monkeypatch.setattr(run_webhooks, "_default_post", post)
    monkeypatch.setattr(run_webhooks, "log_counter", record_outcome)
    owned_thread = None
    try:
        for index in range(8):
            assert worker.submit(_delivery(f"held-{index}", timeout_seconds=0.1))
            if owned_thread is None:
                owned_thread = _owned_worker_thread(worker)
            with outcomes_changed:
                assert outcomes_changed.wait_for(
                    lambda index=index: len(outcomes) > index, 1.0
                )
        assert len(native_futures) == 2, "timed-out waiters kept submitting native jobs"
        assert all(not future.done() for future in native_futures)
        assert len(calls) == (2 if stall_at == "resolver" else 0)
        assert len(auxiliary_threads) == 2
        assert all(thread.is_alive() for thread in auxiliary_threads)
        assert posts == []
        assert not release.is_set()

        # Register after the timeouts, hence after the executor's ownership
        # callback. These events prove physical completion and released slots.
        for future, done in zip(native_futures, settled, strict=True):
            future.add_done_callback(lambda _future, done=done: done.set())
        release.set()
        assert all(done.wait(1.0) for done in settled)
        assert worker.submit(
            _delivery("healthy", url="https://healthy.example/x", timeout_seconds=1.0)
        )
        with outcomes_changed:
            assert outcomes_changed.wait_for(lambda: len(outcomes) == 9, 1.0)
        assert posts == ["healthy"]
        assert _owned_worker_thread(worker) is owned_thread
    finally:
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)
        for thread in auxiliary_threads:
            _join_owned_thread(thread)
    assert worker._queue.unfinished_tasks == 0


def test_failed_native_thread_start_does_not_release_ambiguous_queued_slots(
    monkeypatch,
):
    """A submit exception may follow enqueueing; queued jobs must stay bounded."""
    from tldw_chatbook.Agents import run_webhooks
    from tldw_chatbook.Utils import egress

    allow_start = threading.Event()
    outcomes_changed = threading.Condition()
    outcomes = []
    calls = []
    auxiliary_threads = []
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.5)
    real_start = threading.Thread.start

    def start(thread):
        if threading.current_thread() is worker._thread:
            if not allow_start.is_set():
                raise RuntimeError("native resolver thread refused to start")
            auxiliary_threads.append(thread)
        return real_start(thread)

    def resolve(host, port, family=0, type=0, proto=0, flags=0):
        calls.append(host)
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", 0),
            )
        ]

    async def post(url, body, headers, timeout):
        return None

    def record_outcome(name, **kwargs):
        if name in (
            "run_webhook_failed",
            "run_webhook_blocked",
            "run_webhook_delivered",
        ):
            with outcomes_changed:
                outcomes.append(name)
                outcomes_changed.notify_all()

    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(
        egress, "get_cli_setting", lambda section, key, default: default
    )
    monkeypatch.setattr(run_webhooks, "_default_post", post)
    monkeypatch.setattr(run_webhooks, "log_counter", record_outcome)
    owned_thread = None
    try:
        for index in range(2):
            assert worker.submit(_delivery(f"failed-{index}", timeout_seconds=0.1))
            if owned_thread is None:
                owned_thread = _owned_worker_thread(worker)
            with outcomes_changed:
                assert outcomes_changed.wait_for(
                    lambda index=index: len(outcomes) > index, 1.0
                )
        allow_start.set()
        assert worker.submit(_delivery("third", timeout_seconds=0.1))
        with outcomes_changed:
            assert outcomes_changed.wait_for(lambda: len(outcomes) == 3, 1.0)
        assert calls == [], "failed submissions released slots for queued native jobs"
    finally:
        allow_start.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)
        for thread in auxiliary_threads:
            _join_owned_thread(thread)
    assert calls == []
    assert worker._thread is None


def test_worker_reuses_one_thread_and_event_loop_without_blocking_submit(monkeypatch):
    """Per-event thread/loop scheduling or waiting in submit breaks this test."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    observed = []

    async def held_delivery(config, event, run_id, **kwargs):
        observed.append((threading.get_ident(), id(asyncio.get_running_loop()), run_id))
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return True

    monkeypatch.setattr(run_webhooks, "deliver_webhook", held_delivery)
    worker = _WebhookDeliveryWorker(queue_capacity=2, idle_seconds=0.02)
    owned_thread = None
    try:
        assert worker.submit(_delivery("one")) is True
        owned_thread = _owned_worker_thread(worker)
        assert entered.wait(1.0)
        started = time.monotonic()
        assert worker.submit(_delivery("two")) is True
        assert time.monotonic() - started < 0.1
    finally:
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)

    assert [item[2] for item in observed] == ["one", "two"]
    assert len({item[0] for item in observed}) == 1
    assert len({item[1] for item in observed}) == 1


def test_capacity_one_refuses_third_delivery_without_exposing_canaries(monkeypatch):
    """An unbounded queue or sensitive saturation diagnostic breaks this test."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    warnings = []
    metrics = []

    async def held_delivery(config, event, run_id, **kwargs):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return True

    monkeypatch.setattr(run_webhooks, "deliver_webhook", held_delivery)
    monkeypatch.setattr(
        run_webhooks.logger,
        "warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    monkeypatch.setattr(
        run_webhooks,
        "log_counter",
        lambda name, **kwargs: metrics.append((name, kwargs)),
    )
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    canaries = ("https://secret.example/token", "signing-secret", "run-sensitive")
    owned_thread = None
    try:
        assert worker.submit(_delivery(canaries[2])) is True
        owned_thread = _owned_worker_thread(worker)
        assert entered.wait(1.0)
        assert worker.submit(_delivery("waiting")) is True
        assert (
            worker.submit(
                _WebhookDelivery(
                    config=WebhookConfig(
                        enabled=True,
                        url=canaries[0],
                        secret=canaries[1],
                        events=("completed",),
                    ),
                    event="completed",
                    run_id="overflow",
                    agent_id=None,
                    timestamp=None,
                    extra_ids=None,
                )
            )
            is False
        )
    finally:
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)

    diagnostic_blob = repr((warnings, metrics))
    assert all(canary not in diagnostic_blob for canary in canaries)
    assert metrics == [
        (
            "run_webhook_dropped",
            {"labels": {"reason": "queue_full", "event": "completed"}},
        )
    ]


def test_worker_retires_then_restarts_with_a_new_generation(monkeypatch):
    """Keeping a retired generation or failing to restart breaks this test."""
    from tldw_chatbook.Agents import run_webhooks

    observed_threads = []
    release_by_run = {"one": threading.Event(), "two": threading.Event()}

    async def record_delivery(config, event, run_id, **kwargs):
        observed_threads.append(threading.current_thread())
        while not release_by_run[run_id].is_set():
            await asyncio.sleep(0.005)
        return True

    monkeypatch.setattr(run_webhooks, "deliver_webhook", record_delivery)
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    owned_threads = []
    try:
        assert worker.submit(_delivery("one")) is True
        owned_threads.append(_owned_worker_thread(worker))
        release_by_run["one"].set()
        _join_owned_thread(owned_threads[-1])
        assert worker.submit(_delivery("two")) is True
        owned_threads.append(_owned_worker_thread(worker))
    finally:
        for release in release_by_run.values():
            release.set()
        for thread in owned_threads:
            _join_owned_thread(thread)
    assert len(observed_threads) == 2
    assert observed_threads[0] is not observed_threads[1]


def test_submission_during_runner_close_is_refused_then_restart_succeeds(monkeypatch):
    """Publishing retirement before Runner close can strand an admitted event."""
    from tldw_chatbook.Agents import run_webhooks

    real_runner = asyncio.Runner
    closing = threading.Event()
    release_close = threading.Event()
    release_after = threading.Event()
    delivered = []

    class GatedRunner:
        def __init__(self):
            self._runner = real_runner()

        def __enter__(self):
            self._runner.__enter__()
            return self

        def run(self, coro):
            return self._runner.run(coro)

        def get_loop(self):
            return self._runner.get_loop()

        def __exit__(self, *args):
            closing.set()
            assert release_close.wait(1.0)
            return self._runner.__exit__(*args)

    async def record_delivery(config, event, run_id, **kwargs):
        delivered.append(run_id)
        if run_id == "after-close":
            while not release_after.is_set():
                await asyncio.sleep(0.005)
        return True

    monkeypatch.setattr(run_webhooks.asyncio, "Runner", GatedRunner)
    monkeypatch.setattr(run_webhooks, "deliver_webhook", record_delivery)
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.01)
    owned_threads = []
    try:
        assert worker.submit(_delivery("first")) is True
        owned_threads.append(_owned_worker_thread(worker))
        assert closing.wait(1.0)
        assert worker.submit(_delivery("during-close")) is False
        release_close.set()
        _join_owned_thread(owned_threads[-1])
        assert worker.submit(_delivery("after-close")) is True
        owned_threads.append(_owned_worker_thread(worker))
    finally:
        release_close.set()
        release_after.set()
        for thread in owned_threads:
            _join_owned_thread(thread)
    assert delivered == ["first", "after-close"]


def test_failed_thread_start_is_sanitized_and_later_submit_recovers(monkeypatch):
    """A failed start must not strand work or poison later generations."""
    from tldw_chatbook.Agents import run_webhooks

    real_thread = threading.Thread
    warnings = []
    delivered = []
    release = threading.Event()

    class FailedThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("run-sensitive secret-url signing-secret")

    async def record_delivery(config, event, run_id, **kwargs):
        delivered.append(run_id)
        while not release.is_set():
            await asyncio.sleep(0.005)
        return True

    monkeypatch.setattr(
        run_webhooks.logger,
        "warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    monkeypatch.setattr(run_webhooks.threading, "Thread", FailedThread)
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.01)
    owned_thread = None
    try:
        assert worker.submit(_delivery("not-admitted")) is False
        monkeypatch.setattr(run_webhooks.threading, "Thread", real_thread)
        monkeypatch.setattr(run_webhooks, "deliver_webhook", record_delivery)
        assert worker.submit(_delivery("recovered")) is True
        owned_thread = _owned_worker_thread(worker)
    finally:
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)
    assert delivered == ["recovered"]
    assert "run-sensitive" not in repr(warnings)
    assert "secret-url" not in repr(warnings)
    assert "signing-secret" not in repr(warnings)


def test_callback_failure_does_not_stop_fifo_and_admission_copies_inputs(monkeypatch):
    """A callback escape or caller mutation must not alter later admitted work."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    observed = []

    async def delivery(config, event, run_id, **kwargs):
        if run_id == "held":
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.005)
            raise KeyboardInterrupt("identifier-canary")
        observed.append((run_id, config.events, dict(kwargs["extra_ids"])))
        return True

    monkeypatch.setattr(run_webhooks, "deliver_webhook", delivery)
    mutable_events = ["completed"]
    mutable_ids = {"workspace_id": "original"}
    config = WebhookConfig(
        enabled=True,
        url="https://hook.example/x",
        secret="secret",
        events=mutable_events,  # type: ignore[arg-type]
    )
    worker = _WebhookDeliveryWorker(queue_capacity=2, idle_seconds=0.02)
    owned_thread = None
    try:
        assert worker.submit(_delivery("held")) is True
        owned_thread = _owned_worker_thread(worker)
        assert entered.wait(1.0)
        assert (
            worker.submit(
                _WebhookDelivery(
                    config=config,
                    event="completed",
                    run_id="copied",
                    agent_id=None,
                    timestamp=None,
                    extra_ids=mutable_ids,
                )
            )
            is True
        )
        mutable_events.append("failed")
        mutable_ids["workspace_id"] = "mutated"
    finally:
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)
    assert observed == [("copied", ("completed",), {"workspace_id": "original"})]


def test_diagnostic_failures_do_not_escape_admission_or_stop_fifo(monkeypatch):
    """A broken diagnostic sink must not strand admitted webhook work."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    delivered = []

    async def delivery(config, event, run_id, **kwargs):
        if run_id == "held":
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.005)
            raise RuntimeError("callback")
        delivered.append(run_id)
        return True

    def diagnostic_failure(*args, **kwargs):
        raise RuntimeError("diagnostic")

    monkeypatch.setattr(run_webhooks, "deliver_webhook", delivery)
    monkeypatch.setattr(run_webhooks.logger, "warning", diagnostic_failure)
    monkeypatch.setattr(run_webhooks, "log_counter", diagnostic_failure)
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    owned_thread = None
    try:
        assert worker.submit(_delivery("held")) is True
        owned_thread = _owned_worker_thread(worker)
        assert entered.wait(1.0)
        assert worker.submit(_delivery("waiting")) is True
        assert worker.submit(_delivery("overflow")) is False
    finally:
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)
    assert delivered == ["waiting"]


def test_worker_metrics_bound_unknown_event_labels(monkeypatch):
    """Caller-supplied event names must not create unbounded metric labels."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    metrics = []

    async def held_delivery(config, event, run_id, **kwargs):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return True

    monkeypatch.setattr(run_webhooks, "deliver_webhook", held_delivery)
    monkeypatch.setattr(
        run_webhooks,
        "log_counter",
        lambda name, **kwargs: metrics.append((name, kwargs)),
    )
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    owned_thread = None
    try:
        assert worker.submit(_delivery("held")) is True
        owned_thread = _owned_worker_thread(worker)
        assert entered.wait(1.0)
        assert worker.submit(_delivery("waiting")) is True
        unknown = _WebhookDelivery(
            config=_delivery("unused").config,
            event="user-supplied-sensitive-event",
            run_id="overflow",
            agent_id=None,
            timestamp=None,
            extra_ids=None,
        )
        assert worker.submit(unknown) is False
    finally:
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)
    assert metrics == [
        (
            "run_webhook_dropped",
            {"labels": {"reason": "queue_full", "event": "other"}},
        )
    ]


# --- payload carries identifiers + outcome only (AC#3) ---


def test_payload_has_ids_and_outcome_no_content():
    payload = build_webhook_payload(
        event="completed",
        run_id="run-123",
        agent_id="agent-9",
        timestamp="2026-09-01T00:00:00Z",
    )
    assert payload["event"] == "completed"
    assert payload["run_id"] == "run-123"
    assert payload["agent_id"] == "agent-9"
    # never leak content-shaped keys
    blob = str(payload).lower()
    for forbidden in (
        "message",
        "content",
        "tool_arg",
        "api_key",
        "secret",
        "token",
        "prompt",
    ):
        assert forbidden not in blob, f"payload leaks {forbidden}: {payload}"


# --- HMAC signing is verifiable with a documented scheme (AC#2) ---


def test_signature_is_verifiable_hmac_sha256():
    secret = "s3cr3t"
    body = b'{"event":"completed"}'
    sig = sign_payload(secret, body)
    assert sig.startswith("sha256=")
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert sig == expected


# --- config gating: default off (AC#7) ---


def test_config_default_disabled():
    cfg = webhook_config_from_settings({})
    assert cfg.enabled is False
    assert cfg.url == ""


def test_config_enabled_when_configured():
    cfg = webhook_config_from_settings(
        {
            "webhooks": {
                "enabled": True,
                "url": "https://hook.example/x",
                "secret": "s",
                "events": ["completed"],
            }
        }
    )
    assert cfg.enabled is True
    assert cfg.url == "https://hook.example/x"
    assert "completed" in cfg.events


def _run(coro):
    return asyncio.run(coro)


# --- delivery gating + egress + fire-and-forget (AC#1/#4/#5/#6/#7) ---


def test_disabled_config_makes_no_request(monkeypatch):
    """AC#7: no endpoint configured => no request ever."""
    posted = {"n": 0}

    async def fake_post(url, body, headers, timeout):
        posted["n"] += 1

    result = _run(
        deliver_webhook(
            WebhookConfig(enabled=False, url="", secret="", events=("completed",)),
            "completed",
            "run-1",
            post_fn=fake_post,
        )
    )
    assert posted["n"] == 0
    assert result is False


def test_event_not_subscribed_makes_no_request():
    posted = {"n": 0}

    async def fake_post(url, body, headers, timeout):
        posted["n"] += 1

    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="s", events=("failed",)
            ),
            "completed",
            "run-1",
            post_fn=fake_post,
        )
    )
    assert posted["n"] == 0  # "completed" not in subscribed events
    assert result is False


def test_egress_blocked_url_is_not_posted(monkeypatch):
    """AC#6: destination subject to the SSRF egress policy."""
    from tldw_chatbook.Agents import run_webhooks

    async def blocked(url, **k):
        from tldw_chatbook.Utils.egress import EgressBlockedError

        raise EgressBlockedError(url, "private ip")

    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", blocked)
    posted = {"n": 0}

    async def fake_post(url, body, headers, timeout):
        posted["n"] += 1

    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True,
                url="http://169.254.169.254/x",
                secret="s",
                events=("completed",),
            ),
            "completed",
            "run-1",
            post_fn=fake_post,
        )
    )
    assert posted["n"] == 0, "an egress-blocked URL must not be POSTed"
    assert result is False


def test_successful_delivery_signs_and_posts(monkeypatch):
    """AC#1/#2: a configured, allowed endpoint gets a signed POST."""
    from tldw_chatbook.Agents import run_webhooks

    async def allowed(url, **k):
        return None

    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)
    captured = {}

    async def fake_post(url, body, headers, timeout):
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers
        captured["timeout"] = timeout

    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="sec", events=("completed",)
            ),
            "completed",
            "run-1",
            post_fn=fake_post,
        )
    )
    assert result is True
    assert captured["url"] == "https://h/x"
    assert WEBHOOK_SIGNATURE_HEADER in captured["headers"]
    expected = sign_payload("sec", captured["body"])
    assert captured["headers"][WEBHOOK_SIGNATURE_HEADER] == expected
    assert captured["timeout"] > 0  # AC#4 bounded


def test_delivery_failure_is_visible_not_raised(monkeypatch):
    """AC#4/#5: a dead endpoint never raises into the run; failure is logged."""
    from tldw_chatbook.Agents import run_webhooks

    async def allowed(url, **k):
        return None

    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)

    async def boom(url, body, headers, timeout):
        raise TimeoutError("dead endpoint")

    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="s", events=("completed",)
            ),
            "completed",
            "run-1",
            post_fn=boom,
        )
    )
    assert result is False  # swallowed, run unaffected


def test_scheduler_gates_before_spawning_a_thread():
    """AC#7/#4: disabled or unsubscribed => no thread, no delivery."""
    from tldw_chatbook.Agents.run_webhooks import schedule_run_webhook

    assert (
        schedule_run_webhook(
            WebhookConfig(enabled=False, url="", secret="", events=("completed",)),
            "completed",
            "run-1",
        )
        is False
    )
    assert (
        schedule_run_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="s", events=("failed",)
            ),
            "completed",
            "run-1",
        )
        is False
    )


def test_scheduler_delivers_when_enabled(monkeypatch):
    """The scheduler starts a delivery that reaches the endpoint (AC#1)."""
    import time

    from tldw_chatbook.Agents import run_webhooks

    delivered = {"ok": False}
    release = threading.Event()

    async def allowed(url, **k):
        return None

    async def fake_post(url, body, headers, timeout):
        delivered["ok"] = True
        while not release.is_set():
            await asyncio.sleep(0.005)

    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)
    monkeypatch.setattr(run_webhooks, "_default_post", fake_post)
    monkeypatch.setattr(run_webhooks, "_WEBHOOK_DELIVERY_WORKER", worker)
    owned_thread = None
    try:
        started = run_webhooks.schedule_run_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="s", events=("completed",)
            ),
            "completed",
            "run-1",
        )
        assert started is True
        owned_thread = _owned_worker_thread(worker)
        for _ in range(50):
            if delivered["ok"]:
                break
            time.sleep(0.02)
    finally:
        release.set()
        if owned_thread is not None:
            _join_owned_thread(owned_thread)
    assert delivered["ok"] is True


# --- terminal-seam wiring in AgentService (AC#1 completed/failed) ---


def test_agent_service_terminal_seam_fires_completed_and_failed(monkeypatch):
    """AC#1: a fresh terminal transition maps to a lifecycle webhook event;
    non-notify terminal states (cancelled/superseded) do not fire."""
    from tldw_chatbook.Agents import agent_service as svc_mod
    from tldw_chatbook.Agents.agent_models import (
        RUN_CANCELLED,
        RUN_DONE,
        RUN_ERROR,
    )
    from tldw_chatbook.Agents.agent_service import AgentService

    captured = []
    monkeypatch.setattr(
        svc_mod, "safe_utc_timestamp", lambda *_a, **_k: "2026-09-01T00:00:00Z"
    )

    def fake_schedule(config, event, run_id, **kwargs):
        captured.append((event, run_id))
        return True

    import tldw_chatbook.Agents.run_webhooks as rw

    monkeypatch.setattr(rw, "schedule_run_webhook", fake_schedule)
    monkeypatch.setattr(
        rw,
        "webhook_config_from_settings",
        lambda *_a, **_k: rw.WebhookConfig(
            enabled=True, url="https://h/x", secret="s", events=rw.WEBHOOK_EVENTS
        ),
    )

    svc = AgentService.__new__(AgentService)
    svc.wall_clock = None

    svc._maybe_emit_run_webhook("run-done", RUN_DONE)
    svc._maybe_emit_run_webhook("run-err", RUN_ERROR)
    svc._maybe_emit_run_webhook("run-cancel", RUN_CANCELLED)

    assert ("completed", "run-done") in captured
    assert ("failed", "run-err") in captured
    assert not any(rid == "run-cancel" for _, rid in captured), (
        "cancelled must not notify"
    )


# --- Qodo review round (PR #2301) ---


def test_qodo9_string_false_does_not_enable():
    """Qodo #9: 'false'/'0' strings must not enable the outbound gate."""
    for junk in ("false", "0", "no", "off", "FALSE"):
        cfg = webhook_config_from_settings(
            {"webhooks": {"enabled": junk, "url": "https://h/x", "secret": "s"}}
        )
        assert cfg.enabled is False, junk
    cfg = webhook_config_from_settings(
        {"webhooks": {"enabled": "true", "url": "https://h/x", "secret": "s"}}
    )
    assert cfg.enabled is True


def test_qodo10_enabled_without_secret_never_delivers(monkeypatch):
    """Qodo #10: no signing secret => fail closed, no POST."""
    from tldw_chatbook.Agents import run_webhooks

    async def allowed(url, **k):
        return None

    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)
    posted = {"n": 0}

    async def fake_post(url, body, headers, timeout):
        posted["n"] += 1

    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="", events=("completed",)
            ),
            "completed",
            "run-1",
            post_fn=fake_post,
        )
    )
    assert result is False and posted["n"] == 0


def test_qodo11_timeout_rejects_nan_and_infinity():
    """Qodo #11: NaN/inf timeouts fall back to a finite bound."""
    for junk in ("inf", "nan", float("inf"), float("nan"), -5, 1e12):
        cfg = webhook_config_from_settings(
            {
                "webhooks": {
                    "enabled": True,
                    "url": "https://h/x",
                    "secret": "s",
                    "timeout_seconds": junk,
                }
            }
        )
        import math

        assert math.isfinite(cfg.timeout_seconds), junk
        assert 0.1 <= cfg.timeout_seconds <= 120.0, (junk, cfg.timeout_seconds)


def test_qodo8_http_error_status_is_a_failed_delivery(monkeypatch):
    """Qodo #8: a 4xx/5xx response is a FAILED delivery, not a success."""
    from tldw_chatbook.Agents import run_webhooks

    async def allowed(url, **k):
        return None

    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)

    class _Resp:
        status_code = 500

        def raise_for_status(self):
            raise RuntimeError("500 Server Error")

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, content=None, headers=None):
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="s", events=("completed",)
            ),
            "completed",
            "run-1",  # no post_fn: exercise the REAL default transport
        )
    )
    assert result is False


def test_qodo12_failure_log_never_contains_full_url(monkeypatch, caplog):
    """Qodo #12: a failing delivery logs a sanitized origin, never credentials
    embedded in the URL path/query/userinfo."""
    from tldw_chatbook.Agents import run_webhooks

    async def allowed(url, **k):
        return None

    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)

    async def boom(url, body, headers, timeout):
        raise TimeoutError("dead")

    captured = []
    monkeypatch.setattr(
        run_webhooks.logger,
        "warning",
        lambda msg, *a, **k: captured.append(msg.format(*a) if a else str(msg)),
    )
    _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True,
                url="https://user:tok3n@h.example/hook?apikey=sekret",
                secret="s",
                events=("completed",),
            ),
            "completed",
            "run-1",
            post_fn=boom,
        )
    )
    joined = " ".join(captured)
    assert "tok3n" not in joined and "sekret" not in joined, joined
    assert "h.example" in joined  # origin still identifiable
