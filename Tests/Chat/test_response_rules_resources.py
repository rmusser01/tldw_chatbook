"""Native helper limits follow physical adapter completion, including late usage."""

import asyncio
import time
from threading import Event
from types import SimpleNamespace

import httpx
import pytest

from Tests.Chat.response_rules_fixtures import source
from Tests.Chat.test_console_provider_gateway import (
    _auxiliary_request,
    _auxiliary_resolution,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.response_rules.resources import RuleHelperPool


@pytest.mark.asyncio
async def test_native_helper_refuses_known_context_overflow_before_adapter_entry(
    monkeypatch,
):
    from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError

    pool, charged = pool_with_usage()
    lease = pool.try_acquire(
        source(), purpose="checking", deadline=time.monotonic() + 30
    )
    calls = []
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **kwargs: calls.append(kwargs) or "answer"
    )
    monkeypatch.setattr(
        gateway,
        "prepare_chat_request",
        lambda *_args, **_kwargs: SimpleNamespace(known_overflow=True),
    )
    with pytest.raises(ChatBadRequestError):
        await gateway.complete_auxiliary(_auxiliary_request(), rule_lease=lease)
    assert calls == [] and charged == [] and pool.unsettled_count == 0


def pool_with_usage(*, current=lambda _source: True, clock=time.monotonic):
    usage = []
    pool = RuleHelperPool(
        usage_sink=lambda *entry: usage.append(entry), current=current, clock=clock
    )
    return pool, usage


async def await_event(event):
    assert await asyncio.to_thread(event.wait, 2)


async def await_settlement(pool):
    async with asyncio.timeout(2):
        while pool.unsettled_count:
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_cancelled_provider_retains_capacity_until_worker_finishes():
    pool, usage = pool_with_usage()
    origin = source()
    started, release = Event(), Event()

    def adapter(**_kwargs):
        started.set()
        assert release.wait(2)
        return "LATE-CONTENT-CANARY"

    lease = pool.try_acquire(origin, purpose="checking", deadline=time.monotonic() + 30)
    gateway = ConsoleProviderGateway(chat_api_call_fn=adapter)
    task = asyncio.create_task(
        gateway.complete_auxiliary(_auxiliary_request(), rule_lease=lease)
    )
    try:
        await await_event(started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lease.acceptance_current
        assert (
            pool.try_acquire(origin, purpose="checking", deadline=time.monotonic() + 30)
            is None
        )
        assert pool.unsettled_count == 1
        assert usage == []
    finally:
        release.set()
        await await_settlement(pool)
    assert len(usage) == 1 and usage[0][2] is None
    assert (
        pool.try_acquire(origin, purpose="checking", deadline=time.monotonic() + 30)
        is not None
    )


@pytest.mark.asyncio
async def test_slow_usage_write_does_not_block_other_chat_admission():
    charging, release = Event(), Event()

    def sink(*_args):
        charging.set()
        assert release.wait(2)

    pool = RuleHelperPool(
        usage_sink=sink, current=lambda _origin: True, clock=time.monotonic
    )
    lease = pool.try_acquire(
        source(), purpose="checking", deadline=time.monotonic() + 30
    )
    task = asyncio.create_task(
        ConsoleProviderGateway(
            chat_api_call_fn=lambda **_kwargs: "answer"
        ).complete_auxiliary(_auxiliary_request(), rule_lease=lease)
    )
    try:
        await await_event(charging)
        other = await asyncio.wait_for(
            asyncio.to_thread(
                pool.try_acquire,
                source(session_id="other"),
                purpose="checking",
                deadline=time.monotonic() + 30,
            ),
            timeout=0.1,
        )
        assert other is not None
        other.release_unused()
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_repeated_cancel_cannot_grow_unsettled_work():
    pool, _usage = pool_with_usage()
    started, release = Event(), Event()

    def adapter(**_kwargs):
        started.set()
        assert release.wait(2)
        return "answer"

    lease = pool.try_acquire(
        source(), purpose="learning", deadline=time.monotonic() + 30
    )
    task = asyncio.create_task(
        ConsoleProviderGateway(chat_api_call_fn=adapter).complete_auxiliary(
            _auxiliary_request(), rule_lease=lease
        )
    )
    try:
        await await_event(started)
        for _ in range(20):
            lease.cancel_acceptance("stop")
            assert (
                pool.try_acquire(
                    source(), purpose="checking", deadline=time.monotonic() + 30
                )
                is None
            )
        for i in range(3):
            assert (
                pool.try_acquire(
                    source(session_id=f"other-{i}"),
                    purpose="checking",
                    deadline=time.monotonic() + 30,
                )
                is not None
            )
        assert (
            pool.try_acquire(
                source(session_id="fifth"),
                purpose="checking",
                deadline=time.monotonic() + 30,
            )
            is None
        )
        assert pool.unsettled_count == 4
    finally:
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        pool.close_admission()
    assert not lease.acceptance_current


@pytest.mark.asyncio
async def test_late_usage_charged_once_to_original_profile():
    current_profile = ["profile"]
    pool, usage = pool_with_usage(
        current=lambda origin: origin.profile_id == current_profile[0]
    )
    origin = source()
    started, release = Event(), Event()

    def adapter(**_kwargs):
        started.set()
        assert release.wait(2)
        return {
            "choices": [{"message": {"content": "SECRET-ANSWER"}}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7},
        }

    lease = pool.try_acquire(origin, purpose="learning", deadline=time.monotonic() + 30)
    task = asyncio.create_task(
        ConsoleProviderGateway(chat_api_call_fn=adapter).complete_auxiliary(
            _auxiliary_request(), rule_lease=lease
        )
    )
    try:
        await await_event(started)
        current_profile[0] = "another-profile"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await await_settlement(pool)
    assert len(usage) == 1
    assert usage[0][0] == origin and usage[0][1] == lease.usage_id
    assert usage[0][2].uncached_input == 13 and usage[0][2].output == 7
    assert "SECRET-ANSWER" not in repr(pool) + repr(lease)


@pytest.mark.asyncio
async def test_native_helpers_override_retry_settings_only_for_their_call():
    pool, _ = pool_with_usage(clock=lambda: 10)
    seen = []
    resolution = _auxiliary_resolution(
        request_retries=5, request_timeout=80, max_tokens=3000
    )
    request = _auxiliary_request(resolution=resolution, max_output_tokens=8000)
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **kwargs: seen.append(kwargs) or "answer"
    )
    lease = pool.try_acquire(source(), purpose="checking", deadline=17)
    result = await gateway.complete_auxiliary(request, rule_lease=lease)
    assert result.text == "answer"
    assert seen[0]["request_retries"] == 0
    assert seen[0]["request_timeout"] == 7
    assert seen[0]["max_tokens"] == 3000
    await gateway.complete_auxiliary(request)
    assert seen[1]["request_retries"] == 5 and seen[1]["request_timeout"] == 80
    assert request.resolution == resolution and request.max_output_tokens == 8000


@pytest.mark.asyncio
async def test_close_admission_rejects_late_content_and_retains_cleanup():
    pool, usage = pool_with_usage()
    started, release = Event(), Event()

    def adapter(**_kwargs):
        started.set()
        assert release.wait(2)
        return "late"

    lease = pool.try_acquire(
        source(), purpose="checking", deadline=time.monotonic() + 30
    )
    task = asyncio.create_task(
        ConsoleProviderGateway(chat_api_call_fn=adapter).complete_auxiliary(
            _auxiliary_request(), rule_lease=lease
        )
    )
    try:
        await await_event(started)
        pool.close_admission()
        assert not lease.acceptance_current and pool.unsettled_count == 1
        assert (
            pool.try_acquire(
                source(session_id="new"),
                purpose="checking",
                deadline=time.monotonic() + 30,
            )
            is None
        )
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pool.unsettled_count == 0 and len(usage) == 1


def test_expired_and_closed_leases_never_admit_a_request():
    pool, usage = pool_with_usage(clock=lambda: 10)
    assert pool.try_acquire(source(), purpose="checking", deadline=10) is None
    lease = pool.try_acquire(source(), purpose="learning", deadline=12)
    assert lease.acceptance_current
    pool.close_admission()
    assert not lease.acceptance_current
    assert usage == []


@pytest.mark.asyncio
async def test_cancelling_retained_task_is_not_a_thread_completion_witness():
    pool, usage = pool_with_usage()
    started, release = Event(), Event()

    def adapter(**_kwargs):
        started.set()
        assert release.wait(2)
        return "late"

    lease = pool.try_acquire(
        source(), purpose="checking", deadline=time.monotonic() + 30
    )
    task = asyncio.create_task(
        ConsoleProviderGateway(chat_api_call_fn=adapter).complete_auxiliary(
            _auxiliary_request(), rule_lease=lease
        )
    )
    try:
        await await_event(started)
        for retained in tuple(pool._tasks):
            retained.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.unsettled_count == 1 and not lease.physical_settled
        assert usage == []
    finally:
        release.set()
        await await_settlement(pool)
    assert lease.physical_settled and len(usage) == 1


@pytest.mark.asyncio
async def test_async_llama_cleanup_retains_late_usage_and_discards_content():
    pool, charged = pool_with_usage()
    started, release = asyncio.Event(), asyncio.Event()
    observed_timeouts = []

    async def handler(request):
        observed_timeouts.append(request.extensions["timeout"]["read"])
        started.set()
        await release.wait()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "late"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    resolution = _auxiliary_resolution(
        provider="llama_cpp",
        execution_key="llama_cpp",
        readiness_key="llama_cpp",
        base_url="http://127.0.0.1:9099",
        request_timeout=60,
    )
    lease = pool.try_acquire(
        source(), purpose="checking", deadline=time.monotonic() + 30
    )
    task = asyncio.create_task(
        ConsoleProviderGateway(http_client=client).complete_auxiliary(
            _auxiliary_request(resolution=resolution), rule_lease=lease
        )
    )
    try:
        async with asyncio.timeout(2):
            await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lease.acceptance_current and pool.unsettled_count == 1
        assert charged == []
    finally:
        release.set()
        await await_settlement(pool)
        await client.aclose()
    assert len(charged) == 1
    assert charged[0][2].uncached_input == 11 and charged[0][2].output == 5
    assert 0 < observed_timeouts[0] <= 30


@pytest.mark.asyncio
async def test_native_error_copy_omits_private_adapter_error_and_releases_capacity(
    caplog,
):
    from tldw_chatbook.Chat.Chat_Deps import ChatProviderError

    pool, usage = pool_with_usage()

    def adapter(**_kwargs):
        raise RuntimeError("SECRET-COMPLAINT-FIXTURE")

    lease = pool.try_acquire(
        source(), purpose="checking", deadline=time.monotonic() + 30
    )
    with pytest.raises(ChatProviderError) as raised:
        await ConsoleProviderGateway(chat_api_call_fn=adapter).complete_auxiliary(
            _auxiliary_request(), rule_lease=lease
        )
    assert "SECRET-COMPLAINT-FIXTURE" not in str(raised.value) + caplog.text
    assert pool.unsettled_count == 0
    assert len(usage) == 1 and usage[0][2] is None
