"""Real gateway and SQLite enforce each accepted automatic generation."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace

import httpx
import pytest

from Tests.DB.test_automatic_wake_attempts import claim, survivor
from Tests.DB.test_automatic_work_budget import _automatic_work_db, chain  # noqa: F401
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionRequest,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)


def context_for(db, *, accepted=True, **limits):
    from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext

    chain_id = chain(db, **limits)
    claim(db, chain_id, [survivor(db, chain_id)])
    context = AutomaticWorkContext(db.automatic_work, chain_id, "owner", "attempt")
    if accepted:
        assert db.automatic_work.accept_wake("attempt", owner_id="owner")
        context.mark_accepted()
    return context


def resolution(**changes):
    selected = replace(
        ConsoleProviderResolution(
            provider="openai",
            execution_key="openai",
            base_url="",
            model="gpt-4o",
            ready=True,
            max_tokens=16000,
        ),
        **changes,
    )

    from tldw_chatbook.Chat.console_library_destination import resolve_console_destination
    return replace(selected, resolved_destination=resolve_console_destination(selected))

def response(usage=True):
    result = {"choices": [{"message": {"content": "finished"}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 7, "completion_tokens": 3}
    return result


async def collect(gateway, selected=None, *, signals=None, messages=None):
    return [
        chunk
        async for chunk in gateway.stream_chat(
            selected or resolution(),
            messages or [{"role": "user", "content": "Hi"}],
            signals=signals,
        )
    ]


@pytest.mark.asyncio
async def test_prepared_attempt_cannot_generate_or_self_authorize(db):
    context = context_for(db, accepted=False)
    dispatched = []
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **kw: dispatched.append(kw)
    )
    try:
        with pytest.raises(AutomaticWorkRefused, match="acceptance_required"):
            context.mark_accepted()
        with (
            context.scope(),
            pytest.raises(AutomaticWorkRefused, match="acceptance_required"),
        ):
            await collect(gateway)
        assert dispatched == []
        assert db.automatic_work.snapshot(context.chain_id).used["model_call"] == 0
        with pytest.raises(FrozenInstanceError):
            context.chain_id = "different"
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_plain_and_scoped_signals_share_exact_call_limit_and_settle_usage(db):
    context = context_for(db, model_calls=2, output_tokens=11)
    dispatched = []

    def provider(**kwargs):
        dispatched.append(kwargs)
        return response()

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider)
    try:
        with context.scope():
            assert await collect(gateway) == ["finished"]
            signals = ConsoleProviderStreamSignals().new_usage_call()
            assert await collect(gateway, signals=signals) == ["finished"]
            with pytest.raises(AutomaticWorkRefused, match="model_call_budget"):
                await collect(gateway)
        assert len(dispatched) == 2
        assert [call["max_tokens"] for call in dispatched] == [11, 11]
        snapshot = db.automatic_work.snapshot(context.chain_id)
        assert snapshot.used["model_call"] == 2
        assert snapshot.used["tokens"] == 20
        assert snapshot.reserved["tokens"] == 0
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_auxiliary_uses_shared_allowance_and_unknown_usage_pauses_chain(db):
    context = context_for(db, model_calls=2, output_tokens=17)
    dispatched = []
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **kw: (dispatched.append(kw), response(False))[1]
    )
    try:
        with context.scope():
            result = await gateway.complete_auxiliary(
                AuxiliaryCompletionRequest(
                    resolution=resolution(),
                    messages=({"role": "user", "content": "Summarize"},),
                    response_format=None,
                    max_output_tokens=256,
                )
            )
            assert result.text == "finished"
            with pytest.raises(AutomaticWorkRefused, match="usage_unknown"):
                await collect(gateway)
        assert dispatched[0]["max_tokens"] == 17
        snapshot = db.automatic_work.snapshot(context.chain_id)
        assert snapshot.used["model_call"] == 1
        assert snapshot.reserved["tokens"] > 17
        assert snapshot.status == "review_required"
    finally:
        await gateway.aclose()


def test_grouped_reservation_is_atomic_and_duplicate_never_dispatches(db):
    context = context_for(db, model_calls=5, budget_tokens=100)
    ledger = db.automatic_work
    with pytest.raises(AutomaticWorkRefused, match="tokens_budget"):
        ledger.admit_call(
            context.chain_id,
            call_id="too-large",
            owner_id="owner",
            input_tokens=95,
            output_tokens=6,
        )
    assert ledger.snapshot(context.chain_id).used["model_call"] == 0
    first = ledger.admit_call(
        context.chain_id,
        call_id="call",
        owner_id="owner",
        input_tokens=60,
        output_tokens=40,
    )
    assert first
    assert (
        ledger.admit_call(
            context.chain_id,
            call_id="call",
            owner_id="owner",
            input_tokens=60,
            output_tokens=40,
        )
        is None
    )
    assert ledger.snapshot(context.chain_id).used["model_call"] == 1


def test_concurrent_calls_cannot_spend_same_remainder(db):
    context = context_for(db, budget_tokens=100)
    ready = threading.Barrier(2)

    def reserve_call(index):
        ready.wait()
        try:
            return db.automatic_work.admit_call(
                context.chain_id,
                call_id=str(index),
                owner_id="owner",
                input_tokens=50,
                output_tokens=30,
            )
        except AutomaticWorkRefused:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(reserve_call, [1, 2]))
    assert sum(result is not None for result in results) == 1
    assert db.automatic_work.snapshot(context.chain_id).used["tokens"] == 80


@pytest.mark.asyncio
async def test_each_local_fallback_generation_has_its_own_call_and_token_charge(db):
    context = context_for(db, model_calls=2, output_tokens=13)
    dispatched = []

    def transport(request):
        payload = json.loads(request.content)
        dispatched.append(payload)
        if payload["stream"]:
            body = 'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\ndata: [DONE]\n\n'
            return httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=response())

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport))
    )
    try:
        with context.scope():
            assert await collect(
                gateway,
                resolution(
                    provider="llama_cpp",
                    execution_key="",
                    base_url="http://localhost:9099",
                ),
            ) == ["finished"]
        assert [item["max_tokens"] for item in dispatched] == [13, 13]
        snapshot = db.automatic_work.snapshot(context.chain_id)
        assert snapshot.used["model_call"] == 2
        assert snapshot.used["tokens"] == 13
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_cancellation_retains_estimate_while_provider_worker_still_runs(db):
    context = context_for(db, output_tokens=31)
    entered, release = threading.Event(), threading.Event()

    def provider(**kwargs):
        entered.set()
        assert release.wait(5)
        return response()

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider)
    try:
        with context.scope():
            task = asyncio.create_task(collect(gateway))
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        snapshot = db.automatic_work.snapshot(context.chain_id)
        assert snapshot.status == "review_required"
        assert snapshot.reserved["tokens"] > 31
        assert snapshot.used["model_call"] == 1
    finally:
        release.set()
        await gateway.aclose()


def test_live_reductions_deadline_and_review_pause_remain_enforced(db, monkeypatch):
    context = context_for(db, output_tokens=100)
    monkeypatch.setenv("TLDW_AGENTS_MAX_AUTOWAKE_OUTPUT_TOKENS", "40")
    assert context.output_cap(200) == 40
    monkeypatch.setenv("TLDW_AGENTS_MAX_AUTOWAKE_OUTPUT_TOKENS", "400")
    assert context.output_cap(None) == 100
    db.automatic_work.pause(
        context.chain_id, "completion_write_failed", review_required=True
    )
    assert context.should_cancel()
    with pytest.raises(AutomaticWorkRefused, match="completion_write_failed"):
        context.check()
    assert (
        db.automatic_work.read_attempt("attempt", owner_id="owner").state == "accepted"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 7},
        {"input_tokens": 7},
        {"prompt_tokens": -1, "completion_tokens": 4},
    ],
)
async def test_incomplete_or_malformed_usage_never_refunds_reserved_input(db, usage):
    context = context_for(db, output_tokens=30)
    data = response(False)
    data["usage"] = usage
    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **kw: data)
    try:
        with context.scope():
            await collect(gateway)
        snapshot = db.automatic_work.snapshot(context.chain_id)
        assert snapshot.status == "review_required"
        assert snapshot.reserved["tokens"] > 30
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_real_prepared_input_and_tools_are_reserved_without_repreparing(db):
    from tldw_chatbook.Chat.console_history_budget import count_console_messages_tokens

    context = context_for(db, output_tokens=29)
    observed = []
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]

    def provider(**kwargs):
        wire = [
            {"role": "system", "content": kwargs["system_message"]},
            *kwargs["messages_payload"],
        ]
        input_tokens = count_console_messages_tokens(wire, "gpt-4o")
        input_tokens += count_console_messages_tokens(
            [{"role": "system", "content": json.dumps(tools, separators=(",", ":"))}],
            "gpt-4o",
        )
        observed.append(
            (
                input_tokens + 29,
                db.automatic_work.snapshot(context.chain_id).used["tokens"],
            )
        )
        return response()

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider)
    try:
        prepared = gateway.prepare_chat_request(
            resolution(),
            [
                {"role": "system", "content": "  first  "},
                {"role": "system", "content": "second"},
                {"role": "user", "content": "Hi"},
            ],
            tools=tools,
        )
        with context.scope():
            await collect(gateway, messages=prepared)
        assert len(observed) == 1
        assert observed[0][0] == observed[0][1]
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_actual_overage_stops_next_call_without_changing_reported_usage(db):
    context = context_for(db, budget_tokens=100, output_tokens=20)
    data = response(False)
    data["usage"] = {"prompt_tokens": 150, "completion_tokens": 20}
    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **kw: data)
    try:
        with context.scope():
            await collect(gateway)
            with pytest.raises(AutomaticWorkRefused, match="tokens_budget"):
                await collect(gateway)
        snapshot = db.automatic_work.snapshot(context.chain_id)
        assert snapshot.used["tokens"] == 170
        assert snapshot.used["model_call"] == 1
        assert data["usage"]["prompt_tokens"] == 150
    finally:
        await gateway.aclose()


def test_elapsed_check_pauses_without_resetting_at_completed_primary(db):
    from tldw_chatbook.DB.automatic_work import AutomaticWorkLedger

    clocks = [1000.0, 0.0]
    db.automatic_work = AutomaticWorkLedger(
        db,
        wall_clock=lambda: clocks[0],
        monotonic_clock=lambda: clocks[1],
    )
    context = context_for(db, wall_seconds=10)
    assert db.automatic_work.complete_wake("attempt", owner_id="owner")
    assert not context.should_cancel()
    clocks[:] = [1001.0, 11.0]
    assert context.should_cancel()
    db.close()
    assert db.automatic_work.snapshot(context.chain_id).pause_reason == "wall_budget"


def test_grouped_write_failure_rolls_back_both_resources_under_full_sync(db):
    import sqlite3

    context = context_for(db)
    observed = []
    with db.connection() as conn:
        conn.create_function(
            "capture_sync",
            0,
            lambda: (
                observed.append(conn.execute("PRAGMA synchronous").fetchone()[0]) or 1
            ),
        )
        conn.execute(
            "CREATE TRIGGER fail_token_reservation BEFORE INSERT ON automatic_work_reservations WHEN NEW.kind='tokens' BEGIN SELECT capture_sync(); SELECT RAISE(ABORT, 'token write failed'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="token write failed"):
        context.begin_call(20, 20)
    snapshot = db.automatic_work.snapshot(context.chain_id)
    assert snapshot.used["model_call"] == 0
    assert snapshot.used["tokens"] == 0
    assert observed == [2]
    with db.connection() as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_exhaustion_pause_cannot_be_cleared_by_cancellation_poll(db):
    context = context_for(db, model_calls=1)
    reservation = context.begin_call(20, 20)
    context.settle_call(reservation, 10)
    with pytest.raises(AutomaticWorkRefused, match="model_call_budget"):
        context.begin_call(20, 20)
    assert context.should_cancel()
    assert (
        db.automatic_work.snapshot(context.chain_id).pause_reason == "model_call_budget"
    )


@pytest.mark.asyncio
async def test_local_fallback_is_refused_after_first_calls_exhaustion(db):
    context = context_for(db, model_calls=1)
    count = []

    def transport(request):
        count.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n',
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport))
    )
    try:
        with (
            context.scope(),
            pytest.raises(AutomaticWorkRefused, match="model_call_budget"),
        ):
            await collect(
                gateway,
                resolution(
                    provider="llama_cpp",
                    execution_key="",
                    base_url="http://localhost:9099",
                ),
            )
        assert len(count) == 1
        assert db.automatic_work.snapshot(context.chain_id).used["tokens"] == 3
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_explicit_manual_scope_overrides_inherited_automatic_authority(db):
    from tldw_chatbook.Agents.automatic_work_runtime import (
        current_automatic_work,
        manual_work_scope,
    )

    context = context_for(db, model_calls=0)
    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **kw: response())
    try:
        with context.scope():
            with manual_work_scope():
                assert current_automatic_work() is None
                assert await collect(gateway) == ["finished"]
            assert current_automatic_work() is context
        assert current_automatic_work() is None
        assert db.automatic_work.snapshot(context.chain_id).used["model_call"] == 0
    finally:
        await gateway.aclose()


@pytest.mark.parametrize("value", [False, "false", "off", "0"])
def test_kill_switch_uses_native_string_and_boolean_coercion(db, monkeypatch, value):
    from tldw_chatbook.Agents import automatic_work_runtime

    context = context_for(db)
    monkeypatch.setattr(automatic_work_runtime, "_setting", lambda key, default: value)
    with pytest.raises(AutomaticWorkRefused, match="autowake_disabled"):
        context.check()


@pytest.mark.asyncio
async def test_automatic_provider_consumption_uses_existing_no_retry_policy(db):
    from tldw_chatbook.Utils.sensitive_llm_logging import llm_retry_count

    context = context_for(db)
    retries = []

    def provider(**kwargs):
        retries.append(llm_retry_count(3))
        yield response()

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider)
    try:
        with context.scope():
            await collect(gateway)
        assert retries == [0]
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("auxiliary", [False, True])
async def test_queued_provider_rechecks_pause_before_physical_dispatch(
    db, monkeypatch, auxiliary
):
    from tldw_chatbook.Chat.console_provider_gateway import ChatProviderError

    context = context_for(db)
    admitted = asyncio.Event()
    original_admit = db.automatic_work.admit_call

    def admit(*args, **kwargs):
        result = original_admit(*args, **kwargs)
        admitted.set()
        return result

    monkeypatch.setattr(db.automatic_work, "admit_call", admit)
    loop = asyncio.get_running_loop()
    # Ensure the prior default executor exists so this test can restore it.
    await asyncio.to_thread(lambda: None)
    previous = loop._default_executor
    pool = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    blocker = pool.submit(release.wait)
    loop.set_default_executor(pool)
    dispatched = []
    gateway = ConsoleProviderGateway(
        chat_api_call_fn=lambda **kw: (dispatched.append(kw), response())[1]
    )

    async def request():
        with context.scope():
            if auxiliary:
                return await gateway.complete_auxiliary(
                    AuxiliaryCompletionRequest(
                        resolution=resolution(),
                        messages=({"role": "user", "content": "hi"},),
                        max_output_tokens=20,
                        response_format=None,
                    )
                )
            return await collect(gateway)

    pending = asyncio.create_task(request())
    try:
        await asyncio.wait_for(admitted.wait(), 1)
        db.automatic_work.pause(context.chain_id, "wall_budget")
        release.set()
        with pytest.raises((AutomaticWorkRefused, ChatProviderError)):
            await asyncio.wait_for(pending, 2)
        assert dispatched == []
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await asyncio.wrap_future(blocker)
        loop.set_default_executor(previous)
        pool.shutdown(wait=True)
        await gateway.aclose()


@pytest.mark.asyncio
async def test_lazy_provider_rechecks_pause_before_iteration(db):
    from tldw_chatbook.Chat.console_provider_gateway import ChatProviderError

    context = context_for(db)
    dispatched = []

    def provider(**kwargs):
        db.automatic_work.pause(context.chain_id, "wall_budget")

        def generation():
            dispatched.append(kwargs)
            yield response()

        return generation()

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider)
    try:
        with context.scope(), pytest.raises(ChatProviderError):
            await collect(gateway)
        assert dispatched == []
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_local_stream_rechecks_after_client_acquisition(db, monkeypatch):
    context = context_for(db)
    dispatched = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: (
                dispatched.append(request),
                httpx.Response(
                    200, text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                ),
            )[1]
        )
    )
    gateway = ConsoleProviderGateway(http_client=client)

    def acquire():
        db.automatic_work.pause(context.chain_id, "wall_budget")
        return client

    monkeypatch.setattr(gateway, "_active_http_client", acquire)
    try:
        with context.scope(), pytest.raises(AutomaticWorkRefused, match="wall_budget"):
            await collect(
                gateway,
                resolution(
                    provider="llama_cpp",
                    execution_key="",
                    base_url="http://localhost:9099",
                ),
            )
        assert dispatched == []
    finally:
        await gateway.aclose()
