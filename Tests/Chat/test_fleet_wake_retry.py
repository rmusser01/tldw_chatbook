"""Real-controller regressions for refused wake retry rate and fairness.

Private-profile tests (TASK-41138): a wake turn is a real Console submit,
which reads config. Settle deadlines: measured at most 4.1 s at load ~100
on 18 CPUs (first resolve 1.2-1.5 s, first payload 3.6 s, delivery after the
provider turns ready 4.1 s), so 10 s is over 2x that.
"""

import asyncio
import contextvars

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _close_controller_rig,
    _controller_rig,
    _drain,
    _settle,
    _survivor,
    _terminal_subagent_run,
)
from Tests.console_provider_doubles import provider_resolution
from Tests.private_profile import private_profile_test

_SETTLE_SECONDS = 10.0


@pytest.mark.asyncio
@private_profile_test
async def test_refused_wake_backs_off_then_recovers_without_an_external_poke(tmp_path, request):
    chacha, _app, db, _store, session, gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    calls = []
    real_resolve = gateway.resolve_for_send

    async def counted(selection):
        calls.append(asyncio.get_running_loop().time())
        return await real_resolve(selection)

    gateway.resolve_for_send = counted
    gateway.ready = False
    _parent, run_id = _terminal_subagent_run(db, session.id)
    wake = controller.fleet_wake
    try:
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: calls, seconds=_SETTLE_SECONDS)
        await asyncio.sleep(0.15)
        assert len(calls) == 1, "a refused provider must not be polled in a tight loop"
        assert wake.has_pending(session.id)
        assert not db.get_run(run_id).get("wake_delivered_at")
        gateway.ready = True
        assert await _settle(
            lambda: not wake.has_pending(session.id), seconds=_SETTLE_SECONDS
        )
        assert len(gateway.payloads) == 1
        assert db.get_run(run_id).get("wake_delivered_at")
    finally:
        await _close_controller_rig(chacha, db, controller)


@pytest.mark.asyncio
@private_profile_test
async def test_refused_first_conversation_does_not_starve_a_ready_second(tmp_path, request):
    chacha, _app, db, store, first, gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    second = store.create_session(title="ready second")
    submit = controller.submit_draft
    submitted = []
    # Per submit, not one shared flag: the two wake attempts interleave, so a
    # flag the second set could reach the first's resolve (TASK-41138).
    resolving = contextvars.ContextVar("resolving", default=None)

    async def routed(*args, session_id=None, **kwargs):
        submitted.append(session_id)
        resolving.set(session_id)
        return await submit(*args, session_id=session_id, **kwargs)

    async def ready_for_second(selection):
        return provider_resolution(ready=resolving.get() == second.id)

    controller.submit_draft = routed
    gateway.resolve_for_send = ready_for_second
    wake = controller.fleet_wake
    try:
        for session in (first, second):
            _parent, run_id = _terminal_subagent_run(
                db, session.id, result=session.title
            )
            wake.on_fleet_drained(
                _drain(session.id, _survivor(run_id, session_id=session.id))
            )
        # Fairness by order, not by a 0.5 s window: the durable pre-submit
        # hops (7212692c8b, b5251e9a6e) put the first payload about 1 s
        # after the drain under load. The second never waits out a retry of
        # the refused first (TASK-39044).
        assert await _settle(lambda: gateway.payloads, seconds=_SETTLE_SECONDS)
        assert "ready second" in gateway.payloads[0][-1]["content"]
        assert second.id in submitted, submitted
        assert submitted[: submitted.index(second.id)].count(first.id) <= 1, (
            f"the ready second waited behind the refused first's retries: {submitted}"
        )
        assert wake.has_pending(first.id)
        assert await _settle(
            lambda: not wake.has_pending(second.id), seconds=_SETTLE_SECONDS
        )
    finally:
        await _close_controller_rig(chacha, db, controller)


@pytest.mark.asyncio
@private_profile_test
async def test_disposed_controller_does_not_retry_a_refused_wake(tmp_path, request):
    chacha, _app, db, _store, session, gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    calls = []
    real_resolve = gateway.resolve_for_send

    async def counted(selection):
        calls.append(True)
        return await real_resolve(selection)

    gateway.resolve_for_send = counted
    gateway.ready = False
    _parent, run_id = _terminal_subagent_run(db, session.id)
    wake = controller.fleet_wake
    try:
        wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: calls, seconds=_SETTLE_SECONDS)
        controller._disposed = True
        before = len(calls)
        await asyncio.sleep(1.1)
        assert len(calls) == before
    finally:
        await _close_controller_rig(chacha, db, controller)
