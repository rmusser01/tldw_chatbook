"""Real-controller regressions for refused wake retry rate and fairness."""

import asyncio

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _controller_rig,
    _drain,
    _settle,
    _survivor,
    _terminal_subagent_run,
)


@pytest.mark.asyncio
async def test_refused_wake_backs_off_then_recovers_without_an_external_poke(tmp_path):
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
        assert await _settle(lambda: calls)
        await asyncio.sleep(0.15)
        assert len(calls) == 1, "a refused provider must not be polled in a tight loop"
        assert wake.has_pending(session.id)
        assert not db.get_run(run_id).get("wake_delivered_at")
        gateway.ready = True
        assert await _settle(lambda: not wake.has_pending(session.id), seconds=3)
        assert len(gateway.payloads) == 1
        assert db.get_run(run_id).get("wake_delivered_at")
    finally:
        controller._disposed = True
        chacha.close()


@pytest.mark.asyncio
async def test_refused_first_conversation_does_not_starve_a_ready_second(tmp_path):
    chacha, _app, db, store, first, gateway, _bridge, controller = _controller_rig(
        tmp_path
    )
    second = store.create_session(title="ready second")
    submit = controller.submit_draft

    async def routed(*args, session_id=None, **kwargs):
        gateway.ready = session_id == second.id
        return await submit(*args, session_id=session_id, **kwargs)

    controller.submit_draft = routed
    wake = controller.fleet_wake
    try:
        for session in (first, second):
            _parent, run_id = _terminal_subagent_run(
                db, session.id, result=session.title
            )
            wake.on_fleet_drained(
                _drain(session.id, _survivor(run_id, session_id=session.id))
            )
        assert await _settle(lambda: gateway.payloads, seconds=0.5)
        assert "ready second" in gateway.payloads[0][-1]["content"]
        assert wake.has_pending(first.id)
        assert await _settle(lambda: not wake.has_pending(second.id))
    finally:
        controller._disposed = True
        chacha.close()


@pytest.mark.asyncio
async def test_disposed_controller_does_not_retry_a_refused_wake(tmp_path):
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
        assert await _settle(lambda: calls)
        controller._disposed = True
        before = len(calls)
        await asyncio.sleep(1.1)
        assert len(calls) == before
    finally:
        controller._disposed = True
        chacha.close()
