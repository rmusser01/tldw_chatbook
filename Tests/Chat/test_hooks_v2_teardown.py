"""Real Interrupt observation after synchronous cancellation, mounted or viewless."""

import asyncio
import json
from uuid import uuid4

import pytest

from Tests.Agents.test_hooks_v2_execution import command
from Tests.Chat.test_console_fleet_wake import _controller_rig
from Tests.Chat.test_console_runtime_lifetime import _View
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("revoked", [False, True])
async def test_interrupt_once_after_seal_and_revocation(
    tmp_path, mounted, revoked, slow
):
    db, app, _, store, session, gateway, _, controller = _controller_rig(tmp_path)
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    if mounted:
        runtime.attach_view(_View({}))
    marker = tmp_path / "interrupt.jsonl"
    linger = (
        ";import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)"
        if slow
        else ""
    )
    handler = command(
        "import json,sys;from pathlib import Path;event=json.load(sys.stdin);"
        f"p=Path({str(marker)!r});p.write_text((p.read_text() if p.exists() else '')+json.dumps(event)+'\\n')"
        + linger,
        name="Interrupt",
    )
    authority = [True]
    engine = runtime.ensure_hooks_v2(session.id, (handler,), lambda *_: authority[0])
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_stream(*args, **kwargs):
        started.set()
        await release.wait()
        yield "done"

    normal_stream = gateway.stream_chat
    gateway.stream_chat = blocked_stream
    request = ConsoleTurnCustodyRequest(
        turn_id=str(uuid4()),
        session_id=session.id,
        draft="hello",
        configuration=controller.resolve_turn_configuration_snapshot(session.id),
    )
    turn = runtime.accept_turn(request)
    try:
        await asyncio.wait_for(started.wait(), 3)
        authority[0] = not revoked
        assert controller.stop_active_run()
        controller.stop_active_run()
        assert controller._active_cancel_events[session.id].is_set()
        await runtime.wait_for_turn(turn)
        for _ in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.exists() is (not revoked)
        if not revoked:
            events = [json.loads(line) for line in marker.read_text().splitlines()]
            assert len(events) == 1
            assert events[0]["event"] == "Interrupt"
            assert events[0]["turn_id"] == request.turn_id
        release.set()
        authority[0] = True
        gateway.stream_chat = normal_stream
        next_request = ConsoleTurnCustodyRequest(
            turn_id=str(uuid4()),
            session_id=session.id,
            draft="new ordinary work",
            configuration=controller.resolve_turn_configuration_snapshot(session.id),
        )
        followup = runtime.accept_turn(next_request)
        assert (await runtime.wait_for_turn(followup)).accepted
        assert gateway.payloads
        if not revoked:
            assert len(marker.read_text().splitlines()) == 1
        if slow and not revoked:
            for _ in range(2):
                waiter = asyncio.create_task(runtime.close_hooks_v2())
                await asyncio.sleep(0.01)
                deadline = engine.teardown_deadline
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
                assert runtime._hooks_v2_cleanup_task is not None
                assert not runtime._hooks_v2_cleanup_task.cancelled()
                assert engine.teardown_deadline == deadline
            await asyncio.wait_for(runtime.close_hooks_v2(), 3)
            assert not engine.cleanup_pending
            assert not engine.processes.records
    finally:
        release.set()
        await runtime.close_hooks_v2()
        db.close()


@pytest.mark.asyncio
async def test_real_mounted_console_stop_entry_emits_interrupt(tmp_path):
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    db, rig_app, _, store, session, gateway, _, controller = _controller_rig(tmp_path)
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.local_chat_conversation_service = rig_app.local_chat_conversation_service
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    marker = tmp_path / "mounted-interrupt"
    engine = runtime.ensure_hooks_v2(
        session.id,
        (
            command(
                f"from pathlib import Path;Path({str(marker)!r}).touch()",
                name="Interrupt",
            ),
        ),
        lambda *_: True,
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def stream(*args, **kwargs):
        entered.set()
        await release.wait()
        yield "done"

    gateway.stream_chat = stream
    try:
        async with ConsoleHarness(app).run_test(size=(160, 48)) as pilot:
            screen = pilot.app.screen
            assert isinstance(screen, ChatScreen)
            await _wait_for_selector(screen, pilot, "#console-native-composer")
            assert screen._ensure_console_chat_controller() is controller
            request = ConsoleTurnCustodyRequest(
                turn_id=str(uuid4()),
                session_id=session.id,
                draft="mounted input",
                configuration=controller.resolve_turn_configuration_snapshot(
                    session.id
                ),
            )
            turn = runtime.accept_turn(request)
            await asyncio.wait_for(entered.wait(), 5)
            await screen._stop_console_generation_from_visible_action()
            await runtime.wait_for_turn(turn)
            for _ in range(200):
                if marker.exists():
                    break
                await asyncio.sleep(0.005)
            assert marker.exists()
            assert (
                session.id in controller.prompt_queue_coordinator._sealed_continuations
            )
        await runtime.close_hooks_v2()
        assert not engine.cleanup_pending
    finally:
        release.set()
        await runtime.close_hooks_v2()
        db.close()
