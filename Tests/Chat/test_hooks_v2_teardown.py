"""Real Interrupt observation after synchronous cancellation, mounted or viewless."""

import asyncio
import json
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.Agents.test_hooks_v2_execution import command
from Tests.Chat.test_console_fleet_wake import _controller_rig
from Tests.Chat.test_console_runtime_lifetime import _View
from Tests.conftest import _close_database_instance
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest


@pytest.fixture(autouse=True)
async def _close_mounted_app_owners(monkeypatch):
    """The Console harness does not run the backing app's shutdown lifecycle."""
    from Tests.conftest import _close_database_instance
    from Tests.UI import app_factory

    build = app_factory._build_test_app
    resources = []

    def build_owned(*args, **kwargs):
        app = build(*args, **kwargs)
        resources.append(
            (
                app,
                (
                    app.chachanotes_db,
                    app.local_library_collections_db,
                    app.evaluation_orchestrator.db,
                    app.local_workspace_db,
                    app.subscriptions_db,
                ),
                app._instance_lock_status.handle,
            )
        )
        return app

    monkeypatch.setattr(app_factory, "_build_test_app", build_owned)
    yield
    for app, databases, lock in resources:
        await app._shutdown_app_owned_lifecycles()
        for database in (*databases, app.chachanotes_db):
            _close_database_instance(database)
        if lock is not None:
            lock.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("revoked", [False, True])
async def test_interrupt_once_after_seal_and_revocation(
    tmp_path, mounted, revoked, slow
):
    db, app, runs_db, store, session, gateway, _, controller = _controller_rig(tmp_path)
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
        cancellation = controller._active_cancel_events[session.id]
        assert controller.stop_active_run()
        controller.stop_active_run()
        assert cancellation.is_set()
        assert controller._interrupt_host._hook_interrupts == {cancellation}
        await runtime.wait_for_turn(turn)
        for _ in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        # The host emits once; optional command execution keeps its one-second bound.
        observed = marker.exists()
        notification_failures = dict(engine.notification_failures)
        assert not revoked or not observed
        if not revoked and not observed:
            assert notification_failures in (
                {"event_deadline": 1},
                {"cancelled": 1},
            )
        if observed:
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
        assert controller._interrupt_host._hook_interrupts == {cancellation}
        if observed:
            assert len(marker.read_text().splitlines()) == 1
        else:
            assert not marker.exists()
            assert dict(engine.notification_failures) == notification_failures
        if slow and not revoked:
            for _ in range(2):
                waiter = asyncio.create_task(runtime.close_hooks_v2())
                await asyncio.sleep(0)
                deadline = engine.teardown_deadline
                if not waiter.done():
                    waiter.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await waiter
                else:
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
        await runtime.dispose()
        runs_db.close()
        _close_database_instance(db)


@pytest.mark.asyncio
async def test_real_mounted_console_stop_entry_emits_interrupt(tmp_path):
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_destination_shells import _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    db, rig_app, runs_db, store, session, gateway, _, controller = _controller_rig(
        tmp_path
    )
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.local_chat_conversation_service = rig_app.local_chat_conversation_service
    app.chachanotes_db = db
    app.local_chat_dictionary_service.db = db
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
            pending = asyncio.create_task(runtime.wait_for_turn(turn))
            await asyncio.wait_for(entered.wait(), 10)
            await screen._stop_console_generation_from_visible_action()
            await pending
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
        await runtime.dispose()
        runs_db.close()
        _close_database_instance(db)


@pytest.mark.asyncio
@pytest.mark.parametrize("collapsed", [False, True])
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_mounted_stop_button_cancels_pending_stop_proposal(
    tmp_path, collapsed, size
):
    from textual.widgets import Button

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
    from Tests.UI.test_destination_shells import _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    db, rig_app, runs_db, store, session, gateway, _, controller = _controller_rig(
        tmp_path
    )
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    app.local_chat_conversation_service = rig_app.local_chat_conversation_service
    app.chachanotes_db = db
    app.local_chat_dictionary_service.db = db
    runtime = ConsoleRuntime(app=app)
    app.console_runtime = runtime
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(gateway)
    runtime.set_chat_controller(controller)
    entered, release = tmp_path / "stop-entered", tmp_path / "stop-release"
    handler = command(
        "import time;from pathlib import Path;"
        f"Path({str(entered)!r}).touch();"
        f"exec({f'while not Path({str(release)!r}).exists(): time.sleep(0.005)'!r});"
        'print(\'{"version":2,"decision":"pass","continuation":{"message":"late proposal"}}\')',
        name="Stop",
        effects=["continuation"],
        # Keep the controlled process pending through mounted tab navigation.
        timeout_seconds=60,
    )
    notifications = tmp_path / "pending-notifications.jsonl"
    observe = (
        "import json,sys;from pathlib import Path;event=json.load(sys.stdin);"
        f"p=Path({str(notifications)!r});p.write_text((p.read_text() if p.exists() else '')+json.dumps(event)+'\\n')"
    )
    engine = runtime.ensure_hooks_v2(
        session.id,
        (
            handler,
            command(observe, name="Interrupt", id="pending-interrupt"),
            command(observe, name="SessionEnd", id="pending-end"),
        ),
        lambda *_: True,
    )
    original_stream = gateway.stream_chat
    cancellation = []
    interrupts = []
    notify = engine.notify_teardown

    def capture_interrupt(event):
        if event.event == "Interrupt":
            interrupts.append((event.event, event.turn_id))
        return notify(event)

    engine.notify_teardown = capture_interrupt

    async def capture_cancel(*args, **kwargs):
        cancellation.append(controller._active_cancel_events[session.id])
        async for chunk in original_stream(*args, **kwargs):
            yield chunk

    gateway.stream_chat = capture_cancel
    try:
        async with ConsoleHarness(app).run_test(size=size) as pilot:
            screen = pilot.app.screen
            await _wait_for_selector(screen, pilot, "#console-native-composer")
            assert screen._ensure_console_chat_controller() is controller
            composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
            _select_llamacpp_console(screen)
            selector = (
                "#console-collapsed-stop-generation"
                if collapsed
                else "#console-stop-generation"
            )
            composer.set_collapsed(collapsed)
            await pilot.pause()
            assert not screen.query_one(selector, Button).display
            composer.set_collapsed(False)
            composer.load_draft("mounted pending Stop")
            screen.query_one("#console-send-message", Button).press()
            for _ in range(2000):
                if entered.exists():
                    break
                await asyncio.sleep(0.005)
            assert entered.exists()
            composer.set_collapsed(collapsed)
            await pilot.pause()
            selector = (
                "#console-collapsed-stop-generation"
                if collapsed
                else "#console-stop-generation"
            )
            stop = screen.query_one(selector, Button)
            assert controller.is_stop_allowed
            settlement = controller.prompt_queue_coordinator._chains[session.id]
            parent_turn_id = settlement.request.turn_id
            for _ in range(40):
                if (
                    stop.display
                    and not screen.query_one(
                        "#console-redirect-generation", Button
                    ).display
                ):
                    break
                await pilot.pause(0.05)
            assert stop.display and not stop.disabled
            assert not screen.query_one("#console-redirect-generation", Button).display
            other = store.create_session(title="Other idle tab", activate=False)
            await screen._session._activate_native_console_session(other.id)
            await pilot.pause()
            assert not controller.is_stop_allowed
            for _ in range(40):
                if not screen.query_one(selector, Button).display:
                    break
                await pilot.pause(0.05)
            assert not screen.query_one(selector, Button).display
            assert not controller.stop_active_run()
            await screen._session._activate_native_console_session(session.id)
            await pilot.pause()
            for _ in range(40):
                if screen.query_one(selector, Button).display:
                    break
                await pilot.pause(0.05)
            assert screen.query_one(selector, Button).display
            assert await pilot.click(selector)
            await pilot.pause()
            assert (
                session.id in controller.prompt_queue_coordinator._sealed_continuations
            )
            assert cancellation[0].is_set()
            controller.stop_active_run()
            for _ in range(100):
                if not controller.prompt_queue_coordinator._chains:
                    break
                await pilot.pause(0.02)
            assert not controller.prompt_queue_coordinator._chains
            for _ in range(100):
                if notifications.exists() and not engine.cleanup_pending:
                    break
                await pilot.pause(0.02)
            assert interrupts == [("Interrupt", parent_turn_id)]
            # Interrupt observers are best effort within one second. Expiry can
            # precede dispatch or cancel delivery; emission/cancellation stay exact.
            notification_failures = dict(engine.notification_failures)
            observer_ran = notifications.exists()
            if observer_ran:
                observed = [
                    json.loads(line) for line in notifications.read_text().splitlines()
                ]
                assert [(event["event"], event["turn_id"]) for event in observed] == [
                    ("Interrupt", parent_turn_id)
                ]
            else:
                assert notification_failures in (
                    {"event_deadline": 1},
                    {"cancelled": 1},
                )
            assert settlement.hook_cancel_event is None
            assert settlement.pending_stop_task is None
            assert settlement.pending_stop_key is None
            assert not release.exists()
            assert engine.lifecycle_owner.live
            assert not engine.cleanup_pending
            assert len(gateway.payloads) == 1
            assert (
                db.get_connection()
                .execute("SELECT COUNT(*) FROM console_hook_continuation_receipts")
                .fetchone()[0]
                == 0
            )
            release.touch()
            composer.set_collapsed(False)
            composer.load_draft("fresh ordinary work")
            screen.query_one("#console-send-message", Button).press()
            for _ in range(300):
                if (
                    len(gateway.payloads) >= 2
                    and not controller.prompt_queue_coordinator._chains
                ):
                    break
                await pilot.pause(0.02)
            assert 2 <= len(gateway.payloads) <= 5
            assert not controller.prompt_queue_coordinator._chains
            assert all(not event.is_set() for event in cancellation[1:])
            assert interrupts == [("Interrupt", parent_turn_id)]
            assert dict(engine.notification_failures) == notification_failures
            assert notifications.exists() is observer_ran
            if observer_ran:
                assert len(notifications.read_text().splitlines()) == 1
    finally:
        release.touch()
        await runtime.close_hooks_v2()
        assert not engine.processes.records
        await runtime.dispose()
        _close_database_instance(db)
        runs_db.close()
