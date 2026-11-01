"""PR3300: exercise Terminal return and input through mounted Console owners."""

from types import SimpleNamespace

import pytest
from textual.events import Paste
from textual.screen import ModalScreen
from textual.widgets import Input

from Tests.UI.test_console_terminal_integration import (
    _create_running_terminal,
    _terminal_app,
    _wait_until,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_terminal_workspace import TerminalViewport
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

pytestmark = pytest.mark.bootstrap_profile


async def _settled_console(host, pilot):
    screen = host.screen_stack[-1]
    await _wait_for_selector(screen, pilot, "#console-native-composer", timeout=10)

    def attach_details():
        names = (
            "_console_attach_reconciled",
            "_console_attach_sync_complete",
            "_console_attach_runtime_reconciled",
            "_console_attach_view_started",
            "_console_attach_reconcile_running",
            "_console_attach_reconcile_retry_count",
            "_console_attach_reconcile_retry_exhausted",
            "_console_sync_in_progress",
            "_console_sync_requested",
            "_console_sync_maintenance_paused",
            "_console_control_bar_replay_whole_sync",
            "_resume_navigation_startup_in_progress",
        )
        details = [f"{name}={getattr(screen, name, None)!r}" for name in names]
        return "\n".join(details)

    await _wait_until(
        pilot,
        lambda: screen._console_attach_reconciled
        and not screen._console_attach_resume_in_progress,
        timeout=20,
        detail=attach_details,
    )
    return screen


async def _take_terminal_input(manager, session_id, pilot):
    """Observe the real input actor; the inert backend has no runtime bridge."""
    received = []

    def has_input():
        event = manager.take_input(session_id)
        if event is not None:
            received.append(event)
        return bool(received)

    await _wait_until(pilot, has_input)
    return received[0].data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deferral,switch_before_callback",
    [("maintenance", False), ("maintenance", True), ("projection", False)],
)
async def test_return_reading_waits_for_publication_and_rechecks_owner(
    monkeypatch,
    deferral,
    switch_before_callback,
):
    app, manager, _ = _terminal_app(permitted=True)
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            screen = await _settled_console(host, pilot)
            store = screen._ensure_console_chat_store()
            first = store.ensure_session()
            second = store.create_session(title="Second")
            for session in (first, second):
                for i in range(24):
                    store.append_message(
                        session.id,
                        role=ConsoleMessageRole.USER,
                        content=f"{session.id} message {i}\nsecond line",
                    )
            store.switch_session(first.id)
            await screen._sync_native_console_chat_ui()
            transcript = screen.query_one(
                "#console-native-transcript", ConsoleTranscript
            )
            await _wait_until(pilot, lambda: transcript.max_scroll_y > 4)
            transcript.release_anchor()
            transcript.scroll_to(y=2, animate=False)
            await _wait_until(pilot, lambda: transcript.scroll_y == 2)
            screen.action_open_console_terminal()
            await _wait_for_selector(screen, pilot, "#console-terminal-viewport")
            await _wait_until(pilot, lambda: not screen._console_sync_in_progress)
            projection = screen._change_review_projection
            prepare = projection.prepare

            async def held_prepare():
                return False

            if deferral == "maintenance":
                screen._console_sync_maintenance_paused = True
            else:
                monkeypatch.setattr(projection, "prepare", held_prepare)
            assert await pilot.click("#console-terminal-return")
            await _wait_for_selector(screen, pilot, "#console-native-transcript")
            await pilot.pause()
            assert (
                screen._console_terminal_return_reading is not None
            ), "Deferred publication discarded the return position"
            restores = []
            restore = screen._restore_console_transcript_reading_state

            def record_restore(state, **kwargs):
                restores.append(store.active_session_id)
                restore(state, **kwargs)

            monkeypatch.setattr(
                screen, "_restore_console_transcript_reading_state", record_restore
            )
            after_refresh = screen.call_after_refresh
            callbacks = []
            if switch_before_callback:
                monkeypatch.setattr(
                    screen,
                    "call_after_refresh",
                    lambda callback, *args, **kw: callbacks.append(
                        (callback, args, kw)
                    ),
                )
            screen._console_sync_maintenance_paused = False
            monkeypatch.setattr(projection, "prepare", prepare)
            await screen._sync_native_console_chat_ui()
            if switch_before_callback:
                store.switch_session(second.id)
                monkeypatch.setattr(screen, "call_after_refresh", after_refresh)
                for callback, args, kw in callbacks:
                    after_refresh(callback, *args, **kw)
                await pilot.pause()
                assert (
                    restores == []
                ), "Queued return restored the previous owner's offset"
                await screen._sync_native_console_chat_ui()
            else:
                await _wait_until(pilot, lambda: bool(restores))
                restored = screen.query_one(
                    "#console-native-transcript", ConsoleTranscript
                )
                await _wait_until(pilot, lambda: restored.scroll_y == 2)
                assert restores == [first.id]
    finally:
        manager.finalize_shutdown()


class _PasteModal(ModalScreen):
    def compose(self):
        yield Input(id="modal-paste")


@pytest.mark.asyncio
async def test_terminal_paste_uses_bounded_input_and_respects_release_and_modal(
    tmp_path,
):
    app, manager, _backends = _terminal_app(permitted=True)
    session_id = _create_running_terminal(manager, tmp_path)
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            screen = await _settled_console(host, pilot)
            composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
            draft = composer.draft_text()
            screen.action_open_console_terminal()
            await _wait_for_selector(screen, pilot, "#console-terminal-viewport")
            viewport = screen.query_one("#console-terminal-viewport", TerminalViewport)
            viewport.focus_input()
            await _wait_until(pilot, lambda: host.focused is viewport)
            host.post_message(Paste("plain paste\n"))
            assert (
                await _take_terminal_input(manager, session_id, pilot)
                == b"plain paste\n"
            )
            assert manager.offer_output(session_id, b"\x1b[?2004h").accepted
            manager.process_output(session_id, visible=True)
            await pilot.pause()
            host.post_message(Paste("bracketed paste\n"))
            assert await _take_terminal_input(manager, session_id, pilot) == (
                b"\x1b[200~bracketed paste\n\x1b[201~"
            )
            host.post_message(Paste("PRIVATE-PASTE-SENTINEL\x1bcontrol"))
            await pilot.pause()
            assert manager.take_input(session_id) is None
            status = str(screen.query_one("#console-terminal-status").renderable)
            assert "paste" in status.lower() and "PRIVATE-PASTE-SENTINEL" not in status
            assert "control" in status.lower()
            await pilot.press("ctrl+right_square_bracket")
            assert not viewport.input_focused
            host.post_message(Paste("released paste"))
            await pilot.pause()
            assert manager.take_input(session_id) is None
            viewport.focus_input()
            await pilot.pause()
            modal = _PasteModal()
            await host.push_screen(modal)
            modal.query_one(Input).focus()
            await pilot.pause()
            host.post_message(Paste("modal paste"))
            await _wait_until(
                pilot, lambda: modal.query_one(Input).value == "modal paste"
            )
            assert manager.take_input(session_id) is None
            assert composer.draft_text() == draft
            modal.dismiss()
    finally:
        manager.finalize_shutdown()


@pytest.mark.asyncio
async def test_terminal_capture_gets_escape_before_active_voice(tmp_path):
    app, manager, _backends = _terminal_app(permitted=True)
    session_id = _create_running_terminal(manager, tmp_path)
    host = ConsoleHarness(app)
    exits = []
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            screen = await _settled_console(host, pilot)
            screen.action_open_console_terminal()
            await _wait_for_selector(screen, pilot, "#console-terminal-viewport")
            viewport = screen.query_one("#console-terminal-viewport", TerminalViewport)
            viewport.focus_input()
            await _wait_until(pilot, lambda: host.focused is viewport)
            screen._console_realtime = SimpleNamespace(
                controller=SimpleNamespace(on_exit_request=lambda: exits.append(True)),
            )
            try:
                await pilot.press("escape")
                await pilot.pause()
                assert exits == [], "Priority voice binding stole the shell's Escape"
                assert await _take_terminal_input(manager, session_id, pilot) == b"\x1b"
                await pilot.press("ctrl+right_square_bracket", "escape")
                await _wait_until(pilot, lambda: exits == [True])
                assert manager.take_input(session_id) is None
            finally:
                screen._console_realtime = None
    finally:
        manager.finalize_shutdown()


@pytest.mark.asyncio
async def test_deferred_return_sync_does_not_schedule_a_restore(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import terminal

    callbacks = []
    workers = []
    state = object()

    async def held_sync():
        return False

    screen = SimpleNamespace(
        _console_terminal_open=False,
        _console_terminal_return_reading=("owner", state),
        _console_chat_store=SimpleNamespace(active_session_id="owner"),
        _sync_console_rail_visibility_if_changed=lambda state: None,
        _current_console_rail_state=lambda: None,
        _sync_native_console_chat_ui=held_sync,
        _restore_console_transcript_reading_state=lambda state: None,
        _focus_console_workbench_target=lambda target: None,
        run_worker=lambda coro, **kw: workers.append(coro),
        call_after_refresh=lambda fn, *args: callbacks.append((fn, args)),
    )
    monkeypatch.setattr(terminal, "sync_terminal_center_chrome", lambda screen: None)
    terminal.finish_center_swap(screen)
    for worker in workers:
        await worker
    assert not any(
        fn == screen._restore_console_transcript_reading_state for fn, _ in callbacks
    ), "Held sync restored an empty center"
    assert screen._console_terminal_return_reading == ("owner", state)


def test_priority_voice_escape_yields_to_actual_capturing_viewport():
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    viewport = TerminalViewport(id="console-terminal-viewport")
    screen = SimpleNamespace(
        focused=viewport,
        _console_terminal_open=True,
        _hands_free=SimpleNamespace(console_hands_free_exit_available=lambda: True),
        _console_row_action_menu_open=lambda: False,
    )
    assert ChatScreen.check_action(screen, "exit_console_hands_free", ()) is False
    viewport.input_focused = False
    assert ChatScreen.check_action(screen, "exit_console_hands_free", ()) is True


@pytest.mark.parametrize("bracketed", [False, True])
def test_paste_on_capturing_terminal_reaches_controller(bracketed):
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Terminal.screen_model import TerminalScreenModel

    model = TerminalScreenModel(columns=20, rows=2)
    if bracketed:
        model.feed(b"\x1b[?2004h")
    viewport = TerminalViewport(id="console-terminal-viewport")
    viewport._snapshot = model.snapshot()
    composer = SimpleNamespace(collapsed=True)
    calls = []
    screen = SimpleNamespace(
        focused=viewport,
        _console_terminal_open=True,
        _console_setup_modal_blocking=lambda: False,
        query_one=lambda *args: composer,
        _terminal=SimpleNamespace(
            send_paste=lambda text, **kw: calls.append((text, kw))
        ),
    )
    screen._should_capture_console_input = (
        lambda widget: ChatScreen._should_capture_console_input(screen, widget)
    )
    ChatScreen.on_paste(screen, Paste("paste sentinel"))
    assert calls == [("paste sentinel", {})]  # The controller resolves current mode.


def test_return_waits_for_transcript_publication_not_just_outer_sync():
    from tldw_chatbook.UI.Console_Modules.terminal import (
        restore_returned_reading_after_publication,
    )

    callbacks = []
    transcript = SimpleNamespace(_session_identity=object())
    region = SimpleNamespace(
        is_attached=True,
        reading_session_id=transcript._session_identity,
        _transcript_or_none=lambda: transcript,
    )
    reading = ("owner", object())
    screen = SimpleNamespace(
        _console_terminal_return_reading=reading,
        _console_terminal_open=False,
        _console_chat_store=SimpleNamespace(active_session_id="owner"),
        _console_transcript_region_or_none=lambda: region,
        call_after_refresh=lambda fn: callbacks.append(fn),
    )
    restore_returned_reading_after_publication(screen)
    assert callbacks == [], "Outer sync success is not transcript publication"
    assert screen._console_terminal_return_reading is reading


def test_return_keeps_pending_until_deferred_reveal_applies():
    from tldw_chatbook.UI.Console_Modules.terminal import (
        restore_returned_reading_after_publication,
    )

    callbacks = []
    deferred = []
    transcript = SimpleNamespace(_session_identity="owner")
    region = SimpleNamespace(
        is_attached=True,
        reading_session_id=transcript._session_identity,
        _transcript_or_none=lambda: transcript,
    )
    reading = ("owner", object())
    screen = SimpleNamespace(
        _console_terminal_return_reading=reading,
        _console_terminal_open=False,
        _console_chat_store=SimpleNamespace(active_session_id="owner"),
        _console_transcript_region_or_none=lambda: region,
        call_after_refresh=lambda fn: callbacks.append(fn),
        _restore_console_transcript_reading_state=lambda state, **kw: deferred.append(
            kw
        ),
    )
    restore_returned_reading_after_publication(screen)
    callbacks.pop()()
    assert (
        screen._console_terminal_return_reading is reading
    ), "Reveal has not applied yet"
    region.is_attached = False
    assert not deferred[0]["is_current"]()
    assert screen._console_terminal_return_reading is reading


def test_paste_binds_live_selected_mode_before_viewport_repaint():
    from dataclasses import replace
    from Tests.UI.test_console_terminal_workspace import (
        _Runtime,
        _ProjectionSink,
        _controller,
        _session,
    )
    from tldw_chatbook.Terminal.session_manager import TerminalViewState
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    old = _session("old", "Old")
    live = _session("live", "Live")
    live = replace(live, screen=replace(live.screen, bracketed_paste=True))
    runtime = _Runtime(
        TerminalViewState(selected_session_id="old", sessions=(old, live))
    )
    controller = _controller(runtime, _ProjectionSink())
    assert controller.open_workspace()
    viewport = TerminalViewport(id="console-terminal-viewport")
    viewport._snapshot = old.screen
    runtime.state = replace(runtime.state, selected_session_id="live")
    screen = SimpleNamespace(
        focused=viewport,
        _console_terminal_open=True,
        _terminal=controller,
        _console_setup_modal_blocking=lambda: False,
    )
    ChatScreen.on_paste(screen, Paste("paste after selection"))
    assert runtime.pastes == [("live", "paste after selection", True)]


def test_refused_paste_surfaces_only_the_runtime_safe_reason(monkeypatch):
    from Tests.UI.test_console_terminal_workspace import (
        _Runtime,
        _ProjectionSink,
        _controller,
        _session,
    )
    from tldw_chatbook.Terminal.session_manager import TerminalViewState
    from tldw_chatbook.Terminal.io_actors import InputOfferResult

    runtime = _Runtime(
        TerminalViewState(selected_session_id="one", sessions=(_session("one", "One"),))
    )
    sink = _ProjectionSink()
    controller = _controller(runtime, sink)
    assert controller.open_workspace()
    monkeypatch.setattr(
        runtime,
        "send_paste",
        lambda *args, **kw: InputOfferResult(
            safe_message="Paste refused: unsupported control."
        ),
    )
    assert not controller.send_paste("secret\x1b", bracketed=False)
    assert sink.statuses == ["Paste refused: unsupported control."]


def test_open_terminal_does_not_relabel_a_still_painted_old_conversation():
    from tldw_chatbook.UI.Console_Modules.terminal import capture_center_reading

    screen = SimpleNamespace(
        _console_chat_store=SimpleNamespace(active_session_id="activating-B"),
        _console_terminal_return_reading=None,
    )
    center = SimpleNamespace(
        reading_session_id="painted-A",
        capture_reading_state=lambda: object(),
    )
    capture_center_reading(screen, center)
    assert screen._console_terminal_return_reading is None


@pytest.mark.parametrize("invalidate", ["owner", "transcript"])
def test_deferred_reveal_rechecks_owner_before_position_and_completion(invalidate):
    from tldw_chatbook.UI.Console_Modules.transcript import ConsoleTranscriptRegion

    queued, applied, completed = [], [], []
    current = [True]
    transcript = SimpleNamespace(
        reveal_message=lambda message: True,
        refresh_messages=lambda: None,
        selected_message_id=None,
        max_scroll_y=20,
        call_later=lambda callback: None,
        call_after_refresh=lambda callback: queued.append(callback),
        release_anchor=lambda: None,
        scroll_to=lambda **kw: applied.append(kw),
    )
    mounted = [transcript]
    region = SimpleNamespace(_transcript_or_none=lambda: mounted[0])
    state = SimpleNamespace(selected_message_id="selected", anchored=False, scroll_y=3)
    ConsoleTranscriptRegion.restore_reading_state(
        region,
        state,
        is_current=lambda: current[0],
        on_restored=lambda: completed.append(True),
    )
    assert not applied and not completed
    if invalidate == "owner":
        current[0] = False
    else:
        mounted[0] = object()
    queued.pop()()
    assert not applied and not completed
    current[0] = True
    mounted[0] = transcript
    ConsoleTranscriptRegion.restore_reading_state(
        region,
        state,
        is_current=lambda: current[0],
        on_restored=lambda: completed.append(True),
    )
    queued.pop()()
    assert applied == [{"y": 3, "animate": False}]
    assert completed == [True]


@pytest.mark.parametrize("painted_owner", [None, "owner"])
def test_reopening_terminal_preserves_the_pending_unrestored_position(painted_owner):
    from tldw_chatbook.UI.Console_Modules.terminal import capture_center_reading

    reading = ("owner", object())
    screen = SimpleNamespace(
        _console_chat_store=SimpleNamespace(active_session_id="owner"),
        _console_terminal_return_reading=reading,
    )
    center = SimpleNamespace(
        reading_session_id=painted_owner, capture_reading_state=lambda: object()
    )
    capture_center_reading(screen, center)
    assert screen._console_terminal_return_reading is reading
