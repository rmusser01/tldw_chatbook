"""TASK-340: Enter must snapshot the draft synchronously at the keypress.

The Console Enter branch posts a ``Button.Pressed`` message and the send
handler used to read ``composer.draft_text()`` only when that message was
finally processed — printable keys handled in between mutated the draft and
were folded into the sent message (UX review finding
j6-send-captures-late-keystrokes). These tests deliver Enter synchronously
via ``ChatScreen.on_key`` and interleave typing before the message pump runs,
which is exactly the interleave a fast typist produces.
"""

import pytest
from textual.events import Key
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import (
    BlockedGateway,
    _build_console_send_test_app,
    _persist_console_provider_config,
    _select_llamacpp_console,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

DUMMY_OPENAI_API_KEY = "DUMMY_OPENAI_API_KEY"


async def _wait_for_text(console, pilot, needle: str, tries: int = 40) -> None:
    for _ in range(tries):
        if needle in _visible_text(console):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {needle!r}")


def _ready_openai_app(monkeypatch, reply: str):
    app = _build_console_send_test_app()
    _persist_console_provider_config(
        app,
        provider="openai",
        model="gpt-4.1",
        provider_settings={"api_key": DUMMY_OPENAI_API_KEY},
    )

    def fake_chat_api_call(**_kwargs):
        return reply

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    return app


def _press_enter_synchronously(console) -> None:
    """Deliver Enter to the screen key handler without pumping messages."""
    console.on_key(Key(key="enter", character="\r"))


@pytest.mark.asyncio
async def test_console_enter_snapshots_draft_before_late_keystrokes(monkeypatch):
    app = _ready_openai_app(monkeypatch, "snapshot reply")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("line one")

        _press_enter_synchronously(console)
        # Keystrokes arriving before the Button.Pressed message is processed:
        composer.insert_text("line two")

        await _wait_for_text(console, pilot, "snapshot reply")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        user_messages = [
            m for m in messages if m.role is ConsoleMessageRole.USER
        ]
        assert user_messages[-1].content == "line one"
        # The late keystrokes belong to the NEXT draft — and the
        # accepted-submit clear must not eat them either.
        assert composer.draft_text() == "line two"


@pytest.mark.asyncio
async def test_mouse_send_completion_preserves_text_typed_after_acceptance(monkeypatch):
    """A late mouse-send cleanup must not erase the user's next draft."""

    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.ensure_session()
        store.switch_session(session.id)
        console._session._sync_console_session_draft()
        composer.load_draft("original mouse draft")
        store.set_session_draft(session.id, composer.draft_text())

        async def accepted_then_user_types(draft, *, session_id=None):
            assert draft == "original mouse draft"
            assert session_id == session.id
            assert controller.on_submission_accepted is not None
            controller.on_submission_accepted()
            composer.insert_text("newer draft")
            store.set_session_draft(session.id, composer.draft_text())
            return ConsoleSubmitResult(accepted=True, should_clear_draft=True)

        monkeypatch.setattr(controller, "run_prompt_chain", accepted_then_user_types)

        await console._submit_console_native_draft(
            "original mouse draft", session.id
        )

        assert composer.draft_text() == "newer draft"
        assert store.session_draft(session.id) == "newer draft"


@pytest.mark.asyncio
async def test_console_blocked_send_restores_snapshot_before_late_keystrokes():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = BlockedGateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("keep me")

        _press_enter_synchronously(console)
        composer.insert_text("!")
        await _wait_for_text(console, pilot, "Provider blocked")
        await pilot.pause()

        # Blocked send restores the snapshot ahead of the late typing —
        # original text first, later keystrokes appended after it.
        assert composer.draft_text() == "keep me!"


@pytest.mark.asyncio
async def test_console_unknown_command_hint_restores_draft(monkeypatch):
    app = _ready_openai_app(monkeypatch, "never sent")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("/nosuchcommand")

        _press_enter_synchronously(console)
        await pilot.pause()
        await pilot.pause()

        # The unknown-command hint path must put the draft back so the
        # armed second-Enter flow still compares against the same text.
        assert composer.draft_text() == "/nosuchcommand"


@pytest.mark.asyncio
async def test_console_blocked_send_restore_preserves_paste_segments():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = BlockedGateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.insert_text_as_paste("pasted payload " * 20)
        composer.insert_text(" tail")
        # Unfurl the token fully (collapsed -> confirm -> expanded); Enter
        # only sends once no token is awaiting the unfurl flow, and expanded
        # segments retain their paste provenance.
        assert composer.activate_focused_paste_token()
        assert composer.activate_focused_paste_token()
        assert not composer.activate_focused_paste_token()
        assert composer.has_paste_segments()
        expected = composer.draft_text()

        _press_enter_synchronously(console)
        await _wait_for_text(console, pilot, "Provider blocked")
        await pilot.pause()

        assert composer.draft_text() == expected
        assert composer.has_paste_segments()


@pytest.mark.asyncio
async def test_console_double_enter_sends_once_and_loses_nothing(monkeypatch):
    """A second Enter before the first Pressed handler runs must not
    overwrite the pending stash with None (that ate the message)."""
    app = _ready_openai_app(monkeypatch, "double reply")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("line one")

        _press_enter_synchronously(console)
        _press_enter_synchronously(console)

        await _wait_for_text(console, pilot, "double reply")

        store = console._ensure_console_chat_store()
        messages = store.messages_for_session(store.active_session_id)
        user_messages = [m for m in messages if m.role is ConsoleMessageRole.USER]
        assert [m.content for m in user_messages] == ["line one"]
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_submit_exception_restores_draft_and_keeps_app_alive(
    monkeypatch,
):
    """If submit_draft raises, the keypress-cleared draft must come back."""
    app = _ready_openai_app(monkeypatch, "never used")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        controller = console._ensure_console_chat_controller()

        async def exploding_submit(draft):
            raise RuntimeError("provider imploded")

        monkeypatch.setattr(controller, "submit_draft", exploding_submit)
        composer.load_draft("precious draft")

        _press_enter_synchronously(console)
        for _ in range(10):
            await pilot.pause(0.05)

        assert composer.draft_text() == "precious draft"
        # App survived the worker exception (queries still work).
        assert console.query_one("#console-native-composer", ConsoleComposerBar)


# ---------------------------------------------------------------------------
# TASK-4 (D2 fix wave): the swallowed send -- resolve the session at
# dispatch, no silent refusals, guard the no-op press.
#
# Fresh-profile arrangement below: no active session before the send, the
# gap the historical "" sentinel exploited. Additive beside the six tests
# above, which all assume an active session already exists.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_fresh_profile_first_send_resolves_real_session_not_sentinel(
    monkeypatch,
):
    """D2 regression guard: a fresh profile's first send (no active session
    at the moment Enter is pressed) must resolve a REAL session id at
    dispatch time -- never the historical `""` sentinel -- so the stash
    map, the worker group, and the double-send gate all key on the same,
    resolvable id instead of silently starting a separate "no session"
    bucket."""
    app = _ready_openai_app(monkeypatch, "fresh reply")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        store = console._ensure_console_chat_store()
        # Fresh-profile arrangement: no active session at the moment Enter
        # is pressed (whatever the mount-time sync worker created so far is
        # deliberately orphaned here -- the point is the STORE STATE at
        # Enter, not how a real user would reach it).
        store.active_session_id = None

        groups: list[str] = []
        real_run_worker = console.run_worker

        def spying_run_worker(work, **kwargs):
            groups.append(kwargs.get("group", ""))
            return real_run_worker(work, **kwargs)

        monkeypatch.setattr(console, "run_worker", spying_run_worker)

        composer.load_draft("hello fresh")
        _press_enter_synchronously(console)
        await _wait_for_text(console, pilot, "fresh reply")

        assert store.active_session_id is not None
        # Only the console-run worker's own group matters here -- an
        # unrelated periodic "console-sync" worker also fires during the
        # waits above and is not part of what this test is guarding.
        console_run_groups = [g for g in groups if g.startswith("console-run-")]
        assert console_run_groups == [f"console-run-{store.active_session_id}"]
        assert "" not in console._console_inflight_send_stashes
        messages = store.messages_for_session(store.active_session_id)
        user_messages = [m for m in messages if m.role is ConsoleMessageRole.USER]
        assert user_messages and user_messages[-1].content == "hello fresh"


@pytest.mark.asyncio
async def test_console_active_run_rejection_appends_visible_system_row():
    """The controller's defense-in-depth double-send guard
    (`_active_run_rejection`) used to reject a send with no transcript row
    and no toast -- silent from the user's point of view (the SCREEN's own
    earlier gate already passed, or this dispatch wouldn't exist at all).
    It must now leave a visible SYSTEM row, exactly like `_block` does for
    every other pre-echo gate in `submit_draft`."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.ensure_session()
        # Simulate the race `_active_run_rejection` defends against: by the
        # time the dispatched worker's `submit_draft` actually runs, another
        # send already put this session's run state into STREAMING.
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "already streaming"),
            session_id=session.id,
        )

        await console._submit_console_native_draft("hello", session.id)

        messages = store.messages_for_session(session.id)
        system_messages = [m for m in messages if m.role is ConsoleMessageRole.SYSTEM]
        assert any(
            "already running in this tab" in m.content for m in system_messages
        ), system_messages


@pytest.mark.asyncio
async def test_console_session_closed_mid_dispatch_notifies_instead_of_silent_swallow():
    """`_session_closed_result` fires when the dispatched session was closed
    during the gap between dispatch and the worker actually running -- it is
    `accepted=True` (so the composer-restore branch never fires) and the
    owning session no longer exists to hold a SYSTEM row. Before this fix,
    nothing told the user their message never went anywhere; now a toast
    must, and (fix-round-2 I2/M2) it must be the INFORMATIVE copy -- not the
    generic "Session closed." every other call site of this method uses --
    since this is the one case where the user's message specifically never
    sent."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.ensure_session()
        closed_id = session.id
        controller.close_session(closed_id)
        assert all(s.id != closed_id for s in store.sessions())

        notices: list[tuple[str, str]] = []
        console.app_instance.notify = lambda message, **kwargs: notices.append(
            (str(message), kwargs.get("severity", ""))
        )

        await console._submit_console_native_draft("hello", closed_id)

        assert any(
            "before your message could send" in note for note, _sev in notices
        ), notices


@pytest.mark.asyncio
async def test_console_enter_no_op_press_restores_draft_and_unblocks_next_send(
    monkeypatch,
):
    """Textual 8.2.7's `Button.press()` returns immediately -- without
    posting `Button.Pressed` -- when the button is `disabled` or not
    `display`ed. Before this fix, the Enter handler stashed-and-cleared the
    draft unconditionally, so a no-op press left the stash set with an
    empty composer AND the duplicate-guard just above permanently swallowed
    every subsequent Enter (the pending-stash slot never went back to
    `None` on its own)."""
    app = _ready_openai_app(monkeypatch, "after reenable")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("no-op test")

        send_button = console.query_one("#console-send-message", Button)
        # `.disabled` is composer-managed (`_sync_current_action_state`
        # unconditionally re-derives it from draft/readiness state on every
        # `clear_draft()`, including the one `stash_draft_for_send()` does
        # internally) -- setting it directly here would just be clobbered
        # before `Enter`'s own `.press()` call ever sees it. `display`,
        # in contrast, is untouched by that sync and is exactly what goes
        # `False` while the button is being pruned (the brief's own named
        # real-world trigger), so it is the realistic way to arrange a
        # genuine no-op `.press()`.
        send_button.styles.display = "none"

        _press_enter_synchronously(console)
        await pilot.pause()

        # Not silently swallowed: the draft is back in the composer and the
        # pending-stash slot is clear (the duplicate guard cannot latch).
        assert composer.draft_text() == "no-op test"
        assert console._console_pending_send_stash is None

        send_button.styles.display = "block"
        _press_enter_synchronously(console)
        await _wait_for_text(console, pilot, "after reenable")
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_send_watchdog_recovers_stash_the_pressed_handler_never_consumed():
    """Fix-round-2 (I3): the no-op-press check only catches the case where
    the button is ALREADY disabled/hidden at the instant `on_key` reads it.
    `.press()` itself just POSTS `Button.Pressed` for the message pump to
    deliver later -- if a prune begins in the gap between that post and
    delivery, the message is dropped and nothing ever consumes
    `_console_pending_send_stash`, latching the duplicate-send guard shut
    forever. The watchdog scheduled right after `.press()` is the backstop:
    if the stash is still exactly the object it was scheduled for once the
    window passes, nothing consumed it, so it must be recovered."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        composer.load_draft("watchdog test")
        stash = composer.stash_draft_for_send()
        console._console_pending_send_stash = stash
        assert composer.draft_text() == ""

        # Simulate the Pressed handler never running at all (the message
        # was dropped by a prune racing delivery) by invoking the watchdog
        # directly -- exactly how `on_key`'s Enter branch schedules it via
        # `set_timer`, just without waiting out the real delay.
        console._recover_stuck_console_send_stash(stash)

        assert composer.draft_text() == "watchdog test"
        assert console._console_pending_send_stash is None


@pytest.mark.asyncio
async def test_console_send_watchdog_is_a_noop_once_the_stash_is_consumed():
    """The common, non-buggy case: the Pressed handler already consumed the
    stash (cleared the slot back to `None`) before the watchdog fires -- it
    must not resurrect a stale draft into a composer that has since moved
    on."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        composer.load_draft("already sent")
        stash = composer.stash_draft_for_send()
        console._console_pending_send_stash = stash
        # The Pressed handler runs first and consumes the slot (the normal,
        # non-buggy path) -- simulate that here, ahead of the watchdog.
        console._console_pending_send_stash = None

        console._recover_stuck_console_send_stash(stash)

        # Nothing resurrected: the composer stays exactly as the normal
        # accept/refuse path already left it.
        assert composer.draft_text() == ""
