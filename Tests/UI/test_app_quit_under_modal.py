"""Ctrl+Q reaches the app's quit flow while a Console modal is open (TASK-33622.10).

ADR-031 makes Ctrl+Q app-global and the footer advertises it on every screen.
``TldwCli`` rebinds ``ctrl+q``, and Textual merges a subclass's BINDINGS per key
by REPLACEMENT, so the rebinding silently dropped the ``priority=True`` that
Textual's own ``ctrl+q`` carries. Non-priority bindings are looked up along
``Screen._modal_binding_chain``, which stops at the first modal screen -- so
any open modal (the Ctrl+K switcher, the Conversation Inspector, the hook
review modal ...) swallowed the key and the app never started quitting.

These drive the real ``TldwCli`` (not a ``ConsoleHarness``: the defect lives in
the app's own bindings) and press the real key. Console reports pending loss so
the quit flow's own "Quit Chatbook?" confirmation appears -- that dialog is the
observable proof the quit flow started, and choosing Stay keeps the app alive
without running the irreversible shutdown in the test process.

The next tests pin the review follow-ups: a modal's own discard guard asks
before quitting -- a dirty ReminderForm, the generated-video choice and a
memory-only profile interview -- and the walk reaches unsaved work on the
destination BENEATH a hook-less modal (Settings' theme editor).

The last group pins the orphaned-prompt hazard Ctrl+Q-under-a-modal exposes.
``Screen.dismiss()`` pops the app's TOP screen, whoever calls it, and
``App.pop_screen`` drops the popped screen's result callback unresolved. So a
covered modal that closes itself from a timer or worker while a quit prompt
sits above it pops that prompt unanswered: before the fix the quit worker
waited on it forever, ``_quit_in_progress`` stayed set, and Ctrl+Q was dead
for the rest of the session. Each case covers a different prompt route: the
app-owned Console prompt, a dirty modal's discard prompt, and a destination's
own prompt beneath a modal. Those covered modals are plain ``ModalScreen``s:
the prompt vanishes, the flow ends as Stay, and the helper finishes the
covered modal's own interrupted close -- left on the stack with its result
already spent, that modal's next close raised ``InvalidStateError`` and
exited the app (review follow-up). A modal on ``SafeModalDismissMixin`` (the
switcher, the video player, most Console modals) cannot pop the prompt at
all: its covered dismiss is refused, which the switcher journey pins.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Chat.console_chat_models import ConsoleLifecycleImpact
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    ConsoleConversationInspector,
)
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

_SETTLE_SECONDS = 15.0


async def _until(pilot, predicate, what: str, timeout: float = _SETTLE_SECONDS):
    """Pump the app until ``predicate()`` holds, or fail naming ``what``."""
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


def _dialogs_titled(app, title: str) -> list[ConfirmationDialog]:
    return [
        screen
        for screen in app.screen_stack
        if isinstance(screen, ConfirmationDialog) and screen.title == title
    ]


def _quit_dialogs(app) -> list[ConfirmationDialog]:
    return _dialogs_titled(app, "Quit Chatbook?")


async def _mounted_console(app, pilot):
    await _until(
        pilot,
        lambda: (
            type(app.screen).__name__ == "ChatScreen"
            and bool(app.screen.query("#console-native-composer"))
        ),
        "the Console composer",
    )
    await pilot.pause(0.2)
    return app.screen


def _arm_unsent_console_work(console, monkeypatch) -> None:
    """Make the app-owned quit confirmation see Console work it would discard.

    Only the loss derivation is replaced (a queued prompt needs a live run to
    queue behind); the dialog, its routing and the quit worker are real.
    """
    controller = console._ensure_console_chat_controller()
    assert controller is console.app.console_runtime.chat_controller
    impact = ConsoleLifecycleImpact(
        revision=7,
        live_run_count=0,
        queued_session_count=1,
        unsent_prompt_count=1,
    )
    monkeypatch.setattr(controller, "lifecycle_impact", lambda **_kwargs: impact)


@pytest.mark.parametrize(
    ("opener", "modal_type"),
    [
        ("ctrl+k", ConsoleSessionSwitcherModal),
        ("ctrl+shift+p", ConsoleConversationInspector),
    ],
    ids=["session-switcher", "conversation-inspector"],
)
async def test_ctrl_q_under_a_console_modal_starts_the_quit_confirmation(
    monkeypatch, opener, modal_type
):
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        _arm_unsent_console_work(console, monkeypatch)

        await pilot.press(opener)
        await _until(
            pilot,
            lambda: isinstance(app.screen, modal_type),
            f"{modal_type.__name__} to open from {opener}",
        )
        modal = app.screen
        await pilot.pause(0.2)
        assert app._quit_in_progress is False

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_quit_dialogs(app)),
            f"Ctrl+Q under {modal_type.__name__} to start the quit flow",
            timeout=5.0,
        )
        assert app._quit_in_progress is True
        assert app.screen is _quit_dialogs(app)[0]
        assert modal in app.screen_stack, "the quit prompt replaced the modal"

        # Ctrl+Q again while the quit prompt is up: one prompt, no crash.
        await pilot.press("ctrl+q")
        await pilot.pause(0.3)
        assert len(_quit_dialogs(app)) == 1
        assert app.is_running

        # Stay: the prompt closes, the modal the user was in is back on top,
        # and the next Ctrl+Q is allowed to ask again.
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is modal, "Stay to restore the modal")
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Stay",
        )
        assert app.is_running
        assert not _quit_dialogs(app)


async def test_ctrl_q_under_a_dirty_form_asks_before_discarding_its_edits(
    monkeypatch,
):
    """Review finding (TASK-33622.10): a dirty modal's own work is guarded too.

    ReminderForm asks "Discard changes?" before Escape drops typed edits.
    Before the fix Ctrl+Q was swallowed under it, which protected that work
    by accident; once Ctrl+Q reached the quit flow it quit straight past the
    guard. The quit flow now asks the form first: Keep editing keeps the app,
    the form and the typed text; Discard and quit lets the quit proceed.
    """
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups: list[bool] = []

    async def _record_cleanup() -> None:
        # Stands in for the irreversible shutdown, as the other quit tests do.
        cleanups.append(True)

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_cleanup)
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        form = ReminderForm()
        await app.push_screen(form)
        await _until(
            pilot,
            lambda: app.screen is form and form._ready,
            "the reminder form to finish its mount-time prefill",
        )
        title = form.query_one("#reminder-title", Input)
        title.focus()
        await pilot.press(*"Pay")
        await _until(pilot, lambda: form._dirty, "typing to mark the form dirty")

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, "Discard changes and quit?")),
            "Ctrl+Q under a dirty form to ask before discarding its edits",
            timeout=5.0,
        )
        assert app.screen is _dialogs_titled(app, "Discard changes and quit?")[0]
        assert form in app.screen_stack
        assert cleanups == []

        # Keep editing: nothing quits and nothing is lost.
        await pilot.press("escape")
        await _until(
            pilot, lambda: app.screen is form, "Keep editing to restore the form"
        )
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Keep editing",
        )
        assert cleanups == []
        assert app._shutting_down is False
        assert app.is_running
        assert title.value == "Pay"

        # Discard and quit: the user chose to drop the edits, so the quit runs.
        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, "Discard changes and quit?")),
            "the second Ctrl+Q to ask again",
            timeout=5.0,
        )
        await pilot.click("#confirm-button")
        await _until(
            pilot,
            lambda: cleanups == [True],
            "Discard and quit to reach the approved shutdown",
            timeout=5.0,
        )


async def _ask_stay_then_quit(app, pilot, modal, title: str, cleanups, kept) -> None:
    """Ctrl+Q over ``modal`` asks ``title``; Stay keeps it; the confirm quits.

    Args:
        app: The running app.
        pilot: Its pilot.
        modal: The open modal whose own close guard Ctrl+Q must honour.
        title: The quit prompt that modal must raise.
        cleanups: Filled by the stand-in for the irreversible shutdown.
        kept: Asserts what the modal holds is intact after Stay.
    """
    await pilot.press("ctrl+q")
    await _until(
        pilot,
        lambda: bool(_dialogs_titled(app, title)) or bool(cleanups),
        f"Ctrl+Q over {type(modal).__name__} to ask {title!r}",
        timeout=5.0,
    )
    assert cleanups == [], f"quit straight past {type(modal).__name__}"
    assert app.screen is _dialogs_titled(app, title)[0]
    assert modal in app.screen_stack

    # Stay: nothing quits and nothing is lost.
    await pilot.press("escape")
    await _until(pilot, lambda: app.screen is modal, "Stay to restore the modal")
    await _until(
        pilot,
        lambda: app._quit_in_progress is False,
        "the quit guard to clear after Stay",
    )
    assert cleanups == []
    assert app._shutting_down is False
    assert app.is_running
    kept()

    # The confirm: the user chose to lose it, so the quit runs.
    await pilot.press("ctrl+q")
    await _until(
        pilot,
        lambda: bool(_dialogs_titled(app, title)),
        "the second Ctrl+Q to ask again",
        timeout=5.0,
    )
    await pilot.click("#confirm-button")
    await _until(
        pilot,
        lambda: cleanups == [True],
        "the confirm to reach the approved shutdown",
        timeout=5.0,
    )


async def test_ctrl_q_over_the_generated_video_choice_asks_before_discarding_it(
    monkeypatch,
):
    """Review follow-up (TASK-33622.10): the video choice guards its close.

    Escape on ConsoleVideoCapacityModal asks "Discard generated video?" -- the
    result "cannot be recovered". Once Ctrl+Q reached the quit flow under
    modals, it quit straight past that guard. Now the choice asks first.
    """
    from tldw_chatbook.Widgets.Console.console_video_capacity_modal import (
        ConsoleVideoCapacityModal,
    )

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups: list[bool] = []

    async def _record_cleanup() -> None:
        cleanups.append(True)

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_cleanup)
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        results: list[object] = []
        modal = ConsoleVideoCapacityModal(
            reason="over_capacity",
            size_bytes=300 * 1024 * 1024,
            max_bytes=256 * 1024 * 1024,
        )
        await app.push_screen(modal, callback=results.append)
        await _until(pilot, lambda: app.screen is modal, "the video choice")

        def _kept() -> None:
            assert results == [], "Stay must not decide the video's fate"

        await _ask_stay_then_quit(
            app, pilot, modal, "Discard generated video and quit?", cleanups, _kept
        )


async def test_ctrl_q_over_a_memory_only_interview_asks_before_discarding_it(
    monkeypatch,
):
    """Review follow-up (TASK-33622.10): a memory-only interview is lost on quit.

    Its own Leave prompt says the interview "exists only in memory" and
    offers only Continue or Discard. Ctrl+Q now asks before discarding it;
    Continue interview keeps the screen, the typed answer and the session.
    """
    from dataclasses import replace

    from Tests.UI.test_profile_interview_screen import _Coordinator, _session_base
    from tldw_chatbook.UI.Screens.profile_interview_screen import (
        ProfileInterviewScreen,
    )

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups: list[bool] = []

    async def _record_cleanup() -> None:
        cleanups.append(True)

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_cleanup)
    coordinator = _Coordinator(replace(_session_base(), draft_is_memory_only=True))
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        results: list[object] = []
        screen = ProfileInterviewScreen(
            coordinator, kind="personal", scope_id="scope-global", mode="fixed"
        )
        await app.push_screen(screen, callback=results.append)
        await _until(
            pilot,
            lambda: (
                app.screen is screen
                and screen._session is not None
                and not screen._busy
            ),
            "the interview to load",
        )
        answer = screen.query_one("#profile-interview-answer", Input)
        answer.focus()
        await pilot.press(*"Kim")
        await _until(pilot, lambda: answer.value == "Kim", "the typed answer")

        def _kept() -> None:
            assert answer.value == "Kim"
            assert results == []
            assert not [call for call in coordinator.calls if call[0] == "discard"]

        await _ask_stay_then_quit(
            app, pilot, screen, "Discard interview and quit?", cleanups, _kept
        )


class _HooklessModal(ModalScreen[None]):
    """A modal with no quit hooks of its own, like most pickers."""

    def compose(self):
        yield Static("covering modal")


async def test_ctrl_q_under_a_modal_still_asks_the_destination_beneath_it(
    monkeypatch,
):
    """TASK-33622.10: the quit flow walks past the modal to the destination.

    The unsaved work can live on the screen BENEATH the modal -- here
    Settings' theme editor. Ctrl+Q under a hook-less modal must still raise
    Settings' own Save / Discard / Stay prompt, and Stay keeps the app, the
    modal and the edits.
    """
    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    app = _build_test_app(configured_default="settings")
    cleanups: list[bool] = []

    async def _record_cleanup() -> None:
        cleanups.append(True)

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_cleanup)
    async with app.run_test(size=(190, 55)) as pilot:
        _settings, editor = await _modified_theme_editor(app, pilot)

        modal = _HooklessModal()
        await app.push_screen(modal)
        await _until(pilot, lambda: app.screen is modal, "the covering modal")

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: isinstance(app.screen, ThemeLeaveModal),
            "Ctrl+Q under the modal to raise Settings' own unsaved-theme prompt",
            timeout=5.0,
        )
        assert modal in app.screen_stack

        await pilot.click("#settings-theme-leave-stay")
        await _until(pilot, lambda: app.screen is modal, "Stay to restore the modal")
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Stay",
        )
        assert cleanups == []
        assert app.is_running
        assert editor.is_modified


# --- The orphaned-prompt hazard (TASK-33622.10) ------------------------------


def _close_from_its_own_timer(modal) -> None:
    """Make ``modal`` call a bare ``self.dismiss()`` from its own timer.

    This is what an async poll or a worker's completion callback does when
    it decides the modal is finished -- without knowing another screen now
    covers it. The return value (Textual's ``AwaitComplete``) is deliberately
    not awaited, exactly as such callbacks do.
    """

    def _fire() -> None:
        modal.dismiss(None)

    modal.set_timer(0.05, _fire)


def _notified(app, message: str) -> bool:
    return any(note.message == message for note in app._notifications)


def _record_cleanup_then_exit(app, cleanups: list[bool]):
    """Stand in for the irreversible shutdown, then exit as the real one does."""

    async def _cleanup() -> None:
        cleanups.append(True)
        app.exit()

    return _cleanup


async def _until_exited(app, cleanups: list[bool], what: str) -> None:
    # Plain sleeps: once the app starts exiting, Pilot's screen waits raise.
    try:
        async with asyncio.timeout(5.0):
            while not cleanups:
                await asyncio.sleep(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


async def _assert_quit_ended_as_stay(app, pilot, prompt, covered) -> None:
    """The prompt left unanswered; the flow must end as Stay, never hang.

    And the covered modal, whose own dismiss went to the prompt, is closed as
    it asked -- not left behind as a zombie whose next close crashes the app.
    """
    await _until(
        pilot,
        lambda: prompt not in app.screen_stack,
        "the covered modal's dismiss to pop the quit prompt",
        timeout=5.0,
    )
    await _until(
        pilot,
        lambda: app._quit_in_progress is False,
        "the quit flow to end as Stay once its prompt vanished unanswered",
        timeout=5.0,
    )
    assert app.is_running
    assert app._shutting_down is False
    # App.notify posts a Notify message; the toast lands a moment later.
    await _until(
        pilot,
        lambda: _notified(app, "Quit cancelled."),
        'the "Quit cancelled." toast',
        timeout=5.0,
    )
    # The covered modal's dismiss popped the prompt instead of itself; its
    # close is finished, so nothing is left to re-dismiss.
    await _until(
        pilot,
        lambda: covered not in app.screen_stack,
        "the covered modal's interrupted close to be finished",
        timeout=5.0,
    )
    await pilot.pause()
    assert app._exception is None
    assert app.is_running


async def test_ctrl_q_survives_a_covered_modal_popping_the_console_quit_prompt(
    monkeypatch,
):
    """The app-owned "Quit Chatbook?" prompt, over a plain modal."""
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups: list[bool] = []
    monkeypatch.setattr(
        app, "_run_approved_quit_cleanup", _record_cleanup_then_exit(app, cleanups)
    )
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        _arm_unsent_console_work(console, monkeypatch)
        modal = _HooklessModal()
        await app.push_screen(modal)
        await _until(pilot, lambda: app.screen is modal, "the covering modal")

        await pilot.press("ctrl+q")
        await _until(
            pilot, lambda: bool(_quit_dialogs(app)), "the quit prompt", timeout=5.0
        )
        prompt = _quit_dialogs(app)[0]
        assert app.screen is prompt

        _close_from_its_own_timer(modal)
        await _assert_quit_ended_as_stay(app, pilot, prompt, modal)
        assert app.screen is console

        # Ctrl+Q is still live: it asks again, and Quit quits.
        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_quit_dialogs(app)),
            "a second Ctrl+Q to ask again",
            timeout=5.0,
        )
        assert cleanups == []
        await pilot.click("#confirm-button")
        await _until_exited(app, cleanups, "Quit to reach the approved shutdown")
    assert cleanups == [True]
    assert app.return_code == 0


async def test_a_covered_switcher_cannot_pop_the_console_quit_prompt(monkeypatch):
    """The switcher is a ``SafeModalDismissMixin`` modal: its covered close is refused.

    So the prompt stays up for the user to answer, nothing says "Quit
    cancelled.", Stay brings the switcher back intact, and it still closes.
    """
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups: list[bool] = []
    monkeypatch.setattr(
        app, "_run_approved_quit_cleanup", _record_cleanup_then_exit(app, cleanups)
    )
    async with app.run_test(size=(140, 44)) as pilot:
        console = await _mounted_console(app, pilot)
        _arm_unsent_console_work(console, monkeypatch)
        await pilot.press("ctrl+k")
        await _until(
            pilot,
            lambda: isinstance(app.screen, ConsoleSessionSwitcherModal),
            "the session switcher",
        )
        switcher = app.screen
        await pilot.pause(0.2)

        await pilot.press("ctrl+q")
        await _until(
            pilot, lambda: bool(_quit_dialogs(app)), "the quit prompt", timeout=5.0
        )
        prompt = _quit_dialogs(app)[0]

        _close_from_its_own_timer(switcher)
        await pilot.pause(0.5)  # the timer, then past the watch interval + grace
        assert app.screen is prompt, "the covered switcher popped the quit prompt"
        assert app._quit_in_progress is True
        assert not _notified(app, "Quit cancelled.")

        await pilot.click("#cancel-button")  # Stay
        await _until(pilot, lambda: app.screen is switcher, "Stay to restore it")
        await _until(
            pilot, lambda: app._quit_in_progress is False, "the guard to clear"
        )
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is console, "Escape to close it")
        assert app._exception is None

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_quit_dialogs(app)),
            "a second Ctrl+Q to ask again",
            timeout=5.0,
        )
        await pilot.click("#confirm-button")
        await _until_exited(app, cleanups, "Quit to reach the approved shutdown")
    assert cleanups == [True]
    assert app.return_code == 0


async def test_ctrl_q_survives_a_dirty_form_popping_its_own_discard_prompt(
    monkeypatch,
):
    """A dirty modal's "Discard changes and quit?" prompt, over that modal."""
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    cleanups: list[bool] = []
    monkeypatch.setattr(
        app, "_run_approved_quit_cleanup", _record_cleanup_then_exit(app, cleanups)
    )
    async with app.run_test(size=(140, 44)) as pilot:
        await _mounted_console(app, pilot)
        form = ReminderForm()
        await app.push_screen(form)
        await _until(
            pilot,
            lambda: app.screen is form and form._ready,
            "the reminder form to finish its mount-time prefill",
        )
        title = form.query_one("#reminder-title", Input)
        title.focus()
        await pilot.press(*"Pay")
        await _until(pilot, lambda: form._dirty, "typing to mark the form dirty")

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, "Discard changes and quit?")),
            "the discard-and-quit prompt",
            timeout=5.0,
        )
        prompt = _dialogs_titled(app, "Discard changes and quit?")[0]
        assert app.screen is prompt

        _close_from_its_own_timer(form)
        await _assert_quit_ended_as_stay(app, pilot, prompt, form)
        # The form closed itself, as it asked to, so its edits are gone with
        # it and nothing is left to ask about: Ctrl+Q is live and just quits.
        assert type(app.screen).__name__ == "ChatScreen"
        assert cleanups == []

        await pilot.press("ctrl+q")
        await _until_exited(app, cleanups, "a second Ctrl+Q to reach the shutdown")
        assert not _dialogs_titled(app, "Discard changes and quit?")
    assert cleanups == [True]
    assert app.return_code == 0


async def _modified_theme_editor(app, pilot):
    """Open Settings ▸ Theme, clone a theme and make one real edit."""
    from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

    await _until(
        pilot,
        lambda: type(app.screen).__name__ == "SettingsScreen",
        "the Settings screen",
    )
    settings = app.screen
    settings._select_category(SettingsCategoryId.THEME.value)
    await _until(
        pilot,
        lambda: bool(settings.query("#settings-theme-list")),
        "the theme picker",
    )
    settings.query_one("#settings-theme-list").focus()
    await pilot.press("c")
    await _until(
        pilot,
        lambda: bool(settings.query("#settings-theme-editor")),
        "Clone to open the theme editor",
    )
    editor = settings.query_one("#settings-theme-editor")
    editor.query_one("#settings-theme-name", Input).value = "quit_under_modal"
    await _until(pilot, lambda: editor.is_modified, "a real theme edit")
    return settings, editor


async def test_ctrl_q_survives_a_covered_modal_popping_settings_theme_prompt(
    monkeypatch,
):
    """A destination's own prompt (Settings' unsaved theme) beneath a modal."""
    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    app = _build_test_app(configured_default="settings")
    cleanups: list[bool] = []
    monkeypatch.setattr(
        app, "_run_approved_quit_cleanup", _record_cleanup_then_exit(app, cleanups)
    )
    async with app.run_test(size=(190, 55)) as pilot:
        settings, editor = await _modified_theme_editor(app, pilot)
        modal = _HooklessModal()
        await app.push_screen(modal)
        await _until(pilot, lambda: app.screen is modal, "the covering modal")

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: isinstance(app.screen, ThemeLeaveModal),
            "Settings' unsaved-theme prompt",
            timeout=5.0,
        )
        prompt = app.screen

        _close_from_its_own_timer(modal)
        await _assert_quit_ended_as_stay(app, pilot, prompt, modal)
        assert app.screen is settings
        assert editor.is_modified
        # Settings' one-prompt-at-a-time latch must not stay stuck either.
        assert settings._theme_leave_in_progress is False

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: isinstance(app.screen, ThemeLeaveModal),
            "a second Ctrl+Q to ask again",
            timeout=5.0,
        )
        assert cleanups == []
        await pilot.click("#settings-theme-leave-discard")
        await _until_exited(app, cleanups, "Discard to reach the approved shutdown")
    assert cleanups == [True]
    assert app.return_code == 0
