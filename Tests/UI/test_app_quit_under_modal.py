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

The next group pins the orphaned-prompt hazard Ctrl+Q-under-a-modal exposes.
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

The final group pins ruling C-N1 (TASK-33007): a Settings draft kept after
leaving Settings is asked about from any screen, and only while Settings is
not open to ask for itself; Keep editing under a modal stays in it.
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


# --- A Settings draft kept after leaving Settings (TASK-33007, ruling C-N1) --

_SETTINGS_QUIT_TITLE = "Quit and discard unsaved Settings changes?"


async def _settings_with_a_kept_draft(app, pilot):
    """Stage a Providers & Models change, then leave Settings for Home.

    Leaving keeps the draft in ``app.screen_state_store`` (the rail's * shows
    it on return); Home is not Settings, so nothing on screen holds it.
    """
    from tldw_chatbook.UI.Screens.settings_config_models import (
        SettingsCategoryId,
        SettingsDraft,
    )

    await _until(
        pilot,
        lambda: type(app.screen).__name__ == "SettingsScreen",
        "the Settings screen",
    )
    draft = SettingsDraft(category=SettingsCategoryId.PROVIDERS_MODELS)
    draft.set_value("provider", "openai", "anthropic")
    app.screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS] = draft
    await _go_home(app, pilot)


async def _go_home(app, pilot) -> None:
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    app.post_message(NavigateToScreen("home"))
    await _until(
        pilot, lambda: type(app.screen).__name__ == "HomeScreen", "the Home screen"
    )


def _kept_provider_draft(app):
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    return app.screen._settings_drafts.get(SettingsCategoryId.PROVIDERS_MODELS)


async def test_ctrl_q_from_another_screen_asks_about_a_kept_settings_draft(
    monkeypatch,
):
    """Ruling C-N1 (b): Ctrl+Q from Home quit at once and dropped a Settings
    draft kept from an earlier visit. It now asks as a quit from Settings
    does: Keep editing returns to Settings with the draft, Discard and quit
    quits."""
    app = _build_test_app(configured_default="settings")
    cleanups: list[bool] = []
    monkeypatch.setattr(
        app, "_run_approved_quit_cleanup", _record_cleanup_then_exit(app, cleanups)
    )
    async with app.run_test(size=(190, 55)) as pilot:
        await _settings_with_a_kept_draft(app, pilot)

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, _SETTINGS_QUIT_TITLE)) or bool(cleanups),
            "Ctrl+Q on Home to ask about the kept Settings draft",
            timeout=5.0,
        )
        assert cleanups == [], "quit straight past the kept Settings draft"
        assert "Providers & Models" in app.screen.message

        # Keep editing: back in Settings, the draft still staged.
        await pilot.press("escape")
        await _until(
            pilot,
            lambda: type(app.screen).__name__ == "SettingsScreen",
            "Keep editing to return to Settings",
        )
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Keep editing",
        )
        kept = _kept_provider_draft(app)
        assert kept is not None and kept.is_dirty
        assert cleanups == [] and app.is_running

        # Discard and quit from another screen: the quit runs.
        await _go_home(app, pilot)
        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, _SETTINGS_QUIT_TITLE)),
            "the second Ctrl+Q to ask again",
            timeout=5.0,
        )
        await pilot.click("#confirm-button")
        await _until_exited(app, cleanups, "Discard and quit to reach the shutdown")
    assert cleanups == [True]


async def test_ctrl_q_on_settings_never_asks_about_its_own_stale_snapshot(
    monkeypatch,
):
    """Back on Settings the store still holds the snapshot it left with; once
    the draft is gone (saved or discarded) Ctrl+Q must not ask about it."""
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    app = _build_test_app(configured_default="settings")
    cleanups: list[bool] = []
    monkeypatch.setattr(
        app, "_run_approved_quit_cleanup", _record_cleanup_then_exit(app, cleanups)
    )
    async with app.run_test(size=(190, 55)) as pilot:
        await _settings_with_a_kept_draft(app, pilot)
        app.post_message(NavigateToScreen("settings"))
        await _until(
            pilot,
            lambda: (
                type(app.screen).__name__ == "SettingsScreen"
                and _kept_provider_draft(app) is not None
            ),
            "Settings with the kept draft",
        )
        app.screen._settings_drafts.clear()  # as a Save leaves it

        await pilot.press("ctrl+q")
        # A prompt would hold the quit, so reaching the shutdown proves none.
        await _until_exited(app, cleanups, "Ctrl+Q to quit without asking")
    assert cleanups == [True]


async def test_keep_editing_on_a_kept_draft_under_a_modal_keeps_the_modal(
    monkeypatch,
):
    """Capture fix 3 round 1, finding 1: Keep editing on the kept-draft
    prompt went back to Settings, and navigating dismisses every modal with
    None, past its own close guard. Under a modal the user stays in it."""
    app = _build_test_app(configured_default="settings")
    cleanups: list[bool] = []
    monkeypatch.setattr(
        app, "_run_approved_quit_cleanup", _record_cleanup_then_exit(app, cleanups)
    )
    async with app.run_test(size=(190, 55)) as pilot:
        await _settings_with_a_kept_draft(app, pilot)
        home = app.screen
        results: list[object] = []
        modal = _HooklessModal()
        await app.push_screen(modal, callback=results.append)
        await _until(pilot, lambda: app.screen is modal, "the covering modal")

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, _SETTINGS_QUIT_TITLE)),
            "Ctrl+Q under the modal to ask about the kept Settings draft",
            timeout=5.0,
        )
        await pilot.press("escape")  # Keep editing
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Keep editing",
        )
        # A navigation, were one posted, dismisses the modal within this.
        for _ in range(50):
            if modal not in app.screen_stack:
                break
            await pilot.pause(0.02)
        assert results == [], "the modal was closed past its own guard"
        assert app.screen is modal
        assert app.screen_stack[-2] is home
        assert cleanups == []


async def test_a_raw_edit_left_before_its_change_lands_is_asked_about():
    """Capture fix 3 round 1, finding 2: leaving Settings by a shortcut before
    the raw editor's TextArea.Changed lands, save_state named the unsaved
    categories before it captured the editor's text, so Ctrl+Q from another
    screen quit without asking about Advanced Config and the edit was lost."""
    from textual.widgets import TextArea

    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
    from tldw_chatbook.UI.Settings_Modules.settings_quit_guard import (
        KEPT_UNSAVED_TITLES,
    )

    app = _build_test_app(configured_default="settings")
    async with app.run_test(size=(190, 55)) as pilot:
        await _until(
            pilot,
            lambda: type(app.screen).__name__ == "SettingsScreen",
            "the Settings screen",
        )
        settings = app.screen
        settings._select_category(SettingsCategoryId.ADVANCED_CONFIG.value)
        await _until(
            pilot,
            lambda: (
                bool(settings.query("#settings-advanced-config-editor"))
                and settings._advanced_config_settings is not None
                and not settings._advanced_config_settings.busy
            ),
            "the raw editor to load",
        )
        assert not settings._category_has_unsaved_changes(
            SettingsCategoryId.ADVANCED_CONFIG
        )
        editor = settings.query_one("#settings-advanced-config-editor", TextArea)
        editor.text = editor.text + "\n# typed, its Changed still queued\n"
        state = settings.save_state()  # no await: Changed is not delivered
    assert "Advanced Config" in state[KEPT_UNSAVED_TITLES]


async def _settings_snapshot_keep(app, pilot):
    """Open Providers & Models and return its snapshot Keep count field."""
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    await _until(
        pilot,
        lambda: type(app.screen).__name__ == "SettingsScreen",
        "the Settings screen",
    )
    app.screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
    await _until(
        pilot,
        lambda: bool(app.screen.query("#settings-snapshot-keep")),
        "the snapshot Keep count field",
    )
    return app.screen.query_one("#settings-snapshot-keep", Input)


async def test_a_snapshot_preference_edit_is_kept_and_asked_about_from_console(
    monkeypatch,
):
    """Ruling cf3r1-N1 (b): leaving Settings dropped an unsaved Prompt-cache
    snapshot preference without notice, and Ctrl+Q elsewhere quit past it.
    Rewritten on purpose (it pinned that the edit was not kept, so not asked
    about): the edit is kept with the draft, Ctrl+Q from Console names
    Providers & Models, and Keep editing returns to the edit, still unsaved."""
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    app = _build_test_app(configured_default="settings")
    _configure_native_ready_console(app)
    cleanups: list[bool] = []
    monkeypatch.setattr(
        app, "_run_approved_quit_cleanup", _record_cleanup_then_exit(app, cleanups)
    )
    async with app.run_test(size=(190, 55)) as pilot:
        keep = await _settings_snapshot_keep(app, pilot)
        edited = str(int(keep.value) + 1)
        keep.value = edited
        await _until(
            pilot, app.screen._snapshot_preferences_dirty, "the edit to be staged"
        )
        app.post_message(NavigateToScreen("chat"))
        await _mounted_console(app, pilot)

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: bool(_dialogs_titled(app, _SETTINGS_QUIT_TITLE)) or bool(cleanups),
            "Ctrl+Q on Console to ask about the kept snapshot preference",
            timeout=5.0,
        )
        assert cleanups == [], "quit straight past the kept snapshot preference"
        assert "Providers & Models" in app.screen.message

        await pilot.press("escape")  # Keep editing
        await _until(
            pilot,
            lambda: (
                type(app.screen).__name__ == "SettingsScreen"
                and bool(app.screen.query("#settings-snapshot-keep"))
            ),
            "Keep editing to return to Settings",
        )
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Keep editing",
        )
        settings = app.screen
        assert settings.query_one("#settings-snapshot-keep", Input).value == edited
        status = settings.query_one("#settings-selected-category-draft-status")
        assert str(status.render()) == "Unsaved changes"
        rail = settings.query_one(
            f"#settings-category-{SettingsCategoryId.PROVIDERS_MODELS.value}"
        )
        assert str(rail.label).endswith(" *"), rail.label
    assert cleanups == []


def _in_memory_snapshot_preferences(monkeypatch, enabled: bool, keep_count: int):
    """Keep the snapshot preferences in a list instead of the config.

    The save refuses a stale ``expected`` as the real one does. Returns the
    one-item list holding the saved value.
    """
    from tldw_chatbook.LLM_Management import snapshot_settings as preferences

    saved = [preferences.SnapshotPreferences(enabled=enabled, keep_count=keep_count)]

    def save(value, *, expected=None):
        if expected is not None and expected != saved[0]:
            raise preferences.SnapshotPreferencesConflict("changed")
        saved[0] = value
        return True

    monkeypatch.setattr(preferences, "load_snapshot_preferences", lambda: saved[0])
    monkeypatch.setattr(preferences, "save_snapshot_preferences", save)
    return saved


async def _back_to_settings_snapshot_keep(app, pilot):
    """Leave Settings for Home, come back, and return the Keep count field."""
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    await _go_home(app, pilot)
    app.post_message(NavigateToScreen("settings"))
    keep = await _settings_snapshot_keep(app, pilot)
    await pilot.pause(0.2)
    return keep


@pytest.mark.parametrize("resolve", ["revert", "save"])
async def test_a_kept_snapshot_edit_is_forgotten_once_saved_or_reverted(
    monkeypatch, resolve
):
    """Capture 13c: once a leave kept a snapshot edit, every later return
    restored it, even after it was reverted or saved. save_state started from
    the dict the visit was restored from, and the kept pair is written only
    while unsaved, so it outlived its Revert or Save: Settings came back
    "Unsaved changes" on the old value, and Save said it changed elsewhere."""
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
    from tldw_chatbook.UI.Settings_Modules.settings_quit_guard import (
        KEPT_SNAPSHOT_PREFERENCES,
    )

    saved = _in_memory_snapshot_preferences(monkeypatch, True, 30)
    app = _build_test_app(configured_default="settings")
    async with app.run_test(size=(190, 55)) as pilot:
        keep = await _settings_snapshot_keep(app, pilot)
        assert keep.value == "30"
        keep.value = "25"
        await _until(
            pilot, app.screen._snapshot_preferences_dirty, "the edit to be staged"
        )
        keep = await _back_to_settings_snapshot_keep(app, pilot)
        assert keep.value == "25" and app.screen._snapshot_preferences_dirty()

        settings = app.screen
        if resolve == "revert":
            settings.action_settings_revert_category(allow_text_entry_focus=True)
            await _until(
                pilot,
                lambda: (
                    app.screen in _dialogs_titled(app, "Revert Settings changes")
                    and bool(app.screen.query("#confirm-button"))
                ),
                "the Revert confirmation",
            )
            await pilot.click("#confirm-button")
            expected = "30"
        else:
            settings.action_settings_save_category(allow_text_entry_focus=True)
            expected = "25"
        await _until(
            pilot,
            lambda: (
                not settings._snapshot_preferences_dirty()
                and settings.query_one("#settings-snapshot-keep", Input).value
                == expected
            ),
            f"the {resolve} to settle",
        )
        assert saved[0].keep_count == int(expected)

        keep = await _back_to_settings_snapshot_keep(app, pilot)
        settings = app.screen
        state = settings.save_state()
        assert keep.value == expected, "a resolved snapshot edit came back"
        assert not settings._snapshot_preferences_dirty()
        status = settings.query_one("#settings-selected-category-draft-status")
        assert str(status.render()) == "No unsaved changes"
        rail = settings.query_one(
            f"#settings-category-{SettingsCategoryId.PROVIDERS_MODELS.value}"
        )
        assert not str(rail.label).endswith(" *"), rail.label
    assert KEPT_SNAPSHOT_PREFERENCES not in state


async def test_a_snapshot_edit_left_before_its_change_lands_is_kept():
    """Ruling cf3r1-N1 (b), as finding 2 of round 1 for the raw editor: left
    by a shortcut before Input.Changed lands, the typed keep count must still
    be what Settings keeps and names."""
    from tldw_chatbook.UI.Settings_Modules.settings_quit_guard import (
        KEPT_UNSAVED_TITLES,
    )

    app = _build_test_app(configured_default="settings")
    async with app.run_test(size=(190, 55)) as pilot:
        keep = await _settings_snapshot_keep(app, pilot)
        edited = str(int(keep.value) + 1)
        keep.value = edited
        state = app.screen.save_state()  # no await: Changed is not delivered
        fresh = type(app.screen)(app)  # the next visit's screen
        fresh.restore_state(state)
    assert "Providers & Models" in state[KEPT_UNSAVED_TITLES]
    assert fresh._snapshot_preferences_raw[1] == edited
    assert fresh._snapshot_preferences_dirty()


def _held_snapshot_write(monkeypatch):
    """Hold the snapshot preferences' config write until ``release`` is set.

    Returns ``(entered, release, writes)``; nothing reaches a real config.
    """
    import threading

    from tldw_chatbook.LLM_Management import snapshot_settings as preferences

    entered, release = threading.Event(), threading.Event()
    writes: list = []

    def save(value, **kwargs):
        writes.append(value)
        entered.set()
        assert release.wait(5)
        return True

    monkeypatch.setattr(preferences, "save_snapshot_preferences", save)
    return entered, release, writes


async def _save_a_snapshot_edit(app, pilot, entered) -> None:
    """Edit the snapshot Keep count and press Save; return once it writes."""
    keep = await _settings_snapshot_keep(app, pilot)
    keep.value = str(int(keep.value) + 1)
    await _until(pilot, app.screen._snapshot_preferences_dirty, "the edit to be staged")
    app.screen.action_settings_save_category(allow_text_entry_focus=True)
    await _until(pilot, entered.is_set, "the snapshot write to start")


async def test_leaving_settings_waits_for_a_snapshot_save_in_flight(monkeypatch):
    """Post-rebase round 1, finding 1: Save, then leaving Settings before the
    config write returned, kept the edit as unsaved (the saved value is noted
    only once the write returns) and unmounting cancelled the save's report.
    Back in Settings the edit read 'Unsaved changes' and Save said it had
    changed elsewhere. Leaving now waits for the save, as for the theme's."""
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Settings_Modules.settings_quit_guard import (
        KEPT_SNAPSHOT_PREFERENCES,
        KEPT_UNSAVED_TITLES,
    )

    entered, release, writes = _held_snapshot_write(monkeypatch)
    app = _build_test_app(configured_default="settings")
    try:
        async with app.run_test(size=(190, 55)) as pilot:
            await _save_a_snapshot_edit(app, pilot, entered)
            app.post_message(NavigateToScreen("home"))
            await pilot.pause(0.2)
            release.set()
            await _until(
                pilot, lambda: type(app.screen).__name__ == "HomeScreen", "Home"
            )
            state = (
                app.screen_state_store.restore(
                    "settings", app._current_runtime_identity()
                )
                or {}
            )
            assert _notified(app, "Snapshot preferences saved.")
    finally:
        release.set()
    assert len(writes) == 1
    assert KEPT_SNAPSHOT_PREFERENCES not in state
    assert "Providers & Models" not in state.get(KEPT_UNSAVED_TITLES, [])


async def test_ctrl_q_on_settings_waits_for_a_snapshot_save_in_flight(monkeypatch):
    """Post-rebase round 1, finding 1 on the quit path: Ctrl+Q while the
    snapshot save was writing asked about Providers & Models, which was being
    saved. The quit now waits for the save; with nothing else unsaved it
    quits without asking."""
    entered, release, writes = _held_snapshot_write(monkeypatch)
    app = _build_test_app(configured_default="settings")
    cleanups: list[bool] = []
    monkeypatch.setattr(
        app, "_run_approved_quit_cleanup", _record_cleanup_then_exit(app, cleanups)
    )
    try:
        async with app.run_test(size=(190, 55)) as pilot:
            await _save_a_snapshot_edit(app, pilot, entered)
            await pilot.press("ctrl+q")
            await pilot.pause(0.2)
            release.set()
            # A prompt would hold the quit, so reaching the shutdown proves none.
            await _until_exited(app, cleanups, "Ctrl+Q to quit without asking")
    finally:
        release.set()
    assert cleanups == [True]
    assert len(writes) == 1


async def test_the_unsaved_work_check_reads_a_kept_settings_visit():
    """Recovery mode and maintenance refuse while ``probe_unsaved_editors``
    reports work. It knew four of the keys Settings' save_state writes, so
    every kept visit (``hooks_snapshot`` is always written) read "Retained
    screens: unknown-editor-state", which no save or discard clears. A clean
    visit is now clean, and a kept snapshot preference edit is retained
    Settings work to save or discard."""
    from tldw_chatbook.Backup_Recovery.unsaved_editors import (
        UnsavedEditor,
        probe_unsaved_editors,
    )
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    app = _build_test_app(configured_default="settings")
    async with app.run_test(size=(190, 55)) as pilot:
        await _settings_snapshot_keep(app, pilot)
        await _go_home(app, pilot)
        clean = probe_unsaved_editors(screen_state_store=app.screen_state_store)

        app.post_message(NavigateToScreen("settings"))
        keep = await _settings_snapshot_keep(app, pilot)
        keep.value = str(int(keep.value) + 1)
        await _until(
            pilot, app.screen._snapshot_preferences_dirty, "the edit to be staged"
        )
        await _go_home(app, pilot)
        kept = probe_unsaved_editors(screen_state_store=app.screen_state_store)
    assert clean == ()
    assert kept == (UnsavedEditor("Retained Settings", "needs-user-save-discard"),)
