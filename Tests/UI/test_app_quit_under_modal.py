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

The last two tests pin the review follow-ups: a dirty modal's own discard
guard (a ReminderForm) asks before quitting, and the walk reaches unsaved work
on the destination BENEATH a hook-less modal (Settings' theme editor).
"""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from textual.screen import ModalScreen
from textual.widgets import Input, Static

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
    from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId
    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    app = _build_test_app(configured_default="settings")
    cleanups: list[bool] = []

    async def _record_cleanup() -> None:
        cleanups.append(True)

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_cleanup)
    async with app.run_test(size=(190, 55)) as pilot:
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
