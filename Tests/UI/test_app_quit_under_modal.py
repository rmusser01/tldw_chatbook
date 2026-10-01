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
"""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Chat.console_chat_models import ConsoleLifecycleImpact
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


def _quit_dialogs(app) -> list[ConfirmationDialog]:
    return [
        screen
        for screen in app.screen_stack
        if isinstance(screen, ConfirmationDialog) and screen.title == "Quit Chatbook?"
    ]


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
