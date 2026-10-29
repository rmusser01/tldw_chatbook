"""A raw-CLI draft's composer state follows the arm, and arms in place (TASK-33622.9).

Live on dev 38c9d2edcf (160x45, ``raw_cli_permitted = true``, nothing armed),
typing ``! git status --short`` turned the composer to "RAW CLI · HOST
ACCESS" with an enabled "Run": the same as the armed state. Enter then
refused the command. The draft-doubling half of the finding was already fixed
on dev (1393efb623); these tests pin it with repeated refusals.

The owner decided on 2026-10-27 that the composer may arm raw CLI in place
behind the same danger-disclosure confirm the Terminal uses. These tests drive
the real ChatScreen, composer and raw-CLI controller with real keys; the app's
raw-CLI runtime is the real ``RawCliRuntime`` with its persisted unlock read
replaced.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_console_send_acknowledgement import (
    build,
    eager_tasks,
    press,
    ready_console,
    until,
)
from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

# Mounted Console tests retain the configuration admitted at collection time.
pytestmark = pytest.mark.bootstrap_profile

COMMAND = "! pwd"


def _install_runtime(host, *, permitted: bool) -> RawCliRuntime:
    runtime = RawCliRuntime(lambda: permitted)
    host.app_instance.raw_cli_runtime = runtime
    return runtime


def _type_command(host) -> None:
    names = {"!": "exclamation_mark", " ": "space"}
    for char in COMMAND:
        press(host, names.get(char, char), char)


def _status(composer) -> str:
    status = composer.query_one("#console-raw-cli-status", Static)
    return str(status.renderable)


def _send(composer) -> Button:
    return composer.query_one("#console-send-message", Button)


def _record_starts(console, *, accept: bool) -> list:
    """Spy on the controller's start; ``accept`` replaces it with a success."""
    started: list = []
    real_start = console._raw_cli.start_user_command

    def start(stash):
        started.append(stash)
        return True if accept else real_start(stash)

    console._raw_cli.start_user_command = start
    return started


async def _typed_raw_draft(host, pilot, gateway):
    console, composer = await ready_console(host, pilot, gateway)
    composer.load_draft("")
    await pilot.pause()
    _type_command(host)
    await until(lambda: composer.draft_text() == COMMAND)
    await pilot.pause()
    return console, composer


async def _dialog(host) -> ConfirmationDialog:
    await until(lambda: isinstance(host.screen, ConfirmationDialog), timeout=10)
    return host.screen


@pytest.mark.asyncio
async def test_a_locked_raw_command_is_refused_and_restored_exactly_on_every_enter():
    """AC#1/#4/#5/#6: no "Run" while Locked; each refusal restores it once.

    Before 1393efb623 each Enter prepended the restored capture to the live
    copy (1, 2, 4 copies); the pre-fix composer also read "RAW CLI · HOST
    ACCESS" and "Run" here.
    """
    host, gateway, _timeline = build()
    _install_runtime(host, permitted=False)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await _typed_raw_draft(host, pilot, gateway)
            assert _status(composer) == "RAW CLI · LOCKED"
            assert str(_send(composer).label) == "Locked"
            started = _record_starts(console, accept=False)

            for attempt in (1, 2, 3):
                press(host, "enter", "\r")
                await until(lambda: len(started) == attempt)
                await pilot.pause(0.2)
                assert composer.draft_text() == COMMAND, attempt

            assert [stash.text for stash in started] == [COMMAND] * 3
            assert not isinstance(host.screen, ConfirmationDialog)
            assert _status(composer) == "RAW CLI · LOCKED"


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["enter", "button"])
async def test_an_unarmed_raw_command_arms_in_place_behind_the_disclosure(trigger):
    """AC#2/#4/#6: "Arm…" (Enter or a click) opens the arm disclosure.

    Cancel changes nothing; Confirm arms this launch and leaves the draft for
    an explicit Run, and that Run clears the composer.
    """
    host, gateway, _timeline = build()
    runtime = _install_runtime(host, permitted=True)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await _typed_raw_draft(host, pilot, gateway)
            assert _status(composer) == "RAW CLI · NOT ARMED"
            assert str(_send(composer).label) == "Arm…"
            started = _record_starts(console, accept=True)

            def choose_primary() -> None:
                if trigger == "enter":
                    press(host, "enter", "\r")
                else:
                    _send(composer).press()

            choose_primary()
            dialog = await _dialog(host)
            assert dialog.title == "Arm raw CLI for this launch"
            assert "shell_exec" in dialog.message
            assert "not a sandbox" in dialog.message
            await pilot.pause()
            assert host.focused is not None
            assert host.focused.id == "cancel-button"
            assert composer.draft_text() == COMMAND
            assert started == []

            press(host, "escape")
            await until(lambda: not isinstance(host.screen, ConfirmationDialog))
            await pilot.pause(0.2)
            assert runtime.armed is False
            assert composer.draft_text() == COMMAND
            assert str(_send(composer).label) == "Arm…"

            choose_primary()
            dialog = await _dialog(host)
            dialog.query_one("#confirm-button", Button).press()
            await until(lambda: runtime.armed is True)
            await until(lambda: str(_send(composer).label) == "Run")
            assert _status(composer) == "RAW CLI · HOST ACCESS"
            assert composer.draft_text() == COMMAND
            assert started == []

            press(host, "enter", "\r")
            await until(lambda: bool(started))
            await pilot.pause(0.2)
            assert [stash.text for stash in started] == [COMMAND]
            assert composer.draft_text() == ""
