"""The Delete-profile journey, driven through a real mounted Textual app.

Qodo review of PR #2799, finding 3 ("Profile deletion lacks a ui journey").
`Tests/TTS/test_voice_cloning_delete_confirm.py` allocates the window with
`__new__` and replaces `VoiceCloningWindow.app` with a fake whose
`push_screen_wait` just records the screen -- so it asserts on the dialog's
constructed `message` and nothing else.

That fake hid a second, live defect: `App.push_screen_wait` raises
`NoActiveWorker` unless it is awaited from inside a Textual worker, and
NEITHER production entry point into `_delete_profile` is one --
`on_button_pressed` runs on the message pump, and `action_delete_profile`
spawns a bare `asyncio.create_task`. TASK-32892 replaced an `AttributeError`
raised inside a nested modal's `compose` with a `NoActiveWorker` raised at
the push: the Delete action still showed nothing and still deleted nothing.

These tests mount the real window, press the real button (and fire the real
binding action), and drive the real `ConfirmationDialog` to both outcomes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = pytest.mark.bootstrap_profile

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _Harness(App[None]):
    CSS_PATH = str(BUNDLE)

    def compose(self) -> ComposeResult:
        yield VoiceCloningWindow()


class _Manager:
    """Records whether the irreversible half actually ran."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_profile(self, name: str) -> tuple[bool, str]:
        self.deleted.append(name)
        return True, f"Deleted {name}"


async def _ready(pilot: Any) -> tuple[VoiceCloningWindow, _Manager]:
    """Mount, let the on_mount profile-load timer settle, select a profile.

    The name deliberately contains `[old]`: an irreversible prompt must name
    the profile literally, not feed it to a markup parser.
    """
    widget = pilot.app.query_one(VoiceCloningWindow)
    await pilot.pause(0.2)
    await pilot.app.workers.wait_for_complete()

    manager = _Manager()
    widget.backend_managers = {"higgs": manager}
    widget.current_backend = "higgs"
    widget.selected_profile = "[old] narrator-01"
    widget._update_button_states()
    await pilot.pause()
    return widget, manager


async def _open_dialog(pilot: Any) -> ConfirmationDialog:
    await pilot.click("#delete-profile-btn")
    await pilot.pause()
    screen = pilot.app.screen
    assert isinstance(screen, ConfirmationDialog), (
        f"Delete showed {type(screen).__name__}, not a confirmation dialog"
    )
    return screen


@pytest.mark.asyncio
async def test_delete_button_opens_a_dialog_naming_the_profile_literally() -> None:
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        await _ready(pilot)
        dialog = await _open_dialog(pilot)

        rendered = "\n".join(str(node.render()) for node in dialog.query("Static,Label"))
        assert "[old] narrator-01" in rendered, rendered
        assert "Delete" in str(dialog.query_one("#confirm-button", Button).label)


@pytest.mark.asyncio
async def test_cancelling_the_dialog_deletes_nothing() -> None:
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        _widget, manager = await _ready(pilot)
        await _open_dialog(pilot)

        await pilot.click("#cancel-button")
        await pilot.pause()

        assert manager.deleted == []
        assert not isinstance(pilot.app.screen, ConfirmationDialog)


@pytest.mark.asyncio
async def test_confirming_the_dialog_deletes_exactly_the_named_profile() -> None:
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        _widget, manager = await _ready(pilot)
        await _open_dialog(pilot)

        await pilot.click("#confirm-button")
        await pilot.pause()
        await pilot.pause()

        assert manager.deleted == ["[old] narrator-01"]
        assert not isinstance(pilot.app.screen, ConfirmationDialog)


@pytest.mark.asyncio
async def test_the_keyboard_action_reaches_the_same_dialog() -> None:
    """`action_delete_profile` spawns a bare `asyncio.create_task`, which is
    not a Textual worker -- the other half of the `NoActiveWorker` trap."""
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        _widget, manager = await _ready(pilot)

        _widget.action_delete_profile()
        await pilot.pause()
        assert isinstance(pilot.app.screen, ConfirmationDialog)

        await pilot.click("#confirm-button")
        await pilot.pause()
        await pilot.pause()
        assert manager.deleted == ["[old] narrator-01"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
