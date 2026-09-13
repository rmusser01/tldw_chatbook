"""PasswordDialog UX contract (TASK-21141, UAT findings K-1..K-6).

The UAT incident behind K-3: in the real app the dialog's error Static wore
the app-wide ``.error-message`` class, whose blanket rule in _wizards.tcss
(round border + padding + margin) inflated the one-line error to ~7 rows and
pushed Cancel/Submit past the container's max-height clip — the buttons
stayed functional but invisible, and the error never cleared. These tests pin
the corrected contract at the widget level; the class rename
(``password-dialog-error``) is what keeps the blanket rule away.
"""

from typing import Optional

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, Input, Static

from tldw_chatbook.Widgets.password_dialog import PasswordDialog


class _Host(App):
    def __init__(self, mode: str = "setup") -> None:
        super().__init__()
        self._mode = mode
        self.result: Optional[str] = "UNSET"

    def on_mount(self) -> None:
        self.push_screen(PasswordDialog(mode=self._mode), self._record)

    def _record(self, value) -> None:
        self.result = value


async def _submit(pilot, dialog) -> None:
    dialog.query_one("#submit-button", Button).press()
    await pilot.pause(0.1)


@pytest.mark.asyncio
async def test_escape_cancels_the_dialog() -> None:
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.1)
        assert isinstance(app.screen, PasswordDialog)
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not isinstance(app.screen, PasswordDialog)
        assert app.result is None


@pytest.mark.asyncio
async def test_failed_submit_keeps_buttons_visible_and_error_clears_on_edit() -> None:
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.1)
        dialog = app.screen
        dialog.query_one("#password-input", Input).value = "a"
        dialog.query_one("#confirm-input", Input).value = "a"
        await _submit(pilot, dialog)

        error = dialog.query_one("#error-message", Static)
        assert error.has_class("visible")
        assert "at least 8 characters" in str(error.renderable)
        # The dialog-scoped class keeps the app-wide .error-message blanket
        # rule (border+padding) off this widget.
        assert error.has_class("password-dialog-error")
        assert not error.has_class("error-message")
        # Buttons remain on screen after the error appears (the K-3 clip).
        for button_id in ("#cancel-button", "#submit-button"):
            button = dialog.query_one(button_id, Button)
            assert button.region.height > 0, f"{button_id} clipped out of view"

        # Editing any field retires the stale error.
        dialog.query_one("#password-input", Input).value = "ab"
        await pilot.pause(0.1)
        assert not error.has_class("visible")


@pytest.mark.asyncio
async def test_requirements_and_consequence_stated_before_submit() -> None:
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.1)
        dialog = app.screen
        message = str(dialog.query_one(".dialog-message", Static).renderable)
        assert "At least 8 characters" in message
        assert "forget" in message and "cannot be recovered" in message
        title = str(dialog.query_one(".dialog-title").renderable)
        assert title == "Set up master password"


@pytest.mark.asyncio
async def test_show_password_toggle_reveals_both_fields() -> None:
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.1)
        dialog = app.screen
        password = dialog.query_one("#password-input", Input)
        confirm = dialog.query_one("#confirm-input", Input)
        assert password.password and confirm.password
        toggle = dialog.query_one("#show-password-toggle", Checkbox)
        toggle.value = True
        await pilot.pause(0.1)
        assert not password.password and not confirm.password
        toggle.value = False
        await pilot.pause(0.1)
        assert password.password and confirm.password
