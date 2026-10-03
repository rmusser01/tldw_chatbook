"""PasswordDialog focus, glyph toggle and change mode (TASK-34100.4).

protect-summary-05: the dialog opened with focus on its VerticalScroll (the
first focusable widget under App.AUTO_FOCUS='*'), so typed characters went
nowhere and the scroll box drew a stray focus frame. protect-summary-03: the
'change' mode had one password field plus a confirm, so it could not drive
``change_encryption_password(old, new)``.
"""

from __future__ import annotations

from typing import Optional

import pytest
from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.password_dialog import PasswordChange, PasswordDialog
from tldw_chatbook.Widgets.state_checkbox import StateCheckbox


class _Host(App):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self._mode = mode
        self.result: Optional[object] = "UNSET"

    def on_mount(self) -> None:
        self.push_screen(PasswordDialog(mode=self._mode), self._record)

    def _record(self, value) -> None:
        self.result = value


class _NoAutoFocusHost(_Host):
    """A host that focuses nothing by itself (review round 1, F7).

    Under the default ``App.AUTO_FOCUS = '*'`` the first input gets focus
    even without the dialog's own ``AUTO_FOCUS``; here only the dialog's
    setting can put the caret in its first field.
    """

    AUTO_FOCUS = None


FIRST_FIELD = {
    "setup": "#password-input",
    "unlock": "#password-input",
    "change": "#current-password-input",
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "host", [_Host, _NoAutoFocusHost], ids=["app-default", "no-app-autofocus"]
)
@pytest.mark.parametrize("mode", ["setup", "unlock", "change"])
async def test_typing_without_tab_fills_the_first_password_field(mode, host) -> None:
    app = host(mode)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        dialog = app.screen
        assert isinstance(dialog, PasswordDialog)
        await pilot.press(*"abcdefgh")
        await pilot.pause(0.05)
        field = dialog.query_one(FIRST_FIELD[mode], Input)
        assert field.value == "abcdefgh"
        assert dialog.focused is field
        # No stray focus frame: the scroll wrapper never takes focus.
        scroll = dialog.query_one(VerticalScroll)
        assert scroll.can_focus is False
        assert scroll not in dialog.focus_chain


@pytest.mark.asyncio
async def test_show_password_toggle_is_the_shared_glyph_checkbox() -> None:
    app = _Host("setup")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        toggle = app.screen.query_one("#show-password-toggle")
        assert isinstance(toggle, StateCheckbox)
        # What the widget actually paints, not the glyph seam it reads
        # (review round 1, F6/F8): the box part precedes the label.
        off = _painted_box(toggle)
        toggle.value = True
        await pilot.pause(0.05)
        on = _painted_box(toggle)
        assert "✓" in on
        assert "✓" not in off and "X" not in off
        assert "X" not in on


def _painted_box(toggle) -> str:
    rendered = toggle.render()
    text = getattr(rendered, "plain", None) or str(rendered)
    return text.split("Show password", 1)[0]


async def _fill_change(pilot, dialog, current: str, new: str, confirm: str) -> None:
    dialog.query_one("#current-password-input", Input).value = current
    dialog.query_one("#password-input", Input).value = new
    dialog.query_one("#confirm-input", Input).value = confirm
    dialog.query_one("#submit-button", Button).press()
    await pilot.pause(0.1)


@pytest.mark.asyncio
async def test_change_mode_returns_current_and_new_password() -> None:
    app = _Host("change")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        dialog = app.screen
        message = str(dialog.query_one(".dialog-message", Static).render())
        assert "current master password" in message
        await _fill_change(
            pilot, dialog, "old-password", "new-password", "new-password"
        )
        assert app.result == PasswordChange(current="old-password", new="new-password")
        assert "old-password" not in repr(app.result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "new", "confirm", "error"),
    [
        ("", "new-password", "new-password", "current master password"),
        ("old-password", "new-password", "different-password", "do not match"),
        ("old-password", "old-password", "old-password", "different"),
        ("old-password", "short", "short", "at least 8"),
    ],
)
async def test_change_mode_rejects_bad_input_inline(
    current, new, confirm, error
) -> None:
    app = _Host("change")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        dialog = app.screen
        await _fill_change(pilot, dialog, current, new, confirm)
        assert isinstance(app.screen, PasswordDialog)
        shown = str(dialog.query_one("#error-message", Static).render())
        assert error in shown
        assert app.result == "UNSET"


@pytest.mark.asyncio
async def test_show_password_reveals_every_field_in_change_mode() -> None:
    app = _Host("change")
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        dialog = app.screen
        fields = list(dialog.query(".password-input").results(Input))
        assert len(fields) == 3 and all(field.password for field in fields)
        dialog.query_one("#show-password-toggle", StateCheckbox).value = True
        await pilot.pause(0.05)
        assert not any(field.password for field in fields)


def _post_burst(app: App, text: str) -> None:
    """Queue every key at once, the way auto-type (a password manager, or a
    pasted burst over tmux) delivers them -- no pause for the app to settle
    between keys, unlike ``pilot.press``."""
    from textual import events

    for char in text:
        if char == "\t":
            app.post_message(events.Key("tab", None))
        else:
            app.post_message(events.Key(char, char))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["setup", "change"])
async def test_a_burst_of_keys_with_tabs_fills_each_field(mode) -> None:
    # Review round 2 (F-R2-5): 'current<Tab>new<Tab>new' arriving as one
    # burst dropped the second Tab, so both copies of the new password landed
    # in the New field and submit said "Passwords do not match". Tab is a
    # non-priority binding: the App forwards the next keys to the OLD focus
    # before the Tab, queued behind them, bubbles back and moves focus.
    app = _Host(mode)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        dialog = app.screen
        assert isinstance(dialog, PasswordDialog)
        prefix = "currentpw\t" if mode == "change" else ""
        _post_burst(app, prefix + "newpassword\tnewpassword")
        await pilot.pause(0.5)
        if mode == "change":
            current = dialog.query_one("#current-password-input", Input)
            assert current.value == "currentpw"
        assert dialog.query_one("#password-input", Input).value == "newpassword"
        assert dialog.query_one("#confirm-input", Input).value == "newpassword"
