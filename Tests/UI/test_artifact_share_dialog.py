"""Dialog behavior tests for artifact share."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, SelectionList, Static

from tldw_chatbook.UI.Screens.artifact_share_dialog import ArtifactShareDialog

pytestmark = pytest.mark.ui

_CONFIRM_PHRASE = "share"


class _DialogHost(App[None]):
    def __init__(self, dialog: ArtifactShareDialog) -> None:
        super().__init__()
        self._dialog = dialog
        self.result: object = "unset"

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(self._dialog, self._accept)

    def _accept(self, result: object) -> None:
        self.result = result


def _records() -> list[dict]:
    return [
        {"id": "1", "chatbook_id": 1, "name": "With Bundle", "description": "d1", "file_path": "/tmp/a.zip"},
        {"id": "2", "chatbook_id": 2, "name": "No Bundle", "description": "d2", "file_path": None},
    ]


async def test_cancel_dismisses_with_none():
    dialog = ArtifactShareDialog(_records())
    app = _DialogHost(dialog)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


async def test_start_requires_selection():
    dialog = ArtifactShareDialog(_records())
    app = _DialogHost(dialog)
    async with app.run_test(size=(100, 60)) as pilot:
        await pilot.click("#share-start")
        await pilot.pause()
        assert app.result == "unset"  # still open
        status = dialog.query_one("#share-dialog-status", Static)
        assert "select" in status.renderable.lower() if status.renderable else True


async def test_valid_submission_returns_options():
    dialog = ArtifactShareDialog(_records())
    app = _DialogHost(dialog)
    async with app.run_test(size=(100, 60)) as pilot:
        options = dialog.query_one("#share-artifact-list", SelectionList)
        # select the first (enabled) option; the second is disabled (no bundle)
        options.select(options.get_option_at_index(0).value)
        await pilot.pause()
        share_name = dialog.query_one("#share-name", Input)
        share_name.value = "Field kit"
        await pilot.click("#share-auth-toggle")
        await pilot.pause()
        dialog.query_one("#share-username").value = "alice"
        dialog.query_one("#share-password").value = "secret-pass"
        await pilot.click("#share-start")
        await pilot.pause()
    assert isinstance(app.result, dict)
    assert app.result["share_name"] == "Field kit"
    assert [r["name"] for r in app.result["selected_records"]] == ["With Bundle"]
    assert app.result["username"] == "alice"
    assert app.result["bind"] == "127.0.0.1"
    assert app.result["port"] == 0


async def test_lan_without_password_requires_typed_confirmation():
    dialog = ArtifactShareDialog(_records())
    app = _DialogHost(dialog)
    async with app.run_test(size=(100, 60)) as pilot:
        options = dialog.query_one("#share-artifact-list", SelectionList)
        options.select(options.get_option_at_index(0).value)
        await pilot.pause()
        await pilot.click("#share-bind-lan")
        await pilot.pause()
        await pilot.click("#share-start")
        await pilot.pause()
        assert app.result == "unset"  # blocked: confirmation required
        confirm = dialog.query_one("#share-confirm")
        assert confirm.display
        confirm.value = _CONFIRM_PHRASE
        # A second rapid pilot.click on the same Button is swallowed: press() adds
        # the -active animation class and Button._on_click no-ops while it is set.
        # Button.press() posts the same Button.Pressed message deterministically.
        dialog.query_one("#share-start", Button).press()
        await pilot.pause()
    assert isinstance(app.result, dict)
    assert app.result["bind"] == "0.0.0.0"
    assert app.result["password"] == ""


async def test_username_with_colon_is_rejected():
    # Qodo #14: Basic auth splits on the first ':', so a colon username is
    # refused at the dialog instead of starting an unusable share.
    dialog = ArtifactShareDialog(_records())
    app = _DialogHost(dialog)
    async with app.run_test(size=(100, 60)) as pilot:
        options = dialog.query_one("#share-artifact-list", SelectionList)
        options.select(options.get_option_at_index(0).value)
        await pilot.pause()
        await pilot.click("#share-auth-toggle")
        await pilot.pause()
        dialog.query_one("#share-username").value = "alice:admin"
        dialog.query_one("#share-password").value = "secret-pass"
        dialog.query_one("#share-start", Button).press()
        await pilot.pause()
        assert app.result == "unset"  # still open
        status = dialog.query_one("#share-dialog-status", Static)
        assert ":" in str(status.renderable)


async def test_oversized_fields_are_rejected():
    # Qodo #2: share_name (200), username (64), and password (256) are
    # bounded; an over-long value keeps the dialog open with a message.
    dialog = ArtifactShareDialog(_records())
    app = _DialogHost(dialog)
    async with app.run_test(size=(100, 60)) as pilot:
        options = dialog.query_one("#share-artifact-list", SelectionList)
        options.select(options.get_option_at_index(0).value)
        await pilot.pause()
        share_name = dialog.query_one("#share-name")
        username = dialog.query_one("#share-username")
        password = dialog.query_one("#share-password")

        share_name.value = "n" * 201
        dialog.query_one("#share-start", Button).press()
        await pilot.pause()
        assert app.result == "unset"

        share_name.value = "Field kit"
        await pilot.click("#share-auth-toggle")
        await pilot.pause()
        username.value = "u" * 65
        password.value = "secret-pass"
        dialog.query_one("#share-start", Button).press()
        await pilot.pause()
        assert app.result == "unset"

        username.value = "alice"
        password.value = "p" * 257
        dialog.query_one("#share-start", Button).press()
        await pilot.pause()
        assert app.result == "unset"

        # values at the bound itself are accepted
        share_name.value = "n" * 200
        username.value = "u" * 64
        password.value = "p" * 256
        dialog.query_one("#share-start", Button).press()
        await pilot.pause()
    assert isinstance(app.result, dict)
    assert len(app.result["share_name"]) == 200
    assert len(app.result["username"]) == 64
    assert len(app.result["password"]) == 256
