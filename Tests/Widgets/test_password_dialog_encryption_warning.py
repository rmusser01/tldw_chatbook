"""Regression test for task-851 review finding 3.

Config encryption rewrites config.toml through a TOML parse/serialize
round-trip (``tomllib.load`` + ``toml.dumps``): every value and type
round-trips losslessly, but comments and table ordering do not survive.
That was already true before this branch, but task-851 pointed encryption
at whichever profile file is actually active (previously it silently
no-op'd under a profile) and task-852 makes the "you should encrypt this"
prompt actually fire -- so users now hit the comment/formatting loss where
they didn't before. The fix is to warn in the enable-encryption
confirmation flow rather than silently rewrite an annotated config; see
``Widgets/password_dialog.py``'s ``EncryptionSetupDialog`` and
``UI/Tools_Settings_Window.py``'s checkbox-toggle enable path.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Widgets.password_dialog import (
    COMMENT_LOSS_WARNING,
    EncryptionSetupDialog,
)


class _EncryptionSetupDialogHostApp(App):
    def compose(self) -> ComposeResult:
        yield from ()

    async def show_dialog(self) -> None:
        await self.push_screen(
            EncryptionSetupDialog(detected_providers=["openai"])
        )


@pytest.mark.asyncio
async def test_encryption_setup_dialog_warns_about_comment_loss():
    """The confirmation dialog shown before the very first "enable
    encryption" write must tell the user comments/formatting will not
    survive the rewrite."""
    app = _EncryptionSetupDialogHostApp()
    async with app.run_test() as pilot:
        await app.show_dialog()
        await pilot.pause()

        dialog = app.screen
        assert isinstance(dialog, EncryptionSetupDialog)
        rendered_texts = [
            str(widget.render()) for widget in dialog.query(Static)
        ]
        assert any(COMMENT_LOSS_WARNING in text for text in rendered_texts), (
            "EncryptionSetupDialog no longer warns about comment/formatting "
            f"loss; rendered Static widgets: {rendered_texts!r}"
        )


def test_tools_settings_window_checkbox_toggle_path_warns_about_comment_loss():
    """The OTHER enable-encryption entry point (the General Settings
    "encryption enabled" checkbox toggle in ``Tools_Settings_Window.py``)
    saves directly via a ``PasswordDialog`` and never shows
    ``EncryptionSetupDialog``, so it needs its own copy of the warning
    rather than relying on the dialog test above.

    A source-level check (matching this codebase's existing
    grep-gate idiom, e.g. ``test_unified_mcp_panel_modules_have_zero_importers_repo_wide``)
    rather than a full UI mount: driving that flow end-to-end requires
    toggling a mounted Checkbox, intercepting ``push_screen`` for two
    separate modal dialogs, and stubbing ``enable_config_encryption``,
    which is disproportionate to what is being asserted here.
    """
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "UI"
        / "Tools_Settings_Window.py"
    )
    source = module_path.read_text(encoding="utf-8")
    checkbox_toggle_marker = 'title="Setup Config Encryption"'
    assert checkbox_toggle_marker in source, (
        "checkbox-toggle encryption setup dialog call site not found; "
        "update this test's anchor if it was renamed"
    )
    anchor = source.index(checkbox_toggle_marker)
    # The warning must live in the same PasswordDialog(...) call as the
    # anchor, not merely somewhere else in this large file.
    window = source[anchor : anchor + 800]
    assert "comments or custom formatting" in window, (
        "the checkbox-toggle encryption enable path lost its "
        "comment/formatting-loss warning"
    )
