# Tests/UI/test_mcp_profile_form.py
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPProfileForm


class FormApp(App):
    def __init__(self, profile=None) -> None:
        super().__init__()
        self.profile = profile
        self.events: list = []

    def compose(self) -> ComposeResult:
        yield MCPProfileForm(profile=self.profile, id="form")

    def on_mcp_profile_form_submit_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_profile_form_cancelled(self, event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_build_payload_splits_env_into_placeholders_and_literals():
    app = FormApp()
    async with app.run_test() as pilot:
        form = app.query_one(MCPProfileForm)
        app.query_one("#mcp-form-id", Input).value = "docs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one("#mcp-form-args", TextArea).text = "-y\n@modelcontextprotocol/server-filesystem"
        # UNBAL_A/UNBAL_B: unbalanced-brace values ($VAR} / ${VAR) are NOT
        # placeholders -- they fall to the literals path, where the store's
        # own validation gives honest copy if they turn out secret-shaped.
        app.query_one("#mcp-form-env", TextArea).text = (
            "API_KEY=$MY_KEY\nDEBUG=true\nUNBAL_A=$MY_KEY}\nUNBAL_B=${MY_KEY"
        )
        payload = form.build_payload()
        assert payload == {
            "profile_id": "docs", "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "env_placeholders": {"API_KEY": "$MY_KEY"},
            "env_literals": {
                "DEBUG": "true",
                "UNBAL_A": "$MY_KEY}",
                "UNBAL_B": "${MY_KEY",
            },
        }


@pytest.mark.asyncio
async def test_malformed_env_line_raises_with_line_number():
    app = FormApp()
    async with app.run_test() as pilot:
        form = app.query_one(MCPProfileForm)
        app.query_one("#mcp-form-id", Input).value = "docs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one("#mcp-form-env", TextArea).text = "API_KEY=$K\nnot-an-env-line"
        with pytest.raises(ValueError, match="line 2"):
            form.build_payload()


@pytest.mark.asyncio
async def test_save_posts_submit_and_edit_mode_locks_id():
    profile = {"profile_id": "docs", "command": "npx", "args": ["-y"],
               "env_placeholders": {"K": "$V"}, "env_literals": {}}
    app = FormApp(profile=profile)
    async with app.run_test() as pilot:
        assert app.query_one("#mcp-form-id", Input).disabled
        assert app.query_one("#mcp-form-command", Input).value == "npx"
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        assert app.events and app.events[-1].payload["profile_id"] == "docs"


@pytest.mark.asyncio
async def test_show_error_renders_store_copy():
    app = FormApp()
    async with app.run_test() as pilot:
        form = app.query_one(MCPProfileForm)
        form.show_error("Secret-bearing env key 'API_KEY' cannot be stored as a literal")
        await pilot.pause()
        assert "cannot be stored" in str(app.query_one("#mcp-form-error", Static).renderable)


@pytest.mark.asyncio
async def test_save_disables_button_and_show_error_reenables():
    """State-driven buttons: a valid submit disables Save (blocking a second
    real click at the press() gate) until the host reports an outcome --
    show_error() (validation/save failure) re-enables it for a retry."""
    profile = {"profile_id": "docs", "command": "npx", "args": [],
               "env_placeholders": {}, "env_literals": {}}
    app = FormApp(profile=profile)
    async with app.run_test() as pilot:
        form = app.query_one(MCPProfileForm)
        save_button = app.query_one("#mcp-form-save", Button)
        assert not save_button.disabled
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        assert app.events, "valid submit must still post SubmitRequested"
        assert save_button.disabled, "Save must disable while a save is pending"
        form.show_error("Secret-bearing env key 'API_KEY' cannot be stored as a literal")
        await pilot.pause()
        assert not save_button.disabled, "show_error must re-enable Save for retry"
