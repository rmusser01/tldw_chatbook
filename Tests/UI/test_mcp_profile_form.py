# Tests/UI/test_mcp_profile_form.py
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static, TextArea

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
        app.query_one("#mcp-form-env", TextArea).text = "API_KEY=$MY_KEY\nDEBUG=true"
        payload = form.build_payload()
        assert payload == {
            "profile_id": "docs", "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "env_placeholders": {"API_KEY": "$MY_KEY"},
            "env_literals": {"DEBUG": "true"},
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
