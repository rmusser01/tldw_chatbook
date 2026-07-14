# Tests/UI/test_mcp_server_mutations.py
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Input, Select, Static

from tldw_chatbook.UI.MCP_Modules.mcp_server_mutations import MCPServerMutationsPanel


class MutationsApp(App):
    def __init__(self, record=None, slots=()) -> None:
        super().__init__()
        self.record = record
        self.slots = list(slots)
        self.posted: list = []

    def compose(self) -> ComposeResult:
        yield MCPServerMutationsPanel(record=self.record, slots=self.slots, id="panel")

    def on_mcp_server_mutations_submit_requested(self, event) -> None:
        self.posted.append((event.action, event.payload))


@pytest.mark.asyncio
async def test_add_mode_posts_create_with_exact_payload():
    app = MutationsApp()
    async with app.run_test() as pilot:
        app.query_one("#mcp-srv-id", Input).value = "web-search"
        app.query_one("#mcp-srv-name", Input).value = "Web Search"
        app.query_one("#mcp-srv-transport", Select).value = "http"
        app.query_one("#mcp-srv-url", Input).value = "https://mcp.example/api"
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        action, payload = app.posted[-1]
        assert action == "external_server.create"
        assert payload == {
            "server_id": "web-search", "name": "Web Search", "transport": "http",
            "config": {"url": "https://mcp.example/api"}, "enabled": True,
        }
        assert "owner_scope_type" not in payload and "owner_scope_id" not in payload


@pytest.mark.asyncio
async def test_edit_mode_posts_update_with_name_and_enabled_only():
    record = {"server_id": "web-search", "name": "Web Search", "enabled": True}
    app = MutationsApp(record=record)
    async with app.run_test() as pilot:
        app.query_one("#mcp-srv-name", Input).value = "Search"
        app.query_one("#mcp-srv-enabled", Checkbox).value = False
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        action, payload = app.posted[-1]
        assert action == "external_server.update"
        assert payload == {"server_id": "web-search", "name": "Search", "enabled": False}


@pytest.mark.asyncio
async def test_slot_create_posts_five_fields():
    record = {"server_id": "web-search", "name": "Web Search", "enabled": True}
    app = MutationsApp(record=record)
    async with app.run_test() as pilot:
        app.query_one("#mcp-slot-name", Input).value = "token_readonly"
        app.query_one("#mcp-slot-display", Input).value = "Read-only token"
        app.query_one("#mcp-slot-kind", Select).value = "bearer_token"
        app.query_one("#mcp-slot-privilege", Select).value = "read"
        app.query_one("#mcp-slot-required", Checkbox).value = True
        await pilot.click("#mcp-slot-add")
        await pilot.pause()
        action, payload = app.posted[-1]
        assert action == "external_server.slot.create"
        assert payload == {
            "server_id": "web-search", "slot_name": "token_readonly",
            "display_name": "Read-only token", "secret_kind": "bearer_token",
            "privilege_class": "read", "is_required": True,
        }


@pytest.mark.asyncio
async def test_slot_secret_set_posts_and_clears_input():
    record = {"server_id": "web-search", "name": "Web Search", "enabled": True}
    slots = [{"slot_name": "token_readonly", "display_name": "Read-only token"}]
    app = MutationsApp(record=record, slots=slots)
    async with app.run_test() as pilot:
        secret_input = app.query_one("#mcp-slot-secret-0", Input)
        assert secret_input.password is True
        secret_input.value = "sk-value"
        await pilot.click("#mcp-slot-secret-set-0")
        await pilot.pause()
        action, payload = app.posted[-1]
        assert action == "external_server.slot.secret.set"
        assert payload == {"server_id": "web-search", "slot_name": "token_readonly",
                           "secret": "sk-value"}
        assert secret_input.value == ""


# -- F2: URL field validated via the shared input_validation module --------


@pytest.mark.asyncio
async def test_add_mode_rejects_unsupported_url_scheme_inline():
    """F2 (Qodo compliance finding): `build_payload()` must reject an
    invalid/unsupported-scheme URL via `input_validation.validate_url()`
    rather than forwarding it -- surfaced the same way every other
    `build_payload()` `ValueError` is (`on_button_pressed`'s
    `show_error()`), and critically WITHOUT posting `SubmitRequested`
    (a bad payload must never reach `run_action()`).
    """
    app = MutationsApp()
    async with app.run_test() as pilot:
        app.query_one("#mcp-srv-id", Input).value = "web-search"
        app.query_one("#mcp-srv-name", Input).value = "Web Search"
        app.query_one("#mcp-srv-transport", Select).value = "http"
        app.query_one("#mcp-srv-url", Input).value = "javascript:alert(1)"
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        assert app.posted == [], f"expected no SubmitRequested, got: {app.posted!r}"
        error_text = str(app.query_one("#mcp-srv-error", Static).renderable)
        assert error_text == "URL must be a valid http(s) address.", error_text


@pytest.mark.asyncio
async def test_add_mode_accepts_valid_https_url():
    """F2: a well-formed https URL must still pass `validate_url()` and
    reach the payload unchanged -- the positive-path companion to the
    rejection test above.
    """
    app = MutationsApp()
    async with app.run_test() as pilot:
        app.query_one("#mcp-srv-id", Input).value = "web-search"
        app.query_one("#mcp-srv-name", Input).value = "Web Search"
        app.query_one("#mcp-srv-transport", Select).value = "http"
        app.query_one("#mcp-srv-url", Input).value = "https://mcp.example/api"
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        action, payload = app.posted[-1]
        assert action == "external_server.create"
        assert payload["config"] == {"url": "https://mcp.example/api"}


@pytest.mark.asyncio
async def test_add_mode_blank_url_still_posts_empty_config():
    """F2: a blank URL must stay allowed (unchanged pre-fix behavior) --
    `validate_url()` only runs when the field is non-empty, so `config`
    stays `{}` rather than being rejected as an invalid URL.
    """
    app = MutationsApp()
    async with app.run_test() as pilot:
        app.query_one("#mcp-srv-id", Input).value = "web-search"
        app.query_one("#mcp-srv-name", Input).value = "Web Search"
        app.query_one("#mcp-srv-transport", Select).value = "http"
        await pilot.click("#mcp-srv-save")
        await pilot.pause()
        action, payload = app.posted[-1]
        assert action == "external_server.create"
        assert payload["config"] == {}
