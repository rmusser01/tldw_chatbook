# Tests/UI/test_mcp_profile_form.py
from __future__ import annotations

import json

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel, MCPProfileForm


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
async def test_form_error_static_carries_status_error_class():
    """A2: `#mcp-form-error` must carry `mcp-status-error` at compose time
    (the color rule for it has existed since T13, but nothing applied the
    class) so a validation/save failure actually renders in the error color
    instead of plain body text.
    """
    app = FormApp()
    async with app.run_test() as pilot:
        error = app.query_one("#mcp-form-error", Static)
        assert "mcp-status-error" in error.classes


@pytest.mark.asyncio
async def test_form_args_warning_static_carries_status_warning_class():
    """A2: `#mcp-form-args-warning` must carry `mcp-status-warning` at
    compose time so the secret-lint warning renders in the warning color."""
    app = FormApp()
    async with app.run_test() as pilot:
        warning = app.query_one("#mcp-form-args-warning", Static)
        assert "mcp-status-warning" in warning.classes


@pytest.mark.asyncio
async def test_save_with_secret_shaped_arg_shows_warning_but_still_submits():
    """I4/spec §7: a secret-looking args value (visible in `ps`) must warn,
    NON-BLOCKING -- Save still posts `SubmitRequested`. Reuses the store's
    own secret-value shapes (`local_store._looks_like_raw_secret_value`,
    imported not re-implemented) so the two checks never drift.
    """
    app = FormApp()
    async with app.run_test() as pilot:
        app.query_one("#mcp-form-id", Input).value = "docs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one("#mcp-form-args", TextArea).text = "-y\nsk-1234567890abcdef"
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        warning = str(app.query_one("#mcp-form-args-warning", Static).renderable)
        assert "line 2" in warning
        assert "secret" in warning.lower()
        assert "process listings" in warning.lower()
        assert "env" in warning.lower()
        # Non-blocking: the submit still went through despite the warning.
        assert app.events
        assert app.events[-1].payload["args"] == ["-y", "sk-1234567890abcdef"]


@pytest.mark.asyncio
async def test_save_with_clean_args_shows_no_secret_warning():
    app = FormApp()
    async with app.run_test() as pilot:
        app.query_one("#mcp-form-id", Input).value = "docs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        app.query_one("#mcp-form-args", TextArea).text = (
            "-y\n@modelcontextprotocol/server-filesystem"
        )
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        warning = str(app.query_one("#mcp-form-args-warning", Static).renderable)
        assert warning == ""
        assert app.events


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


# -- A4: Save is state-driven (disabled until profile-id + command are set) --


@pytest.mark.asyncio
async def test_add_mode_save_starts_disabled_with_empty_fields():
    """A fresh add-mode form (both fields empty) must not let the user
    submit an incomplete profile -- Save starts disabled with a tooltip
    explaining why."""
    app = FormApp()
    async with app.run_test() as pilot:
        save_button = app.query_one("#mcp-form-save", Button)
        assert save_button.disabled
        assert save_button.tooltip == "Enter a profile id and command first."


@pytest.mark.asyncio
async def test_edit_mode_save_starts_enabled_with_prefilled_fields():
    """Edit mode pre-fills profile_id/command (the id Input is disabled but
    still carries a value) -- the same non-empty rule holds, so Save starts
    enabled rather than disabled-by-default."""
    profile = {"profile_id": "docs", "command": "npx", "args": [],
               "env_placeholders": {}, "env_literals": {}}
    app = FormApp(profile=profile)
    async with app.run_test() as pilot:
        save_button = app.query_one("#mcp-form-save", Button)
        assert not save_button.disabled


@pytest.mark.asyncio
async def test_typing_both_fields_enables_save():
    app = FormApp()
    async with app.run_test() as pilot:
        save_button = app.query_one("#mcp-form-save", Button)
        assert save_button.disabled

        app.query_one("#mcp-form-id", Input).value = "docs"
        await pilot.pause()
        assert save_button.disabled, "one field alone must not enable Save"

        app.query_one("#mcp-form-command", Input).value = "npx"
        await pilot.pause()
        assert not save_button.disabled
        assert save_button.tooltip == "Validate and save this profile."


@pytest.mark.asyncio
async def test_in_flight_save_disabled_tooltip_does_not_claim_fields_are_empty():
    """Review fix: `_refresh_save_enabled()` must not collapse the two
    distinct disable-reasons (missing fields vs. a submit already in
    flight) into the same tooltip -- while a valid submit is pending, both
    fields ARE filled in, so "Enter a profile id and command first." is
    factually wrong and misleading.
    """
    app = FormApp()
    async with app.run_test() as pilot:
        app.query_one("#mcp-form-id", Input).value = "docs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        await pilot.pause()
        save_button = app.query_one("#mcp-form-save", Button)
        await pilot.click("#mcp-form-save")
        await pilot.pause()
        assert save_button.disabled
        assert save_button.tooltip != "Enter a profile id and command first."


@pytest.mark.asyncio
async def test_whitespace_only_fields_keep_save_disabled():
    """"empty/whitespace" per the spec -- a field that is only spaces must
    not count as filled in."""
    app = FormApp()
    async with app.run_test() as pilot:
        save_button = app.query_one("#mcp-form-save", Button)
        app.query_one("#mcp-form-id", Input).value = "   "
        app.query_one("#mcp-form-command", Input).value = "npx"
        await pilot.pause()
        assert save_button.disabled


@pytest.mark.asyncio
async def test_error_path_reenables_only_when_fields_still_valid():
    """`show_error()` routes through `_refresh_save_enabled()` -- if the
    fields are no longer both filled in (e.g. the user cleared one while a
    save was in flight), the error path must NOT blindly re-enable Save.
    """
    app = FormApp()
    async with app.run_test() as pilot:
        form = app.query_one(MCPProfileForm)
        save_button = app.query_one("#mcp-form-save", Button)
        app.query_one("#mcp-form-id", Input).value = "docs"
        app.query_one("#mcp-form-command", Input).value = "npx"
        await pilot.pause()
        assert not save_button.disabled

        await pilot.click("#mcp-form-save")
        await pilot.pause()
        assert save_button.disabled, "Save must disable while a save is pending"

        # Field cleared while the (fake) save was in flight.
        app.query_one("#mcp-form-command", Input).value = ""
        await pilot.pause()

        form.show_error("Some store-side failure.")
        await pilot.pause()
        assert save_button.disabled, "show_error must not re-enable Save with an empty field"
        assert save_button.tooltip == "Enter a profile id and command first."


# -- T8: MCPImportPanel (paste/file mcpServers JSON import) -----------------


class ImportApp(App):
    def __init__(self, *, existing_ids=None) -> None:
        super().__init__()
        self.existing_ids = existing_ids
        self.events: list = []

    def compose(self) -> ComposeResult:
        yield MCPImportPanel(existing_ids=self.existing_ids, id="import-panel")

    def on_mcp_import_panel_file_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_import_panel_import_requested(self, event) -> None:
        self.events.append(event)

    def on_mcp_import_panel_cancelled(self, event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_preview_renders_candidates_and_warnings_as_plain_text():
    """Preview renders one Static per candidate with id/command/warnings, and
    the content is literal, unstyled text (`markup=False`) even when a
    server name looks like Rich markup -- same defense as the overview
    table's user-controlled cells in mcp_servers_mode.py.
    """
    app = ImportApp()
    async with app.run_test() as pilot:
        text = json.dumps({"mcpServers": {
            "[/bold]evil": {"command": "npx", "env": {"API_KEY": "sk-live-123456"}},
        }})
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()
        statics = list(app.query("#mcp-import-list Static"))
        assert len(statics) == 1
        body = str(statics[0].renderable)
        assert "[/bold]evil" in body  # literal text, not parsed as markup
        assert "npx" in body
        assert "export it before connecting" in body
        apply_button = app.query_one("#mcp-import-apply", Button)
        assert not apply_button.disabled
        assert apply_button.label.plain == "Import 1 server"


@pytest.mark.asyncio
async def test_import_list_container_does_not_expand_past_content():
    """A1: `Vertical(id="mcp-import-list")` had no height override, so it
    silently inherited Textual's Vertical default (`height: 1fr`) and
    expanded to consume whatever vertical space the panel had left -- pushing
    the "Import N servers"/Cancel action row far below the preview list on a
    real terminal window (same bug class as `#mcp-detail-builtin-toggles`,
    fixed in commit 83a5fa98). A larger viewport is needed to reproduce this
    (the bare 80x24 default doesn't leave much room for the `1fr` to expand
    into): confirmed red pre-fix at `size=(100, 60)` with a single preview
    candidate -- the container claimed all 16 rows of leftover space instead
    of the ~1 its one rendered line actually needs.
    """
    app = ImportApp()
    async with app.run_test(size=(100, 60)) as pilot:
        text = json.dumps({"mcpServers": {"docs": {"command": "npx"}}})
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()

        preview_list = app.query_one("#mcp-import-list")
        apply_button = app.query_one("#mcp-import-apply", Button)

        # One candidate is a single rendered line -- nowhere near the height
        # of an expanding 1fr container consuming whatever vertical space
        # the panel has left (16 rows pre-fix at this viewport).
        assert preview_list.size.height < 16

        # The action row must sit directly under the preview list, not
        # dozens of rows further down the panel.
        gap = apply_button.region.y - (preview_list.region.y + preview_list.region.height)
        assert 0 <= gap <= 2


@pytest.mark.asyncio
async def test_preview_existing_id_warns_overwrite():
    app = ImportApp(existing_ids={"docs"})
    async with app.run_test() as pilot:
        text = json.dumps({"mcpServers": {"docs": {"command": "npx"}}})
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()
        body = str(app.query_one("#mcp-import-list Static").renderable)
        assert "overwrite" in body


@pytest.mark.asyncio
async def test_preview_invalid_json_shows_error_and_disables_apply():
    app = ImportApp()
    async with app.run_test() as pilot:
        app.query_one("#mcp-import-text", TextArea).text = "{nope"
        await pilot.click("#mcp-import-preview")
        await pilot.pause()
        error_text = str(app.query_one("#mcp-import-error", Static).renderable)
        assert "Not valid JSON" in error_text
        assert app.query_one("#mcp-import-apply", Button).disabled
        assert not list(app.query("#mcp-import-list Static"))


@pytest.mark.asyncio
async def test_import_error_static_carries_status_error_class():
    """A2: `#mcp-import-error` must carry `mcp-status-error` at compose
    time, mirroring `#mcp-form-error`, so an invalid-JSON preview error
    renders in the error color instead of plain body text."""
    app = ImportApp()
    async with app.run_test() as pilot:
        error = app.query_one("#mcp-import-error", Static)
        assert "mcp-status-error" in error.classes


@pytest.mark.asyncio
async def test_apply_posts_import_requested_with_previewed_candidates_and_disables():
    app = ImportApp()
    async with app.run_test() as pilot:
        text = json.dumps({"mcpServers": {"docs": {"command": "npx", "args": ["-y", "pkg"]}}})
        app.query_one("#mcp-import-text", TextArea).text = text
        await pilot.click("#mcp-import-preview")
        await pilot.pause()
        await pilot.click("#mcp-import-apply")
        await pilot.pause()
        assert app.events
        event = app.events[-1]
        assert len(event.candidates) == 1
        assert event.candidates[0].profile_id == "docs"
        assert event.candidates[0].to_payload() == {
            "profile_id": "docs", "command": "npx", "args": ["-y", "pkg"],
            "env_placeholders": {}, "env_literals": {},
        }
        assert app.query_one("#mcp-import-apply", Button).disabled


@pytest.mark.asyncio
async def test_file_button_posts_file_requested():
    app = ImportApp()
    async with app.run_test() as pilot:
        await pilot.click("#mcp-import-file")
        await pilot.pause()
        assert app.events and isinstance(app.events[-1], MCPImportPanel.FileRequested)


@pytest.mark.asyncio
async def test_set_file_text_populates_textarea():
    app = ImportApp()
    async with app.run_test() as pilot:
        panel = app.query_one(MCPImportPanel)
        panel.set_file_text('{"mcpServers": {"docs": {"command": "npx"}}}')
        await pilot.pause()
        assert app.query_one("#mcp-import-text", TextArea).text == (
            '{"mcpServers": {"docs": {"command": "npx"}}}'
        )


@pytest.mark.asyncio
async def test_cancel_button_posts_cancelled():
    app = ImportApp()
    async with app.run_test() as pilot:
        await pilot.click("#mcp-import-cancel")
        await pilot.pause()
        assert app.events and isinstance(app.events[-1], MCPImportPanel.Cancelled)
