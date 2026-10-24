"""A malformed import remains editable under the installed keepalive policy."""

import json

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("args", 42, id="args-number"),
        pytest.param("env", ["bad"], id="env-list"),
        pytest.param("args", {}, id="args-empty-object"),
        pytest.param("env", [], id="env-empty-list"),
    ],
)
@private_profile_test
async def test_preview_rejects_malformed_fields_and_recovers(
    request, tmp_path, field, value
):
    from textual.app import App
    from textual.widgets import Button, Static, TextArea

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.app_lifecycle import LifecycleMixin
    from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel

    class ImportHost(LifecycleMixin, ConsolidatedCSSApp):
        # This host owns only the real panel, not TldwCli's service lifecycles.
        # Keep native App cleanup and the unmodified production error policy.
        _shutdown = App._shutdown
        _keep_screen_alive_on_handler_error = True

        def __init__(self):
            super().__init__()
            self.errors = []
            self.imports = []

        def compose(self):
            yield MCPImportPanel()

        def on_unmount(self, event):
            event.prevent_default()  # No TldwCli services exist to tear down.

        def _handle_exception(self, error):
            self.errors.append(error)
            super()._handle_exception(error)

        def on_mcp_import_panel_import_requested(self, event):
            self.imports.append(event.candidates)

    app = ImportHost()
    async with app.run_test(size=(140, 55)) as pilot:
        await pilot.pause()
        panel = app.query_one(MCPImportPanel)
        editor = panel.query_one("#mcp-import-text", TextArea)
        preview = panel.query_one("#mcp-import-preview", Button)
        apply_button = panel.query_one("#mcp-import-apply", Button)
        error = panel.query_one("#mcp-import-error", Static)

        editor.text = '{"mcpServers": {"stale": {"command": "npx"}}}'
        preview.press()
        await pilot.pause()
        assert [candidate.profile_id for candidate in panel._candidates] == ["stale"]
        assert len(panel.query("#mcp-import-list Static")) == 1
        assert not apply_button.disabled

        rejected = json.dumps(
            {"mcpServers": {"broken": {"command": "npx", field: value}}}
        )
        editor.text = rejected
        preview.press()
        await pilot.pause()
        assert app.errors == []
        assert panel.is_attached and panel.is_running
        assert editor.is_attached and not editor.read_only
        assert editor.text == rejected
        assert field in str(error.renderable)
        assert panel._candidates == []
        assert not list(panel.query("#mcp-import-list Static"))
        assert apply_button.disabled
        assert app._exception is None
        # Even a Pressed already queued before disabling must not submit stale
        # candidates. Use the real message handler, not a direct _preview call.
        panel.post_message(Button.Pressed(apply_button))
        await pilot.pause()
        assert app.imports == []

        editor.focus()
        editor.move_cursor((0, len(rejected)))
        await pilot.press("space")
        assert editor.text == rejected + " "
        editor.text = (
            '{"mcpServers": {"corrected": {"command": "uvx", "args": [], "env": {}}}}'
        )
        preview.press()
        await pilot.pause()
        assert panel.is_attached and panel.is_running
        assert [candidate.profile_id for candidate in panel._candidates] == [
            "corrected"
        ]
        assert "corrected" in str(
            panel.query_one("#mcp-import-list Static", Static).renderable
        )
        assert str(error.renderable) == ""
        assert not apply_button.disabled
        apply_button.press()
        await pilot.pause()
        assert [[c.profile_id for c in batch] for batch in app.imports] == [
            ["corrected"]
        ]
        assert app.errors == []
        assert app._exception is None
