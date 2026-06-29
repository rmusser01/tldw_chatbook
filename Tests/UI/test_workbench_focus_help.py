import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Workbench.focus import WorkbenchFocusRegistry
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel, WorkbenchHelpState
from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchAction


def test_focus_registry_cycles_visible_panes_only():
    registry = WorkbenchFocusRegistry(
        ("context", "transcript", "inspector", "composer")
    )

    assert registry.next_after(None, hidden={"inspector"}) == "context"
    assert registry.next_after("missing", hidden={"inspector"}) == "context"
    assert registry.next_after("context", hidden={"inspector"}) == "transcript"
    assert registry.next_after("transcript", hidden={"inspector"}) == "composer"
    assert registry.next_after("composer", hidden={"inspector"}) == "context"
    assert registry.next_after("context", hidden=set(registry.pane_order)) is None


def test_help_state_lists_visible_actions_not_palette_only():
    help_state = WorkbenchHelpState(
        route_id="chat",
        title="Console",
        actions=(
            WorkbenchAction(id="settings", label="Settings"),
            WorkbenchAction(id="send", label="Send"),
            WorkbenchAction(id="hidden", label="Hidden", disabled=True),
        ),
        shortcuts=(("F6", "next pane"), ("F1", "help")),
    )

    rendered = help_state.render_text()

    assert "Console" in rendered
    assert "Settings" in rendered
    assert "Send" in rendered
    assert "Hidden" not in rendered
    assert "F6" in rendered
    assert "next pane" in rendered
    assert "F1" in rendered
    assert "help" in rendered
    assert "Ctrl+P" not in rendered


class _WorkbenchHelpPanelApp(App[None]):
    def compose(self) -> ComposeResult:
        yield from ()


@pytest.mark.asyncio
async def test_help_panel_renders_body_and_close_button():
    app = _WorkbenchHelpPanelApp()

    async with app.run_test(size=(80, 20)) as pilot:
        app.push_screen(
            WorkbenchHelpPanel(
                WorkbenchHelpState(
                    route_id="chat",
                    title="Console",
                    actions=(WorkbenchAction(id="send", label="Send"),),
                    shortcuts=(("F1", "help"),),
                )
            )
        )
        await pilot.pause()

        assert app.screen.query_one("#workbench-help-panel")
        rendered = app.screen.query_one("#workbench-help-body").renderable
        assert "Send" in str(rendered)

        await pilot.click("#workbench-help-close")
        await pilot.pause()

        assert len(app.screen_stack) == 1
