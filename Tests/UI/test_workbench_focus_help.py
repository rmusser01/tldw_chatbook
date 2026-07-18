from pathlib import Path
from types import SimpleNamespace

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
    assert registry.previous_before(None, hidden={"inspector"}) == "composer"
    assert registry.previous_before("context", hidden={"inspector"}) == "composer"
    assert registry.previous_before("composer", hidden={"inspector"}) == "transcript"
    assert registry.previous_before("context", hidden=set(registry.pane_order)) is None


def test_workbench_css_contains_normal_and_compact_density():
    css = Path("tldw_chatbook/css/components/_workbench.tcss").read_text()

    assert ".density-normal" in css
    assert ".density-compact" in css


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


@pytest.mark.asyncio
async def test_help_panel_escape_dismisses():
    app = _WorkbenchHelpPanelApp()

    async with app.run_test(size=(80, 20)) as pilot:
        app.push_screen(
            WorkbenchHelpPanel(
                WorkbenchHelpState(
                    route_id="chat",
                    title="Console",
                    shortcuts=(("F1", "help"),),
                )
            )
        )
        await pilot.pause()
        assert len(app.screen_stack) == 2

        await pilot.press("escape")
        await pilot.pause()

        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_generic_help_fallback_lists_screen_bindings():
    """Screens without a custom handler get help generated from their BINDINGS."""
    from textual.binding import Binding

    from tldw_chatbook import app as app_module

    class BareScreen:
        BINDINGS = [
            Binding("ctrl+s", "send", "Send message"),
            ("ctrl+n", "new_note", "New note"),
        ]

    class FakeApp:
        # Bind the real fallback builder so the action's self-call resolves.
        _show_generic_screen_help = app_module.TldwCli._show_generic_screen_help

        def __init__(self) -> None:
            self.screen = BareScreen()
            self.current_tab = "library"
            self.pushed: list = []

        def push_screen(self, panel) -> None:
            self.pushed.append(panel)

        def notify(self, *_args, **_kwargs) -> None:
            raise AssertionError("generic fallback must not toast the dead message")

    fake_app = FakeApp()

    await app_module.TldwCli.action_show_workbench_help(fake_app)

    assert len(fake_app.pushed) == 1
    panel = fake_app.pushed[0]
    assert isinstance(panel, WorkbenchHelpPanel)
    assert panel.state.route_id == "library"
    assert ("ctrl+s", "Send message") in panel.state.shortcuts
    assert ("ctrl+n", "New note") in panel.state.shortcuts


@pytest.mark.asyncio
async def test_generic_help_fallback_uses_app_bindings_when_screen_has_none():
    """A screen with no BINDINGS still gets truthful help from the app layer."""
    from textual.binding import Binding

    from tldw_chatbook import app as app_module

    class BareScreen:
        BINDINGS: list = []

    class FakeApp:
        BINDINGS = [Binding("ctrl+q", "quit", "Quit App")]
        # Bind the real fallback builder so the action's self-call resolves.
        _show_generic_screen_help = app_module.TldwCli._show_generic_screen_help

        def __init__(self) -> None:
            self.screen = BareScreen()
            self.current_tab = ""
            self.pushed: list = []

        def push_screen(self, panel) -> None:
            self.pushed.append(panel)

        def notify(self, *_args, **_kwargs) -> None:
            raise AssertionError("generic fallback must not toast the dead message")

    fake_app = FakeApp()

    await app_module.TldwCli.action_show_workbench_help(fake_app)

    assert len(fake_app.pushed) == 1
    panel = fake_app.pushed[0]
    assert ("ctrl+q", "Quit App") in panel.state.shortcuts


@pytest.mark.asyncio
async def test_app_workbench_delegation_awaits_async_screen_actions():
    from tldw_chatbook import app as app_module

    calls: list[str] = []

    class AsyncScreen:
        async def action_show_workbench_help(self) -> None:
            calls.append("help")

        async def action_focus_next_workbench_pane(self) -> None:
            calls.append("focus")

    app = SimpleNamespace(
        screen=AsyncScreen(),
        notify=lambda *_args, **_kwargs: None,
    )

    result = app_module.TldwCli.action_show_workbench_help(app)
    if hasattr(result, "__await__"):
        await result
    result = app_module.TldwCli.action_focus_next_workbench_pane(app)
    if hasattr(result, "__await__"):
        await result

    assert calls == ["help", "focus"]
