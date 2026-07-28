from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.binding import Binding

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook import app as app_module
from tldw_chatbook.UI.Workbench.focus import WorkbenchFocusRegistry
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel, WorkbenchHelpState
from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchAction


@pytest.fixture(autouse=True)
def _disable_full_app_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


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


def test_help_state_renders_grouped_shortcuts():
    """TASK-362: a grouped keyboard map renders group headers with their keys,
    replacing the flat shortcut list."""
    help_state = WorkbenchHelpState(
        route_id="chat",
        title="Console",
        shortcut_groups=(
            ("Transcript", (("j / k", "select"), ("c", "copy"))),
            ("Composer", (("Shift+Enter", "newline"),)),
        ),
    )

    rendered = help_state.render_text()

    assert "Transcript:" in rendered
    assert "j / k" in rendered and "select" in rendered
    assert "c" in rendered and "copy" in rendered
    assert "Composer:" in rendered and "Shift+Enter" in rendered


def test_console_help_map_covers_the_full_keyboard_vocabulary():
    """TASK-362: the Console F1 map must surface the transcript j/k/c/e/r keys,
    Shift+Enter, Alt+M, F2 and Escape (previously undiscoverable anywhere),
    grouped by surface."""
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_WORKBENCH_SHORTCUT_GROUPS,
    )

    rendered = WorkbenchHelpState(
        route_id="chat",
        title="Console",
        shortcut_groups=CONSOLE_WORKBENCH_SHORTCUT_GROUPS,
    ).render_text()

    for token in (
        "Panes:",
        "Transcript:",
        "Composer:",
        "j / k",
        "c",
        "e",
        "r",
        "Shift+Enter",
        "Alt+M",
        "F2",
        "Escape",
    ):
        assert token in rendered, token


@pytest.mark.asyncio
async def test_help_panel_renders_body_and_close_button():
    app = _build_test_app()
    app.app_config["_first_run"] = False

    async with app.run_test(size=(80, 20)) as pilot:
        initial_depth = len(app.screen_stack)
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

        assert len(app.screen_stack) == initial_depth


@pytest.mark.asyncio
async def test_help_panel_escape_dismisses():
    app = _build_test_app()
    app.app_config["_first_run"] = False

    async with app.run_test(size=(80, 20)) as pilot:
        initial_depth = len(app.screen_stack)
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
        assert len(app.screen_stack) == initial_depth + 1

        await pilot.press("escape")
        await pilot.pause()

        assert len(app.screen_stack) == initial_depth


@pytest.mark.asyncio
async def test_generic_help_fallback_lists_screen_bindings():
    """Screens without a custom handler get help generated from their BINDINGS."""
    class BareScreen:
        BINDINGS = [
            Binding("ctrl+s", "send", "Send message"),
            ("ctrl+n", "new_note", "New note"),
        ]

    pushed = []
    function_context = SimpleNamespace(
        screen=BareScreen(),
        current_tab="library",
        push_screen=pushed.append,
    )

    app_module.TldwCli._show_generic_screen_help(function_context)

    assert len(pushed) == 1
    panel = pushed[0]
    assert isinstance(panel, WorkbenchHelpPanel)
    assert panel.state.route_id == "library"
    assert ("ctrl+s", "Send message") in panel.state.shortcuts
    assert ("ctrl+n", "New note") in panel.state.shortcuts


@pytest.mark.asyncio
async def test_generic_help_fallback_uses_app_bindings_when_screen_has_none():
    """A screen with no BINDINGS still gets truthful help from the app layer."""
    app = _build_test_app()
    app.app_config["_first_run"] = False

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        app.screen.BINDINGS = []

        app._show_generic_screen_help()
        await pilot.pause()

        panel = app.screen
        assert isinstance(panel, WorkbenchHelpPanel)
        assert panel.state.shortcuts
        assert panel.state.shortcuts == app_module._bindings_to_shortcuts(
            type(app).BINDINGS
        )


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
