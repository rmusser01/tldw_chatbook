from __future__ import annotations

import pytest
from textual.widgets import Input, TabbedContent

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_INGEST,
    LIBRARY_NAV_CONTEXT_MODE,
    TAB_EVALS,
    TAB_LIBRARY,
    TAB_MCP,
    TAB_SEARCH,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_INGEST_MEDIA,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen
from tldw_chatbook.UI.Screens.search_screen import SearchScreen


REMOVED_ROOT_NAMES = (
    "current_selected_note_id",
    "current_selected_note_version",
    "current_selected_note_title",
    "current_selected_note_content",
    "notes_sort_by",
    "notes_sort_ascending",
    "notes_preview_mode",
    "notes_auto_save_enabled",
    "notes_auto_save_timer",
    "notes_last_save_time",
    "search_active_sub_tab",
    "ingest_active_view",
    "tools_settings_active_view",
    "evals_sidebar_collapsed",
    "_notes_search_timer",
    "_initial_search_sub_tab_view",
    "_initial_ingest_view",
    "_initial_tools_settings_view",
    "_activate_initial_ingest_view",
)


def _disable_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


def _production_app(monkeypatch: pytest.MonkeyPatch) -> TldwCli:
    _disable_splash(monkeypatch)
    app = TldwCli()
    app.app_config["_first_run"] = False
    return app


def _assert_removed_root_state_absent(app: TldwCli) -> None:
    assert all(name not in vars(TldwCli) for name in REMOVED_ROOT_NAMES)
    assert all(not hasattr(app, name) for name in REMOVED_ROOT_NAMES)


async def _wait_for_screen(
    app: TldwCli,
    pilot,
    screen_type: type,
    canonical_tab: str,
    previous_screen=None,
):
    for _ in range(400):
        if (
            type(app.screen) is screen_type
            and app.current_tab == canonical_tab
            and app.screen is not previous_screen
        ):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(
        f"production TldwCli did not route to exact {screen_type.__name__}"
    )


async def _drive_affected_routes(
    app: TldwCli,
    pilot,
    *,
    query: str,
    search_tab: str,
    mcp_mode: str,
    eval_id: str,
    expected_initial_search: tuple[str, str] | None = None,
) -> None:
    app.post_message(NavigateToScreen("library"))
    library = await _wait_for_screen(app, pilot, LibraryScreen, TAB_LIBRARY)
    library.apply_navigation_context({LIBRARY_NAV_CONTEXT_MODE: "notes"})
    await pilot.pause()
    assert library._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
    _assert_removed_root_state_absent(app)

    app.post_message(NavigateToScreen("notes"))
    notes = await _wait_for_screen(
        app,
        pilot,
        LibraryScreen,
        TAB_LIBRARY,
        previous_screen=library,
    )
    assert notes is not library
    assert notes._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
    _assert_removed_root_state_absent(app)

    app.post_message(NavigateToScreen("search"))
    search = await _wait_for_screen(app, pilot, SearchScreen, TAB_SEARCH)
    search_query = search.query_one("#search-query-input", Input)
    search_tabs = search.query_one("#search-tabs", TabbedContent)
    if expected_initial_search is not None:
        expected_query, expected_tab = expected_initial_search
        assert search_query.value == expected_query
        assert search_tabs.active == expected_tab
    search_query.value = query
    search_tabs.active = search_tab
    await pilot.pause()
    assert search_query.value == query
    assert search_tabs.active == search_tab
    _assert_removed_root_state_absent(app)

    app.post_message(NavigateToScreen("ingest"))
    ingest = await _wait_for_screen(app, pilot, LibraryScreen, TAB_LIBRARY)
    ingest.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    await pilot.pause()
    assert ingest._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
    _assert_removed_root_state_absent(app)

    app.post_message(NavigateToScreen("mcp"))
    mcp = await _wait_for_screen(app, pilot, MCPScreen, TAB_MCP)
    assert mcp.workbench is not None
    mcp.action_mcp_mode(mcp_mode)
    await pilot.pause()
    assert mcp.workbench.active_mode == mcp_mode
    _assert_removed_root_state_absent(app)

    app.post_message(NavigateToScreen("tools_settings"))
    tools_settings = await _wait_for_screen(
        app,
        pilot,
        MCPScreen,
        TAB_MCP,
        previous_screen=mcp,
    )
    assert tools_settings is not mcp
    assert tools_settings.workbench is not None
    for _ in range(400):
        if tools_settings.workbench.active_mode == mcp_mode:
            break
        await pilot.pause(0.01)
    assert tools_settings.workbench.active_mode == mcp_mode
    _assert_removed_root_state_absent(app)

    app.post_message(NavigateToScreen("evals"))
    evals = await _wait_for_screen(app, pilot, EvalsScreen, TAB_EVALS)
    evals.select(kind="classic", id=eval_id)
    await pilot.pause()
    assert evals._selection.kind == "classic"
    assert evals._selection.id == eval_id
    _assert_removed_root_state_absent(app)

    app.post_message(NavigateToScreen("search"))
    restored_search = await _wait_for_screen(app, pilot, SearchScreen, TAB_SEARCH)
    assert restored_search is not search
    assert restored_search.query_one("#search-query-input", Input).value == query
    assert restored_search.query_one("#search-tabs", TabbedContent).active == search_tab
    _assert_removed_root_state_absent(app)


@pytest.mark.asyncio
async def test_registered_destinations_own_state_without_retired_root_mirrors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_routes = {
        "library": ("library", TAB_LIBRARY, LibraryScreen),
        "notes": ("library", TAB_LIBRARY, LibraryScreen),
        "search": ("search", TAB_SEARCH, SearchScreen),
        "ingest": ("library", TAB_LIBRARY, LibraryScreen),
        "mcp": ("mcp", TAB_MCP, MCPScreen),
        "tools_settings": ("tools_settings", TAB_MCP, MCPScreen),
        "evals": ("evals", TAB_EVALS, EvalsScreen),
    }
    for route, expected in expected_routes.items():
        assert resolve_screen_target(route) == expected

    app = _production_app(monkeypatch)
    _assert_removed_root_state_absent(app)
    async with app.run_test(size=(180, 55)) as pilot:
        await _drive_affected_routes(
            app,
            pilot,
            query="TASK-904 destination-owned query",
            search_tab="saved-tab",
            mcp_mode="tools",
            eval_id="task-904-destination-owned",
        )

    fresh_app = _production_app(monkeypatch)
    _assert_removed_root_state_absent(fresh_app)
    async with fresh_app.run_test(size=(180, 55)) as fresh_pilot:
        await _drive_affected_routes(
            fresh_app,
            fresh_pilot,
            query="TASK-904 fresh destination query",
            search_tab="history-tab",
            mcp_mode="permissions",
            eval_id="task-904-fresh-destination-owned",
            expected_initial_search=("", "search-tab"),
        )
