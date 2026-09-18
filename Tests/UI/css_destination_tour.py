"""Route-confirmed destination tour shared by CSS source-budget probes."""

import asyncio

# These are actual destination bodies, not shared shell navigation/chrome.
DESTINATION_BODIES = {
    "home": "#home-canvas",
    "console": "#console-native-composer",
    "library": "#library-landing-canvas",
    "personas": "#personas-detail-stack",
    "watchlists_collections": "#wl-workbench-body",
    "artifacts": "#artifacts-detail-pane",
    "schedules": "#scheduling-task-table",
    "workflows": "#workflows-editor",
    "mcp": "#mcp-mode-canvas-servers",
    "acp": "#acp-detail-pane",
    "lab": "#lab-body LLMManagementWindow",
    "logs": "#logs-window",
    "settings": "#settings-detail-pane",
    "research": "#research-chat-pane",
    "meetings": "#meetings-transcript",
}


def build_css_tour_app():
    """Disable startup overlays in the caller's private test profile."""
    from tldw_chatbook.config import save_setting_to_cli_config

    save_setting_to_cli_config("splash_screen", "enabled", False)
    from Tests.UI.app_factory import _build_test_app

    return _build_test_app()


async def visit_all_shell_destinations(app, pilot):
    """Press every shell shortcut and confirm its mounted destination."""
    from tldw_chatbook.UI.Navigation.shell_destinations import (
        SHELL_DESTINATION_SHORTCUTS,
        resolve_shell_route,
    )

    async def wait_until(predicate, description):
        async with asyncio.timeout(15):
            while not predicate():
                await pilot.pause()
                await asyncio.sleep(0.05)
        await pilot.pause()
        assert predicate(), description

    await wait_until(
        lambda: getattr(app, "_initial_screen_pushed", False),
        "initial screen was not pushed",
    )
    assert set(DESTINATION_BODIES) == set(SHELL_DESTINATION_SHORTCUTS)
    visited = []
    for destination, shortcut in SHELL_DESTINATION_SHORTCUTS.items():
        await pilot.press(shortcut)

        def arrived(destination=destination):
            route = getattr(app.screen, "screen_name", "")
            return (
                resolve_shell_route(route).destination_id == destination
                and app.screen.is_mounted
                and any(
                    body.is_mounted
                    and (
                        destination != "settings"
                        or any(child.is_mounted for child in body.children)
                    )
                    for body in app.screen.query(DESTINATION_BODIES[destination])
                )
            )

        await wait_until(arrived, f"{shortcut} did not mount {destination}")
        visited.append(destination)
    assert visited == list(SHELL_DESTINATION_SHORTCUTS)
    return visited
