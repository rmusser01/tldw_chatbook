"""Task-264: every BaseAppScreen must carry its OWN AppFooterStatus instance.

`AppFooterStatus` used to be mounted once on the App's DEFAULT screen
(app.py's own `compose()`), which is occluded the instant any `BaseAppScreen`
is pushed on top -- `App.query_one`/`query` always resolve against
`App.default_screen` by design (see `App._get_dom_base`), so
`self.app.query_one(AppFooterStatus)` from within a pushed screen silently
updated an invisible widget and every `set_workbench_shortcuts()`
registration was a no-op the user could never see.

The fix: `BaseAppScreen.compose()` now yields its own `AppFooterStatus`, and
callers resolve it through the screen (`self.query_one(...)`) instead of the
app. These tests pin that contract directly against the real screens/
registration methods, not a hand-rolled fake.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


class _MinimalScreen(BaseAppScreen):
    """The lightest possible BaseAppScreen subclass -- just enough content
    to mount without pulling in a real destination's dependencies."""

    def compose_content(self) -> ComposeResult:
        yield Static("minimal screen content")


class _MinimalScreenHost(App):
    """Hosts a bare BaseAppScreen subclass with no App-level footer of its
    own, so the only AppFooterStatus in the tree is the one the screen
    itself composes."""

    async def on_mount(self) -> None:
        await self.push_screen(_MinimalScreen(None, "minimal"))


@pytest.mark.asyncio
async def test_base_app_screen_composes_footer_status():
    """Every BaseAppScreen carries its own AppFooterStatus instance."""
    host = _MinimalScreenHost()

    async with host.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]

        footer = screen.query_one(AppFooterStatus)

        assert footer.id == "screen-footer-status"
        assert footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT


class _DefaultScreenFooterHost(App):
    """Mirrors app.py's real shape: an `AppFooterStatus` composed directly
    on the App's own DEFAULT screen (id="app-footer-status", exactly like
    `TldwCli._create_main_ui_widgets`), with a real destination screen
    pushed on top of it. Before task-264 this was the ONLY footer in the
    tree, and it is what `self.app.query_one(AppFooterStatus)` used to
    (mis)resolve to from inside the pushed screen.
    """

    def __init__(self, app_instance, screen_factory):
        super().__init__()
        self.app_instance = app_instance
        self._screen_factory = screen_factory

    def compose(self) -> ComposeResult:
        yield AppFooterStatus(id="app-footer-status")

    async def on_mount(self) -> None:
        await self.push_screen(self._screen_factory(self.app_instance))


@pytest.mark.asyncio
async def test_console_registration_updates_the_screens_own_footer():
    """chat_screen's registration must land on ITS instance, not the app's
    default-screen one; text contains 'F6' and 'Ctrl+K'."""
    app_instance = _build_test_app()
    host = _DefaultScreenFooterHost(app_instance, ChatScreen)

    async with host.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]

        # Real registration path: ChatScreen.on_mount() already called
        # _register_console_footer_shortcuts() during push_screen above.
        screen_footer = screen.query_one(AppFooterStatus)
        assert screen_footer.id == "screen-footer-status"
        assert "F6" in screen_footer.shortcut_text
        assert "Ctrl+K" in screen_footer.shortcut_text

        # The app's default-screen footer (what `self.app.query_one(...)`
        # used to target) must be left untouched at its default text.
        default_screen_footer = host.query_one(AppFooterStatus)
        assert default_screen_footer is not screen_footer
        assert default_screen_footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT


@pytest.mark.asyncio
async def test_mcp_registration_updates_the_screens_own_footer():
    """mcp_screen registration -> its footer text contains 'mode' and
    'a add server'."""
    app_instance = _build_test_app()
    host = _DefaultScreenFooterHost(app_instance, MCPScreen)

    async with host.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]

        # Real registration path: MCPScreen.on_mount() already called
        # _register_footer_shortcuts() during push_screen above.
        screen_footer = screen.query_one(AppFooterStatus)
        assert screen_footer.id == "screen-footer-status"
        assert "mode" in screen_footer.shortcut_text
        assert "a add server" in screen_footer.shortcut_text

        default_screen_footer = host.query_one(AppFooterStatus)
        assert default_screen_footer is not screen_footer
        assert default_screen_footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT


@pytest.mark.asyncio
async def test_registration_survives_screen_recompose():
    """Screen-level recompose (settings' recompose=True reactives, library/
    chat `refresh(recompose=True)`) replaces the footer widget; the
    persisted registration must re-seed the fresh instance (task-264
    review)."""
    host = _MinimalScreenHost()

    async with host.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]
        screen.register_footer_shortcuts(
            source="minimal", shortcuts=(("x", "do the thing"),)
        )
        footer_before = screen.query_one(AppFooterStatus)
        assert "do the thing" in footer_before.shortcut_text

        screen.refresh(recompose=True)
        await pilot.pause()

        footer_after = screen.query_one(AppFooterStatus)
        assert footer_after is not footer_before
        assert "do the thing" in footer_after.shortcut_text

        # And a source-guarded clear drops both the live text and the
        # persisted copy, so a later recompose stays at the default.
        screen.clear_footer_shortcuts(source="minimal")
        screen.refresh(recompose=True)
        await pilot.pause()
        footer_cleared = screen.query_one(AppFooterStatus)
        assert footer_cleared.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT


@pytest.mark.asyncio
async def test_library_registration_updates_the_screens_own_footer():
    """task-264 review Important: Library's `u` hint (previously rendered by
    the retired Textual Footer's show=True binding) must reach its own
    footer via the registration API."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app_instance = _build_test_app()
    host = _DefaultScreenFooterHost(app_instance, LibraryScreen)

    async with host.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]
        screen_footer = screen.query_one(AppFooterStatus)
        assert "u" in screen_footer.shortcut_text
        assert "use Library context in Console" in screen_footer.shortcut_text


@pytest.mark.asyncio
async def test_settings_registration_updates_the_screens_own_footer():
    """task-264 review Important: Settings' s/r/t hints (previously rendered
    by the retired Footer) must reach its own footer via registration."""
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    app_instance = _build_test_app()
    host = _DefaultScreenFooterHost(app_instance, SettingsScreen)

    async with host.run_test(size=(160, 48)) as pilot:
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        screen = host.screen_stack[-1]
        screen_footer = screen.query_one(AppFooterStatus)
        for token in ("save category", "revert category", "test category"):
            assert token in screen_footer.shortcut_text
