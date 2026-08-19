"""Focus mode (task-16320, ADR-067) — config, CLI, and behavior tests."""

import io
from types import SimpleNamespace

import pytest

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.UI.test_destination_shells import _build_test_app

from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME


class TestFocusCliAndConfig:
    def test_arg_parser_accepts_focus_flag(self):
        from tldw_chatbook.app import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(["--focus"])
        assert args.focus is True
        args = parser.parse_args([])
        assert args.focus is False

    def test_config_template_declares_focus_mode(self):
        from tldw_chatbook.config import CONFIG_TOML_CONTENT

        general_block = CONFIG_TOML_CONTENT.split("[general]")[1].split("\n[")[0]
        assert "focus_mode = false" in general_block


class TestInitialRouteResolution:
    """Unbound-method tests: _resolve_initial_shell_route reads only
    app_config / _initial_tab_value / the focus attrs, so a stub works."""

    @staticmethod
    def _stub(**overrides):
        stub = SimpleNamespace(
            app_config={"_first_run": False},
            _initial_tab_value="notes",
            _cli_focus_override=False,
            _focus_mode_config=False,
            focus_mode=False,
        )
        for key, value in overrides.items():
            setattr(stub, key, value)
        return stub

    @pytest.fixture(autouse=True)
    def _wizard_off(self, monkeypatch):
        monkeypatch.setattr(
            "tldw_chatbook.UI.Wizards.first_run_setup_state.setup_recovery_action",
            lambda cfg, env: "skip",
        )

    def test_cli_focus_override_forces_chat(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(_cli_focus_override=True)
        assert TldwCli._resolve_initial_shell_route(stub) == TAB_CHAT
        assert stub.focus_mode is True

    def test_config_focus_mode_forces_chat(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(_focus_mode_config=True, _initial_tab_value="notes")
        assert TldwCli._resolve_initial_shell_route(stub) == TAB_CHAT
        assert stub.focus_mode is True

    def test_cli_flag_wins_over_false_config(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(_cli_focus_override=True)
        assert TldwCli._resolve_initial_shell_route(stub) == TAB_CHAT

    def test_no_focus_respects_default_tab(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(_initial_tab_value="notes")
        assert TldwCli._resolve_initial_shell_route(stub) == "notes"
        assert stub.focus_mode is False

    def test_first_run_onboarding_beats_focus(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(
            _cli_focus_override=True, app_config={"_first_run": True}
        )
        assert TldwCli._resolve_initial_shell_route(stub) == TAB_HOME
        assert stub.focus_mode is False


from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


class FocusConsoleHarness(ConsolidatedCSSApp):
    """Mounts the real ChatScreen against a fake app, like the neighboring
    console contract harnesses do (see test_console_workbench_contract.py)."""

    # The focus rules live in the app bundle, which ConsolidatedCSSApp does
    # not load by default -- mirror the contract harnesses that assert
    # CSS-driven display (ConsolidatedCSSApp brackets this with the screen
    # sheets in production order).
    CSS_PATH = str(BUNDLED_STYLESHEET)

    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _make_app_instance(focus: bool):
    app_instance = _build_test_app()
    app_instance.focus_mode = focus
    return app_instance


#: Tall enough that `-console-compact` (height < 35 rows, TASK-346) never
#: engages — at run_test's 80x24 default it hides the workbench header
#: regardless of focus mode, masking the rules under test.
_FOCUS_TERMINAL_SIZE = (120, 40)


class TestFocusChromeSuppression:
    async def test_focus_mode_hides_chrome_keeps_status_line(self):
        app_instance = _make_app_instance(focus=True)
        harness = FocusConsoleHarness(app_instance)
        async with harness.run_test(size=_FOCUS_TERMINAL_SIZE) as pilot:
            screen = pilot.app.screen
            assert screen.has_class("-focus")
            assert screen.query_one(MainNavigationBar).display is False
            header = screen.query_one("#console-workbench-header")
            assert header.display is False
            # One-line status bar is KEPT (owner decision, ADR-067).
            footer = screen.query_one("#screen-footer-status", AppFooterStatus)
            assert footer.display is not False

    async def test_default_mount_shows_all_chrome(self):
        app_instance = _make_app_instance(focus=False)
        harness = FocusConsoleHarness(app_instance)
        async with harness.run_test(size=_FOCUS_TERMINAL_SIZE) as pilot:
            screen = pilot.app.screen
            assert not screen.has_class("-focus")
            assert screen.query_one(MainNavigationBar).display is not False
            header = screen.query_one("#console-workbench-header")
            assert header.display is not False
            footer = screen.query_one("#screen-footer-status", AppFooterStatus)
            assert footer.display is not False

    async def test_apply_focus_chrome_flips_in_place(self):
        app_instance = _make_app_instance(focus=False)
        harness = FocusConsoleHarness(app_instance)
        async with harness.run_test(size=_FOCUS_TERMINAL_SIZE) as pilot:
            screen = pilot.app.screen
            assert not screen.has_class("-focus")
            app_instance.focus_mode = True
            screen._apply_focus_chrome()
            assert screen.has_class("-focus")
            assert screen.query_one(MainNavigationBar).display is False
            app_instance.focus_mode = False
            screen._apply_focus_chrome()
            assert not screen.has_class("-focus")
            assert screen.query_one(MainNavigationBar).display is not False
