"""Focus mode (task-18812, ADR-071) — config, CLI, and behavior tests."""

from types import SimpleNamespace

import pytest

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.UI.test_destination_shells import _build_test_app

from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME
from tldw_chatbook.UI.Navigation.main_navigation import (
    MainNavigationBar,
    NavigateToScreen,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


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
            _deferred_focus_request=False,
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

    def test_first_run_onboarding_beats_focus_but_defers_request(self):
        """Qodo finding 3: first-run still routes to Home, but the focus
        request is recorded for the wizard's completion, not discarded."""
        from tldw_chatbook.app import TldwCli

        stub = self._stub(_cli_focus_override=True, app_config={"_first_run": True})
        assert TldwCli._resolve_initial_shell_route(stub) == TAB_HOME
        assert stub.focus_mode is False
        assert stub._deferred_focus_request is True

    def test_wizard_recovery_offer_also_defers_focus_request(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(_cli_focus_override=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "tldw_chatbook.UI.Wizards.first_run_setup_state.setup_recovery_action",
                lambda cfg, env: "offer",
            )
            assert TldwCli._resolve_initial_shell_route(stub) == TAB_HOME
        assert stub.focus_mode is False
        assert stub._deferred_focus_request is True

    def test_no_focus_no_deferral(self):
        from tldw_chatbook.app import TldwCli

        stub = self._stub(app_config={"_first_run": True})
        assert TldwCli._resolve_initial_shell_route(stub) == TAB_HOME
        assert stub._deferred_focus_request is False


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


#: Representative desktop size for checking the focus-mode chrome contract.
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
            assert screen.query_one("#console-speech-controls").parent is header
            # One-line status bar is KEPT (owner decision, ADR-071).
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


class TestFocusFooterHint:
    async def test_footer_advertises_toggle_in_both_states(self):
        app_instance = _make_app_instance(focus=True)
        harness = FocusConsoleHarness(app_instance)
        async with harness.run_test(size=_FOCUS_TERMINAL_SIZE) as pilot:
            screen = pilot.app.screen
            source, shortcuts = screen._footer_shortcut_registration
            assert source == "console"
            assert ("Ctrl+Shift+F", "exit focus") in shortcuts

            app_instance.focus_mode = False
            screen._register_console_footer_shortcuts()
            _, shortcuts = screen._footer_shortcut_registration
            assert ("Ctrl+Shift+F", "focus") in shortcuts
            assert ("Ctrl+Shift+F", "exit focus") not in shortcuts


class TestAppToggleAndNavigationExit:
    def test_ctrl_shift_f_binding_registered(self):
        from tldw_chatbook.app import TldwCli

        assert any(binding.key == "ctrl+shift+f" for binding in TldwCli.BINDINGS)

    def test_set_focus_mode_applies_to_console_screen(self):
        from tldw_chatbook.app import TldwCli

        calls = []

        class FakeConsoleScreen:
            def _apply_focus_chrome(self):
                calls.append("applied")

        stub = SimpleNamespace(
            focus_mode=False,
            _navigation_outgoing_screen=lambda: FakeConsoleScreen(),
            post_message=lambda msg: calls.append(msg),
        )
        TldwCli._set_focus_mode(stub, True)
        assert stub.focus_mode is True
        assert calls == ["applied"]

    def test_set_focus_mode_navigates_when_elsewhere(self):
        from tldw_chatbook.app import TldwCli

        posted = []
        stub = SimpleNamespace(
            focus_mode=False,
            _navigation_outgoing_screen=lambda: object(),
            post_message=posted.append,
        )
        TldwCli._set_focus_mode(stub, True)
        assert stub.focus_mode is True
        assert len(posted) == 1
        assert posted[0].screen_name == TAB_CHAT

    def test_set_focus_mode_disable_clears_flag(self):
        from tldw_chatbook.app import TldwCli

        posted = []
        stub = SimpleNamespace(
            focus_mode=True,
            _navigation_outgoing_screen=lambda: object(),
            post_message=posted.append,
        )
        TldwCli._set_focus_mode(stub, False)
        assert stub.focus_mode is False
        assert posted == []  # disabling never navigates

    def test_clear_focus_when_leaving_console(self):
        from tldw_chatbook.app import TldwCli

        leaving = SimpleNamespace(focus_mode=True)
        TldwCli._clear_focus_if_leaving_console(leaving, "settings")
        assert leaving.focus_mode is False

        staying = SimpleNamespace(focus_mode=True)
        TldwCli._clear_focus_if_leaving_console(staying, TAB_CHAT)
        assert staying.focus_mode is True

    def test_action_toggle_flips_state(self):
        from tldw_chatbook.app import TldwCli

        stub = SimpleNamespace(
            focus_mode=True,
            _set_focus_mode=lambda enabled: setattr(stub, "focus_mode", enabled),
        )
        TldwCli.action_toggle_focus_mode(stub)
        assert stub.focus_mode is False


class TestPaletteQuickAction:
    async def test_focus_toggle_is_searchable_and_executable(self):
        from tldw_chatbook.app import QuickActionsProvider

        app_instance = _make_app_instance(focus=False)
        harness = FocusConsoleHarness(app_instance)
        async with harness.run_test(size=_FOCUS_TERMINAL_SIZE) as pilot:
            provider = QuickActionsProvider(pilot.app.screen)
            # Textual 8's Hit has no `display`; `text` is the plain-text
            # field the palette tests elsewhere in this suite assert on.
            hits = [hit async for hit in provider.search("focus")]
            assert any("Toggle Focus Mode" in hit.text for hit in hits)

            called = []
            pilot.app.action_toggle_focus_mode = lambda: called.append(True)
            provider.execute_quick_action("toggle_focus_mode")
            assert called == [True]


class TestDeferredFocusRestore:
    """Qodo finding 3: a first-run launch with --focus must deliver the
    chrome-free Console when the wizard completes and navigates to Chat."""

    async def test_wizard_completion_restores_deferred_focus(self):
        # Boot WITHOUT the focus request (the resolver would apply it
        # immediately on a non-first-run launch); then model the state the
        # first-run path leaves behind: request deferred, Console mounted
        # unfocused beneath the wizard.
        app_instance = _build_test_app(configured_default="chat")
        app_instance._deferred_focus_request = True

        async with app_instance.run_test(size=_FOCUS_TERMINAL_SIZE) as pilot:
            for _ in range(200):
                await pilot.pause(0.02)
                if type(app_instance.screen).__name__ == "ChatScreen":
                    break
            await pilot.pause(0.3)
            assert app_instance.focus_mode is False  # not yet applied
            assert not app_instance.screen.has_class("-focus")

            # Wizard completes and routes the user to the Console.
            app_instance._handle_first_run_wizard_result(
                {"completed": True, "exit_route": TAB_CHAT}
            )
            await pilot.pause(0.3)
            assert app_instance._deferred_focus_request is False
            assert app_instance.focus_mode is True

            # The restore must land on the mounted Console even when the
            # remount shortcut skips navigation (Console already current).
            assert app_instance.screen.has_class("-focus")

    def test_wizard_exit_to_settings_does_not_restore_focus(self):
        app_instance = _build_test_app()
        app_instance._deferred_focus_request = True
        app_instance._handle_first_run_wizard_result(
            {
                "completed": False,
                "exit_route": "settings",
                "exit_context": {"category": "providers-models"},
            }
        )
        # Focus is Console-only; leaving to Settings consumes the request
        # without applying it.
        assert app_instance._deferred_focus_request is False
        assert app_instance.focus_mode is False


class TestNavigationVetoKeepsFocus:
    """Qodo finding 4: an aborted navigation must leave the flag, the
    -focus class, and the footer untouched and synchronized."""

    async def test_flush_veto_keeps_focus_state_synchronized(self):
        app_instance = _build_test_app(configured_default="chat")

        async with app_instance.run_test(size=_FOCUS_TERMINAL_SIZE) as pilot:
            for _ in range(200):
                await pilot.pause(0.02)
                if type(app_instance.screen).__name__ == "ChatScreen":
                    break
            await pilot.pause(0.3)
            screen = app_instance.screen

            await pilot.press("ctrl+shift+f")
            await pilot.pause(0.3)
            assert app_instance.focus_mode is True
            assert screen.has_class("-focus")

            # A screen whose pending-work flush vetoes the navigation.
            async def _veto_flush():
                return False

            screen.flush_pending_work = _veto_flush
            app_instance._initial_screen_pushed = True
            await app_instance.handle_screen_navigation(NavigateToScreen("settings"))
            await pilot.pause(0.3)

            # Navigation aborted: the Console is still resident and focus
            # state must be exactly as before the attempt.
            assert type(app_instance.screen).__name__ == "ChatScreen"
            assert app_instance.focus_mode is True
            assert screen.has_class("-focus")

            # And the next toggle therefore does the right visible action.
            await pilot.press("ctrl+shift+f")
            await pilot.pause(0.3)
            assert app_instance.focus_mode is False
            assert not screen.has_class("-focus")
