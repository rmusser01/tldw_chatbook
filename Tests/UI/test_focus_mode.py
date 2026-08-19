"""Focus mode (task-16320, ADR-067) — config, CLI, and behavior tests."""

import io
from types import SimpleNamespace

import pytest

from Tests.UI.consolidated_css import ConsolidatedCSSApp
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
