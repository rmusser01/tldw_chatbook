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
