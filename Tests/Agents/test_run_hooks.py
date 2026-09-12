"""Tests for the Console run-hooks engine (spec: 2026-09-11-console-run-hooks-design)."""

import sys

import pytest

from tldw_chatbook.Agents.run_hooks import (
    HOOK_DEFAULT_TIMEOUT_S,
    HOOK_EVENTS,
    HookSpec,
    RunHooksConfig,
    load_hooks_config,
)


def _cfg(enabled=True, **hook_kwargs):
    base = {"event": "PreToolUse", "command": ["/bin/true"]}
    base.update(hook_kwargs)
    return {"hooks": {"enabled": enabled, "hook": [base]}}


class TestLoadHooksConfig:
    def test_empty_section_is_no_op(self):
        result = load_hooks_config({})
        assert result == RunHooksConfig(enabled=True, hooks=())

    def test_master_switch_off(self):
        result = load_hooks_config({"hooks": {"enabled": False}})
        assert result.enabled is False

    def test_valid_hook_parsed(self):
        result = load_hooks_config(
            {"hooks": {"hook": [{"event": "PreToolUse", "matcher": "fs_*",
                                  "command": ["/bin/guard", "--strict"], "timeout_s": 5}]}}
        )
        assert result.hooks == (HookSpec("PreToolUse", ("/bin/guard", "--strict"), "fs_*", 5.0),)

    def test_unknown_event_disables_that_hook(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Nope", "command": ["/bin/true"]}]}})
        assert result.hooks == ()

    def test_matcher_rejected_on_non_tool_event(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "matcher": "fs_*",
                                                         "command": ["/bin/true"]}]}})
        assert result.hooks == ()

    def test_empty_command_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": []}]}})
        assert result.hooks == ()

    def test_non_positive_timeout_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": ["/bin/true"],
                                                         "timeout_s": 0}]}})
        assert result.hooks == ()

    def test_string_command_rejected(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": "/bin/true"}]}})
        assert result.hooks == ()

    def test_default_timeout_applied(self):
        result = load_hooks_config({"hooks": {"hook": [{"event": "Stop", "command": ["/bin/true"]}]}})
        assert result.hooks[0].timeout_s == HOOK_DEFAULT_TIMEOUT_S
