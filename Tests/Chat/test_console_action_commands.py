"""TASK-25909: existing Console actions reachable by typed slash command."""

from __future__ import annotations

from tldw_chatbook.Chat.console_command_grammar import (
    CONSOLE_ACTION_COMMAND_HANDLER_ID,
    CONSOLE_ACTION_COMMANDS,
    KIND_COMMAND,
    default_console_registry,
)
from tldw_chatbook.Chat.console_command_suggestions import _COMMAND_DESCRIPTIONS
from tldw_chatbook.Chat.console_help import build_help_listing


def test_every_action_command_parses_to_a_command():
    """AC#1: reachable by typed slash command."""
    reg = default_console_registry()
    for name, _hint in CONSOLE_ACTION_COMMANDS:
        parse = reg.parse(f"/{name}")
        assert parse.kind == KIND_COMMAND, name
        assert parse.name == name


def test_action_commands_all_use_the_shared_handler_id():
    reg = default_console_registry()
    by_name = {c.name: c for c in reg.commands()}
    for name, _hint in CONSOLE_ACTION_COMMANDS:
        assert by_name[name].handler_id == CONSOLE_ACTION_COMMAND_HANDLER_ID


def test_action_commands_appear_in_help_with_descriptions():
    """AC#3: appear in /help."""
    reg = default_console_registry()
    out = build_help_listing(reg.commands(), _COMMAND_DESCRIPTIONS)
    for name, _hint in CONSOLE_ACTION_COMMANDS:
        assert f"/{name}" in out, name
        assert name in _COMMAND_DESCRIPTIONS, f"{name} has no description"


def test_each_action_command_maps_to_a_real_screen_method():
    """AC#1/#2: every added command dispatches to an action method that
    already exists -- no new capability, no dangling target."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    targets = ChatScreen._CONSOLE_ACTION_COMMAND_TARGETS
    for name, _hint in CONSOLE_ACTION_COMMANDS:
        assert name in targets, f"{name} has no action target"
        method_name = targets[name]
        assert hasattr(ChatScreen, method_name), f"{method_name} missing on ChatScreen"
    # no target points at a nonexistent command
    action_names = {n for n, _ in CONSOLE_ACTION_COMMANDS}
    assert set(targets) == action_names
