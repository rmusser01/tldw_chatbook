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


def test_model_takes_a_query_and_help_lists_it():
    """TASK-33004.7 AC#1: /help lists `/model [query]`; the other action
    commands still take nothing."""
    reg = default_console_registry()
    hints = {c.name: c.argument_hint for c in reg.commands()}
    assert hints["model"] == "[query]"
    assert {name: hints[name] for name, _ in CONSOLE_ACTION_COMMANDS} == {
        name: ("[query]" if name == "model" else "") for name, _ in CONSOLE_ACTION_COMMANDS
    }
    out = build_help_listing(reg.commands(), _COMMAND_DESCRIPTIONS)
    assert "/model [query]" in out
    parse = reg.parse("/model son")
    assert (parse.kind, parse.name, parse.args) == (KIND_COMMAND, "model", "son")


def test_model_and_settings_descriptions_name_their_surfaces_and_keys():
    """TASK-33004.7 AC#6: the slash popup teaches the same two surfaces."""
    assert "Switch model" in _COMMAND_DESCRIPTIONS["model"]
    assert "Alt+M" in _COMMAND_DESCRIPTIONS["model"]
    assert "Chat settings" in _COMMAND_DESCRIPTIONS["settings"]
    assert "Ctrl+O" in _COMMAND_DESCRIPTIONS["settings"]


async def _run(command: str, screen) -> None:
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    await ChatScreen._console_command_run_action(
        screen, default_console_registry().parse(command)
    )


def test_the_runner_passes_arguments_only_to_commands_that_take_them():
    """TASK-33004.7 AC#1/AC#7: `/model son` reaches the opener with `son`;
    `/new son` still opens a tab and ignores the stray text, as before."""
    import asyncio
    from typing import ClassVar

    calls: list[tuple[str, tuple]] = []

    class _Screen:
        _CONSOLE_ACTION_COMMAND_TARGETS: ClassVar[dict[str, str]] = {
            "model": "action_open_console_model_popover",
            "new": "action_new_console_tab",
        }

        async def action_open_console_model_popover(self, query: str = "") -> None:
            calls.append(("model", (query,)))

        def action_new_console_tab(self) -> None:
            calls.append(("new", ()))

        async def _append_native_console_system_message(self, text: str) -> None:
            calls.append(("message", (text,)))

        def _clear_console_composer_draft(self) -> None:
            calls.append(("clear", ()))

    screen = _Screen()
    asyncio.run(_run("/model son", screen))
    asyncio.run(_run("/model", screen))
    asyncio.run(_run("/new son", screen))
    # A command that ran leaves no "/model son" behind in the composer.
    assert calls == [
        ("clear", ()),
        ("model", ("son",)),
        ("clear", ()),
        ("model", ("",)),
        ("clear", ()),
        ("new", ()),
    ]
