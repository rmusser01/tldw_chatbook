"""TASK-25908: /help lists console commands from the live registry."""

from __future__ import annotations

from tldw_chatbook.Chat.console_command_grammar import (
    ConsoleCommand,
    ConsoleCommandRegistry,
    default_console_registry,
)
from tldw_chatbook.Chat.console_help import (
    build_help_listing,
    build_command_detail,
)

_DESCS = {
    "prompt": "Insert a saved prompt",
    "rewind": "Rewind the session",
}


def test_listing_includes_every_registered_command_with_desc_and_hint():
    """AC#1."""
    reg = ConsoleCommandRegistry()
    reg.register(ConsoleCommand(name="prompt", argument_hint="[name]", handler_id="p"))
    reg.register(ConsoleCommand(name="rewind", argument_hint="<n>", handler_id="r"))
    out = build_help_listing(reg.commands(), _DESCS)
    assert "/prompt" in out and "[name]" in out and "Insert a saved prompt" in out
    assert "/rewind" in out and "<n>" in out and "Rewind the session" in out


def test_newly_registered_command_appears_without_touching_help(  ):
    """AC#2: generated from the live registry."""
    reg = ConsoleCommandRegistry()
    reg.register(ConsoleCommand(name="brandnew", argument_hint="[x]", handler_id="bn"))
    out = build_help_listing(reg.commands(), {"brandnew": "A brand new command"})
    assert "/brandnew" in out and "A brand new command" in out


def test_command_with_no_description_still_listed():
    reg = ConsoleCommandRegistry()
    reg.register(ConsoleCommand(name="nodesc", argument_hint="", handler_id="nd"))
    out = build_help_listing(reg.commands(), {})
    assert "/nodesc" in out  # fallback description, never blank


def test_detail_for_one_command():
    """AC#3."""
    reg = ConsoleCommandRegistry()
    reg.register(ConsoleCommand(name="prompt", argument_hint="[name]", handler_id="p"))
    out = build_command_detail(reg.commands(), _DESCS, "prompt")
    assert "/prompt" in out and "[name]" in out and "Insert a saved prompt" in out


def test_detail_for_unknown_command_is_honest():
    """AC#3."""
    reg = ConsoleCommandRegistry()
    reg.register(ConsoleCommand(name="prompt", argument_hint="", handler_id="p"))
    out = build_command_detail(reg.commands(), _DESCS, "nope")
    assert "nope" in out
    assert "unknown" in out.lower() or "not a" in out.lower() or "no such" in out.lower()


def test_unavailable_command_is_marked_not_silently_usable():
    """AC#5."""
    reg = ConsoleCommandRegistry()
    reg.register(ConsoleCommand(name="generate-image", argument_hint="", handler_id="gi"))
    reg.register(ConsoleCommand(name="prompt", argument_hint="", handler_id="p"))

    def availability(name):
        return "image generation is disabled in temporary chats" if name == "generate-image" else None

    out = build_help_listing(reg.commands(), {"generate-image": "Generate an image", "prompt": "x"}, availability_fn=availability)
    assert "generate-image" in out
    # its unavailability must be stated
    assert "disabled" in out.lower() or "unavailable" in out.lower()


def test_default_registry_help_lists_the_real_commands():
    """Integration with the shipped registry."""
    reg = default_console_registry()
    from tldw_chatbook.Chat.console_command_suggestions import _COMMAND_DESCRIPTIONS
    out = build_help_listing(reg.commands(), _COMMAND_DESCRIPTIONS)
    assert "/rewind" in out and "/prompt" in out and "/emergency-stop" in out
