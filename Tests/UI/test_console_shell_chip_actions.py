"""Console shell-strip chips are actionable and show longer model ids.

tasks 1670-1672: Provider/Model chips open the quick model popover, the
Character chip opens a character picker that also asks where the pick
lands, and the chip width cap fits common model ids.
"""

from pathlib import Path

import pytest

from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
    ConsoleCharacterOption,
    filter_character_options,
)
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleAssistantChip,
    ConsoleChip,
    ConsoleModelChip,
)

_CSS = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook" / "css" / "components" / "_agentic_terminal.tcss"
)


@pytest.mark.unit
def test_provider_and_model_chips_are_action_chips():
    """task-1670: both chips must be activatable, not inert labels."""
    for action in ("enter", "space"):
        assert any(
            binding.key == action for binding in ConsoleModelChip.BINDINGS
        ), action
    assert issubclass(ConsoleModelChip, ConsoleChip)
    assert hasattr(ConsoleModelChip, "OpenRequested")


@pytest.mark.unit
def test_assistant_chip_is_an_action_chip():
    """task-1672: the Character chip opens the picker."""
    for action in ("enter", "space"):
        assert any(
            binding.key == action for binding in ConsoleAssistantChip.BINDINGS
        ), action
    assert hasattr(ConsoleAssistantChip, "OpenRequested")


@pytest.mark.unit
def test_chip_width_fits_25_chars_of_model_name():
    """task-1671: the ask was 25 chars of the model NAME, not 25 cells.

    "Model: " costs 7 cells and padding 2, so the cap must be 34 for the
    name itself to reach 25. Chips are ``width: auto``, so a higher cap
    never widens short chips like "RAG: off".
    """
    block = _CSS.read_text(encoding="utf-8").split(".console-control-chip {", 1)[1]
    block = block.split("}", 1)[0]
    assert "width: auto;" in block
    assert "max-width: 34;" in block
    prefix_and_padding = len("Model: ") + 2
    assert 34 - prefix_and_padding == 25


@pytest.mark.unit
def test_character_filter_ranks_name_matches_first():
    """Typing a name must not bury it under description matches."""
    options = (
        ConsoleCharacterOption(1, "Zara", description="knows Lana well"),
        ConsoleCharacterOption(2, "Lana", description="an artist"),
    )
    result = filter_character_options(options, "lana")
    assert [o.name for o in result] == ["Lana", "Zara"]


@pytest.mark.unit
def test_character_filter_blank_query_and_limit():
    """A blank query lists the head of the library, bounded."""
    options = tuple(
        ConsoleCharacterOption(i, f"C{i}") for i in range(1, 60)
    )
    assert len(filter_character_options(options, "")) == 40
    assert len(filter_character_options(options, "", limit=5)) == 5
    assert filter_character_options(options, "nope") == ()


@pytest.mark.unit
def test_screen_handles_both_new_chip_messages():
    """The screen must subscribe to the chips it now renders."""
    import inspect

    from tldw_chatbook.UI.Screens import chat_screen

    src = inspect.getsource(chat_screen.ChatScreen)
    assert "@on(ConsoleModelChip.OpenRequested)" in src
    assert "@on(ConsoleAssistantChip.OpenRequested)" in src
    # The model chip reuses the existing Alt+M opener rather than a fork.
    assert "await self.action_open_console_model_popover()" in src


@pytest.mark.unit
def test_swap_seeds_greeting_only_into_an_empty_chat():
    """User decision: a greeting must not interrupt a live conversation."""
    import inspect

    from tldw_chatbook.UI.Screens import chat_screen

    src = inspect.getsource(chat_screen.ChatScreen._swap_console_session_character)
    assert "if greeting and not store.messages_for_session(session_id):" in src
