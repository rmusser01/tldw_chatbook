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
    ConsoleRagChip,
    ConsoleSourcesChip,
    ConsoleToolsChip,
)

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

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
def test_rag_chip_is_an_action_chip():
    """The Library-search chip opens the Library search settings modal (user
    request 2026-08-01): "Library search: off" must be an entry point into
    enabling it, not an inert status label."""
    for action in ("enter", "space"):
        assert any(
            binding.key == action for binding in ConsoleRagChip.BINDINGS
        ), action
    assert issubclass(ConsoleRagChip, ConsoleChip)
    assert hasattr(ConsoleRagChip, "OpenRequested")


@pytest.mark.unit
def test_sources_and_tools_chips_are_action_chips():
    """TASK-2154.2 (DS-06): the Sources and Tools chips must be activatable,
    not inert focus traps -- below 150 cols they are the only route to the
    Inspector's staged-sources tray and tool rows."""
    for chip in (ConsoleSourcesChip, ConsoleToolsChip):
        for action in ("enter", "space"):
            assert any(
                binding.key == action for binding in chip.BINDINGS
            ), (chip.__name__, action)
        assert issubclass(chip, ConsoleChip)
        assert hasattr(chip, "OpenRequested")


@pytest.mark.unit
def test_chip_width_fits_25_chars_of_model_name():
    """task-1671: the ask was 25 chars of the model NAME, not 25 cells.

    "Model: " costs 7 cells and padding 2, so the cap must be 34 for the
    name itself to reach 25. Chips are ``width: auto``, so a higher cap
    never widens short chips like "Library search: off".
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
def test_screen_subscribes_to_both_new_chip_messages():
    """The screen must actually handle the messages the chips post.

    cubic PR #1153 P3: this previously asserted on ``inspect.getsource``
    substrings, which break on any cosmetic reformat. Textual records
    ``@on`` subscriptions on the handler itself, so assert the real
    wiring: the handler exists and is registered for that message type.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    for handler_name, chip in (
        ("_console_model_chip_activated", ConsoleModelChip),
        ("_console_assistant_chip_activated", ConsoleAssistantChip),
        ("_console_rag_chip_activated", ConsoleRagChip),
        ("_console_sources_chip_activated", ConsoleSourcesChip),
        ("_console_tools_chip_activated", ConsoleToolsChip),
    ):
        handler = getattr(ChatScreen, handler_name, None)
        assert handler is not None, handler_name
        handlers = getattr(handler, "_textual_on", None)
        assert handlers, f"{handler_name} is not an @on handler"
        assert any(
            message_type is chip.OpenRequested for message_type, _ in handlers
        ), f"{handler_name} does not subscribe to {chip.__name__}.OpenRequested"


@pytest.mark.unit
def test_swap_seeds_greeting_only_into_an_empty_chat():
    """User decision: a greeting must not interrupt a live conversation.

    Drives the real method against a real in-memory store rather than reading
    source text (cubic PR #1153 P3).
    """
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleChatStore,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.UI.Console_Modules.session import (
        CharacterSessionPromptSeed,
        ConsoleSessionController,
    )

    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller.app_instance = SimpleNamespace(notify=lambda *_args, **_kwargs: None)
    controller._manual_reaction_overrides = {}
    seed = CharacterSessionPromptSeed(
        name="Lana",
        system_template="SYS",
        system_prompt="SYS",
        greeting_template="Hello!",
        greeting="Hello!",
    )

    empty = ConsoleChatStore()
    empty_session = empty.create_session(
        settings=ConsoleSessionSettings(provider="openai")
    )
    assert controller._swap_console_session_character(
        empty, 7, seed, global_default="User"
    )
    assert [m.content for m in empty.messages_for_session(empty_session.id)] == [
        "Hello!"
    ]
    assert empty.session_settings(empty_session.id).system_prompt == "SYS"

    busy = ConsoleChatStore()
    busy_session = busy.create_session(
        settings=ConsoleSessionSettings(provider="openai")
    )
    busy.append_message(
        busy_session.id,
        role=ConsoleMessageRole.USER,
        content="an existing message",
        persist=False,
    )
    assert controller._swap_console_session_character(
        busy, 7, seed, global_default="User"
    )
    assert [m.content for m in busy.messages_for_session(busy_session.id)] == [
        "an existing message"
    ], "greeting must not interrupt a live chat"
    assert busy.session_settings(busy_session.id).system_prompt == "SYS"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_changing_the_query_unstages_a_pending_pick():
    """A staged pick must not survive the user looking at something else.

    Qodo PR #1153: after Enter staged a character, typing a new query left
    ``_pending`` on the OLD character while the Swap/New buttons stayed
    visible -- clicking one would swap to a character the user was no
    longer looking at.
    """
    from textual.app import App

    from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
        ConsoleCharacterPickerModal,
    )

    options = (
        ConsoleCharacterOption(1, "Lana"),
        ConsoleCharacterOption(2, "Zara"),
    )

    class _Host(ConsolidatedCSSApp):
        pass

    app = _Host()
    async with app.run_test() as pilot:
        modal = ConsoleCharacterPickerModal(options=options)
        await app.push_screen(modal)
        await pilot.pause()

        modal._select(options[0])
        assert modal._pending is options[0]

        await modal._refresh_results("zara")
        assert modal._pending is None, "stale pick survived a query change"
        placement = modal.query_one("#console-character-picker-placement")
        assert placement.display is False

        modal._select(modal._results[0])
        await modal.action_cursor_down()
        assert modal._pending is None, "stale pick survived a cursor move"
