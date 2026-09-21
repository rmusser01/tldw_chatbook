"""A chat title must not be able to blank a Context rail row, or kill it.

TASK-32802.3. The Console Context rail built its group, row and search-row
labels by interpolating `title` and `character_label` straight into an
f-string, with no escape at all. `Button.label` markup-parses on Textual 8
whatever the widget's markup flag says, and the same strings are reused
verbatim as tooltips, which render through a markup-on Static. Measured on
a real Button before the fix:

    '[TODO] chat' -> ' chat'                  the row loses its name
    '[IMPORTANT]' -> ''                       the row renders blank
    '[/b] chat'   -> MarkupError in compose   the rail does not render

These tests drive the three label builders the rail actually uses and then
put their output on a real Button, because the builder returning the right
string is only half the contract -- what the user sees is the other half.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Label

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationGroup,
    CharacterConversationRow,
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
)
from tldw_chatbook.Widgets.Console.console_character_context import (
    ConsoleCharacterContext,
)


# The shapes a user actually types into a chat title or a character name.
HOSTILE = [
    "[TODO] plan",
    "[IMPORTANT]",
    "[/b] plan",
    "[WIP] draft",
    "[bold]not bold[/bold]",
]


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield Label("host")


def _row(title: str, character_label: str = "Ada") -> CharacterConversationRow:
    """Built through the same factory the production rail's rows come from."""
    key = ResolvedLocalCharacterKey("authority", 1)
    return CharacterConversationRow.resolved(
        LocalCharacterConversationTarget(key, "conversation-1"),
        character_label=character_label,
        title=title,
        last_modified="2026-09-01T12:00:00Z",
        created_at="2026-09-01T00:00:00Z",
    )


def _group(character_label: str) -> CharacterConversationGroup:
    return CharacterConversationGroup(
        ResolvedLocalCharacterKey("authority", 1),
        character_label,
        (),
        3,
        False,
    )


async def _button_renders(label: str) -> str:
    app = _Harness()
    async with app.run_test() as pilot:
        button = Button(label, compact=True)
        await app.mount(button)
        await pilot.pause()
        return button.label.plain


@pytest.mark.asyncio
@pytest.mark.parametrize("title", HOSTILE)
async def test_a_row_label_survives_a_bracketed_title(title):
    label = ConsoleCharacterContext._row_label(_row(title))
    assert title in await _button_renders(label)


@pytest.mark.asyncio
@pytest.mark.parametrize("title", HOSTILE)
async def test_a_search_row_label_survives_both_of_its_user_fields(title):
    label = ConsoleCharacterContext._search_row_label(_row(title, title))
    rendered = await _button_renders(label)
    # Both the title and the character label are user data on this row.
    assert rendered.count(title) == 2, rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("name", HOSTILE)
async def test_a_group_label_survives_a_bracketed_character_name(name):
    label = ConsoleCharacterContext._group_label(_group(name), expanded=True)
    assert name in await _button_renders(label)


@pytest.mark.asyncio
async def test_the_group_tooltip_strips_only_its_glyph():
    """The rail derives the header tooltip with `.lstrip('▾▸ ')`."""
    label = ConsoleCharacterContext._group_label(_group("[TODO] Ada"), expanded=True)
    tooltip = label.lstrip("▾▸ ")
    assert "[TODO] Ada" in await _button_renders(tooltip)
