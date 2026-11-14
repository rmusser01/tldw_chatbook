"""Character labels must reach native Content without a Rich Text round trip."""

from dataclasses import replace

import pytest
from rich.text import Text
from textual.content import Content

from Tests.UI.test_console_character_switcher_geometry import _GeometryApp
from tldw_chatbook.Chat.console_conversation_activation import ConsoleActivationPhase

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
async def test_character_labels_avoid_rich_allocations_without_changing_paint(
    monkeypatch, size
):
    """Reintroducing intermediate Rich Text must fail on the real modal rows."""
    app = _GeometryApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        modal = app.screen
        modal._sync_candidate_labels()
        await pilot.pause()
        entries = modal._entries
        targets = tuple(entry.target for entry in entries)
        focused = app.focused
        frame = tuple(strip.text for strip in modal._compositor.render_strips())
        allocations = []
        original = Text.__init__

        def construct(self, *args, **kwargs):
            allocations.append(1)
            original(self, *args, **kwargs)

        with monkeypatch.context() as scope:
            scope.setattr(Text, "__init__", construct)
            # Exclude detail copy, which is outside this plain row-label branch.
            for _ in range(3):
                for index, button in enumerate(modal._result_buttons()):
                    width = max(
                        1, button.content_size.width - 2 * button.styles.line_pad
                    )
                    button.label = modal._entry_label(
                        index, entries[index], available_width=width
                    )
        assert not allocations
        await pilot.pause()
        assert tuple(strip.text for strip in modal._compositor.render_strips()) == frame
        assert app.focused is focused
        assert tuple(entry.target for entry in modal._entries) == targets
        assert all(button.region.height == 2 for button in modal._result_buttons())


async def test_character_content_keeps_native_unicode_clipping_and_literal_markup():
    """Native two-line paint, including control sanitization, is the oracle."""
    app = _GeometryApp()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        modal = app.screen
        entry = replace(
            modal._entries[0],
            title="[bold]界🙂 é title[/bold]",
            character_label="Ada界",
        )
        for phase, word in (
            (ConsoleActivationPhase.IDLE, "RESUME CHAT"),
            (ConsoleActivationPhase.OPENING_CANCELLABLE, "OPENING"),
            (ConsoleActivationPhase.COMMITTING, "FINISHING"),
        ):
            modal._activation_phase = phase
            modal._committed_character_result = entry
            for width in (1, 2, 12, 24, 48, 96):
                first = Text(f"▸ [{word}] [bold]界🙂 é title[/bold]")
                second = Text(f"  Ada界 · Local · {entry.relative_time}")
                first.truncate(width, overflow="ellipsis", pad=False)
                second.truncate(width, overflow="ellipsis", pad=False)
                expected = Content.from_text(Text.assemble(first, "\n", second))
                actual = modal._entry_label(0, entry, available_width=width)
                assert Content.from_text(actual) == expected
        modal._activation_phase = ConsoleActivationPhase.IDLE
        modal._committed_character_result = None
