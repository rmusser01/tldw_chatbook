"""Character labels must reach native Content without a Rich Text round trip."""

from dataclasses import replace

import pytest
from rich.text import Text
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Button, Static

from Tests.UI.test_console_character_switcher_geometry import _GeometryApp
from tldw_chatbook.Chat.console_conversation_activation import ConsoleActivationPhase
from tldw_chatbook.Chat.console_switcher_state import SwitcherMode

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
async def test_character_activation_keeps_stable_chrome_without_repainting(
    monkeypatch, size
):
    app = _GeometryApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        modal = app.screen
        panel = modal.query_one("#console-switcher-modal", Vertical)
        status = modal.query_one("#console-switcher-page-status", Static)
        confirm = modal.query_one("#console-switcher-confirm-mark-seen", Button)
        original_refresh = panel.refresh
        original_update = status.update
        original_resize = modal._sync_modal_max_height
        repaints = []
        page_updates = []
        resizes = []

        def refresh(*regions, **kwargs):
            repaints.append(1)
            return original_refresh(*regions, **kwargs)

        def update(content="", *, layout=True):
            page_updates.append((content, layout))
            return original_update(content, layout=layout)

        def resize():
            resizes.append(1)
            original_resize()

        monkeypatch.setattr(panel, "refresh", refresh)
        monkeypatch.setattr(status, "update", update)
        monkeypatch.setattr(modal, "_sync_modal_max_height", resize)
        assert panel.border_title == "Switch or resume [Character chats]"
        assert status.content
        assert not confirm.display

        modal._set_activation_controls_disabled(True)
        modal._update_page_controls(None)
        modal._update_page_controls(None)
        modal._clear_mark_seen_confirmation()
        assert not repaints
        assert page_updates == [("", False)]
        assert not resizes

        confirm.display = True
        modal._armed_mark_seen_key = "confirmed-row"
        modal._clear_mark_seen_confirmation()
        assert not confirm.display and not modal._armed_mark_seen_key
        assert resizes == [1]
        modal._mode = SwitcherMode.ACTIVE
        modal._update_mode_controls()
        assert panel.border_title == "Switch or resume [Active]"


@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
async def test_character_activation_copy_repaints_fixed_rows_without_layout(
    monkeypatch, size
):
    app = _GeometryApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        modal = app.screen
        rows = {
            modal.query_one(f"#console-switcher-{name}", Static)
            for name in ("status", "divider", "scope", "hints")
        }
        geometry = {row: row.region for row in rows}
        calls = []
        original = Static.update

        def record_update(row, content="", *, layout=True):
            if row in rows:
                calls.append((row.id, layout))
            return original(row, content, layout=layout)

        monkeypatch.setattr(Static, "update", record_update)
        entry = modal._entries[0]
        modal._activation_phase = ConsoleActivationPhase.OPENING_CANCELLABLE
        modal._set_status("Opening…")
        modal._update_hints(entry)
        await pilot.pause()
        assert "Opening" in modal.query_one("#console-switcher-divider", Static).content
        assert "Opening" in modal.query_one("#console-switcher-hints", Static).content
        assert {row: row.region for row in rows} == geometry
        assert calls and not any(layout for _, layout in calls)


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
