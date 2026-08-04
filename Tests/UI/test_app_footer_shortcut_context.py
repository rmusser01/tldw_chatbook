"""Tests for global footer shortcut context updates."""

import pytest
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.UI.Navigation.shortcut_context import ShortcutAction, ShortcutContext
from tldw_chatbook.Utils.db_status_manager import DBStatusManager
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


@pytest.mark.asyncio
async def test_footer_uses_global_shortcuts_by_default():
    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    app = TestApp()

    async with app.run_test(size=(100, 12)) as pilot:
        await pilot.pause(0.1)
        footer = app.query_one("#footer", AppFooterStatus)

        assert "Ctrl+Q quit" in footer.shortcut_text
        assert "Ctrl+P palette" in footer.shortcut_text


@pytest.mark.asyncio
async def test_footer_replaces_stale_context_shortcuts():
    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    app = TestApp()

    async with app.run_test(size=(100, 12)) as pilot:
        await pilot.pause(0.1)
        footer = app.query_one("#footer", AppFooterStatus)
        footer.set_shortcut_context(
            ShortcutContext(
                source="console",
                actions=(ShortcutAction("Ctrl+Enter", "send"),),
            )
        )
        footer.set_shortcut_context(
            ShortcutContext(
                source="library",
                actions=(ShortcutAction("Ctrl+F", "search"),),
            )
        )

        assert "Ctrl+F search" in footer.shortcut_text
        assert "Ctrl+Enter send" not in footer.shortcut_text


@pytest.mark.asyncio
async def test_footer_renders_workbench_shortcuts():
    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    app = TestApp()

    async with app.run_test(size=(100, 12)) as pilot:
        await pilot.pause(0.1)
        footer = app.query_one("#footer", AppFooterStatus)

        footer.set_workbench_shortcuts(
            source="console",
            shortcuts=(("F6", "next pane"), ("F1", "help")),
        )
        await pilot.pause()

        shortcut_display = footer.query_one("#footer-key-quit", Static)
        rendered = str(shortcut_display.renderable)

        assert "F6 next pane" in rendered
        assert "F1 help" in rendered


@pytest.mark.asyncio
async def test_footer_token_chip_hidden_until_a_real_count_lands():
    """F-003: the Tokens chip is meaningful only where token counts exist
    (chat contexts). It now starts empty and hidden -- never a
    "Tokens: --" placeholder -- appears when a real count is pushed, and
    hides again when the count clears (non-chat tabs write "" via the
    periodic updater)."""

    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    app = TestApp()
    async with app.run_test(size=(100, 12)) as pilot:
        footer = app.query_one("#footer", AppFooterStatus)
        chip = app.query_one("#footer-token-count", Static)

        # Fresh footer: no placeholder text, no chip.
        assert str(chip.renderable) == ""
        assert chip.display is False

        # A real count reveals the chip...
        footer.update_token_count("Tokens: 1,234")
        await pilot.pause()
        assert "Tokens: 1,234" in str(chip.renderable)
        assert chip.display is True

        # ...and clearing it (the non-chat updater path) hides it again.
        footer.update_token_count("")
        await pilot.pause()
        assert chip.display is False


@pytest.mark.asyncio
async def test_footer_db_size_stats_use_readable_labels_and_context_tooltip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spell out production DB labels and describe their local context.

    Args:
        monkeypatch: Pytest fixture used to bind the production manager to the
            production footer function under test.
    """
    footer = AppFooterStatus(id="footer")
    manager = DBStatusManager(app=object())
    monkeypatch.setattr(manager, "_get_db_status_widget", lambda: footer)
    monkeypatch.setattr(manager, "_get_db_size", lambda *_args: "1.0 KB")

    await manager.update_db_sizes()

    rendered = str(footer._db_status_display.renderable)
    tooltip = str(footer._db_status_display.tooltip or "")

    assert "Prompts:" in rendered
    assert "Chats/Notes:" in rendered
    assert "Media:" in rendered
    assert "local" in tooltip.lower()
    assert "database file sizes" in tooltip.lower()


@pytest.mark.asyncio
async def test_footer_memory_stats_yield_to_key_hints_when_narrow():
    """A narrow footer hides the debug memory stats to preserve the key hints."""

    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    app = TestApp()
    async with app.run_test(size=(200, 12)) as pilot:
        footer = app.query_one("#footer", AppFooterStatus)
        footer.set_workbench_shortcuts(
            source="console",
            shortcuts=(
                ("F6", "next pane"),
                ("Shift+F6", "previous pane"),
                ("F1", "help"),
                ("Ctrl+K", "switch session"),
            ),
        )
        footer.update_db_sizes_display("P: 144.0 KB | C/N: 904.0 KB | M: 376.0 KB")
        await pilot.pause()
        db = app.query_one("#internal-db-size-indicator", Static)

        # Wide: both fit -> memory stats shown (AC#2 no regression at normal width).
        assert db.display is True

        # Narrow: not enough room for both -> memory stats yield, hints preserved.
        await pilot.resize_terminal(60, 12)
        await pilot.pause()
        assert db.display is False

        # Widen again -> memory stats return.
        await pilot.resize_terminal(200, 12)
        await pilot.pause()
        assert db.display is True


@pytest.mark.asyncio
async def test_footer_reflows_when_counts_change_without_a_resize():
    """A word/token count change re-runs the priority reflow (Qodo #834)."""

    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    app = TestApp()
    # F-003 recalibration: the width budget below used to include the
    # "Tokens: --" placeholder's 10 cells; the chip now starts hidden and
    # empty, so the same push-over-the-edge exercise runs at 90 cols.
    async with app.run_test(size=(90, 12)) as pilot:
        footer = app.query_one("#footer", AppFooterStatus)
        footer.update_db_sizes_display("P: 144.0 KB | C/N: 904.0 KB | M: 376.0 KB")
        await pilot.pause()
        db = app.query_one("#internal-db-size-indicator", Static)
        assert db.display is True

        # Growing the word count (no resize) can push past the width -> stats yield.
        footer.update_word_count(999_999_999)
        await pilot.pause()
        assert db.display is False

        # Clearing it brings them back.
        footer.update_word_count(0)
        await pilot.pause()
        assert db.display is True
