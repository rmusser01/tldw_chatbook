"""Tests for global footer shortcut context updates."""

import pytest
from types import SimpleNamespace
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
async def test_db_size_telemetry_caches_on_the_app_and_leaves_the_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-014: DB-size telemetry no longer renders in user chrome.

    The manager still computes the spelled-out sizes (task-1714 labels)
    and logs them, but its output is cached on the app for the Library
    Details disclosure instead of pushed into the footer, so a fresh
    install's footer no longer reads "Prompts: N/A | Chats/Notes: N/A |
    Media: N/A".

    Args:
        monkeypatch: Pytest fixture used to stub the size lookups.
    """
    app = SimpleNamespace()
    footer = AppFooterStatus(id="footer")
    manager = DBStatusManager(app=app)
    monkeypatch.setattr(manager, "_get_db_size", lambda *_args: "1.0 KB")

    await manager.update_db_sizes()

    assert app.db_sizes_status == {
        "prompts": "1.0 KB",
        "chachanotes": "1.0 KB",
        "media": "1.0 KB",
    }
    # The footer indicator is never fed -- and stays collapsed (empty).
    assert str(footer._db_status_display.renderable) == ""
    assert footer._db_status_display.display is False


@pytest.mark.asyncio
async def test_footer_db_indicator_collapses_when_empty_and_stays_down():
    """The DB-size indicator only takes footer space while it has content;
    the priority reflow must not resurrect an empty indicator on resize."""

    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    app = TestApp()
    async with app.run_test(size=(200, 12)) as pilot:
        footer = app.query_one("#footer", AppFooterStatus)
        db = app.query_one("#internal-db-size-indicator", Static)
        # Never fed: collapsed from the start.
        assert db.display is False

        # Feeding content reveals it; clearing collapses it again.
        footer.update_db_sizes_display("P: 144.0 KB | C/N: 904.0 KB | M: 376.0 KB")
        await pilot.pause()
        assert db.display is True
        footer.update_db_sizes_display("")
        await pilot.pause()
        assert db.display is False

        # A resize re-runs the priority reflow -- the empty indicator
        # must stay down rather than reappear as blank chrome.
        await pilot.resize_terminal(60, 12)
        await pilot.pause()
        assert db.display is False
        await pilot.resize_terminal(200, 12)
        await pilot.pause()
        assert db.display is False


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
    async with app.run_test(size=(100, 12)) as pilot:
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
