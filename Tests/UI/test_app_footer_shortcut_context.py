"""Tests for global footer shortcut context updates."""

from types import SimpleNamespace

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
            shortcuts=(("F6", "next pane"), ("F1", "help"), ("Ctrl+K", "switch session")),
        )
        await pilot.pause()

        shortcut_display = footer.query_one("#footer-key-quit", Static)
        rendered = str(shortcut_display.renderable)

        # Reserved global keys (F1/F6/Ctrl+P/Ctrl+Q) are never duplicated by
        # the screen context — the always-present global strip covers them.
        assert "F6 next pane" not in rendered
        assert rendered.count("F1") == 1
        # Non-reserved context hints render ahead of the globals.
        assert "Ctrl+K switch session" in rendered
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
    # Merge recalibration (82 cols): the merged footer's no-context hint set
    # is the four-key global strip ("F1 help · F6 panes · Ctrl+P palette ·
    # Ctrl+Q quit"), wider than the two-key default this budget was tuned
    # against. 82 terminal cols puts the widget at exactly the DB-stats
    # minimum width (80, after the footer's 2-cell padding): the stats fit
    # with the compact hints (71 cells) and the grown word count pushes
    # them over (90 cells).
    async with app.run_test(size=(82, 12)) as pilot:
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


def _rendered_footer_text(footer: AppFooterStatus) -> str:
    """Return what the footer ACTUALLY displays (post width-fitting).

    ``footer.shortcut_text`` is the stored, unfitted "logical" text --
    ``_apply_responsive_footer`` picks a (possibly shrunk) variant and
    writes it straight to the ``#footer-key-quit`` Static without ever
    reassigning ``_shortcut_text``, so that property alone cannot tell
    what a real terminal at this width would show.
    """
    return str(footer.query_one("#footer-key-quit").renderable)


@pytest.mark.asyncio
async def test_footer_compacts_globals_before_dropping_screen_hints_when_narrow():
    """LIB-18: reproduces the live 100/80-column Library footer finding --
    at a narrow width the screen's OWN hints (what the user came here to
    discover) used to drop first, replaced by a bare "… <globals>"; the
    fix compacts the always-present globals (muscle-memory keys most users
    already know) BEFORE the screen-specific hints disappear.

    Mirrors the live-verification recipe (fresh session per width) rather
    than resizing one running app, matching how the width sweep is
    actually driven live.
    """

    def _library_shortcuts() -> tuple[tuple[str, str], ...]:
        # Approximates the Library landing's real registered hint set
        # (screen_registry's LIBRARY_LANDING_SHORTCUTS shape).
        return (
            ("/", "focus search"),
            ("i", "import content"),
            ("n", "new note"),
        )

    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    # Wide (170): both the screen hints and the full globals fit.
    wide_app = TestApp()
    async with wide_app.run_test(size=(170, 12)) as pilot:
        footer = wide_app.query_one("#footer", AppFooterStatus)
        footer.set_workbench_shortcuts(source="library", shortcuts=_library_shortcuts())
        await pilot.pause()
        wide_text = _rendered_footer_text(footer)
        assert "focus search" in wide_text
        assert "F1 help" in wide_text

    # 100 columns: the live Library repro width.
    narrow_app = TestApp()
    async with narrow_app.run_test(size=(100, 12)) as pilot:
        footer = narrow_app.query_one("#footer", AppFooterStatus)
        footer.set_workbench_shortcuts(source="library", shortcuts=_library_shortcuts())
        await pilot.pause()
        narrow_text = _rendered_footer_text(footer)
        # The screen's own hints must still be legible -- not replaced by
        # a bare "… F1 help · ...".
        assert "focus search" in narrow_text, narrow_text
        assert "import content" in narrow_text, narrow_text
        assert "new note" in narrow_text, narrow_text
        assert not narrow_text.startswith("…"), narrow_text
        # The GLOBAL half compacts instead (still present, just shorter).
        assert "F1 " in narrow_text
        assert "F6 panes" not in narrow_text  # only the compact form survives

    # 80 columns (the other live-verified width) still holds.
    narrowest_app = TestApp()
    async with narrowest_app.run_test(size=(80, 12)) as pilot:
        footer = narrowest_app.query_one("#footer", AppFooterStatus)
        footer.set_workbench_shortcuts(source="library", shortcuts=_library_shortcuts())
        await pilot.pause()
        narrowest_text = _rendered_footer_text(footer)
        assert "focus search" in narrowest_text, narrowest_text


@pytest.mark.asyncio
async def test_footer_control_reproduces_the_historical_ellipsis_drop():
    """Control case: confirms the width chosen above genuinely exercises
    the historical bug shape (screen hints dropping to a bare ellipsis)
    when the new compact-globals step is removed -- i.e. this test would
    have failed before the fix, proving the fix (not the width choice)
    is what changed the outcome."""

    class TestApp(App):
        def compose(self):
            yield AppFooterStatus(id="footer")

    app = TestApp()
    async with app.run_test(size=(100, 12)) as pilot:
        footer = app.query_one("#footer", AppFooterStatus)
        footer.set_workbench_shortcuts(
            source="library",
            shortcuts=(
                ("/", "focus search"),
                ("i", "import content"),
                ("n", "new note"),
            ),
        )
        await pilot.pause()
        rendered = _rendered_footer_text(footer)
        # With the fix, the compact-globals step is what actually renders
        # at this width -- confirms it is REACHED (not merely defined).
        assert rendered == (
            "/ focus search | i import content | n new note | "
            f"{footer.GLOBAL_HINTS_COMPACT}"
        ), rendered
