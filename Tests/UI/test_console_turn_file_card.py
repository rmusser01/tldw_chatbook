"""Turn file card: header, async rows, expandable capped diffs.

Runs on the REAL app CSS stack (screen css + bundle): geometry measured
without the bundle is not measured (task-15110's lesson).
"""
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.css import build_css
from tldw_chatbook.Widgets.Console.console_turn_file_card import (
    ConsoleTurnFileCard,
)

_CSS_DIR = Path(build_css.__file__).parent
_SELF, _SCOPED = build_css.screen_css_paths(_CSS_DIR)

MARKER = "✎ Edited 2 files  +8 −3 — review with `v`"


class _FakeProvider:
    diff_display_max_lines = 2000

    def __init__(self):
        from tldw_chatbook.UI.Screens.change_review_screen import ReviewTurn

        self._row = {"root": "/ws", "kind": "turn", "tracking_error": None,
                     "files_changed": 2, "adds": 8, "dels": 3,
                     "baseline_sha": "b", "end_sha": "e", "run_id": "run-1"}
        self._turn = ReviewTurn(run_id="run-1", label="t", rows=(self._row,))

    def turns(self):
        return [self._turn]

    def changed_files(self, row):
        from tldw_chatbook.Workspaces.change_tracking import ChangedFile

        assert row is self._row
        return [ChangedFile(path="a.py", status="M", adds=5, dels=3),
                ChangedFile(path="b.md", status="A", adds=3, dels=0)]

    def diff_text(self, row, path):
        assert path in ("a.py", "b.md")
        return "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old line\n+new line\n"


class _Host(App):
    CSS_PATH = [str(_SELF), str(_CSS_DIR / "tldw_cli_modular.tcss"), str(_SCOPED)]

    def compose(self) -> ComposeResult:
        yield ConsoleTurnFileCard(
            MARKER, "run-1", lambda: _FakeProvider(),
            id="card-under-test",
        )


async def _settled_card(pilot):
    card = pilot.app.query_one("#card-under-test", ConsoleTurnFileCard)
    for _ in range(60):
        if card.query(".console-turn-file-row"):
            break
        await pilot.pause(0.02)
    return card


@pytest.mark.asyncio
async def test_header_shows_marker_and_rows_load_async():
    async with _Host().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        header = card.query_one(".console-turn-file-header")
        assert MARKER.split(" — ")[0] in str(header.render())
        rows = list(card.query(".console-turn-file-row"))
        assert len(rows) == 2
        assert "a.py" in str(rows[0].render())
        assert "+5" in str(rows[0].render()) and "−3" in str(rows[0].render())


@pytest.mark.asyncio
async def test_expand_shows_capped_scrolling_diff():
    async with _Host().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        row = card.query(".console-turn-file-row").first()
        row.focus()
        await pilot.press("enter")
        body = None
        for _ in range(60):
            bodies = card.query(".console-turn-file-diff")
            if bodies and bodies.first().display:
                body = bodies.first()
                break
            await pilot.pause(0.02)
        assert body is not None, "diff body never displayed"
        # `body` is the VerticalScroll container -- its own render() is a
        # Blank placeholder (containers paint children, not self-content).
        # The diff text lives on the mounted Static child.
        diff_text_widget = body.query_one(".console-turn-file-diff-text")
        assert "+new line" in str(diff_text_widget.render())
        assert str(body.styles.overflow_y) == "auto"
        assert body.styles.max_height is not None
        # collapse again: display-managed, never unmounted
        row.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not body.display and body.is_mounted


@pytest.mark.asyncio
async def test_expand_provider_construction_failure_never_crashes_app():
    """Pins the Critical fix: the factory succeeds on its FIRST call (used
    by `_load_rows`, which was already guarded) but raises on its SECOND
    call (used by `on_button_pressed` on first expand, which was NOT). An
    exception escaping a Textual `on_*` handler propagates to
    `app._handle_exception()`, which unconditionally exits the app -- so
    this must degrade the row instead, leaving the app fully responsive.
    """
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeProvider()
        raise RuntimeError("shadow repo transiently unavailable")

    class _FlakyHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", factory, id="card-under-test"
            )

    async with _FlakyHost().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        rows = list(card.query(".console-turn-file-row"))
        assert len(rows) == 2

        rows[0].focus()
        await pilot.press("enter")
        await pilot.pause(0.3)

        bodies = list(card.query(".console-turn-file-diff"))
        assert bodies[0].display is False, (
            "diff body must stay hidden when provider construction fails"
        )
        assert card.is_mounted
        assert pilot.app.is_running, (
            "a provider-construction failure on expand must not crash the app"
        )

        # The app must still be responsive after the failure -- a press on
        # the OTHER row (a fresh factory call) is handled the same way,
        # not just tolerated once by accident.
        rows[1].focus()
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert bodies[1].display is False
        assert card.is_mounted
        assert pilot.app.is_running


@pytest.mark.asyncio
async def test_provider_failure_degrades_to_marker_only():
    class _Broken(_FakeProvider):
        def turns(self):
            raise RuntimeError("shadow repo unavailable")

    class _BrokenHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: _Broken(), id="card-under-test"
            )

    async with _BrokenHost().run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        card = pilot.app.query_one("#card-under-test", ConsoleTurnFileCard)
        assert not list(card.query(".console-turn-file-row"))
        assert MARKER.split(" — ")[0] in str(
            card.query_one(".console-turn-file-header").render()
        )
