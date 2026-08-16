"""ChapterEditorWidget must not recompose away a high-volume population (task-15773).

`chapters` was `reactive([], recompose=True)` on a widget whose `compose()`
is entirely static. Every `set_chapters` therefore did two broken things:

1. **Threw the population away.** `watch_chapters` populated the CURRENT
   DataTable synchronously, then the scheduled recompose removed that whole
   subtree and mounted a fresh, EMPTY table -- the settled table had 0 rows
   after every update, at any row count.
2. **Reopened `Select`'s mount window on every data arrival.** The remount
   re-ran `#chapter-voice-select`'s Compose->Mount sequence; a high-volume
   population (hundreds of `add_row` calls in one reactive update, plus the
   teardown of the old populated table) clogs the loop and widens the gap
   between the fresh Select's registration and its Compose dispatch. Any
   teardown landing in that gap (app/test shutdown, a Speech view
   transition) marks the Select `_pruning`, which turns its child mount into
   a silent no-op while its Mount event still fires -- `Select._on_mount ->
   _setup_options_renderables -> query_one(SelectOverlay)` then raises
   `NoMatches: No nodes match 'SelectOverlay'` through
   `app._handle_exception`. This is the task-15478 review-round-3 flake
   ("observed once in a full-file run"), reproduced deterministically here
   by gating that exact window.

The fix drops `recompose=True`: population lands in place on the persistent,
already-mounted children, and no mount sequence ever re-runs for a data
update. Pre-mount data stays queued in the reactives (the `is_mounted`
guards) and is replayed by `on_mount`.

All three tests were born red against the pre-fix widget:
- population: settled row_count was 0 (and the table/Select identities had
  changed);
- gated teardown: `NoMatches: No nodes match 'SelectOverlay'` raised out of
  the test's app teardown;
- pre-mount replay: row_count 0 and an empty preview after mount.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import DataTable, Select, TextArea

from tldw_chatbook.TTS.audiobook_generator import Chapter
from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditorWidget

# The original flake's shape: task-15478's pre-workaround `_make_large_book`
# density produced ~999 chapters for a 3M-word book, populated into the
# table in one reactive update.
FLAKE_ROW_COUNT = 999


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield Container(id="slot")


def _chapters(n: int) -> list[Chapter]:
    return [
        Chapter(
            number=i + 1,
            title=f"Chapter {i + 1}",
            content="word " * 60,
            start_position=0,
            end_position=0,
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_high_volume_population_lands_and_children_stay_mounted():
    """One reactive update with 999 rows must (a) actually end up in the
    settled table, byte-identical to what `_refresh_chapter_table` computes,
    and (b) leave every child the SAME mounted instance -- no teardown, no
    re-mount, no reopened Select mount sequence."""
    app = _Host()
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        slot = app.query_one("#slot", Container)
        editor = ChapterEditorWidget(id="chapter-editor-widget")
        await slot.mount(editor)
        await pilot.pause()

        table_before = editor.query_one("#chapter-table", DataTable)
        select_before = editor.query_one("#chapter-voice-select", Select)

        editor.set_chapters(_chapters(FLAKE_ROW_COUNT))
        for _ in range(10):
            await pilot.pause()

        table_after = editor.query_one("#chapter-table", DataTable)
        select_after = editor.query_one("#chapter-voice-select", Select)

        assert table_after.row_count == FLAKE_ROW_COUNT, (
            f"population was thrown away: settled table has "
            f"{table_after.row_count} rows for {FLAKE_ROW_COUNT} chapters"
        )
        # Content parity pin: exactly what `_refresh_chapter_table` writes
        # ("word " * 60 -> 60 words -> 60/150 min narration -> 24s).
        assert table_after.get_row_at(0) == ["≡", "1", "Chapter 1", "60", "24s"]
        assert table_after.get_row_at(FLAKE_ROW_COUNT - 1) == [
            "≡",
            f"{FLAKE_ROW_COUNT}",
            f"Chapter {FLAKE_ROW_COUNT}",
            "60",
            "24s",
        ]

        assert table_after is table_before, (
            "the DataTable was torn down and remounted by a data update"
        )
        assert select_after is select_before, (
            "the Select was torn down and remounted by a data update -- "
            "its mount sequence must not re-run on population"
        )
        assert table_before.is_attached and select_before.is_attached


@pytest.mark.asyncio
async def test_teardown_racing_a_population_cannot_break_select_mount(monkeypatch):
    """Deterministic interleave for the task-15478 flake, no repetition
    needed: gate any post-population `Select` at the exact point the race
    fires -- registered, parked right before its Compose dispatch -- then
    land a teardown inside that window and release.

    Pre-fix, `set_chapters` recomposes a fresh Select, the gate engages, and
    `editor.remove()` inside the window makes the released mount sequence
    raise `NoMatches: No nodes match 'SelectOverlay'` out of the app (it
    surfaces from `run_test`'s teardown). Post-fix a population mounts
    nothing, so the gate can never engage and the removal is clean.
    """
    gate = asyncio.Event()
    entered = asyncio.Event()
    armed = False

    orig_on_compose = Widget._on_compose

    async def gated_on_compose(self, event):
        if armed and isinstance(self, Select):
            entered.set()
            await gate.wait()
        await orig_on_compose(self, event)

    monkeypatch.setattr(Select, "_on_compose", gated_on_compose)

    app = _Host()
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        slot = app.query_one("#slot", Container)
        editor = ChapterEditorWidget(id="chapter-editor-widget")
        await slot.mount(editor)
        await pilot.pause()
        armed = True

        editor.set_chapters(_chapters(500))

        # Pre-fix: the population's recompose registers a fresh Select and
        # this loop parks it in the race window within a few ticks.
        # Post-fix: no Select is ever (re)composed by a population, so this
        # is a short bounded spin and the gate stays disengaged.
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0)

        editor.remove()  # the teardown that lands inside the window
        await asyncio.sleep(0)
        gate.set()

        for _ in range(20):
            await pilot.pause()

        assert not entered.is_set(), (
            "a chapter population re-entered Select's mount sequence -- "
            "the task-15773 race window is open again"
        )
    # Exiting run_test re-raises anything that hit app._handle_exception:
    # pre-fix that is `NoMatches: No nodes match 'SelectOverlay'`.


@pytest.mark.asyncio
async def test_chapters_set_before_mount_are_replayed_when_ready():
    """Data arriving before the widget is ready must be queued, not lost:
    `on_mount` replays both the table population and the selected chapter's
    preview once the children exist."""
    app = _Host()
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        slot = app.query_one("#slot", Container)

        editor = ChapterEditorWidget(id="chapter-editor-widget")
        chapters = _chapters(300)
        editor.set_chapters(chapters)  # before mount: widget not ready yet

        await slot.mount(editor)
        for _ in range(10):
            await pilot.pause()

        table = editor.query_one("#chapter-table", DataTable)
        assert table.row_count == 300
        preview = editor.query_one("#chapter-preview", TextArea)
        assert preview.text == chapters[0].content
