"""Changed-files rail section: pure render from precomputed state.

Runs on the REAL app CSS stack (screen css + bundle): geometry measured
without the bundle is not measured (task-15110's lesson) -- load-bearing
here because row labels are middle-elided to the section's actual mounted
width.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import ConversationFileEntry
from tldw_chatbook.css import build_css
from tldw_chatbook.Widgets.Console.console_changed_files_section import (
    MAX_VISIBLE_ROWS,
    ConsoleChangedFilesSection,
    ConsoleChangedFilesState,
)

_CSS_DIR = Path(build_css.__file__).parent
_SCOPED, _SELF = build_css.screen_css_paths(_CSS_DIR)


def _entry(
    *,
    root: str = "/workspace",
    path: str = "src/app.py",
    label: str | None = None,
    status: str = "M",
    adds: int = 5,
    dels: int = 2,
    run_id: str = "run-1",
    snapshot_id: int = 101,
    note_count: int = 0,
) -> ConversationFileEntry:
    return ConversationFileEntry(
        root=root,
        path=path,
        label=label if label is not None else path,
        status=status,
        adds=adds,
        dels=dels,
        run_id=run_id,
        snapshot_id=snapshot_id,
        note_count=note_count,
    )


class _Host(App):
    CSS_PATH = [str(_SCOPED), str(_CSS_DIR / "tldw_cli_modular.tcss"), str(_SELF)]

    def __init__(self, state: ConsoleChangedFilesState) -> None:
        super().__init__()
        self._state = state
        self.captured: list[ConsoleChangedFilesSection.FileSelected] = []

    def compose(self) -> ComposeResult:
        yield ConsoleChangedFilesSection(self._state, id="section-under-test")

    @on(ConsoleChangedFilesSection.FileSelected)
    def _capture(self, event: ConsoleChangedFilesSection.FileSelected) -> None:
        self.captured.append(event)


@pytest.mark.asyncio
async def test_rows_render_status_deltas_badge_and_elision():
    long_path = "very/deeply/nested/directory/structure/that/is/long/module.py"
    state = ConsoleChangedFilesState(
        entries=(
            _entry(
                path=long_path,
                label=long_path,
                status="M",
                adds=7,
                dels=3,
                note_count=2,
            ),
        )
    )
    async with _Host(state).run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        section = pilot.app.query_one(
            "#section-under-test", ConsoleChangedFilesSection
        )
        row = section.query_one(".console-changed-files-row", Button)
        assert row.active_effect_duration == 0
        rendered = str(row.label)
        assert long_path not in rendered, (
            "a long path in a narrow row must be elided, not shown whole"
        )
        assert "…" in rendered
        assert "M" in rendered
        assert "+7" in rendered and "−3" in rendered
        assert "✎" in rendered and "2" in rendered
        # The full, un-elided label always stays reachable via the tooltip.
        assert row.tooltip == long_path

        header = section.query_one(
            "#console-changed-files-header", Static
        )
        assert "Changed files (1)" in str(header.renderable)
        assert "+7" in str(header.renderable) and "−3" in str(header.renderable)


@pytest.mark.asyncio
async def test_file_selected_posts_the_pressed_entrys_exact_identity():
    entries = (
        _entry(
            root="/workspace",
            path="a.py",
            label="a.py",
            run_id="run-a",
            snapshot_id=10,
        ),
        _entry(
            root="/workspace/other-root",
            path="b/nested.py",
            label="other-root/b/nested.py",
            run_id="run-b",
            snapshot_id=20,
        ),
    )
    state = ConsoleChangedFilesState(entries=entries)
    async with _Host(state).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rows = list(pilot.app.query(".console-changed-files-row"))
        assert len(rows) == 2
        # Press the SECOND row -- proves the posted identity is read from
        # the pressed row's own index, not hardcoded to the first entry.
        rows[1].focus()
        await pilot.press("enter")
        await pilot.pause()
        assert len(pilot.app.captured) == 1
        event = pilot.app.captured[0]
        assert event.run_id == "run-b"
        assert event.snapshot_id == 20
        assert event.path == "b/nested.py"
        assert event.root == "/workspace/other-root"


@pytest.mark.asyncio
async def test_cap_enforced_at_twelve_rows_with_honest_tail_count():
    entries = tuple(
        _entry(
            path=f"file_{idx}.py",
            label=f"file_{idx}.py",
            run_id="run-1",
            snapshot_id=100 + idx,
        )
        for idx in range(15)
    )
    state = ConsoleChangedFilesState(entries=entries)
    async with _Host(state).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rows = list(pilot.app.query(".console-changed-files-row"))
        assert len(rows) == MAX_VISIBLE_ROWS == 12
        tail = pilot.app.query_one("#console-changed-files-tail", Static)
        assert str(tail.renderable) == "+3 more — open Review"

        # Regression pin: the header's file count and +A -D totals cover
        # the FULL 15-entry set, not just the 12 rendered rows -- each
        # entry defaults to +5/-2, so a header that (wrongly) summed only
        # the visible rows would read "Changed files (12) ... +60 -24"
        # instead of the honest full-set totals below.
        header = pilot.app.query_one("#console-changed-files-header", Static)
        header_text = str(header.renderable)
        assert "Changed files (15)" in header_text
        assert "+75" in header_text and "−30" in header_text


@pytest.mark.asyncio
async def test_pruned_rows_render_a_dim_honest_tail_line():
    state = ConsoleChangedFilesState(
        entries=(_entry(),),
        pruned_rows=3,
    )
    async with _Host(state).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pruned = pilot.app.query_one("#console-changed-files-pruned", Static)
        assert str(pruned.renderable) == "history pruned for 3 turns"

    singular_state = ConsoleChangedFilesState(entries=(), pruned_rows=1)
    async with _Host(singular_state).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pruned = pilot.app.query_one("#console-changed-files-pruned", Static)
        assert str(pruned.renderable) == "history pruned for 1 turn"


@pytest.mark.asyncio
async def test_empty_state_renders_nothing_and_stays_display_false():
    state = ConsoleChangedFilesState(entries=(), pruned_rows=0)
    async with _Host(state).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        section = pilot.app.query_one(
            "#section-under-test", ConsoleChangedFilesSection
        )
        assert section.display is False
        assert list(section.children) == []


@pytest.mark.asyncio
async def test_update_state_resyncs_the_same_instance_in_place():
    initial = ConsoleChangedFilesState(
        entries=(
            _entry(path="a.py", label="a.py", run_id="run-1", snapshot_id=1),
            _entry(path="b.py", label="b.py", run_id="run-1", snapshot_id=2),
        )
    )
    async with _Host(initial).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        section = pilot.app.query_one(
            "#section-under-test", ConsoleChangedFilesSection
        )
        original_identity = id(section)
        rows = list(section.query(".console-changed-files-row"))
        assert len(rows) == 2
        assert any("a.py" in str(row.label) for row in rows)
        assert any("b.py" in str(row.label) for row in rows)

        updated = ConsoleChangedFilesState(
            entries=(
                _entry(path="c.py", label="c.py", run_id="run-2", snapshot_id=3),
            )
        )
        section.update_state(updated)
        await pilot.pause()

        # Same widget instance -- the owner never tore this down and
        # remounted a new one.
        resynced = pilot.app.query_one(
            "#section-under-test", ConsoleChangedFilesSection
        )
        assert id(resynced) == original_identity
        rows = list(section.query(".console-changed-files-row"))
        assert len(rows) == 1
        assert "c.py" in str(rows[0].label)
        assert not any("a.py" in str(row.label) for row in rows)
        assert not any("b.py" in str(row.label) for row in rows)

        # An empty update also flips display back off, in place.
        section.update_state(ConsoleChangedFilesState(entries=(), pruned_rows=0))
        await pilot.pause()
        assert id(section) == original_identity
        assert section.display is False
        assert list(section.children) == []


@pytest.mark.asyncio
async def test_stale_press_after_reorder_resolves_the_original_entry():
    """Qodo round: a row press must resolve by IDENTITY, not by position.

    A row `Button.Pressed` message can still be in flight when
    `update_state()` recomposes the section onto a REORDERED (or
    otherwise different) `entries` tuple -- Textual delivers the message
    against the ALREADY-CAPTURED button object, whose position in the new
    layout says nothing about which entry it was drawn for. Capturing the
    row up front and pressing it only AFTER the reorder reproduces exactly
    that race without relying on message-queue timing.
    """
    original_first = _entry(
        root="/workspace",
        path="a.py",
        label="a.py",
        run_id="run-a",
        snapshot_id=10,
    )
    original_second = _entry(
        root="/workspace",
        path="b.py",
        label="b.py",
        run_id="run-b",
        snapshot_id=20,
    )
    state = ConsoleChangedFilesState(entries=(original_first, original_second))
    async with _Host(state).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rows = list(pilot.app.query(".console-changed-files-row"))
        assert len(rows) == 2
        # Capture the FIRST row (drawn for `original_first`, index 0) --
        # this is the stale button a late press event still targets.
        stale_button = rows[0]

        section = pilot.app.query_one(
            "#section-under-test", ConsoleChangedFilesSection
        )
        # Recompose onto a REORDERED entries tuple: `original_first` is now
        # at index 1, a brand-new entry now occupies index 0. Index-based
        # re-resolution would read the row's stale `entry_index == 0` and
        # find the NEW entry instead of `original_first`.
        reordered_new = _entry(
            root="/workspace",
            path="c.py",
            label="c.py",
            run_id="run-c",
            snapshot_id=30,
        )
        section.update_state(
            ConsoleChangedFilesState(entries=(reordered_new, original_first))
        )
        await pilot.pause()

        # Deliver the press directly against the captured (now-detached,
        # stale) button object -- this is exactly what a late-arriving
        # `Button.Pressed` message carries: `event.button` pointing at the
        # widget instance that was pressed, regardless of whether it is
        # still mounted by the time the handler runs.
        await section.on_button_pressed(Button.Pressed(stale_button))
        await pilot.pause()

        assert len(pilot.app.captured) == 1
        event = pilot.app.captured[0]
        assert event.run_id == "run-a"
        assert event.snapshot_id == 10
        assert event.path == "a.py"
        assert event.root == "/workspace", (
            "a stale press must resolve to the ORIGINAL entry it was drawn "
            "for, not whatever entry now occupies its old positional index"
        )


@pytest.mark.asyncio
async def test_same_relative_path_under_two_roots_renders_two_distinct_rows():
    """TASK-2 pin, carried into this widget: a path shared by two roots must
    render as two rows, distinguished via the multi-root prefix already
    baked into each entry's own ``label``.
    """
    state = ConsoleChangedFilesState(
        entries=(
            _entry(
                root="/workspace/repo-a",
                path="src/app.py",
                label="repo-a/src/app.py",
                run_id="run-1",
                snapshot_id=1,
            ),
            _entry(
                root="/workspace/repo-b",
                path="src/app.py",
                label="repo-b/src/app.py",
                run_id="run-1",
                snapshot_id=2,
            ),
        )
    )
    async with _Host(state).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rows = list(pilot.app.query(".console-changed-files-row"))
        assert len(rows) == 2
        labels = [str(row.label) for row in rows]
        assert any("repo-a/src/app.py" in label for label in labels)
        assert any("repo-b/src/app.py" in label for label in labels)
