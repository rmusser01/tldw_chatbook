"""B17: activity log keystroke rebuilds are debounced, bounded, and cheap to refresh.

Covers three efficiency defects in :class:`ActivityLogWidget`:

1. Typing in the search box must not tear down and remount the whole entry
   list per keystroke -- a settled query rebuilds exactly once (0.3 s
   debounce, house ``set_timer`` handle-cancel pattern).
2. A full display rebuild must mount a bounded number of entry rows
   (``_RENDER_CAP``), even when the in-memory store holds ``MAX_ENTRIES``
   entries; data retention is unchanged.
3. The 60 s timestamp refresh must patch the already-mounted timestamp
   labels in place -- zero ``mount`` calls.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Input, Static

# The autouse Tests/UI catalog-refresh fixture lazily imports
# `tldw_chatbook.app`; under the per-test config redirect that first import
# fails the raw-source admission (RecoveryRequired). Importing it here, at
# collection time while the bootstrap config is still bound, is the repo's
# established standalone-run pattern (see Tests/UI/conftest.py's notes).
import tldw_chatbook.app  # noqa: F401,E402
from tldw_chatbook.Widgets.activity_log import (
    ActivityEntry,
    ActivityLogWidget,
)


class _WidgetHostApp(App[None]):
    """Minimal host app mounting a single widget under test."""

    def __init__(self, widget) -> None:
        super().__init__()
        self._hosted_widget = widget

    def compose(self) -> ComposeResult:
        with Container():
            yield self._hosted_widget


def _seed_entries(widget: ActivityLogWidget, count: int) -> None:
    """Populate the backing store directly (newest first, like add_entry)."""
    now = datetime.now()
    for i in range(count):
        widget.entries.appendleft(
            ActivityEntry(
                timestamp=now,
                level="info",
                category="cat",
                message=f"entry-{i}",
            )
        )


class TestActivityLogSearchDebounce:
    """A typing burst rebuilds the display exactly once."""

    @pytest.mark.asyncio
    async def test_five_char_burst_debounces_to_one_rebuild(self):
        log = ActivityLogWidget()
        app = _WidgetHostApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            rebuilds: list[int] = []
            real_update = log._update_display

            def spy_update() -> None:
                rebuilds.append(1)
                real_update()

            log._update_display = spy_update

            search = log.query_one("#search-logs", Input)
            search.focus()
            await pilot.pause()

            await pilot.press(*"abcde")
            # Nothing rebuilds while the debounce window is still open.
            assert rebuilds == []
            await pilot.pause(0.5)

            assert len(rebuilds) == 1, (
                f"expected one debounced rebuild, got {len(rebuilds)}"
            )
            assert log.search_query == "abcde"

    @pytest.mark.asyncio
    async def test_debounce_rearm_collapses_burst_into_single_fire(self):
        log = ActivityLogWidget()
        app = _WidgetHostApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            rebuilds: list[int] = []
            real_update = log._update_display

            def spy_update() -> None:
                rebuilds.append(1)
                real_update()

            log._update_display = spy_update

            search = log.query_one("#search-logs", Input)
            search.focus()
            await pilot.pause()

            # Two bursts separated by less than the debounce window must
            # collapse: the second keystroke cancels the pending timer.
            await pilot.press("x")
            await pilot.pause(0.1)
            await pilot.press("y")
            await pilot.pause(0.1)
            await pilot.press("z")
            await pilot.pause(0.5)

            assert len(rebuilds) == 1
            assert log.search_query == "xyz"


class TestActivityLogRenderCap:
    """A full rebuild mounts at most _RENDER_CAP rows; retention is unchanged."""

    @pytest.mark.asyncio
    async def test_full_rebuild_mounts_at_most_render_cap(self):
        log = ActivityLogWidget(max_entries=1000)
        app = _WidgetHostApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            _seed_entries(log, ActivityLogWidget.MAX_ENTRIES)
            assert len(log.entries) == ActivityLogWidget.MAX_ENTRIES

            log._update_display()
            await pilot.pause()

            rendered = app.query(".log-entry")
            assert len(rendered) <= ActivityLogWidget._RENDER_CAP
            assert len(rendered) == ActivityLogWidget._RENDER_CAP

    @pytest.mark.asyncio
    async def test_render_cap_keeps_newest_entries_and_retention(self):
        log = ActivityLogWidget(max_entries=1000)
        app = _WidgetHostApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            _seed_entries(log, ActivityLogWidget.MAX_ENTRIES)
            log._update_display()
            await pilot.pause()

            # Data retention unchanged.
            assert len(log.entries) == ActivityLogWidget.MAX_ENTRIES

            # The view keeps the newest slice: newest message rendered,
            # oldest message dropped from the view only.
            messages = [
                str(row.query_one(".log-message", Static).render())
                for row in app.query(".log-entry")
            ]
            assert any("entry-999" in m for m in messages)
            assert not any("entry-0" in m for m in messages)


class TestActivityLogTimestampRefresh:
    """The 60 s refresh patches timestamp labels in place (zero mounts)."""

    @pytest.mark.asyncio
    async def test_update_timestamps_mounts_nothing_and_patches_text(self):
        log = ActivityLogWidget()
        app = _WidgetHostApp(log)
        async with app.run_test() as pilot:
            await pilot.pause()

            log.add_entry("hello", "info")
            await pilot.pause()
            assert len(app.query(".log-entry")) == 1

            # Backdate so the relative time actually changes.
            log.entries[0].timestamp = datetime.now() - timedelta(minutes=5)

            container = log.query_one("#log-entries", Container)
            mount_calls: list[tuple] = []
            real_mount = container.mount

            def spy_mount(*args, **kwargs):
                mount_calls.append(args)
                return real_mount(*args, **kwargs)

            container.mount = spy_mount
            try:
                log._update_timestamps()
                await pilot.pause()
            finally:
                del container.mount

            assert mount_calls == [], "timestamp refresh must not mount widgets"

            row = app.query_one(".log-entry")
            timestamp_label = row.query_one(".log-timestamp", Static)
            rendered = str(timestamp_label.render())
            assert "5m ago" in rendered
            # The row itself survived (no teardown/remount churn).
            assert len(app.query(".log-entry")) == 1
