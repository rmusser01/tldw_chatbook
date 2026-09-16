"""Import retains the user's live control through Library resize transitions."""

from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Collapsible, Input

from Tests.UI.test_library_ingest_queue_journeys import _queue_host, _stage_draft
from Tests.UI.test_library_ingest_recent_journeys import RECENT_TITLE
from Tests.UI.test_library_prompt_collection_journeys import _focus
from tldw_chatbook.Widgets.Library.library_ingest_canvas import LibraryIngestCanvas


def _painted(host, widget):
    region = widget.content_region
    strips = list(host.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["title", "encoding", "details", "recent"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_import_resize_preserves_visible_control_and_draft_without_data_work(
    tmp_path, monkeypatch, control, theme
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    source, app, host = _queue_host(tmp_path, theme)
    registry = app.library_ingest_jobs
    failed = registry.submit(source_path=str(tmp_path / "failed.txt"))
    registry.mark_failed(failed.job_id, error="Fixture failure")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = host.screen
        title = await _stage_draft(screen, pilot, source)
        selectors = {
            "title": ("#library-ingest-title", "Unsaved next import"),
            "encoding": ("#opt-generic-encoding", "Auto-detect"),
            "details": (f"#library-ingest-details-{failed.job_id}", "Hide details"),
            "recent": (RECENT_TITLE, "Recent imports"),
        }
        selector, label = selectors[control]
        field = screen.query_one(selector)
        field.focus()
        await pilot.pause()
        if control in {"details", "recent"}:
            await pilot.press("enter")
            field = screen.query_one(selector)
        await _focus(screen, host, pilot, selector, label)
        title.selection = type(title.selection)(2, 7)
        selection = title.selection
        canvas = screen.query_one(LibraryIngestCanvas)
        snapshot = Mock(wraps=screen._refresh_local_source_snapshot)
        persistence = AsyncMock(wraps=screen._persist_library_reader_preference)
        preflight = Mock(wraps=screen._trigger_library_ingest_preflight)
        monkeypatch.setattr(screen, "_refresh_local_source_snapshot", snapshot)
        monkeypatch.setattr(screen, "_persist_library_reader_preference", persistence)
        monkeypatch.setattr(screen, "_trigger_library_ingest_preflight", preflight)
        jobs = registry.jobs()
        observations = []
        for size in ((80, 24), (170, 48), (170, 24), (170, 48)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            observations.append(
                (
                    size,
                    screen.focused is field,
                    label in _painted(host, field),
                    repr(screen.focused),
                    tuple(field.region),
                )
            )
            assert screen.query_one(selector) is field
            assert screen.query_one(LibraryIngestCanvas) is canvas
            assert screen.query_one("#library-ingest-title", Input) is title
            assert title.value == "Unsaved next import"
            assert title.selection == selection
            assert screen._ingest_state.form.path == str(source)
            if control == "recent":
                assert not screen.query_one(
                    "#library-ingest-recent", Collapsible
                ).collapsed
        assert all(focused and painted for _, focused, painted, *_ in observations), (
            observations
        )
        assert registry.jobs() == jobs
        snapshot.assert_not_called()
        persistence.assert_not_called()
        preflight.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("outside_import", [False, True])
@pytest.mark.parametrize("size", [(80, 24), (170, 24)])
async def test_newer_focus_wins_over_deferred_import_resize(
    tmp_path, monkeypatch, outside_import, size
):
    source, _, host = _queue_host(tmp_path, "textual-dark")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = host.screen
        title = await _stage_draft(screen, pilot, source)
        title.focus()
        await _focus(
            screen, host, pilot, "#library-ingest-title", "Unsaved next import"
        )
        canvas = screen.query_one(LibraryIngestCanvas)
        pending = []

        def defer(callback, *args, **kwargs):
            pending.append((callback, args, kwargs))
            return True

        monkeypatch.setattr(canvas, "call_after_refresh", defer)
        await pilot.resize_terminal(*size)
        assert pending
        if outside_import:
            target = screen.query_one(
                "#library-rail-open" if size[0] == 80 else "#library-search-input"
            )
        else:
            target = screen.query_one("#library-ingest-author")
        target.focus()
        await pilot.pause()
        assert screen.focused is target
        offset = canvas.scroll_offset
        for callback, args, kwargs in tuple(pending):
            callback(*args, **kwargs)
        await pilot.pause()
        assert screen.focused is target
        if outside_import:
            assert canvas.scroll_offset == offset
        else:
            assert "e.g. Ada Lovelace" in _painted(host, target)
        assert title.value == "Unsaved next import"
        assert screen._ingest_state.form.path == str(source)
