"""TASK-22500 Task 6: the three scroller lookups must resolve for real.

``_capture_library_media_loaded_progress``, ``_restore_library_media_loaded_progress``,
and ``_scroll_library_media_content_to_line`` in ``library_screen.py`` each do
``self.query_one("#library-media-viewer-content", VerticalScroll)`` inside a
``try/except (NoMatches, QueryError)``. Since Task 5 the body mounted at that id
is a ``LibraryMediaContentBody`` (a plain ``Container``), so that lookup raises
``WrongType`` -- which IS a ``QueryError`` subclass -- and the except swallows
it. The reader then silently loses scroll capture, scroll restore, and match
scrolling: no exception, no log, just a no-op.

The existing unit tests covering the capture/restore sites
(``Tests/UI/test_library_media_reader_flow.py``) call the bound methods on a
``SimpleNamespace`` fake whose ``query_one`` is ``lambda *_args, **_kwargs:
SimpleNamespace(...)`` -- it ignores the expected-type argument entirely, so
it cannot observe a ``WrongType`` regression. That is why two of these three
breakages were invisible to CI.

The tests below instead drive a REAL mounted ``LibraryScreen`` and call the
screen's own (unmocked) ``query_one``. If any of the three sites regresses --
e.g. back to querying ``VerticalScroll`` -- the real ``query_one`` raises
``WrongType``, the site's except swallows it, and the assertions below fail
because the observable effect (a populated capture dict, a moved scroll
offset, an exact wrap-mapped scroll target) never happens.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_library_media_reader_flow import (
    _flow_app,
    _row_identity,
    _wait_for_detail_call,
)
from Tests.UI.test_library_media_side_by_side import WIDE_SIZE, _open_media_list
from Tests.UI.test_library_shell import LibraryProductionCSSHarness, _wait_for_condition
from tldw_chatbook.Widgets.Library.library_media_content import LibraryMediaContentBody


def _document(lines: int = 400) -> str:
    """Build a plain-text document of short, non-wrapping lines.

    Copied from ``Tests/UI/test_library_media_reader_match_nav_t22209.py``
    (module-level helper, no shared fixture to import instead).
    """
    return "\n".join(f"line {index}: ordinary reading material" for index in range(lines))


def _wrapping_document(
    target_line: int = 40, long_line_length: int = 400, trailing_lines: int = 60
) -> str:
    """Build a document whose first line wraps across several virtual rows.

    ``target_line`` is an ordinary short line, so treating its source-line
    index as a screen row (the pre-Task-6 behavior) lands above where it
    actually renders once line 0 wraps. Line 0 is padded well past any
    plausible content-panel width so it wraps regardless of layout, and
    ``trailing_lines`` pads the document well past the Raw view's
    ``max_visible_rows`` (18) so the view actually has something to scroll
    (a document that fits entirely within the viewport clamps every
    ``scroll_to`` to a no-op).
    """
    lines = ["A" * long_line_length]
    lines.extend(f"filler line {index}" for index in range(1, target_line))
    lines.append(f"line {target_line}: the target line")
    lines.extend(
        f"filler line {index}"
        for index in range(target_line + 1, target_line + trailing_lines)
    )
    return "\n".join(lines)


def _seed_row_document(screen, service, index: int, content: str):
    """Give row ``index``'s backing item ``content`` before it is fetched.

    Copied from ``test_library_media_reader_match_nav_t22209.py``.
    """
    row = screen.query_one(f"#library-media-row-{index}", Button)
    canonical_id, backing_id, title = _row_identity(row)
    source = next(
        item for item in service.media_items if item["id"] == f"media-{backing_id}"
    )
    source["content"] = content
    return canonical_id, backing_id, title


async def _load_row_with_document(screen, pilot, service, index: int, content: str):
    """Seed row ``index`` with ``content`` and open it from the Items list.

    Copied from ``test_library_media_reader_match_nav_t22209.py``.
    """
    canonical_id, backing_id, title = _seed_row_document(screen, service, index, content)
    screen.query_one(f"#library-media-row-{index}", Button).press()
    await _wait_for_detail_call(service, backing_id)
    service.release(backing_id)
    await _wait_for_condition(
        pilot,
        lambda: screen._library_media_reader_session.loaded_id == canonical_id,
        message=f"Row {index} never settled its detail.",
    )
    return canonical_id, backing_id, title


async def _open_raw_view_ready(screen, pilot) -> LibraryMediaContentBody:
    """Return the mounted body once its Raw view has a built wrap index.

    Session state settling (what ``_load_row_with_document`` waits for) can
    land a pump or two before the Raw view's own ``on_mount``/``on_resize``
    layout pass gives it a real size and builds ``wrap_index`` -- without
    this, an immediate ``scroll_to`` right after load can be clamped to 0
    against a still-zero ``virtual_size``.
    """
    body = screen.query_one("#library-media-viewer-content", LibraryMediaContentBody)
    await _wait_for_condition(
        pilot,
        lambda: body.raw_view is not None and body.raw_view.wrap_index is not None,
        message="Raw view's wrap index never built (no layout pass yet).",
    )
    return body


@pytest.mark.asyncio
async def test_capture_progress_resolves_the_real_scroller_and_snapshots_its_offset():
    """``_capture_library_media_loaded_progress`` must find the real scroller.

    Scrolls the mounted Raw view for real, then calls the screen's actual
    (unmocked) capture method and asserts the exact offset it recorded. A
    regression to querying ``VerticalScroll`` makes ``query_one`` raise
    ``WrongType``, the capture method returns early, and
    ``_library_media_read_scroll_by_id`` stays empty -- this assertion fails.
    """
    app, service = _flow_app(count=4)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        canonical_id, _backing_id, _title = await _load_row_with_document(
            screen, pilot, service, 0, _document()
        )
        body = await _open_raw_view_ready(screen, pilot)

        body.scroller.scroll_to(y=37, animate=False, immediate=True)
        await pilot.pause()
        assert body.scroller.scroll_y > 0, "Fixture scroll did not move -- test setup is broken."

        screen._library_media_read_scroll_by_id.clear()
        screen._capture_library_media_loaded_progress()

        assert screen._library_media_read_scroll_by_id.get(canonical_id) == (
            int(body.scroller.scroll_x),
            int(body.scroller.scroll_y),
        )


@pytest.mark.asyncio
async def test_restore_progress_resolves_the_real_scroller_and_applies_the_saved_offset():
    """``_restore_library_media_loaded_progress`` must scroll the real target.

    Seeds a saved offset directly (mirroring what capture would have
    written) and calls the screen's actual restore method. A regression to
    querying ``VerticalScroll`` makes ``query_one`` raise ``WrongType``, the
    restore method returns before calling ``scroll_to``, and the scroller
    stays at its rest position -- this assertion fails.
    """
    app, service = _flow_app(count=4)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        canonical_id, _backing_id, _title = await _load_row_with_document(
            screen, pilot, service, 0, _document()
        )
        body = await _open_raw_view_ready(screen, pilot)
        assert body.scroller.scroll_y == 0, "Fixture must start unscrolled."

        screen._library_media_read_scroll_by_id[canonical_id] = (0, 41)
        screen._restore_library_media_loaded_progress(canonical_id)
        await pilot.pause()

        assert body.scroller.scroll_y == 41


@pytest.mark.asyncio
async def test_match_scroll_resolves_the_real_scroller_and_maps_through_the_wrap_index():
    """``_scroll_library_media_content_to_line`` must land exactly, post-wrap.

    Line 0 of the fixture wraps across several virtual rows, so the target
    line's true virtual row is strictly greater than its source-line index.
    The old code passed the source-line index straight to ``scroll_to(y=...)``
    as if it were already a screen row, which drifts once anything wraps --
    this asserts the scroller lands on the WRAP-INDEX-mapped row, not the
    raw source-line index. A regression to querying ``VerticalScroll`` (or
    to the old raw ``scroll_to(y=line_index)`` call) makes this fail either
    by leaving the scroller un-moved or by landing on the wrong row.
    """
    app, service = _flow_app(count=4)
    host = LibraryProductionCSSHarness(app)
    target_line = 40

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_with_document(
            screen, pilot, service, 0, _wrapping_document(target_line)
        )
        body = await _open_raw_view_ready(screen, pilot)
        raw_view = body.raw_view
        expected_row = raw_view.wrap_index.line_start_row(target_line)
        assert expected_row > target_line, (
            "Fixture line 0 must actually wrap for this check to mean anything "
            f"(expected_row={expected_row}, target_line={target_line})."
        )
        assert expected_row <= raw_view.max_scroll_y, (
            "Fixture must be tall enough for the target row to be reachable "
            f"(expected_row={expected_row}, max_scroll_y={raw_view.max_scroll_y})."
        )

        screen._scroll_library_media_content_to_line(target_line)
        # `scroll_to_source_line` calls `scroll_to(..., animate=False)`
        # without `immediate=True`, so Textual defers applying it to the
        # next screen refresh -- poll rather than assume one pump suffices.
        await _wait_for_condition(
            pilot,
            lambda: raw_view.scroll_offset.y == expected_row,
            message=(
                f"Raw view never scrolled to row {expected_row} "
                f"(stuck at {raw_view.scroll_offset.y})."
            ),
        )
        assert raw_view.scroll_offset.y != target_line
