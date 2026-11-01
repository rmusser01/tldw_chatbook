"""TASK-34000.18 fix round 2: an open-by-link under a sticky filter.

In the PR lane census too, appended at its tail after fix round 2 (the lane
file proper is ``test_library_media_actions_target_opened_item.py``; these
boots are its slower siblings, and the one CI guard for a defect the review
found twice). The re-review's probe cases A, B and E: a user filter
is applied FIRST, then an item is opened by link (the Search/RAG route keeps
the current scope, so a filtered-out or off-page item is the common case).
The round-1 scope-based hold failed all three -- row 0 kept the ``▸`` and
the Reader keys went dead, because the key gate needs
``loaded_id == selected_media_id``.

The ruling is ORDER-based: the Reader holds the highlight from the moment an
item is opened or selected for it; only the user's scope-change seams (a
type choice, a typed query, the filter Clear, a sort or page change) release
it. A filter applied AFTER an open is the list moving on (TASK-15457,
unchanged); an open made AFTER the filter is the Reader moving on: no row
is highlighted, the selection equals the Reader's item, and ``l`` acts on
it. Each case asserts exactly those four facts, the last against the
service's record (the test app's DB).
"""

from __future__ import annotations

import pytest

from Tests.UI.test_library_media_actions_target_opened_item import (
    SIZE,
    _highlighted_rows,
    _painted,
    _reader_title,
    _rows,
)
from Tests.UI.test_library_shell import (
    LibraryGlobalKeyProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _items(count: int) -> list[dict[str, object]]:
    """Odd ids are documents, even ids videos; the newest (highest) id is row 0."""
    return [
        {
            "id": str(index),
            "title": f"Item {index:02d}",
            "type": "video" if index % 2 == 0 else "document",
            "last_modified": f"2026-07-{(index // 24) + 1:02d}T{index % 24:02d}:00:00Z",
            "author": "fixture",
            "url": f"file:///fixture/{index}.md",
            "keywords": ["fixture"],
            "content": f"Item {index} body text for the Reader.",
            "version": 1,
        }
        for index in range(1, count + 1)
    ]


def _host(count: int):
    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, _two_conversations(), media=_items(count))
    # The global-key harness: F6 is an app-level key (case E leaves the
    # filter box with it), exactly as ``test_media_global_f6_reaches_
    # content_scroller`` drives it.
    return LibraryGlobalKeyProductionCSSHarness(app), app.media_reading_scope_service


async def _open_and_settle(screen, pilot, record_id: str) -> str:
    canonical = f"local:media:{record_id}"
    await screen._open_library_item_by_id("media", record_id)
    await _wait_for_selector(screen, pilot, "#library-media-row-0")
    browse = screen._library_media_browse_controller

    def landed() -> bool:
        session = screen._media_state.reader_session
        return (
            session.loaded_id == canonical
            and session.pending_request is None
            and not browse.loading
            and browse.applied_scope is not None
            and browse.applied_scope == browse.requested_scope
            and len(_rows(screen)) > 0
        )

    await _wait_for_condition(pilot, landed, message=f"open of {record_id!r} never settled")
    for _ in range(4):
        await pilot.pause()
    return canonical


async def _apply_type_filter(screen, pilot, value: str) -> None:
    """The user's gesture: the type chooser, through the real option list."""
    button = await _wait_for_selector(screen, pilot, "#library-media-type-filter")
    button.focus()
    await pilot.pause()
    button.press()
    chooser = await _wait_for_selector(screen, pilot, "#library-media-type-choices")
    chooser.highlighted = next(
        index
        for index, option in enumerate(chooser.options)
        if getattr(option, "choice_value", None) == value
    )
    chooser.action_select()
    browse = screen._library_media_browse_controller
    await _wait_for_condition(
        pilot,
        lambda: browse.applied_scope is not None
        and browse.applied_scope.media_type == value
        and not browse.loading
        and browse.applied_scope == browse.requested_scope,
        message=f"type filter {value!r} never applied",
    )
    for _ in range(4):
        await pilot.pause()


async def _apply_query(screen, pilot, query: str, *, rows: int) -> None:
    """A typed query through the filter box's own submit handler."""
    box = await _wait_for_selector(screen, pilot, "#library-media-filter")
    box.value = query
    await pilot.pause()
    screen._request_library_media_filter(query)
    browse = screen._library_media_browse_controller
    await _wait_for_condition(
        pilot,
        lambda: browse.applied_scope is not None
        and browse.applied_scope.query == query
        and not browse.loading
        and browse.applied_scope == browse.requested_scope
        and len(_rows(screen)) == rows,
        message=f"query {query!r} never applied",
    )
    for _ in range(4):
        await pilot.pause()


async def _assert_reader_holds_the_opened_item(host, screen, pilot, service, record_id: str):
    canonical = f"local:media:{record_id}"
    assert screen._media_state.reader_session.loaded_id == canonical
    assert _reader_title(screen) == f"Item {int(record_id):02d}"
    # 1. no row painted as highlighted
    assert _highlighted_rows(screen) == [], [str(r.media_id) for r in _highlighted_rows(screen)]
    painted = [_painted(host, row.region) for row in _rows(screen)]
    assert not any("▸" in text for text in painted), painted
    # 2. the selection equals the Reader's item
    assert screen._media_state.selected_media_id == canonical
    # 3. the accelerator is live
    assert screen.check_action("library_media_read_later", ()) is True
    # 4. a real keypress saves the OPENED item
    before = len(service.read_it_later_calls)
    await pilot.press("l")
    await _wait_for_condition(
        pilot,
        lambda: len(service.read_it_later_calls) > before,
        message="l never reached the service",
    )
    assert service.read_it_later_calls[-1] == {"action": "save", "media_id": int(record_id)}


async def test_case_a_type_filter_then_open_a_filtered_out_item_by_link():
    """Case A: 6 items, filter video, open document 5 by bare id."""
    host, service = _host(6)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_and_settle(screen, pilot, "6")
        await _apply_type_filter(screen, pilot, "video")
        assert [str(r.media_id) for r in _rows(screen)] == [
            "local:media:6", "local:media:4", "local:media:2"
        ]

        await _open_and_settle(screen, pilot, "5")

        await _assert_reader_holds_the_opened_item(host, screen, pilot, service, "5")


async def test_case_b_type_filter_then_open_an_off_page_matching_item_by_link():
    """Case B: 50 items, filter video (25 match, page size 20), open video 2."""
    host, service = _host(50)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_and_settle(screen, pilot, "50")
        await _apply_type_filter(screen, pilot, "video")
        page_ids = [str(r.media_id) for r in _rows(screen)]
        assert len(page_ids) == 20 and "local:media:2" not in page_ids

        await _open_and_settle(screen, pilot, "2")

        await _assert_reader_holds_the_opened_item(host, screen, pilot, service, "2")


async def test_case_e_query_filter_then_open_a_non_matching_item_by_link():
    """Case E: a typed query narrows the list to one row, then open-by-link."""
    host, service = _host(6)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_and_settle(screen, pilot, "6")
        await _apply_query(screen, pilot, "Item 04", rows=1)
        assert [str(r.media_id) for r in _rows(screen)] == ["local:media:4"]

        await _open_and_settle(screen, pilot, "2")

        # The query request parked focus in the Items filter box, where a
        # printable ``l`` is text, not the accelerator (the guide: "not
        # while the Find box or the Items filter has focus"). The user's
        # documented way out is F6 -- Items -> the Reader's content box
        # (``test_media_global_f6_reaches_content_scroller`` pins that
        # order) -- and only then is ``l`` the Reader's key.
        assert getattr(screen.focused, "id", None) == "library-media-filter"
        await pilot.press("f6")
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None)
            == "library-media-viewer-content-text",
            message=lambda: f"F6 did not reach the Reader: {screen.focused!r}",
        )

        await _assert_reader_holds_the_opened_item(host, screen, pilot, service, "2")


async def test_a_type_filter_after_the_open_releases_the_hold_while_the_viewer_stays_open():
    """Re-review R2's parked pin: the user's scope-change seam
    (``_clear_library_media_selection_for_scope_change``) is the one release
    point of the ORDER-based hold, and it releases it while the viewer is
    still the active view -- the Reader keeps showing the opened document,
    but from this scope change on the selection follows the canvas
    (TASK-15457: row 0 wears the cue, because the opened item is gone from
    the page). Mutation-checked: dropping the release at that seam leaves
    ``highlight_held_by_reader`` True and the selection on the filtered-out
    item, and this fails."""
    host, _service = _host(6)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        canonical = await _open_and_settle(screen, pilot, "5")
        assert screen._media_state.highlight_held_by_reader is True
        assert screen._media_state.selected_media_id == canonical

        await _apply_type_filter(screen, pilot, "video")

        # The viewer is still the stage and still shows document 5 ...
        assert screen._media_state.view == "viewer"
        assert screen._media_state.reader_session.loaded_id == canonical
        assert _reader_title(screen) == "Item 05"
        # ... but the hold is released and the list has moved on: row 0.
        assert screen._media_state.highlight_held_by_reader is False
        assert [str(r.media_id) for r in _rows(screen)] == [
            "local:media:6", "local:media:4", "local:media:2"
        ]
        assert screen._media_state.selected_media_id == "local:media:6"
        assert [str(r.media_id) for r in _highlighted_rows(screen)] == ["local:media:6"]
        assert "▸" in _painted(host, _rows(screen)[0].region)
