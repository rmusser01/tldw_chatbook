"""Continuous Library Media traversal and authoritative filtering contracts."""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_library_media_side_by_side import (
    WIDE_SIZE,
    _build_media_test_app,
    _many_media_items,
    _open_media_list,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    StaticLibraryMediaScopeService,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
)
from tldw_chatbook.Library.library_media_state import (
    MediaBrowseScope,
    build_library_media_browse_state,
    build_media_browse_result,
)


class ControlledDetailMediaService(StaticLibraryMediaScopeService):
    """Gate each detail response independently without time-based sleeps."""

    def __init__(self, media_items):
        super().__init__(media_items)
        self.detail_entered: dict[int, threading.Event] = {}
        self.detail_release: dict[int, threading.Event] = {}
        self.detail_outcomes: dict[int, object] = {}

    def get_media_item(self, *, media_id, **kwargs):
        self.detail_calls.append({"media_id": media_id, **kwargs})
        entered = self.detail_entered.setdefault(media_id, threading.Event())
        release = self.detail_release.setdefault(media_id, threading.Event())
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError(f"Timed out waiting to release detail {media_id}.")
        outcome = self.detail_outcomes.get(media_id)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is not None:
            return outcome
        for index, item in enumerate(self.media_items):
            if self._backing_id(item, index) == media_id:
                return dict(item)
        return None

    def release(self, media_id: int, outcome: object | None = None) -> None:
        if outcome is not None:
            self.detail_outcomes[media_id] = outcome
        self.detail_release.setdefault(media_id, threading.Event()).set()


def _flow_app(count: int = 65):
    app = _build_media_test_app()
    items = _many_media_items(count)
    _seed_conversations(app, _two_conversations(), media=items)
    service = ControlledDetailMediaService(items)
    app.media_reading_scope_service = service
    return app, service


def _row_identity(row: Button) -> tuple[str, int, str]:
    canonical_id = str(row.media_id)
    backing_id = int(canonical_id.rsplit(":", 1)[-1])
    title = str(row._library_media_title)
    return canonical_id, backing_id, title


async def _wait_for_detail_call(
    service: ControlledDetailMediaService, backing_id: int
) -> None:
    try:
        started = await asyncio.to_thread(
            service.detail_entered.setdefault(backing_id, threading.Event()).wait,
            2,
        )
        if not started:
            raise TimeoutError
    except TimeoutError:
        pytest.fail(f"Detail call for backing id {backing_id} did not start.")


@pytest.mark.asyncio
async def test_arrow_traversal_updates_selection_immediately_but_loads_only_settled_row():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        first = screen.query_one("#library-media-row-0", Button)
        first_id, first_backing_id, _ = _row_identity(first)
        first.press()
        await _wait_for_detail_call(service, first_backing_id)
        service.release(first_backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == first_id,
            message="Initial settled row did not load.",
        )
        service.detail_calls.clear()
        row_1 = _row_identity(screen.query_one("#library-media-row-1", Button))
        row_2 = _row_identity(screen.query_one("#library-media-row-2", Button))
        screen._select_library_media_reader_row(row_1[0], row_1[2], immediate=False)
        screen._select_library_media_reader_row(row_2[0], row_2[2], immediate=False)

        selected, expected_backing_id, _ = row_2
        assert screen._selected_media_id == selected
        await _wait_for_condition(
            pilot,
            lambda: bool(service.detail_calls),
            message="Settled traversal never dispatched detail.",
        )
        assert [call["media_id"] for call in service.detail_calls] == [
            expected_backing_id
        ]
        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_enter_bypasses_selection_settle_window():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-1", Button)
        _, expected_backing_id, _ = _row_identity(row)
        row.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: bool(service.detail_calls),
            message="Enter did not dispatch detail immediately.",
        )
        assert service.detail_calls[0]["media_id"] == expected_backing_id
        service.release(expected_backing_id)


@pytest.mark.asyncio
async def test_pending_banner_names_selected_b_and_loaded_a():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row_a = screen.query_one("#library-media-row-0", Button)
        canonical_a, backing_a, title_a = _row_identity(row_a)
        row_a.press()
        await _wait_for_detail_call(service, backing_a)
        service.release(backing_a)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == canonical_a,
            message="First detail did not load.",
        )
        row_b = screen.query_one("#library-media-row-1", Button)
        _, backing_b, title_b = _row_identity(row_b)
        row_b.press()
        await _wait_for_detail_call(service, backing_b)
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-viewer-loading")),
            message="Reader never painted its pending banner.",
        )

        banner = screen.query_one("#library-media-viewer-loading", Static)
        copy = str(banner.renderable)
        assert title_b in copy
        assert title_a in copy
        service.release(backing_b)


@pytest.mark.asyncio
@pytest.mark.parametrize("late_outcome", [None, RuntimeError("late failure")])
async def test_late_completion_for_a_cannot_replace_loaded_b_or_show_error(
    late_outcome,
):
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row_a = screen.query_one("#library-media-row-0", Button)
        row_b = screen.query_one("#library-media-row-1", Button)
        _, backing_a, _ = _row_identity(row_a)
        canonical_b, backing_b, title_b = _row_identity(row_b)
        row_a.press()
        await _wait_for_detail_call(service, backing_a)
        screen._select_library_media_reader_row(canonical_b, title_b, immediate=True)
        await _wait_for_detail_call(service, backing_b)
        service.release(backing_b)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == canonical_b,
            message="B detail did not load.",
        )

        service.release(backing_a, late_outcome)
        await pilot.pause()
        assert screen._library_media_reader_session.loaded_id == canonical_b
        assert screen._library_media_reader_session.error is None
        assert screen._library_media_detail["id"] == f"media-{backing_b}"


@pytest.mark.asyncio
async def test_detail_failure_keeps_items_usable_and_reader_retryable():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        _, backing_id, _ = _row_identity(row)
        row.press()
        await _wait_for_detail_call(service, backing_id)
        service.release(backing_id, RuntimeError("private"))
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.error is not None,
            message="Detail failure did not settle as retryable.",
        )

        assert screen.query_one("#library-media-row-1", Button).disabled is False
        assert screen._library_media_view == "viewer"
        assert screen._library_media_reader_session.pending_request is not None
        retry = screen.query_one("#library-media-reader-retry", Button)
        service.detail_outcomes.pop(backing_id)
        retry.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_backing_id
            == backing_id,
            message="Reader-local Retry did not reload the failed item.",
        )
        assert screen._library_media_reader_session.error is None


def test_rows_expose_textual_loading_and_loaded_state():
    scope = MediaBrowseScope()
    result = build_media_browse_result(
        scope,
        {
            "items": [
                {
                    "id": "local:media:1",
                    "backing_media_id": 1,
                    "title": "A",
                    "media_type": "audio",
                    "updated_at": None,
                },
                {
                    "id": "local:media:2",
                    "backing_media_id": 2,
                    "title": "B",
                    "media_type": "video",
                    "updated_at": None,
                },
            ],
            "total": 2,
            "limit": 20,
            "offset": 0,
        },
    )

    state = build_library_media_browse_state(
        result,
        type_options=(),
        selected_id="local:media:2",
        loading_id="local:media:2",
        loaded_id="local:media:1",
    )

    assert state.rows[0].loaded is True
    assert state.rows[1].loading is True


@pytest.mark.asyncio
async def test_filter_uses_authoritative_search_and_restores_page_three_anchor():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-next")),
            message="Page 1 pager did not settle.",
        )
        screen.query_one("#library-media-next", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_browse_controller.applied_scope.page == 2
                and bool(screen.query("#library-media-next"))
            ),
            message="Page 2 did not apply and settle its pager.",
        )
        screen.query_one("#library-media-next", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_browse_controller.applied_scope.page == 3
                and bool(screen.query("#library-media-row-10"))
            ),
            message="Page 3 did not apply and settle its rows.",
        )
        anchor = screen.query_one("#library-media-row-10", Button)
        anchor_id, anchor_backing_id, _ = _row_identity(anchor)
        anchor.press()
        await _wait_for_detail_call(service, anchor_backing_id)
        service.release(anchor_backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == anchor_id,
            message="Page-3 anchor did not load.",
        )

        media_filter = screen.query_one("#library-media-filter", Input)
        media_filter.value = "Media item 03"
        await _wait_for_condition(
            pilot,
            lambda: any(
                call.get("query") == "Media item 03" for call in service.search_calls
            ),
            message="Filter did not call authoritative search.",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_browse_controller.applied_scope.query
                == "Media item 03"
            ),
            message="Filter result did not apply.",
        )
        filtered_id = str(
            screen._library_media_browse_controller.retained_items[0]["id"]
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_reader_session.selected_id == filtered_id
                and screen._library_media_reader_session.pending_request is not None
            ),
            message="The first authoritative filter result was not selected in Reader.",
        )
        screen.query_one("#library-media-filter-clear", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_browse_controller.applied_scope
                == MediaBrowseScope(page=3)
            ),
            message="Unfiltered page 3 was not restored.",
        )

        assert screen._selected_media_id == anchor_id
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == anchor_id,
            message="Clearing the filter did not restore the Reader anchor.",
        )
        assert (
            len({row.media_id for row in screen._build_library_media_state().rows})
            == 20
        )
        assert [
            call["offset"] for call in service.search_calls if not call.get("query")
        ][-1] == 40
        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_entering_select_mode_cancels_pending_single_item_settlement():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-row-1", Button).focus()
        screen.query_one("#library-media-select-toggle", Button).press()
        await pilot.pause()

        assert screen._library_media_select_mode is True
        assert screen._library_media_reader_session.pending_request is None
        assert service.detail_calls == []
