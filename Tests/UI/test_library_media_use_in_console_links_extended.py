"""TASK-34000.24, extended arms (outside the UI PR-gate lane).

- Review Focus 5: a very long workspace name. The receipt row under the
  Reader's primary toolbar must stay ONE row -- the text ellipsises -- and
  the "Undo link" button must keep its own cells, at the narrow 80x24 Reader
  as well as the wide one.
- Fix round 1, M-9: at the review's primary size, 160x45, the whole default
  receipt ("Linked to Local Default · staged in Console") is painted beside
  "Undo link".
- Fix round 1, I-1: an item OUTSIDE the depth state's first-page sample of
  local media (``LIBRARY_SOURCE_PAGE_SIZES["media"]`` = 50) links on use from
  both entry points. The depth state has no row for it, so the aggregate
  verdict -- which refuses whenever any sampled row is unlinked -- must not
  be consulted; the seam reads memberships itself.

The lane file pins the behaviour at the standard size; these arms add the
sizes, the name length and the 52-item library (one boot each, ~4-6 s apiece
on a quiet box).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import StaticLibraryMediaScopeService
from Tests.UI.test_library_media_use_in_console_links import (
    LINKED_RECEIPT,
    RAW_ID,
    _media_host,
    _memberships,
    _notified,
    _open_media_reader,
    _painted,
    _press_use_in_console,
    _receipt_widget_text,
    _run_search_and_select_first,
    _search_result_for,
    _wait_for_receipt,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    _StaticLibraryRagSearchService,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
)
from tldw_chatbook.UI.Library_Modules.screen_constants import LIBRARY_SOURCE_PAGE_SIZES

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

LONG_NAME = "Thesis chapter four interview analysis pass"  # 43 characters


@pytest.mark.parametrize("size", ((80, 24), LIBRARY_TEST_SIZE))
async def test_a_long_workspace_name_keeps_the_receipt_on_one_row(size):
    host, app = _media_host()
    registry = app.workspace_registry_service
    long = registry.create_workspace(workspace_id="ws-long-name", name=LONG_NAME)
    registry.set_active_workspace(long.workspace_id)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_media_reader(screen, pilot)

        await _press_use_in_console(screen, pilot)

        assert [row.workspace_id for row in _memberships(app)] == ["ws-long-name"]
        full_line = f"Linked to {LONG_NAME} · staged in Console"
        # The widget carries the whole line; what is PAINTED is the head of
        # it, ellipsised to the row the Reader can give it beside "Undo link"
        # (at 170x48 the Reader row has ~45 cells for it, fewer than the
        # 53-character head "Linked to <name>"), so the painted check is the
        # prefix and the first word of the name, never the full sentence.
        await _wait_for_condition(
            pilot,
            lambda: _receipt_widget_text(screen) == full_line
            and "Linked to Thesis" in _painted(screen),
            message=lambda: (
                f"long-name receipt never painted; widget="
                f"{_receipt_widget_text(screen)!r}"
            ),
        )
        receipt = screen.query_one("#library-media-link-receipt", Static)
        undo = screen.query_one("#library-media-link-undo", Button)
        assert receipt.region.height == 1, receipt.region
        assert undo.region.height == 1 and undo.region.width >= len("Undo link"), (
            undo.region
        )
        assert receipt.region.y == undo.region.y, (receipt.region, undo.region)
        assert undo.region.right <= screen.size.width
        painted = _painted(screen)
        assert "Undo link" in painted
        assert "ws-long-name" not in painted


async def test_the_default_receipt_fits_beside_undo_at_160x45():
    """Fix round 1, M-9: the review's primary size paints the whole default
    receipt and the Undo on one row (the first live capture clipped it to
    "staged in Conso…")."""
    host, app = _media_host()

    async with host.run_test(size=(160, 45)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_media_reader(screen, pilot)

        await _press_use_in_console(screen, pilot)

        await _wait_for_condition(
            pilot,
            lambda: _receipt_widget_text(screen) == LINKED_RECEIPT,
            message="receipt never recorded",
        )
        receipt = screen.query_one("#library-media-link-receipt", Static)
        undo = screen.query_one("#library-media-link-undo", Button)
        row_widget = screen.query_one("#library-media-link-receipt-row")
        viewer = screen.query_one("#library-media-viewer")
        geometry = (
            f"viewer={viewer.content_region} row={row_widget.region} "
            f"receipt={receipt.region} undo={undo.region} "
            f"undo.styles.padding={undo.styles.padding} margin={undo.styles.margin}"
        )
        await _wait_for_condition(
            pilot,
            lambda: "Linked to Local Default" in _painted(screen),
            message="receipt head never painted",
        )
        painted_row = next(
            row for row in _painted(screen).splitlines() if "Linked to Local" in row
        )
        assert LINKED_RECEIPT in painted_row, f"{painted_row!r} {geometry}"
        assert "Undo link" in painted_row, painted_row
        assert "…" not in painted_row, painted_row
        assert receipt.region.y == undo.region.y


class _PagingMediaScopeService(StaticLibraryMediaScopeService):
    """The static fake honouring the one paging shape the depth-state loader
    sends (``page`` + ``results_per_page``), so a 52-item library really
    leaves items 51 and 52 outside the first-page sample as production does."""

    async def list_media_items(self, **kwargs):
        self.calls.append(kwargs)
        items = list(self.media_items)
        page = int(kwargs.get("page") or 1)
        per_page = kwargs.get("results_per_page")
        if per_page:
            per_page = int(per_page)
            items = items[(page - 1) * per_page : page * per_page]
        return {
            "items": items,
            "pagination": {"total_items": len(self.media_items)},
        }


def _fifty_two_media_items() -> list[dict]:
    return [
        {
            "id": str(index + 1),
            "title": f"Media item {index + 1:02d}",
            "type": ("video", "audio", "PDF")[index % 3],
            "last_modified": f"2026-07-{(index % 28) + 1:02d}T10:00:00Z",
            "content": f"Transcript for media item {index + 1}.",
            "version": 1,
        }
        for index in range(52)
    ]


async def test_an_item_outside_the_depth_state_sample_links_from_u_and_c():
    """Fix round 1, I-1: items 51 and 52 have no depth-state row; every
    sampled row is unlinked, so the aggregate verdict would refuse. Search
    `u` on 51 and the reader's `c` on 52 must both link and stage."""
    assert LIBRARY_SOURCE_PAGE_SIZES["media"] == 50
    host, app = _media_host(media=_fifty_two_media_items())
    app.media_reading_scope_service = _PagingMediaScopeService(
        _fifty_two_media_items()
    )
    app.library_rag_search_service = _StaticLibraryRagSearchService(
        {"results": [_search_result_for("51")]}
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        state = screen._library_workspace_depth_state(refresh=True)
        sampled = {row.item_id for row in state.source_rows if row.item_type == "media"}
        assert "1" in sampled and "51" not in sampled and "52" not in sampled, sampled
        assert not state.context_handoff_enabled, "control: the aggregate gate is closed"

        # Search `u` on item 51 (no row).
        await _run_search_and_select_first(screen, pilot)
        await pilot.press("u")
        await pilot.pause()
        await pilot.pause()
        app.open_console_for_live_work.assert_called_once()
        assert [(row.workspace_id, row.item_id) for row in _memberships(app, item_id="51")] == [
            (RAW_ID, "51")
        ]
        assert LINKED_RECEIPT in _notified(app), _notified(app)

        # Reader `c` on item 52 (no row), opened by id as the Search deep link
        # and the ingest "Open in Library" fallback do.
        screen._navigate_to_media("52")
        await _wait_for_condition(
            pilot,
            lambda: screen._media_state.reader_session.loaded_id == "local:media:52"
            and screen._media_state.reader_session.pending_request is None,
            message="item 52 never loaded in the Reader",
        )
        # The deep-link open switches the rail row and mounts the Reader
        # after the detail lands; wait for the control itself.
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-use-in-chat")),
            message=lambda: (
                f"Reader control never mounted: row={screen._library_selected_row_id!r} "
                f"view={screen._media_state.view!r} "
                f"canvas={bool(screen.query('#library-media-canvas'))} "
                f"viewer={bool(screen.query('#library-media-viewer'))} "
                f"session={screen._media_state.reader_session!r} "
                f"notified={_notified(app)!r}"
            ),
        )
        await _press_use_in_console(screen, pilot)
        app.open_chat_with_handoff.assert_called_once()
        assert app.open_chat_with_handoff.call_args.args[0].source_id == "52"
        assert [(row.workspace_id, row.item_id) for row in _memberships(app, item_id="52")] == [
            (RAW_ID, "52")
        ]
        await _wait_for_receipt(screen, pilot, LINKED_RECEIPT, app)
        assert not [
            line for line in _notified(app) if "Copy or link blocked" in line
        ], _notified(app)
