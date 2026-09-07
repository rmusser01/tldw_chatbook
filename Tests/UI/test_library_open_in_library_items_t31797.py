"""Regression: the ingest "Open in Library" deep-link populates the Items pane.

task-31797: opening a media item straight into the viewer (the ingest-queue
"Open in Library" deep-link, and the sibling Search/RAG + landing-hub "Open"
routes) used to skip the browse-controller page load that the Media rail-row
path performs, leaving the middle Items pane stuck on
"0 of 0 · type: None / No page loaded · Total unavailable" until the user
clicked "Media (N)" in the left rail. The media branch of
``_open_library_item_by_id`` must now mirror the rail's browse + facets
request so the list lands populated alongside the opened item.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from tldw_chatbook.Library.library_notes_session import (
    NoteFlushOutcome,
    NoteFlushOutcomeKind,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


_REFRESH_SCOPE = object()


def _make_deep_link_screen() -> SimpleNamespace:
    """Build the minimal stub state the media branch of the opener touches.

    Only the attributes/methods reached on the non-``entry_origin`` media
    path (with ``required_database=None``) are stubbed; every method that
    mutates the UI is recorded so the test can assert what the deep-link did.
    """
    calls: list[str] = []
    browse_calls: list[dict[str, object]] = []
    facet_calls: list[str] = []

    async def _flush_note_save() -> NoteFlushOutcome:
        return NoteFlushOutcome(NoteFlushOutcomeKind.PERMITTED)

    async def _apply_active_surface() -> None:
        calls.append("apply_active_surface")

    def _request_browse(scope: object, *, focus_identity: object) -> None:
        browse_calls.append({"scope": scope, "focus_identity": focus_identity})

    def _request_facets() -> None:
        facet_calls.append("facets")

    def _run_worker(work: object, **_kwargs: object) -> None:
        calls.append("run_worker")

    screen = SimpleNamespace(
        _prompts_state=SimpleNamespace(mutation_in_flight=False),
        _library_navigation_context_generation=7,
        _library_media_reader_session=SimpleNamespace(external_detail=False),
        _library_media_browse_controller=SimpleNamespace(
            mutation_refresh_scope=_REFRESH_SCOPE
        ),
        # recorded interaction sinks
        _browse_calls=browse_calls,
        _facet_calls=facet_calls,
        _calls=calls,
        # methods reached on the media branch
        _flush_library_note_save=_flush_note_save,
        _acknowledge_library_destination_change=lambda: calls.append("ack"),
        _clear_library_prompt_selection=lambda **_k: calls.append("clear_prompt"),
        _cancel_pending_review_set_resume=lambda: calls.append("cancel_resume"),
        _library_media_reader_identity=lambda _record_id: None,
        _close_library_media_find=lambda: calls.append("close_find"),
        _refresh_library_media_detail=lambda *a, **k: None,
        run_worker=_run_worker,
        _request_library_media_browse=_request_browse,
        _request_library_media_facets=_request_facets,
        _apply_library_media_active_surface=_apply_active_surface,
        # plain attributes the branch assigns to
        _selected_media_id=None,
        _library_selected_row_id=None,
        _library_media_view=None,
        _library_media_editing=None,
        _library_media_confirming_delete=None,
        _library_media_highlights=None,
        _library_media_editing_analysis=None,
        _library_media_content_mode=None,
    )
    return screen


def test_open_in_library_deep_link_requests_media_browse_page() -> None:
    screen = _make_deep_link_screen()

    result = asyncio.run(
        LibraryScreen._open_library_item_by_id(screen, "media", "5")
    )

    # Deep-link (non-entry_origin) path returns None after applying the surface.
    assert result is None
    # It lands on the Media browse row with the opened item selected.
    assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
    assert screen._selected_media_id == "5"
    assert screen._library_media_view == "viewer"

    # The core regression: the deep-link must load an Items page (the rail's
    # browse + facets request) so the middle pane is not left empty.
    assert len(screen._browse_calls) == 1, (
        "deep-link must issue exactly one media browse request"
    )
    assert screen._browse_calls[0]["scope"] is _REFRESH_SCOPE
    # focus_identity=None keeps focus on the opened viewer, not the list row.
    assert screen._browse_calls[0]["focus_identity"] is None
    assert screen._facet_calls == ["facets"]
    # The reader detail worker and the active-surface sync still run.
    assert "run_worker" in screen._calls
    assert "apply_active_surface" in screen._calls
