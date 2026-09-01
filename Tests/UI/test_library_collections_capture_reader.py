"""Mounted contracts for the Collections capture reader panes."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.collections_capture_models import (
    CAPTURE_CAPABILITY_NAMES,
    CapabilityState,
    CaptureCapabilities,
    CaptureCapability,
    CaptureDetail,
    CaptureIdentity,
    CaptureNoteLink,
    CapturePage,
    CapturePageRequest,
    CaptureSummary,
    ExternalMediaReference,
    ExternalNoteReference,
    ExternalReferenceAvailability,
    ResolvedCaptureDetail,
    SavedCaptureSearch,
)
from tldw_chatbook.UI.Library_Modules.library_collections_capture_controller import (
    CaptureArchiveReceipt,
    CollectionsCaptureControllerState,
)
from tldw_chatbook.Widgets.Library.library_collections_capture_reader import (
    CollectionsCaptureReaderPresentation,
    LibraryCollectionsItemsPane,
    LibraryCollectionsScopeRows,
    LibraryCollectionsWorkPane,
)
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LibraryAdaptiveReaderShell,
)


AUTHORITY = "local:test-authority"


def _identity(value: str) -> CaptureIdentity:
    return CaptureIdentity(AUTHORITY, value)


def _summary(value: str, *, title: str, **overrides) -> CaptureSummary:
    values = {
        "identity": _identity(value),
        "canonical_url": f"https://example.com/{value}?private=secret",
        "title": title,
        "domain": "example.com",
        "summary": "A compact summary.",
        "published_at": "2026-08-30T12:00:00Z",
        "status": "reading",
        "favorite": True,
        "tags": ("research",),
        "processing_state": "ready",
        "created_at": "2026-08-31T12:00:00Z",
        "updated_at": "2026-08-31T12:00:00Z",
    }
    values.update(overrides)
    return CaptureSummary(**values)


def _detail(value: str = "a", *, title: str = "A literal [capture]") -> CaptureDetail:
    summary = _summary(value, title=title)
    return CaptureDetail(
        **summary.__dict__,
        submitted_url=f"https://example.com/{value}?submitted=private",
        freeform_note="Private note [kept literal]",
        text_content="Readable body [not markup].",
        byline="Ada Reader",
        word_count=420,
        media_reference=ExternalMediaReference(AUTHORITY, "media-7"),
    )


def _capabilities(*supported: str) -> CaptureCapabilities:
    supported_set = set(supported)
    return CaptureCapabilities(
        {
            action: CaptureCapability(
                CapabilityState.SUPPORTED
                if action in supported_set
                else CapabilityState.UNSUPPORTED,
                None if action in supported_set else f"{action}_unavailable",
            )
            for action in CAPTURE_CAPABILITY_NAMES
        }
    )


def _presentation(**overrides) -> CollectionsCaptureReaderPresentation:
    request = CapturePageRequest(AUTHORITY, statuses=("reading",))
    selected = _summary("b", title="Selected capture B")
    loaded = _detail()
    note_link = CaptureNoteLink(
        loaded.identity,
        "link-1",
        ExternalNoteReference(AUTHORITY, "note-1"),
        "2026-08-31T12:00:00Z",
    )
    resolved = ResolvedCaptureDetail(
        loaded,
        ExternalReferenceAvailability("unavailable", "media_missing"),
        ((note_link, ExternalReferenceAvailability("unavailable", "note_missing")),),
    )
    state = CollectionsCaptureControllerState(
        authority_key=AUTHORITY,
        requested_scope=request,
        applied_scope=request,
        page=CapturePage(request, (_summary("a", title="A literal [capture]"), selected), 2),
        page_stale=True,
        page_error="refresh_failed",
        selected_identity=selected.identity,
        loaded_detail=resolved,
        detail_loading=True,
        visible_archive_receipts=(
            CaptureArchiveReceipt(loaded.identity, "reading", 2, 1.0),
        ),
    )
    searches = (
        SavedCaptureSearch(
            AUTHORITY,
            "search-1",
            "Long reads",
            CapturePageRequest(AUTHORITY, tags=("research",)),
            "2026-08-31T12:00:00Z",
            "2026-08-31T12:00:00Z",
            1,
        ),
    )
    values = {
        "state": state,
        "capabilities": _capabilities(
            "browse",
            "capture",
            "update",
            "linked_notes",
            "archive",
            "hard_delete",
            "legacy_recovery",
        ),
        "saved_searches": searches,
        "saved_searches_total": 22,
        "active_scope": "reading",
        "authority_label": "Local",
        "mode": "notes",
        "more_open": True,
        "confirming_hard_delete": True,
        "legacy_recovery_rows": 3,
    }
    values.update(overrides)
    return CollectionsCaptureReaderPresentation(**values)


class _ReaderApp(ConsolidatedCSSApp):
    def __init__(self, presentation: CollectionsCaptureReaderPresentation) -> None:
        super().__init__()
        self.presentation = presentation

    def compose(self):
        yield Vertical(
            LibraryCollectionsScopeRows(self.presentation, id="scopes"),
            LibraryCollectionsItemsPane(self.presentation, id="items"),
            LibraryCollectionsWorkPane(self.presentation, id="work"),
        )


async def test_scope_rows_are_contextual_bounded_and_show_only_active_total() -> None:
    app = _ReaderApp(_presentation())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        assert len(app.query(".library-collections-scope-row")) == 7
        assert str(app.query_one("#library-collections-scope-reading", Button).label).startswith(
            "▸ Reading"
        )
        assert "(2)" not in str(
            app.query_one("#library-collections-scope-reading", Button).label
        ), "stale totals must not be presented as exact"
        assert "Long reads" in str(
            app.query_one("#library-collections-saved-search-search-1", Button).label
        )
        assert app.query_one("#library-collections-more-saved-searches", Button)


async def test_items_keep_capture_controls_rows_and_stale_recovery_reachable() -> None:
    app = _ReaderApp(_presentation())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        assert app.query_one("#library-collections-quick-capture", Button)
        assert app.query_one("#library-collections-filter", Input)
        assert app.query_one("#library-collections-sort", Button)
        assert app.query_one("#library-collections-page-previous", Button).disabled
        assert app.query_one("#library-collections-page-next", Button).disabled
        assert app.query_one("#library-collections-page-retry", Button)
        painted = "\n".join(
            row.label.plain for row in app.query(".library-collections-item-row")
        )
        assert "A literal [capture]" in painted
        assert "example.com" in painted
        assert "private=secret" not in painted
        assert "Selected · loading" in painted


async def test_work_keeps_selected_loaded_truth_and_distinct_note_models() -> None:
    app = _ReaderApp(_presentation())

    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()

        assert "Loading “Selected capture B”… showing “A literal [capture]”" in str(
            app.query_one("#library-collections-reader-loading", Static).renderable
        )
        assert str(app.query_one("#library-collections-reader-title", Static).renderable) == (
            "A literal [capture]"
        )
        assert app.query_one("#library-collections-freeform-note", TextArea).text == (
            "Private note [kept literal]"
        )
        assert "Linked Notes" in str(
            app.query_one("#library-collections-linked-notes-heading", Static).renderable
        )
        assert "Unavailable: note missing" in str(
            app.query_one("#library-collections-linked-note-link-1", Static).renderable
        )
        assert app.query_one("#library-collections-archive-undo", Button)
        assert app.query_one("#library-collections-hard-delete-confirm", Button)
        assert app.query_one("#library-collections-legacy-recovery", Button)
        assert "summarize unavailable" in str(
            app.query_one("#library-collections-summarize", Button).tooltip
        )


async def test_read_and_info_modes_render_content_and_provenance_as_inert_text() -> None:
    for mode, selector, expected in (
        ("read", "#library-collections-read-body", "Readable body [not markup]."),
        ("info", "#library-collections-media-provenance", "Unavailable: media missing"),
    ):
        app = _ReaderApp(_presentation(mode=mode, confirming_hard_delete=False))
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            widget = app.query_one(selector, Static)
            assert expected in str(widget.renderable)


async def test_real_library_route_mounts_contextual_three_pane_reader_and_both_grips() -> None:
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")
        shell = screen.query_one(
            "#library-collections-reader-shell", LibraryAdaptiveReaderShell
        )

        assert screen.query_one("#library-collections-scopes").parent.id == (
            "library-rail-section-body-browse"
        )
        assert not screen.query("#library-collections-panel")
        assert shell.work.is_mounted and shell.work.display
        assert shell.library_grip.display and shell.items_grip.display

        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                not shell.library.display
                and shell.items.region.width == 56
                and screen._library_reader_durable_generations["library"] > 0
            ),
            message="Collections Library pane did not collapse",
        )

        items_generation = screen._library_reader_persistence_generations[
            "collections_items"
        ]
        shell.items_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                not shell.items.display
                and screen._library_reader_durable_generations[
                    "collections_items"
                ]
                > items_generation
            ),
            message="Collections Items pane did not collapse",
        )
        assert app.app_config["library"]["collections_reader"]["items_open"] is False
        assert shell.work.is_mounted and shell.work.display
