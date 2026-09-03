"""Mounted contracts for the Collections capture reader panes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

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
    CaptureSaveRequest,
    CaptureSaveOutcome,
    CaptureSummary,
    ExternalMediaReference,
    ExternalNoteReference,
    ExternalReferenceAvailability,
    ResolvedCaptureDetail,
    SavedCaptureSearch,
)
from tldw_chatbook.Library.collections_capture_service import (
    LocalCollectionsCaptureService,
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


def _seed_legacy_records(db, *, count: int) -> None:
    with db.transaction() as connection:
        connection.executemany(
            "INSERT INTO library_collections (collection_id, name, description, "
            "created_at, updated_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                (
                    f"legacy-{index:03d}",
                    f"Legacy collection {index:03d}",
                    "Recovery fixture",
                    "2026-08-01T00:00:00Z",
                    "2026-08-01T00:00:00Z",
                    None,
                )
                for index in range(count)
            ),
        )
        connection.executemany(
            "INSERT INTO library_collection_items (membership_id, collection_id, "
            "source_type, source_id, title, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                (
                    f"membership-{index:03d}",
                    f"legacy-{index:03d}",
                    "note",
                    f"note-{index:03d}",
                    f"Legacy member {index:03d}",
                    "2026-08-01T00:00:00Z",
                )
                for index in range(count)
            ),
        )


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
    selected_matches_loaded = bool(overrides.pop("selected_matches_loaded", False))
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
    if selected_matches_loaded:
        state = replace(
            state,
            selected_identity=loaded.identity,
            detail_loading=False,
            page_stale=False,
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


async def test_capture_and_filter_disclosures_mount_complete_editable_controls() -> None:
    app = _ReaderApp(
        _presentation(
            quick_capture_open=True,
            filters_open=True,
            action_status="Saved locally; extraction continues in the background.",
        )
    )

    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()

        assert app.query_one("#library-collections-capture-url", Input)
        assert app.query_one("#library-collections-capture-title", Input)
        assert app.query_one("#library-collections-capture-tags", Input)
        assert app.query_one("#library-collections-capture-note", TextArea)
        assert app.query_one("#library-collections-capture-save", Button)
        assert app.query_one("#library-collections-capture-cancel", Button)
        assert app.query_one("#library-collections-filter-domain", Input)
        assert app.query_one("#library-collections-filter-tags", Input)
        assert app.query_one("#library-collections-filter-date-from", Input)
        assert app.query_one("#library-collections-filter-date-to", Input)
        assert app.query_one("#library-collections-filters-apply", Button)
        assert app.query_one("#library-collections-filters-clear", Button)
        assert "extraction continues" in str(
            app.query_one("#library-collections-action-status", Static).renderable
        )


async def test_unknown_server_save_keeps_draft_and_requires_explicit_retry() -> None:
    app = _ReaderApp(
        _presentation(
            quick_capture_open=True,
            quick_capture_url="https://example.test/uncertain",
            quick_capture_title="Uncertain capture",
            quick_capture_tags="saved, server",
            quick_capture_note="Keep this draft.",
            save_outcome_unknown=True,
            confirming_save_retry=True,
            quick_capture_saving=True,
        )
    )

    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()

        assert app.query_one("#library-collections-capture-url", Input).value == (
            "https://example.test/uncertain"
        )
        assert app.query_one("#library-collections-capture-note", TextArea).text == (
            "Keep this draft."
        )
        assert "Refresh before retrying" in str(
            app.query_one("#library-collections-capture-unknown", Static).renderable
        )
        assert "clear Favorite" in str(
            app.query_one(
                "#library-collections-capture-retry-warning", Static
            ).renderable
        )
        assert app.query_one(
            "#library-collections-capture-refresh", Button
        ).disabled
        assert app.query_one(
            "#library-collections-capture-retry-confirm", Button
        ).disabled
        assert app.query_one(
            "#library-collections-capture-retry-back", Button
        ).disabled


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


async def test_archive_undo_remains_visible_without_a_loaded_detail() -> None:
    presentation = _presentation()
    presentation = replace(
        presentation,
        state=replace(
            presentation.state,
            selected_identity=None,
            loaded_detail=None,
            detail_loading=False,
        ),
    )
    app = _ReaderApp(presentation)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        button = app.query_one("#library-collections-archive-undo", Button)
        assert getattr(button, "capture_identity") == _identity("a")
        receipt = app.query_one("#library-collections-archive-receipt")
        assert "Moved to Archive" in str(
            receipt.query_one(Static).renderable
        )


async def test_supported_annotation_and_overflow_actions_have_reachable_results() -> None:
    app = _ReaderApp(
        _presentation(
            capabilities=_capabilities(
                "browse",
                "capture",
                "update",
                "highlights",
                "archive",
                "offline_copy",
                "summarize",
                "listen",
                "hard_delete",
            ),
            mode="highlights",
            more_open=True,
            confirming_hard_delete=False,
            action_status="Summary ready.",
            action_content="A bounded generated summary.",
            selected_matches_loaded=True,
        )
    )

    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()

        assert app.query_one("#library-collections-highlight-quote", TextArea)
        assert app.query_one("#library-collections-highlight-note", Input)
        assert app.query_one("#library-collections-highlight-save", Button)
        assert app.query_one("#library-collections-summarize", Button).disabled is False
        assert app.query_one("#library-collections-listen", Button).disabled is False
        assert app.query_one("#library-collections-save-offline", Button).disabled is False
        assert "Summary ready" in str(
            app.query_one("#library-collections-action-status", Static).renderable
        )
        assert "bounded generated summary" in str(
            app.query_one("#library-collections-action-content", Static).renderable
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


async def test_real_library_route_quick_capture_persists_and_selects_capture() -> None:
    app = _build_test_app()
    host = LibraryHarness(app)
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")

        screen.query_one("#library-collections-quick-capture", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-capture-url")
        screen.query_one("#library-collections-capture-url", Input).value = (
            "https://example.test/new-capture"
        )
        screen.query_one("#library-collections-capture-title", Input).value = (
            "Saved from the reader"
        )
        screen.query_one("#library-collections-capture-tags", Input).value = (
            "research, later"
        )
        screen.query_one("#library-collections-capture-note", TextArea).text = (
            "A local capture note."
        )
        screen.query_one("#library-collections-capture-save", Button).press()

        await _wait_for_condition(
            pilot,
            lambda: bool(
                screen._library_collections_capture_controller
                and screen._library_collections_capture_controller.state.loaded_detail
                and screen._library_collections_capture_controller.state.loaded_detail.capture.title
                == "Saved from the reader"
            ),
            message="Quick Capture did not persist and select the new capture",
        )
        detail = screen._library_collections_capture_controller.state.loaded_detail
        assert detail is not None
        assert detail.capture.identity.authority_key == authority.key
        assert detail.capture.tags == ("later", "research")
        assert detail.capture.freeform_note == "A local capture note."
        assert "Saved locally" in str(
            screen.query_one("#library-collections-action-status", Static).renderable
        )


async def test_quick_capture_draft_survives_background_reader_recompose() -> None:
    """An unrelated reader refresh must not erase an in-progress capture."""
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")
        screen.query_one("#library-collections-quick-capture", Button).press()
        original_url = await _wait_for_selector(
            screen, pilot, "#library-collections-capture-url"
        )
        original_url.value = "https://example.test/draft-in-progress"
        screen.query_one("#library-collections-capture-title", Input).value = (
            "Draft title"
        )
        screen.query_one("#library-collections-capture-tags", Input).value = (
            "research, later"
        )
        screen.query_one("#library-collections-capture-note", TextArea).text = (
            "Draft note"
        )
        await pilot.pause()

        screen._refresh_library_collections_capture_reader()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                screen.query("#library-collections-capture-url")
                and screen.query_one("#library-collections-capture-url")
                is not original_url
            ),
            message="Quick Capture form did not recompose",
        )

        assert screen.query_one("#library-collections-capture-url", Input).value == (
            "https://example.test/draft-in-progress"
        )
        assert screen.query_one("#library-collections-capture-title", Input).value == (
            "Draft title"
        )
        assert screen.query_one("#library-collections-capture-tags", Input).value == (
            "research, later"
        )
        assert screen.query_one("#library-collections-capture-note", TextArea).text == (
            "Draft note"
        )


async def test_unknown_quick_capture_preserves_draft_and_does_not_auto_retry(
    monkeypatch,
) -> None:
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    calls: list[CaptureSaveRequest] = []

    async def unknown_save(request: CaptureSaveRequest) -> CaptureSaveOutcome:
        calls.append(request)
        return CaptureSaveOutcome(None, None, outcome_unknown=True)

    monkeypatch.setattr(scope, "save_capture", unknown_save)
    host = LibraryHarness(app)

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")
        screen.query_one("#library-collections-quick-capture", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-capture-url")
        screen.query_one("#library-collections-capture-url", Input).value = (
            "https://example.test/uncertain-save"
        )
        screen.query_one("#library-collections-capture-title", Input).value = (
            "Uncertain save"
        )
        screen.query_one("#library-collections-capture-note", TextArea).text = (
            "Do not lose this draft."
        )
        screen.query_one("#library-collections-capture-save", Button).press()

        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.save_outcome_unknown,
            message="Unknown save state did not settle",
        )
        await _wait_for_selector(screen, pilot, "#library-collections-capture-refresh")
        assert len(calls) == 1
        assert screen.query_one("#library-collections-capture-url", Input).value == (
            "https://example.test/uncertain-save"
        )
        assert screen.query_one("#library-collections-capture-note", TextArea).text == (
            "Do not lose this draft."
        )

        screen.query_one("#library-collections-capture-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.confirming_save_retry,
            message="Explicit retry warning did not open",
        )
        assert len(calls) == 1
        await _wait_for_selector(
            screen,
            pilot,
            "#library-collections-capture-retry-confirm",
        )
        screen.query_one(
            "#library-collections-capture-retry-confirm", Button
        ).press()
        await _wait_for_condition(
            pilot,
            lambda: len(calls) == 2,
            message="Confirmed retry did not issue exactly one new save",
        )


async def test_real_local_capture_actions_persist_reader_results() -> None:
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    service = LocalCollectionsCaptureService(
        authority,
        app.collections_capture_repository,
        offline_store=app.collections_offline_store,
        summarizer=lambda detail: f"Summary of {detail.title}",
        listener=lambda detail: f"audio:{detail.identity.capture_id}",
    )
    app.local_collections_capture_service = service
    scope.activate(authority, service)
    await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://example.test/action-capture",
            title="Action capture",
            text_content="A useful body for action coverage.",
        )
    )
    host = LibraryHarness(app)

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-title")

        screen.query_one("#library-collections-more", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-summarize")
        screen.query_one("#library-collections-summarize", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.action_content
            == "Summary of Action capture",
            message="Summarize result did not reach the reader",
        )
        await _wait_for_selector(screen, pilot, "#library-collections-listen")
        screen.query_one("#library-collections-listen", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.action_status == "Audio is ready.",
            message="Listen result did not reach the reader",
        )
        await _wait_for_selector(screen, pilot, "#library-collections-save-offline")
        screen.query_one("#library-collections-save-offline", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                screen._library_collections_capture_controller.state.loaded_detail
                and screen._library_collections_capture_controller.state.loaded_detail.capture.offline_copy
            ),
            message="Offline copy was not reflected in loaded detail",
        )

        await _wait_for_selector(screen, pilot, "#library-collections-mode-highlights")
        screen.query_one("#library-collections-mode-highlights", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-highlight-quote")
        screen.query_one("#library-collections-highlight-quote", TextArea).text = (
            "A useful body"
        )
        screen.query_one("#library-collections-highlight-note", Input).value = (
            "Remember this"
        )
        screen.query_one("#library-collections-highlight-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: len(screen._collections_state.highlights) == 1,
            message="Highlight was not persisted",
        )

        await _wait_for_selector(screen, pilot, "#library-collections-mode-notes")
        screen.query_one("#library-collections-mode-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-freeform-note")
        screen.query_one("#library-collections-freeform-note", TextArea).text = (
            "Updated capture note"
        )
        screen.query_one("#library-collections-freeform-note-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                screen._library_collections_capture_controller.state.loaded_detail
                and screen._library_collections_capture_controller.state.loaded_detail.capture.freeform_note
                == "Updated capture note"
            ),
            message="Capture note was not persisted",
        )
        await _wait_for_selector(screen, pilot, "#library-collections-linked-note-id")
        screen.query_one("#library-collections-linked-note-id", Input).value = "note-7"
        screen.query_one("#library-collections-linked-note-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                screen._library_collections_capture_controller.state.loaded_detail
                and screen._library_collections_capture_controller.state.loaded_detail.note_links
            ),
            message="Linked Note was not reflected in the reader",
        )


async def test_summarize_result_is_discarded_after_selecting_another_capture(
    monkeypatch,
) -> None:
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    first = await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://example.test/stale-summary-first",
            title="First capture",
            text_content="First body.",
        )
    )
    second = await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://example.test/stale-summary-second",
            title="Second capture",
            text_content="Second body.",
        )
    )
    assert first.capture is not None
    assert second.capture is not None
    service = app.local_collections_capture_service
    service.summarizer = lambda detail: f"Summary of {detail.title}"
    summarize = scope.summarize
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_summary(identity: CaptureIdentity):
        started.set()
        await release.wait()
        return await summarize(identity)

    monkeypatch.setattr(scope, "summarize", delayed_summary)
    host = LibraryHarness(app)

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-title")

        controller = screen._library_collections_capture_controller
        assert controller is not None
        loaded = controller.state.loaded_detail
        assert loaded is not None
        source = loaded.capture.identity
        target = (
            second.capture.identity
            if source == first.capture.identity
            else first.capture.identity
        )

        screen.query_one("#library-collections-more", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-summarize")
        screen.query_one("#library-collections-summarize", Button).press()
        await asyncio.wait_for(started.wait(), timeout=2)

        await screen._select_library_collection_capture(target)
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.loaded_detail
                and controller.state.loaded_detail.capture.identity == target
            ),
            message="The replacement capture did not load",
        )
        release.set()
        await pilot.pause()
        await pilot.pause()

        assert screen._collections_state.action_content == ""
        assert screen._collections_state.action_status == ""


async def test_legacy_recovery_inspector_and_export_reach_every_page(
    tmp_path: Path,
) -> None:
    app = _build_test_app()
    _seed_legacy_records(app.local_library_collections_db, count=45)
    host = LibraryHarness(app)

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")
        await _wait_for_selector(screen, pilot, "#library-collections-legacy-recovery")
        screen.query_one("#library-collections-legacy-recovery", Button).press()
        await _wait_for_selector(
            screen,
            pilot,
            "#library-collections-legacy-recovery-content",
        )
        content = str(
            screen.query_one(
                "#library-collections-legacy-recovery-content", Static
            ).renderable
        )
        assert "Collections: 45 total · showing 20" in content
        assert "Memberships: 45 total · showing 20" in content

        destination = tmp_path / "legacy-recovery.json"
        await screen._export_library_collection_legacy_recovery(destination)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert len(payload["collections"]) == 45
        assert len(payload["memberships"]) == 45
        assert screen._collections_state.action_status == (
            "Legacy recovery export complete."
        )
