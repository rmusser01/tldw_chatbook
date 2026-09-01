"""Production-shaped Local Collections capture reader walkthrough."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.UI.test_library_adaptive_reader_closeout import (
    _assert_inside_items,
    _focus_closeout_work_via_f6,
)
from Tests.UI.test_library_collections_capture_reader import _seed_legacy_records
from Tests.UI.test_library_shell import (
    LibraryGlobalKeyProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.collections_capture_models import (
    CapturePageRequest,
    CaptureSaveRequest,
    CollectionsCaptureError,
)
from tldw_chatbook.Library.collections_capture_service import (
    LocalCollectionsCaptureService,
)


SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))


async def _seed_captures(app, *, count: int = 45) -> None:
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    for index in range(count):
        outcome = await scope.save_capture(
            CaptureSaveRequest(
                authority.key,
                f"https://capture-{index:02d}.example.test/article",
                title=(
                    f"Capture {index:02d} with a deliberately long identifying "
                    "title for the production-shaped Items pane"
                ),
                tags=("live", f"batch-{index % 3}"),
                status=("reading" if index % 4 == 0 else "saved"),
                favorite=index % 7 == 0,
                freeform_note=f"Capture note {index:02d}",
                text_content=(
                    f"Readable capture body {index:02d}. "
                    "This inert text proves the Work reader used authoritative content."
                ),
            )
        )
        assert outcome.capture is not None


async def _open_collections(screen, pilot):
    screen.query_one("#library-row-browse-collections", Button).press()
    shell = await _wait_for_selector(
        screen,
        pilot,
        "#library-collections-reader-shell",
    )
    await _wait_for_condition(
        pilot,
        lambda: bool(
            screen._library_collections_capture_controller
            and screen._library_collections_capture_controller.state.page
            and screen._library_collections_capture_controller.state.page.total == 45
            and screen._library_collections_capture_controller.state.loaded_detail
            and len(screen.query(".library-collections-item-row")) == 20
        ),
        message="The 45-capture Local page did not settle",
    )
    return shell


def _assert_pane_contains_visible_descendants(pane) -> None:
    for widget in pane.walk_children():
        if not widget.display or widget.region.area == 0:
            continue
        assert widget.region.x >= pane.region.x, (widget.id, widget.region, pane.region)
        assert widget.region.right <= pane.region.right, (
            widget.id,
            widget.region,
            pane.region,
        )


@pytest.mark.asyncio
async def test_live_local_45_capture_geometry_paging_collapse_resize_and_focus() -> None:
    """Cover real paging and adaptive-reader behavior under production CSS."""
    app = _build_test_app()
    await _seed_captures(app)
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    database_path = Path(app.collections_capture_repository.db.db_path).resolve()
    before_path_fingerprint = hashlib.sha256(
        str(database_path).encode("utf-8")
    ).hexdigest()
    before = await scope.list_page(CapturePageRequest(authority.key))
    assert before.total == 45
    host = LibraryGlobalKeyProductionCSSHarness(app)

    async with host.run_test(size=SIZES[0]) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        assert not screen.query("#library-collections-reader-shell")
        shell = await _open_collections(screen, pilot)
        controller = screen._library_collections_capture_controller
        assert controller is not None

        for page, expected_rows in ((1, 20), (2, 20), (3, 5)):
            if page > 1:
                screen.query_one("#library-collections-page-next", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda page=page: bool(
                        controller.state.applied_scope
                        and controller.state.applied_scope.page == page
                        and not controller.state.page_loading
                        and len(screen.query(".library-collections-item-row"))
                        == expected_rows
                    ),
                    message=f"Capture page {page} did not settle",
                )
            assert len(screen.query(".library-collections-item-row")) == expected_rows
            assert controller.state.exact_total == 45

        for page in (2, 1):
            screen.query_one("#library-collections-page-previous", Button).press()
            await _wait_for_condition(
                pilot,
                lambda page=page: bool(
                    controller.state.applied_scope
                    and controller.state.applied_scope.page == page
                    and not controller.state.page_loading
                    and len(screen.query(".library-collections-item-row")) == 20
                ),
                message=f"Capture page {page} did not restore",
            )

        shell = screen.query_one("#library-collections-reader-shell")
        assert shell.effective_layout.library_open
        assert shell.effective_layout.items_open
        items_before = shell.items.region.width
        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                not shell.effective_layout.library_open
                and shell.effective_layout.items_open
                and shell.items.region.width > items_before
            ),
            message="Closing Library did not expand Items",
        )
        expanded_items = shell.items.region.width
        shell.items_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: not shell.effective_layout.items_open,
            message="Items did not close",
        )
        work_only_width = shell.work.region.width
        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                shell.effective_layout.library_open
                and not shell.effective_layout.items_open
            ),
            message="Library-only optional-pane posture did not settle",
        )
        shell.items_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                shell.effective_layout.library_open
                and shell.effective_layout.items_open
            ),
            message="Both optional panes did not restore",
        )
        assert expanded_items > items_before
        assert work_only_width > shell.work.region.width

        for width, height in SIZES:
            await pilot.resize_terminal(width, height)
            await _wait_for_condition(
                pilot,
                lambda width=width, height=height: screen.size == (width, height),
                message=f"Collections resize to {width}x{height} did not settle",
            )
            shell = screen.query_one("#library-collections-reader-shell")
            assert shell.work.is_mounted and shell.work.display
            assert shell.content_size.width > 0
            assert sum(child.region.width for child in shell.children) == shell.content_size.width
            if shell.items.display:
                for row in screen.query(".library-collections-item-row"):
                    _assert_inside_items(shell.items, row)
                _assert_pane_contains_visible_descendants(shell.items)
            _assert_pane_contains_visible_descendants(shell.work)
            focus_region, _focus_id = await _focus_closeout_work_via_f6(
                screen,
                pilot,
                shell,
                "collections",
            )
            assert focus_region == "work"

        await pilot.resize_terminal(*SIZES[0])
        await _wait_for_condition(
            pilot,
            lambda: screen.size == SIZES[0],
            message="Collections wide resize did not restore",
        )
        assert screen._library_collections_reader_preferences.library_open is True
        assert screen._library_collections_reader_preferences.items_open is True
        assert screen._library_collections_reader_layout.library_open is True
        assert screen._library_collections_reader_layout.items_open is True

    after = await scope.list_page(CapturePageRequest(authority.key))
    after_database_path = Path(app.collections_capture_repository.db.db_path).resolve()
    after_path_fingerprint = hashlib.sha256(
        str(after_database_path).encode("utf-8")
    ).hexdigest()
    assert after_path_fingerprint == before_path_fingerprint
    assert after.total == before.total == 45


@pytest.mark.asyncio
async def test_live_local_capture_commit_failure_retry_modes_archive_delete_and_recovery(
    tmp_path: Path,
) -> None:
    """Walk one capture through Local lifecycle and complete legacy recovery."""
    app = _build_test_app()
    _seed_legacy_records(app.local_library_collections_db, count=45)
    authority = app.local_collections_capture_authority
    assert authority is not None
    extraction_started = threading.Event()
    extraction_release = threading.Event()
    attempts = 0

    def extractor(_url: str):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            extraction_started.set()
            if not extraction_release.wait(timeout=5):
                raise RuntimeError("test extraction release timed out")
            raise RuntimeError("controlled extraction failure")
        return {
            "title": "Recovered extraction title",
            "author": "Local extractor",
            "content": "Recovered readable content after explicit Retry.",
        }

    service = LocalCollectionsCaptureService(
        authority,
        app.collections_capture_repository,
        offline_store=app.collections_offline_store,
        extractor=extractor,
        summarizer=lambda detail: f"Summary of {detail.title}",
        listener=lambda detail: f"audio:{detail.identity.capture_id}",
        legacy_recovery_available=True,
    )
    app.local_collections_capture_service = service
    app.collections_capture_scope_service.activate(authority, service)
    await _seed_captures(app)
    host = LibraryGlobalKeyProductionCSSHarness(app)

    try:
        async with host.run_test(size=SIZES[0]) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_collections(screen, pilot)
            controller = screen._library_collections_capture_controller
            assert controller is not None

            screen.query_one("#library-collections-quick-capture", Button).press()
            await _wait_for_selector(screen, pilot, "#library-collections-capture-url")
            screen.query_one("#library-collections-capture-url", Input).value = (
                "https://live-capture.example.test/article"
            )
            screen.query_one("#library-collections-capture-title", Input).value = (
                "Commit before extraction"
            )
            screen.query_one("#library-collections-capture-tags", Input).value = (
                "live, retry"
            )
            screen.query_one("#library-collections-capture-note", TextArea).text = (
                "Preserved before extraction starts."
            )
            screen.query_one("#library-collections-capture-save", Button).press()
            await _wait_for_condition(
                pilot,
                extraction_started.is_set,
                message="The controlled extractor did not start",
            )
            await _wait_for_condition(
                pilot,
                lambda: bool(
                    controller.state.loaded_detail
                    and controller.state.loaded_detail.capture.title
                    == "Commit before extraction"
                ),
                message="The committed capture was not selected before extraction",
            )
            capture = controller.state.loaded_detail.capture
            committed = app.collections_capture_repository.get_detail(capture.identity)
            assert committed is not None
            assert committed.processing_state in {"queued", "processing"}
            assert committed.freeform_note == "Preserved before extraction starts."

            extraction_release.set()
            await _wait_for_condition(
                pilot,
                lambda: bool(
                    (detail := app.collections_capture_repository.get_detail(capture.identity))
                    and detail.processing_state == "failed"
                ),
                message="The controlled extraction did not fail durably",
            )
            await screen._select_library_collection_capture(capture.identity)
            assert controller.state.loaded_detail is not None
            assert controller.state.loaded_detail.capture.processing_state == "failed"

            await _wait_for_selector(screen, pilot, "#library-collections-more")
            screen.query_one("#library-collections-more", Button).press()
            await _wait_for_selector(
                screen,
                pilot,
                "#library-collections-retry-extraction",
            )
            screen.query_one("#library-collections-retry-extraction", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: attempts == 2,
                message="Explicit Retry did not start a second extraction",
            )
            await service.drain_extractions()
            await _wait_for_condition(
                pilot,
                lambda: bool(
                    (detail := app.collections_capture_repository.get_detail(capture.identity))
                    and detail.processing_state == "ready"
                ),
                message="Explicit Retry did not finish extraction",
            )
            await screen._run_library_collections_capture_transition(
                controller.refresh_selected_detail()
            )
            assert controller.state.loaded_detail is not None
            assert controller.state.loaded_detail.capture.processing_state == "ready"
            assert (
                controller.state.loaded_detail.capture.text_content
                == "Recovered readable content after explicit Retry."
            )

            for mode in ("highlights", "notes", "info", "read"):
                await _wait_for_selector(
                    screen,
                    pilot,
                    f"#library-collections-mode-{mode}",
                )
                screen.query_one(f"#library-collections-mode-{mode}", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda mode=mode: screen._library_collections_reader_mode == mode,
                    message=f"Collections {mode} mode did not settle",
                )

            await _wait_for_selector(screen, pilot, "#library-collections-archive")
            screen.query_one("#library-collections-archive", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: bool(
                    controller.state.loaded_detail
                    and controller.state.loaded_detail.capture.status == "archived"
                    and controller.state.visible_archive_receipts
                ),
                message="Archive and its receipt did not settle",
            )
            await _wait_for_selector(
                screen,
                pilot,
                "#library-collections-archive-undo",
            )
            screen.query_one("#library-collections-archive-undo", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: bool(
                    controller.state.loaded_detail
                    and controller.state.loaded_detail.capture.status == "saved"
                ),
                message="Archive Undo did not restore the prior status",
            )

            await _wait_for_selector(screen, pilot, "#library-collections-save-offline")
            screen.query_one("#library-collections-save-offline", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: bool(
                    controller.state.loaded_detail
                    and controller.state.loaded_detail.capture.offline_copy
                ),
                message="The managed offline copy did not settle",
            )
            assert app.collections_offline_store.open_copy(capture.identity)

            await _wait_for_selector(screen, pilot, "#library-collections-hard-delete")
            screen.query_one("#library-collections-hard-delete", Button).press()
            await _wait_for_selector(
                screen,
                pilot,
                "#library-collections-hard-delete-confirm",
            )
            screen.query_one("#library-collections-hard-delete-confirm", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: app.collections_capture_repository.get_detail(capture.identity)
                is None,
                message="Hard delete did not remove the capture tombstone lifecycle",
            )
            with pytest.raises(CollectionsCaptureError):
                app.collections_offline_store.open_copy(capture.identity)

            await _wait_for_selector(screen, pilot, "#library-collections-legacy-recovery")
            screen.query_one("#library-collections-legacy-recovery", Button).press()
            await _wait_for_selector(
                screen,
                pilot,
                "#library-collections-legacy-recovery-content",
            )
            destination = tmp_path / "complete-legacy-recovery.json"
            await screen._export_library_collection_legacy_recovery(destination)
            exported = json.loads(destination.read_text(encoding="utf-8"))
            assert len(exported["collections"]) == 45
            assert len(exported["memberships"]) == 45
            assert screen._library_collections_action_status == (
                "Legacy recovery export complete."
            )
    finally:
        extraction_release.set()
        await service.cancel_extractions()
