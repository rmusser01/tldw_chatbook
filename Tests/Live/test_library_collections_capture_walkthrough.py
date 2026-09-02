"""Production-shaped Local Collections capture reader walkthrough."""

from __future__ import annotations

import hashlib
import json
import os
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
    CapabilityState,
    CapturePageRequest,
    CaptureSaveRequest,
    CaptureSaveOutcome,
    CollectionsCaptureError,
)
from tldw_chatbook.Library.collections_capture_service import (
    LocalCollectionsCaptureService,
    build_server_capture_authority,
)
from tldw_chatbook.Library.server_collections_capture_service import (
    ServerCollectionsCaptureService,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.tldw_api.client import TLDWAPIClient


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


async def _open_collections(screen, pilot, *, expected_total: int = 45):
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
            and screen._library_collections_capture_controller.state.page.total
            == expected_total
            and screen._library_collections_capture_controller.state.loaded_detail
            and len(screen.query(".library-collections-item-row")) == 20
        ),
        message=f"The {expected_total}-capture page did not settle",
    )
    return shell


async def _wait_for_stable_button(screen, pilot, selector: str) -> Button:
    """Return one visible button that survives consecutive idle UI cycles."""
    button = await _wait_for_selector(screen, pilot, selector)
    for _ in range(12):
        await pilot.pause()
        current = screen.query_one(selector, Button)
        if current is button and current.display:
            await pilot.pause()
            if screen.query_one(selector, Button) is current:
                return current
        button = current
    raise AssertionError(f"{selector} did not settle on one mounted button")


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
@pytest.mark.loopback_network
async def test_live_server_45_capture_source_replacement_geometry_and_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Walk the real Server Reading API without merging it with Local state."""
    base_url = os.getenv("TLDW_TASK18919_SERVER_URL", "").strip()
    api_key = os.getenv("TLDW_TASK18919_SERVER_API_KEY", "").strip()
    if not base_url or not api_key:
        pytest.skip("TASK-18919 isolated Server endpoint is not configured")

    client = TLDWAPIClient(
        base_url,
        token=api_key,
        timeout=15.0,
        connect_timeout=5.0,
    )
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    local_authority = app.local_collections_capture_authority
    local_service = app.local_collections_capture_service
    assert local_authority is not None
    assert local_service is not None

    try:
        docs_info = await client.get_server_docs_info()
        docs = docs_info.model_dump(mode="json")
        assert docs["capabilities"]["hasReadingSnapshotPagesV1"] is True
        atomic_updates = (
            docs["capabilities"].get("hasReadingOptimisticUpdatesV1") is True
        )

        await _seed_captures(app, count=3)
        local_page = await scope.list_page(CapturePageRequest(local_authority.key))
        assert local_page.total == 3

        server_authority = build_server_capture_authority(
            "task-18919-isolated-profile",
            "task-18919-ephemeral-principal",
        )

        async def server_docs_info():
            return await client.get_server_docs_info()

        server_service = ServerCollectionsCaptureService(
            server_authority,
            client,
            docs_info_provider=server_docs_info,
            credential_fingerprint=hashlib.sha256(api_key.encode("utf-8")).hexdigest()[
                :24
            ],
        )
        capabilities = await server_service.capabilities()
        assert capabilities.for_action("browse").state is CapabilityState.SUPPORTED
        assert capabilities.for_action("capture").state is CapabilityState.SUPPORTED
        archive_capability = capabilities.for_action("archive")
        assert archive_capability.state is (
            CapabilityState.SUPPORTED
            if atomic_updates
            else CapabilityState.UNSUPPORTED
        )
        assert capabilities.for_action("offline_copy").reason == (
            "server_offline_copy_unavailable"
        )
        assert capabilities.for_action("retry_extraction").reason == (
            "server_retry_extraction_unavailable"
        )

        existing_server_page = await server_service.list_page(
            CapturePageRequest(server_authority.key)
        )
        assert existing_server_page.total in {0, 45, 46}
        if existing_server_page.total == 0:
            for index in range(45):
                outcome = await server_service.save_capture(
                    CaptureSaveRequest(
                        server_authority.key,
                        f"https://server-capture-{index:02d}.example.test/article",
                        title=(
                            f"Server capture {index:02d} with a deliberately long "
                            "identifying title for the production-shaped Items pane"
                        ),
                        tags=("live-server", f"batch-{index % 3}"),
                        status=("reading" if index % 4 == 0 else "saved"),
                        favorite=index % 7 == 0,
                        freeform_note=f"Server capture note {index:02d}",
                        text_content=(
                            f"Authoritative Server capture body {index:02d}. "
                            "This inert inline content avoids external retrieval."
                        ),
                    )
                )
                assert outcome.capture is not None
                assert outcome.capture.identity.authority_key == server_authority.key

        server_total = (
            45 if existing_server_page.total == 0 else existing_server_page.total
        )
        for page_number, expected_rows in (
            (1, 20),
            (2, 20),
            (3, server_total - 40),
        ):
            page = await server_service.list_page(
                CapturePageRequest(server_authority.key, page=page_number)
            )
            assert page.total == server_total
            assert len(page.items) == expected_rows
            assert all(
                item.identity.authority_key == server_authority.key
                for item in page.items
            )

        app.runtime_policy.state = RuntimeSourceState(
            active_source="server",
            server_configured=True,
            active_server_id="task-18919-isolated-profile",
        )
        app.server_collections_capture_authority = server_authority
        app.server_collections_capture_service = server_service
        scope.activate(server_authority, server_service)
        host = LibraryGlobalKeyProductionCSSHarness(app)

        async with host.run_test(size=SIZES[0]) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            shell = await _open_collections(
                screen,
                pilot,
                expected_total=server_total,
            )
            controller = screen._library_collections_capture_controller
            assert controller is not None
            assert screen._library_collections_capture_presentation().authority_label == (
                "Server"
            )
            assert controller.state.exact_total == server_total
            assert all(
                item.identity.authority_key == server_authority.key
                for item in controller.state.page.items
            )

            for page_number, expected_rows in (
                (2, 20),
                (3, server_total - 40),
            ):
                screen.query_one("#library-collections-page-next", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda page_number=page_number, expected_rows=expected_rows: bool(
                        controller.state.applied_scope
                        and controller.state.applied_scope.page == page_number
                        and not controller.state.page_loading
                        and len(screen.query(".library-collections-item-row"))
                        == expected_rows
                    ),
                    message=f"Server capture page {page_number} did not settle",
                )
                assert controller.state.exact_total == server_total

            for page_number in (2, 1):
                screen.query_one("#library-collections-page-previous", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda page_number=page_number: bool(
                        controller.state.applied_scope
                        and controller.state.applied_scope.page == page_number
                        and not controller.state.page_loading
                        and len(screen.query(".library-collections-item-row")) == 20
                    ),
                    message=f"Server capture page {page_number} did not restore",
                )

            shell = screen.query_one("#library-collections-reader-shell")
            items_before = shell.items.region.width
            shell.library_grip.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    not screen._library_collections_reader_layout.library_open
                    and screen._library_collections_reader_layout.items_open
                    and screen.query_one(
                        "#library-collections-reader-shell"
                    ).items.region.width
                    > items_before
                ),
                message=lambda: (
                    "Closing Server Library did not expand Items: "
                    f"screen_layout={screen._library_collections_reader_layout!r}, "
                    f"prefs={screen._library_collections_reader_preferences!r}, "
                    f"before={items_before}, after={screen.query_one('#library-collections-reader-shell').items.region.width}"
                ),
            )
            shell = screen.query_one("#library-collections-reader-shell")
            shell.items_grip.press()
            await _wait_for_condition(
                pilot,
                lambda: not screen._library_collections_reader_layout.items_open,
                message="Server Items did not close",
            )
            shell = screen.query_one("#library-collections-reader-shell")
            shell.library_grip.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_collections_reader_layout.library_open
                    and not screen._library_collections_reader_layout.items_open
                ),
                message="Server Library-only posture did not settle",
            )
            shell = screen.query_one("#library-collections-reader-shell")
            shell.items_grip.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_collections_reader_layout.library_open
                    and screen._library_collections_reader_layout.items_open
                ),
                message="Server optional panes did not restore",
            )

            for width, height in SIZES:
                await pilot.resize_terminal(width, height)
                await _wait_for_condition(
                    pilot,
                    lambda width=width, height=height: screen.size == (width, height),
                    message=f"Server Collections resize to {width}x{height} did not settle",
                )
                shell = screen.query_one("#library-collections-reader-shell")
                assert shell.work.is_mounted and shell.work.display
                assert shell.content_size.width > 0
                assert (
                    sum(child.region.width for child in shell.children)
                    == shell.content_size.width
                )
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
                message="Server Collections wide resize did not restore",
            )
            for mode in ("highlights", "notes", "info", "read"):
                selector = f"#library-collections-mode-{mode}"
                await _wait_for_selector(screen, pilot, selector)
                screen.query_one(selector, Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda mode=mode, selector=selector: (
                        screen._library_collections_reader_mode == mode
                        and bool(screen.query(selector))
                    ),
                    message=f"Server Collections {mode} mode did not settle",
                )

            registry = app.workspace_registry_service
            assert registry is not None
            registry.create_workspace(
                workspace_id="task-18919-second-workspace",
                name="TASK-18919 second workspace",
            )
            registry.set_active_workspace("task-18919-second-workspace")
            await screen._load_library_collections_capture_entry()
            await _wait_for_condition(
                pilot,
                lambda: bool(
                    scope.active_authority == server_authority
                    and controller.state.exact_total == server_total
                    and not controller.state.page_loading
                    and not controller.state.detail_loading
                    and controller.state.loaded_detail
                    and not screen._library_entry_reconcile_dirty
                    and screen._library_snapshot_rendered_generation
                    == screen._library_snapshot_state_generation
                    and screen.query("#library-collections-quick-capture")
                ),
                message="Workspace switch changed Server capture ownership",
            )

            quick_capture_button = await _wait_for_stable_button(
                screen,
                pilot,
                "#library-collections-quick-capture",
            )
            quick_capture_button.press()
            await _wait_for_condition(
                pilot,
                lambda: bool(
                    screen._library_collections_quick_capture_open
                    and screen.query("#library-collections-capture-url")
                ),
                message="Server Quick Capture did not open after workspace switch",
            )
            screen.query_one("#library-collections-capture-url", Input).value = (
                f"{base_url}/api/v1/health?task=18919-ui-save"
            )
            screen.query_one("#library-collections-capture-title", Input).value = (
                "Confirmed Server UI save"
            )
            screen.query_one("#library-collections-capture-tags", Input).value = (
                "live-server, ui-save"
            )
            screen.query_one("#library-collections-capture-note", TextArea).text = (
                "Saved through the mounted Server reader."
            )
            (
                await _wait_for_stable_button(
                    screen,
                    pilot,
                    "#library-collections-capture-save",
                )
            ).press()
            await _wait_for_condition(
                pilot,
                lambda: bool(
                    controller.state.exact_total == 46
                    and controller.state.loaded_detail
                    and controller.state.loaded_detail.capture.title
                    == "Confirmed Server UI save"
                    and screen._library_collections_action_status.startswith(
                        "Saved to Server"
                    )
                ),
                message="Confirmed Server UI save did not settle",
            )

            archive_button = await _wait_for_stable_button(
                screen,
                pilot,
                "#library-collections-archive",
            )
            if atomic_updates:
                archive_button.press()
                await _wait_for_condition(
                    pilot,
                    lambda: bool(
                        controller.state.loaded_detail
                        and controller.state.loaded_detail.capture.status == "archived"
                        and controller.state.visible_archive_receipts
                    ),
                    message="Server archive did not settle",
                )
                (
                    await _wait_for_stable_button(
                        screen,
                        pilot,
                        "#library-collections-archive-undo",
                    )
                ).press()
                await _wait_for_condition(
                    pilot,
                    lambda: bool(
                        controller.state.loaded_detail
                        and controller.state.loaded_detail.capture.status == "saved"
                    ),
                    message="Server archive Undo did not restore the prior status",
                )
            else:
                assert archive_button.disabled
                assert "server atomic mutation unavailable" in str(
                    archive_button.tooltip
                )

            real_save_capture = server_service.save_capture
            uncertain_calls = 0

            async def controlled_unknown_save(
                request: CaptureSaveRequest,
            ) -> CaptureSaveOutcome:
                nonlocal uncertain_calls
                uncertain_calls += 1
                return CaptureSaveOutcome(None, None, outcome_unknown=True)

            monkeypatch.setattr(
                server_service,
                "save_capture",
                controlled_unknown_save,
            )
            (
                await _wait_for_stable_button(
                    screen,
                    pilot,
                    "#library-collections-quick-capture",
                )
            ).press()
            await _wait_for_selector(screen, pilot, "#library-collections-capture-url")
            screen.query_one("#library-collections-capture-url", Input).value = (
                "https://server-unknown.example.test/article"
            )
            screen.query_one("#library-collections-capture-title", Input).value = (
                "Controlled unknown Server save"
            )
            (
                await _wait_for_stable_button(
                    screen,
                    pilot,
                    "#library-collections-capture-save",
                )
            ).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_collections_save_outcome_unknown,
                message="Controlled unknown Server save did not retain its draft",
            )
            assert uncertain_calls == 1
            assert screen._library_collections_action_status == (
                "Save outcome unknown. Refresh before retrying."
            )
            (
                await _wait_for_stable_button(
                    screen,
                    pilot,
                    "#library-collections-capture-save",
                )
            ).press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_collections_confirming_save_retry,
                message="Unknown Server save did not require explicit confirmation",
            )
            assert uncertain_calls == 1
            monkeypatch.setattr(server_service, "save_capture", real_save_capture)
            (
                await _wait_for_stable_button(
                    screen,
                    pilot,
                    "#library-collections-capture-retry-back",
                )
            ).press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    not screen._library_collections_confirming_save_retry
                    and bool(screen.query("#library-collections-capture-cancel"))
                ),
                message="Unknown Server save Back action did not restore its draft",
            )
            assert uncertain_calls == 1
            (
                await _wait_for_stable_button(
                    screen,
                    pilot,
                    "#library-collections-capture-cancel",
                )
            ).press()

            app.runtime_policy.state = RuntimeSourceState(
                active_source="local",
                server_configured=True,
            )
            scope.activate(local_authority, local_service)
            await screen._load_library_collections_capture_entry()
            await _wait_for_condition(
                pilot,
                lambda: bool(
                    controller.state.exact_total == 3
                    and controller.state.page
                    and len(controller.state.page.items) == 3
                ),
                message="Switching back to Local did not restore only Local captures",
            )
            assert screen._library_collections_capture_presentation().authority_label == (
                "Local"
            )
            assert all(
                item.identity.authority_key == local_authority.key
                for item in controller.state.page.items
            )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_local_capture_commit_failure_retry_modes_archive_delete_and_recovery(
    tmp_path: Path,
) -> None:
    """Walk one capture through Local lifecycle and complete legacy recovery."""
    app = _build_test_app()
    _seed_legacy_records(app.local_library_collections_db, count=45)
    authority = app.collections_capture_scope_service.active_authority
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
