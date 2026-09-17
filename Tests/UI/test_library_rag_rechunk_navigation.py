"""Re-chunk work and feedback survive replacement of the initiating panel."""

import pytest
from textual.screen import Screen
from textual.widgets import Button, Static

from Tests.UI.test_library_rag_history_keyboard import _host, _open
from Tests.UI.test_library_rag_rechunk_feedback import (
    ACTION,
    RECEIPT,
    REPORT,
    SUMMARY,
    _HeldRechunk,
    _isolate_rechunk,  # noqa: F401 - shared isolation fixture
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_selector,
)
from Tests.UI.test_product_maturity_gate16_library_search_rag import (
    _ready_library_rag_provider,  # noqa: F401 - shared autouse fixture
)
from tldw_chatbook.Library.library_rechunk_service import (
    BACKFILL_SLOT,
    RECHUNK_SLOT,
    RECHUNK_WORKER_GROUP,
    acquire_bulk_rag_slot,
    bulk_rag_slot_in_flight,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("route", ["canvas", "screen"])
@pytest.mark.parametrize("finish_away", [False, True])
async def test_return_to_rechunk_retains_progress_and_receipt(
    theme, size, route, finish_away
):
    host, app, _, _ = _host(theme)
    service = _HeldRechunk()
    host.rag_admin_scope_service = service
    notices = []
    host.notify = lambda message, **kwargs: notices.append(str(message))
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(ACTION, Button).display,
            message="Re-chunk visible",
        )
        original = screen.query_one("#library-search-rag-panel")
        screen.query_one(ACTION, Button).focus()
        await pilot.press("enter")
        try:
            await _wait_for_condition(
                pilot, service.started.is_set, message="Re-chunk started"
            )
            if route == "canvas":
                screen.query_one("#library-row-browse-notes", Button).focus()
                await pilot.press("enter")
                await _wait_for_selector(screen, pilot, "#library-notes-canvas")
            else:
                await host.switch_screen(Screen())
            assert not original.is_attached
            assert bulk_rag_slot_in_flight(RECHUNK_SLOT)
            if finish_away:
                service.release.set()
                await _wait_for_condition(
                    pilot,
                    lambda: any("Re-chunk finished" in s for s in notices),
                    message="Finished while away",
                )
            if route == "screen":
                await host.switch_screen(LibraryScreen(app))
                screen = _active_library_screen(host)
            await _open(screen, pilot)
            assert screen.query_one("#library-search-rag-panel") is not original
            if not finish_away:
                assert screen.query_one(ACTION, Button).disabled
                assert (
                    str(screen.query_one(SUMMARY, Static).renderable) == "Re-chunking…"
                )
                screen.query_one("#library-search-rag-panel")._trigger_rechunk_legacy()
                assert any("already running" in s for s in notices)
                assert acquire_bulk_rag_slot(BACKFILL_SLOT) is not None
                assert service.calls == 1
                service.release.set()
            await _wait_for_condition(
                pilot,
                lambda: str(screen.query_one(SUMMARY, Static).renderable) == RECEIPT,
                message="The returned panel never received the completion receipt",
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    str(screen.query_one(REPORT, Static).renderable)
                    == "Chunked by an older engine: 1 items"
                ),
                message="Updated legacy count",
            )
            assert not screen.query_one(ACTION, Button).disabled
            assert not bulk_rag_slot_in_flight(RECHUNK_SLOT)
            assert service.calls == 1
        finally:
            service.release.set()
            await _wait_for_condition(
                pilot,
                lambda: not bulk_rag_slot_in_flight(RECHUNK_SLOT),
                message="Worker released slot",
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["canvas", "screen"])
@pytest.mark.parametrize("failure", ["policy", "exception"])
async def test_failure_while_away_releases_run_and_return_allows_retry(route, failure):
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError

    host, app, _, _ = _host("textual-dark")
    service = _HeldRechunk()
    service.error = (
        PolicyDeniedError(
            action_id="rag.admin.launch.local",
            reason_code="capability_disabled",
            user_message="Re-chunk disabled for this profile.",
            effective_source="local",
            authority_owner="local",
        )
        if failure == "policy"
        else RuntimeError("controlled backend failure")
    )
    host.rag_admin_scope_service = service
    notices = []
    host.notify = lambda message, **kwargs: notices.append(str(message))
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(ACTION, Button).display,
            message="Action visible",
        )
        screen.query_one(ACTION, Button).focus()
        await pilot.press("enter")
        try:
            await _wait_for_condition(
                pilot, service.started.is_set, message="Run started"
            )
            if route == "canvas":
                screen.query_one("#library-row-browse-notes", Button).focus()
                await pilot.press("enter")
                await _wait_for_selector(screen, pilot, "#library-notes-canvas")
            else:
                await host.switch_screen(Screen())
            service.release.set()
            await _wait_for_condition(
                pilot,
                lambda: not bulk_rag_slot_in_flight(RECHUNK_SLOT),
                message="Failure settled",
            )
            expected = "blocked by policy" if failure == "policy" else "Re-chunk failed"
            assert any(expected in n for n in notices)
            if route == "screen":
                await host.switch_screen(LibraryScreen(app))
                screen = _active_library_screen(host)
            await _open(screen, pilot)
            await _wait_for_condition(
                pilot,
                lambda: screen.query_one(ACTION, Button).display,
                message="Retry visible",
            )
            assert not screen.query_one(ACTION, Button).disabled
            assert not screen.query_one(SUMMARY, Static).display
            service.error = None
            screen.query_one(ACTION, Button).focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: str(screen.query_one(SUMMARY, Static).renderable) == RECEIPT,
                message="Retry finished",
            )
            assert service.calls == 2
        finally:
            service.release.set()
            await _wait_for_condition(
                pilot,
                lambda: not bulk_rag_slot_in_flight(RECHUNK_SLOT),
                message="Slot released",
            )


@pytest.mark.asyncio
async def test_worker_scheduling_failure_releases_admission_for_retry(monkeypatch):
    host, _, _, _ = _host("textual-dark")
    service = _HeldRechunk()
    service.release.set()
    host.rag_admin_scope_service = service
    notices = []
    host.notify = lambda message, **kwargs: notices.append(str(message))
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(ACTION, Button).display,
            message="Action visible",
        )
        original = host.run_worker

        def refuse_rechunk(*args, **kwargs):
            if kwargs.get("group") == RECHUNK_WORKER_GROUP:
                raise RuntimeError("controlled scheduler refusal")
            return original(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(host, "run_worker", refuse_rechunk)
            screen.query_one("#library-search-rag-panel")._trigger_rechunk_legacy()
        assert any("could not start" in n for n in notices)
        assert not bulk_rag_slot_in_flight(RECHUNK_SLOT)
        assert not screen.query_one(ACTION, Button).disabled
        assert not screen.query_one(SUMMARY, Static).display
        assert service.calls == 0
        screen.query_one(ACTION, Button).focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: str(screen.query_one(SUMMARY, Static).renderable) == RECEIPT,
            message="Retry finished",
        )
        assert service.calls == 1
