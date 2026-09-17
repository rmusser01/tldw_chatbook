"""Re-chunk feedback through actual mode/source changes and compact layout."""

import asyncio
import threading

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_library_rag_history_keyboard import _host, _open, _settle, _tab_to
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_library_shell import _active_library_screen, _wait_for_condition
from Tests.UI.test_product_maturity_gate16_library_search_rag import (
    _ready_library_rag_provider,  # noqa: F401 - shared autouse fixture
)
from tldw_chatbook.Library.library_rechunk_service import (
    RECHUNK_SLOT,
    bulk_rag_slot_in_flight,
    reset_bulk_rag_slots_for_tests,
)

ACTION = "#library-rag-rechunk-legacy"
SUMMARY = "#library-rag-rechunk-summary"
REPORT = "#library-rag-legacy-chunk-line"
RECEIPT = (
    "1 re-chunked, 1 skipped, 0 failed; re-index skipped (semantic index unavailable)"
)


class _HeldRechunk:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.count = 2
        self.error = None

    async def get_template_diagnostics(self, **kwargs):
        return {
            "legacy_chunk_report": f"Chunked by an older engine: {self.count} items"
        }

    async def rechunk_legacy_media(self, **kwargs):
        self.calls += 1
        self.started.set()
        assert await asyncio.to_thread(self.release.wait, 10)
        if self.error is not None:
            raise self.error
        self.count = 1
        return {
            "rechunked": 1,
            "skipped": 1,
            "failed": 0,
            "reindex_skipped_reason": "semantic index unavailable",
        }


@pytest.fixture(autouse=True)
def _isolate_rechunk(monkeypatch):
    reset_bulk_rag_slots_for_tests()
    monkeypatch.setattr(
        "tldw_chatbook.RAG_Search.ingestion_indexing.semantic_indexing_available",
        lambda: False,
    )
    yield
    reset_bulk_rag_slots_for_tests()


def _painted_text(screen, widget):
    region = widget.region
    return " ".join(
        " ".join(
            strip.crop(region.x, region.right).text
            for strip in screen._compositor.render_strips()[region.y : region.bottom]
        ).split()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_rechunk_feedback_survives_mode_and_scope_recompose(theme, size):
    host, _, _, _ = _host(theme)
    service = _HeldRechunk()
    host.rag_admin_scope_service = service
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(ACTION, Button).display,
            message="Legacy report did not offer Re-chunk",
        )
        screen.query_one("#library-rag-mode-toggle", Button).focus()
        await pilot.pause()
        await _tab_to(
            screen, pilot, lambda focused: getattr(focused, "id", None) == ACTION[1:]
        )
        await pilot.press("enter")
        try:
            await _wait_for_condition(
                pilot, service.started.is_set, message="Re-chunk did not start"
            )
            assert screen.query_one(ACTION, Button).disabled
            assert str(screen.query_one(SUMMARY, Static).renderable) == "Re-chunking…"
            # Both user gestures rebuild the panel while the worker remains active.
            for selector in (
                "#library-rag-mode-toggle",
                "#library-rag-scope-toggle-notes",
            ):
                screen.query_one(selector, Button).focus()
                await pilot.press("enter")
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
                assert screen.focused is screen.query_one(selector, Button)
                _assert_painted(screen, screen.focused)
                assert (
                    screen.query_one(ACTION, Button).disabled,
                    screen.query_one(SUMMARY, Static).display,
                    str(screen.query_one(SUMMARY, Static).renderable),
                ) == (True, True, "Re-chunking…")
                assert service.calls == 1
        finally:
            service.release.set()
        await _settle(screen, pilot)
        assert not bulk_rag_slot_in_flight(RECHUNK_SLOT)
        assert str(screen.query_one(SUMMARY, Static).renderable) == RECEIPT
        assert (
            str(screen.query_one(REPORT, Static).renderable)
            == "Chunked by an older engine: 1 items"
        )
        screen.query_one("#library-rag-mode-toggle", Button).focus()
        await pilot.press("enter")
        await _settle(screen, pilot)
        summary = screen.query_one(SUMMARY, Static)
        assert summary.display and str(summary.renderable) == RECEIPT
        await _tab_to(
            screen, pilot, lambda focused: getattr(focused, "id", None) == ACTION[1:]
        )
        summary.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        _assert_painted(screen, summary)
        assert RECEIPT in _painted_text(screen, summary)
        assert service.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("recompose", [False, True])
async def test_completed_rechunk_receipt_is_retained_and_wraps(size, recompose):
    host, _, _, _ = _host("textual-dark")
    service = _HeldRechunk()
    service.release.set()
    host.rag_admin_scope_service = service
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        await _settle(screen, pilot)
        screen.query_one(ACTION, Button).focus()
        await pilot.press("enter")
        await _settle(screen, pilot)
        assert service.calls == 1
        if recompose:
            screen.query_one("#library-rag-mode-toggle", Button).focus()
            await pilot.press("enter")
            await _settle(screen, pilot)
        summary = screen.query_one(SUMMARY, Static)
        assert summary.display and str(summary.renderable) == RECEIPT
        summary.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        _assert_painted(screen, summary)
        assert RECEIPT in _painted_text(screen, summary)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "policy", "exception"])
async def test_failure_clears_progress_after_recompose_and_allows_retry(failure):
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError

    host, _, _, _ = _host("textual-dark")
    service = _HeldRechunk()
    host.rag_admin_scope_service = service
    notices = []
    host.notify = lambda message, **kwargs: notices.append(str(message))
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        await _settle(screen, pilot)
        if failure == "missing":
            host.rag_admin_scope_service = None
            expected = "Re-chunk could not start"
        elif failure == "policy":
            service.error = PolicyDeniedError(
                action_id="rag.admin.launch.local",
                reason_code="capability_disabled",
                user_message="Re-chunk is disabled in this profile.",
                effective_source="local",
                authority_owner="local",
            )
            expected = "Re-chunk was blocked by policy"
        else:
            service.error = RuntimeError("controlled backend failure")
            expected = "Re-chunk failed"
        screen.query_one(ACTION, Button).focus()
        await pilot.press("enter")
        try:
            if failure != "missing":
                await _wait_for_condition(
                    pilot, service.started.is_set, message="Work did not start"
                )
                screen.query_one("#library-rag-mode-toggle", Button).focus()
                await pilot.press("enter")
                await pilot.pause()
                assert screen.query_one(ACTION, Button).disabled
        finally:
            service.release.set()
        await _settle(screen, pilot)
        assert any(expected in notice for notice in notices), notices
        assert not bulk_rag_slot_in_flight(RECHUNK_SLOT)
        screen.query_one("#library-rag-mode-toggle", Button).focus()
        await pilot.press("enter")
        await _settle(screen, pilot)
        assert not screen.query_one(SUMMARY, Static).display
        assert not screen.query_one(ACTION, Button).disabled
        service.error = None
        host.rag_admin_scope_service = service
        before = service.calls
        screen.query_one(ACTION, Button).focus()
        await pilot.press("enter")
        await _settle(screen, pilot)
        assert service.calls == before + 1
        assert str(screen.query_one(SUMMARY, Static).renderable) == RECEIPT
