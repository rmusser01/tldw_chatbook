"""Generation and recovery contracts for the Collections capture controller."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable
from dataclasses import replace
from typing import Any

import pytest

from tldw_chatbook.Library.collections_capture_models import (
    CaptureActionResult,
    CaptureAuthority,
    CaptureConflict,
    CaptureConflictError,
    CaptureDetail,
    CaptureIdentity,
    CapturePage,
    CapturePageRequest,
    CaptureSummary,
    CollectionsCaptureError,
    ResolvedCaptureDetail,
)
from tldw_chatbook.UI.Library_Modules.library_collections_capture_controller import (
    LibraryCollectionsCaptureController,
)


AUTHORITY_A = CaptureAuthority("local", "local:a", "fingerprint-a")
AUTHORITY_B = CaptureAuthority("server", "server:b", "fingerprint-b")


def _identity(index: int, authority: CaptureAuthority = AUTHORITY_A) -> CaptureIdentity:
    return CaptureIdentity(authority.key, str(index))


def _summary(
    index: int, *, authority: CaptureAuthority = AUTHORITY_A
) -> CaptureSummary:
    return CaptureSummary(
        _identity(index, authority),
        f"https://example.test/{index}",
        title=f"Capture {index}",
        created_at="2026-09-01T12:00:00Z",
        updated_at="2026-09-01T12:00:00Z",
        revision=index,
    )


def _detail(
    index: int,
    *,
    authority: CaptureAuthority = AUTHORITY_A,
    revision: int | None = None,
    **changes: Any,
) -> CaptureDetail:
    values = {
        **_summary(index, authority=authority).__dict__,
        "submitted_url": f"https://example.test/{index}",
        "text_content": f"Body {index}",
    }
    values.update(changes)
    if revision is not None:
        values["revision"] = revision
    return CaptureDetail(**values)


def _resolved(index: int, **changes: Any) -> ResolvedCaptureDetail:
    return ResolvedCaptureDetail(_detail(index, **changes), None, ())


def _page(
    request: CapturePageRequest,
    indexes: range | tuple[int, ...],
    total: int,
    *,
    revision: str = "snapshot-1",
) -> CapturePage:
    return CapturePage(
        request,
        tuple(_summary(index) for index in indexes),
        total,
        source_revision=revision,
    )


class FakeCaptureScope:
    def __init__(self) -> None:
        self.active_authority: CaptureAuthority | None = None
        self.pages: deque[Any] = deque()
        self.details: dict[CaptureIdentity, Any] = {}
        self.updated: Any = None
        self.archived: Any = None
        self.restored: Any = None
        self.extraction: Any = None
        self.list_calls: list[CapturePageRequest] = []

    def activate(self, authority: CaptureAuthority, _backend: object) -> None:
        self.active_authority = authority

    @staticmethod
    async def _result(value: Any) -> Any:
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, Awaitable):
            return await value
        if callable(value):
            value = value()
            if isinstance(value, Awaitable):
                return await value
        return value

    async def list_page(self, request: CapturePageRequest) -> CapturePage:
        self.list_calls.append(request)
        return await self._result(self.pages.popleft())

    async def get_detail(self, identity: CaptureIdentity) -> ResolvedCaptureDetail:
        return await self._result(self.details[identity])

    async def update_capture(
        self,
        _identity: CaptureIdentity,
        _expected_revision: int,
        _changes: dict[str, Any],
    ) -> CaptureDetail:
        return await self._result(self.updated)

    async def archive(
        self,
        _identity: CaptureIdentity,
        _expected_revision: int,
    ) -> CaptureDetail:
        return await self._result(self.archived)

    async def undo_archive(
        self,
        _identity: CaptureIdentity,
        _expected_revision: int,
    ) -> CaptureDetail:
        return await self._result(self.restored)

    async def retry_extraction(
        self,
        identity: CaptureIdentity,
    ) -> CaptureActionResult:
        return await self._result(
            self.extraction or CaptureActionResult(identity, True)
        )


class ControlledSettle:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, _seconds: float) -> None:
        self.started.set()
        await self.release.wait()


@pytest.mark.asyncio
async def test_authoritative_page_applies_requested_scope_and_first_selection() -> None:
    scope = FakeCaptureScope()
    request = CapturePageRequest(AUTHORITY_A.key, statuses=("saved",))
    scope.pages.append(_page(request, (1, 2), 2))
    controller = LibraryCollectionsCaptureController(scope)
    controller.activate(AUTHORITY_A, object())

    applied = await controller.load_page(request)

    assert applied is True
    assert controller.state.requested_scope == request
    assert controller.state.applied_scope == request
    assert controller.state.selected_identity == _identity(1)
    assert controller.state.loaded_detail is None
    assert controller.state.exact_total == 2
    assert controller.state.page_stale is False


@pytest.mark.asyncio
async def test_settled_selection_retains_loaded_item_and_enter_bypasses_delay() -> None:
    scope = FakeCaptureScope()
    request = CapturePageRequest(AUTHORITY_A.key)
    scope.pages.append(_page(request, (1, 2), 2))
    scope.details[_identity(1)] = _resolved(1)
    scope.details[_identity(2)] = _resolved(2)
    settle = ControlledSettle()
    controller = LibraryCollectionsCaptureController(
        scope,
        detail_settle_seconds=0.25,
        sleep=settle,
    )
    controller.activate(AUTHORITY_A, object())
    await controller.load_page(request)
    assert await controller.load_selected_now()

    pending = asyncio.create_task(controller.select_item(_identity(2)))
    await settle.started.wait()
    assert controller.state.selected_identity == _identity(2)
    assert controller.state.loaded_detail == _resolved(1)
    assert controller.state.identity_actions_enabled is False
    assert controller.state.retained_reader_copy == (
        "Loading “Capture 2”… showing “Capture 1” until ready."
    )

    assert await controller.load_selected_now()
    assert controller.state.loaded_detail == _resolved(2)
    settle.release.set()
    assert await pending is False
    assert controller.state.loaded_detail == _resolved(2)


@pytest.mark.asyncio
async def test_source_switch_clears_state_and_fences_late_page() -> None:
    scope = FakeCaptureScope()
    request = CapturePageRequest(AUTHORITY_A.key)
    future: asyncio.Future[CapturePage] = asyncio.get_running_loop().create_future()
    scope.pages.append(future)
    controller = LibraryCollectionsCaptureController(scope)
    controller.activate(AUTHORITY_A, object())
    pending = asyncio.create_task(controller.load_page(request))
    await asyncio.sleep(0)

    controller.activate(AUTHORITY_B, object())
    future.set_result(_page(request, (1,), 1))

    assert await pending is False
    assert controller.state.authority_key == AUTHORITY_B.key
    assert controller.state.page is None
    assert controller.state.selected_identity is None
    assert controller.state.loaded_detail is None


def test_adopt_active_authority_uses_app_owned_scope_without_reactivation() -> None:
    scope = FakeCaptureScope()
    scope.active_authority = AUTHORITY_A
    controller = LibraryCollectionsCaptureController(scope)

    assert controller.adopt_active_authority() is True
    assert controller.state.authority_key == AUTHORITY_A.key
    assert controller.state.page is None

    generation = controller._generations.copy()
    assert controller.adopt_active_authority() is False
    assert controller._generations == generation

    scope.active_authority = AUTHORITY_B
    assert controller.adopt_active_authority() is True
    assert controller.state.authority_key == AUTHORITY_B.key
    assert controller.state.page is None


def test_adopt_missing_active_authority_exposes_bounded_unavailable_state() -> None:
    scope = FakeCaptureScope()
    controller = LibraryCollectionsCaptureController(scope)

    assert controller.adopt_active_authority() is True
    assert controller.state.authority_key is None
    assert controller.state.page_error == "capture_authority_unavailable"


@pytest.mark.asyncio
async def test_page_shrink_retries_last_page_once_and_applies_it() -> None:
    scope = FakeCaptureScope()
    page_three = CapturePageRequest(AUTHORITY_A.key, page=3)
    page_two = CapturePageRequest(AUTHORITY_A.key, page=2)
    scope.pages.extend(
        (
            _page(page_three, range(41, 46), 45, revision="before"),
            _page(page_three, (), 40, revision="shrink"),
            _page(page_two, range(21, 41), 40, revision="after"),
        )
    )
    controller = LibraryCollectionsCaptureController(scope)
    controller.activate(AUTHORITY_A, object())
    await controller.load_page(page_three)

    assert await controller.load_page(page_three)
    assert [request.page for request in scope.list_calls] == [3, 3, 2]
    assert controller.state.applied_scope == page_two
    assert controller.state.page is not None
    assert controller.state.page.source_revision == "after"
    assert controller.state.exact_total == 40


@pytest.mark.asyncio
async def test_repeated_shrink_keeps_last_good_page_stale_without_looping() -> None:
    scope = FakeCaptureScope()
    page_three = CapturePageRequest(AUTHORITY_A.key, page=3)
    page_two = CapturePageRequest(AUTHORITY_A.key, page=2)
    original = _page(page_three, range(41, 46), 45, revision="before")
    scope.pages.extend(
        (
            original,
            _page(page_three, (), 40, revision="first-shrink"),
            _page(page_two, (), 20, revision="second-shrink"),
        )
    )
    controller = LibraryCollectionsCaptureController(scope)
    controller.activate(AUTHORITY_A, object())
    await controller.load_page(page_three)

    assert await controller.load_page(page_three) is False
    assert [request.page for request in scope.list_calls] == [3, 3, 2]
    assert controller.state.page == original
    assert controller.state.page_stale is True
    assert controller.state.page_error == "page_changed_again"
    assert controller.state.exact_total is None
    assert controller.state.paging_enabled is False


@pytest.mark.asyncio
async def test_successful_mutation_survives_failed_follow_up_as_stale() -> None:
    scope = FakeCaptureScope()
    request = CapturePageRequest(AUTHORITY_A.key)
    scope.pages.extend(
        (
            _page(request, (1,), 1),
            CollectionsCaptureError("server_refresh_failed", retryable=True),
        )
    )
    scope.details[_identity(1)] = _resolved(1)
    scope.updated = replace(_detail(1), favorite=True, revision=2)
    controller = LibraryCollectionsCaptureController(scope)
    controller.activate(AUTHORITY_A, object())
    await controller.load_page(request)
    await controller.load_selected_now()

    assert await controller.update_selected({"favorite": True}) is True
    assert controller.state.loaded_detail is not None
    assert controller.state.loaded_detail.capture.favorite is True
    assert controller.state.loaded_detail.capture.revision == 2
    assert controller.state.mutation_error is None
    assert controller.state.page_stale is True
    assert controller.state.page_error == "server_refresh_failed"
    assert controller.state.exact_total is None
    assert controller.state.identity_actions_enabled is False


@pytest.mark.asyncio
async def test_conflict_preserves_mutation_draft_and_current_metadata() -> None:
    scope = FakeCaptureScope()
    request = CapturePageRequest(AUTHORITY_A.key)
    scope.pages.append(_page(request, (1,), 1))
    scope.details[_identity(1)] = _resolved(1)
    current = replace(_detail(1), freeform_note="Remote", revision=3)
    scope.updated = CaptureConflictError(CaptureConflict(_identity(1), 1, current))
    controller = LibraryCollectionsCaptureController(scope)
    controller.activate(AUTHORITY_A, object())
    await controller.load_page(request)
    await controller.load_selected_now()
    draft = {"freeform_note": "My draft", "tags": ("AI",)}

    assert await controller.update_selected(draft) is False
    assert controller.state.conflict is not None
    assert controller.state.conflict.current == current
    assert dict(controller.state.conflict_draft or {}) == draft
    assert controller.state.loaded_detail == _resolved(1)


@pytest.mark.asyncio
async def test_archive_receipt_is_hidden_by_authority_and_undo_restores_status() -> (
    None
):
    scope = FakeCaptureScope()
    request = CapturePageRequest(AUTHORITY_A.key)
    scope.pages.extend(
        (
            _page(request, (1,), 1),
            _page(request, (), 0, revision="archived"),
            _page(request, (1,), 1, revision="restored"),
        )
    )
    scope.details[_identity(1)] = _resolved(1, status="reading")
    scope.archived = _detail(1, status="archived", revision=2)
    scope.restored = _detail(1, status="reading", revision=3)
    controller = LibraryCollectionsCaptureController(scope, clock=lambda: 42.0)
    controller.activate(AUTHORITY_A, object())
    await controller.load_page(request)
    await controller.load_selected_now()

    assert await controller.archive_selected()
    assert controller.state.visible_archive_receipts[0].identity == _identity(1)
    assert controller.state.visible_archive_receipts[0].created_at == 42.0
    controller.activate(AUTHORITY_B, object())
    assert controller.state.visible_archive_receipts == ()
    controller.activate(AUTHORITY_A, object())
    assert controller.state.visible_archive_receipts[0].previous_status == "reading"

    assert await controller.undo_archive(_identity(1))
    assert controller.state.visible_archive_receipts == ()


@pytest.mark.asyncio
async def test_source_switch_fences_late_confirmed_mutation_result() -> None:
    scope = FakeCaptureScope()
    request = CapturePageRequest(AUTHORITY_A.key)
    scope.pages.append(_page(request, (1,), 1))
    scope.details[_identity(1)] = _resolved(1)
    update_future: asyncio.Future[CaptureDetail] = (
        asyncio.get_running_loop().create_future()
    )
    scope.updated = update_future
    controller = LibraryCollectionsCaptureController(scope)
    controller.activate(AUTHORITY_A, object())
    await controller.load_page(request)
    await controller.load_selected_now()
    pending = asyncio.create_task(controller.update_selected({"favorite": True}))
    await asyncio.sleep(0)

    controller.activate(AUTHORITY_B, object())
    update_future.set_result(replace(_detail(1), favorite=True, revision=2))

    assert await pending is False
    assert controller.state.authority_key == AUTHORITY_B.key
    assert controller.state.loaded_detail is None
    assert controller.state.mutation_error is None


@pytest.mark.asyncio
async def test_unmount_invalidates_detail_and_extraction_completions() -> None:
    scope = FakeCaptureScope()
    request = CapturePageRequest(AUTHORITY_A.key)
    scope.pages.append(_page(request, (1,), 1))
    detail_future: asyncio.Future[ResolvedCaptureDetail] = (
        asyncio.get_running_loop().create_future()
    )
    scope.details[_identity(1)] = detail_future
    controller = LibraryCollectionsCaptureController(scope)
    controller.activate(AUTHORITY_A, object())
    await controller.load_page(request)
    detail_pending = asyncio.create_task(controller.load_selected_now())
    await asyncio.sleep(0)

    controller.unmount()
    detail_future.set_result(_resolved(1))

    assert await detail_pending is False
    assert controller.state.mounted is False
    assert controller.state.loaded_detail is None

    extraction_scope = FakeCaptureScope()
    extraction_scope.pages.append(_page(request, (1,), 1))
    extraction_scope.details[_identity(1)] = _resolved(1)
    extraction_future: asyncio.Future[CaptureActionResult] = (
        asyncio.get_running_loop().create_future()
    )
    extraction_scope.extraction = extraction_future
    extraction_controller = LibraryCollectionsCaptureController(extraction_scope)
    extraction_controller.activate(AUTHORITY_A, object())
    await extraction_controller.load_page(request)
    await extraction_controller.load_selected_now()
    extraction_pending = asyncio.create_task(extraction_controller.retry_extraction())
    await asyncio.sleep(0)

    extraction_controller.unmount()
    extraction_future.set_result(CaptureActionResult(_identity(1), True))

    assert await extraction_pending is False
    assert extraction_controller.state.mounted is False
