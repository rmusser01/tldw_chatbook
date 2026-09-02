"""Shared Local/Server authority seam for the Collections capture reader."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.collections_capture_models import (
    CAPTURE_CAPABILITY_NAMES,
    CapabilityState,
    CaptureCapabilities,
    CapturePage,
    CapturePageRequest,
    CaptureSaveRequest,
    CollectionsCaptureError,
    ExternalNoteReference,
    ExternalReferenceAvailability,
    SavedCaptureSearch,
)
from tldw_chatbook.Library.collections_capture_repository import (
    CollectionsCaptureRepository,
)
from tldw_chatbook.Library.collections_capture_service import (
    CollectionsCaptureScopeService,
    LocalCollectionsCaptureService,
    build_local_capture_authority,
    build_server_capture_authority,
)
from tldw_chatbook.Library.library_content_evidence import LibraryContentEvidence


def _clock_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class MutableClock:
    value: datetime

    def __call__(self) -> str:
        return _clock_text(self.value)


def _local_service(
    tmp_path: Path,
    *,
    clock=None,
    extractor=None,
    heartbeat_interval: float = 0.01,
):
    authority = build_local_capture_authority(
        profile_id="profile-a",
        database_identity=tmp_path / "private-library.db",
    )
    database = LibraryCollectionsDB(tmp_path / "collections.db")
    repository = CollectionsCaptureRepository(
        database,
        authority_key=authority.key,
        clock=clock or (lambda: "2026-09-01T12:00:00Z"),
        extraction_lease_seconds=300,
    )
    service = LocalCollectionsCaptureService(
        authority,
        repository,
        extractor=extractor,
        heartbeat_interval=heartbeat_interval,
        legacy_recovery_available=True,
    )
    return authority, database, repository, service


def test_authority_keys_are_opaque_and_server_ignores_local_workspace(
    tmp_path: Path,
) -> None:
    local_a = build_local_capture_authority(
        profile_id="private-profile-a",
        database_identity=tmp_path / "private-a.db",
    )
    local_b = build_local_capture_authority(
        profile_id="private-profile-a",
        database_identity=tmp_path / "private-b.db",
    )
    server_before = build_server_capture_authority(
        profile_id="private-server-a",
        principal_id="private-user-a",
    )
    server_after_workspace_switch = build_server_capture_authority(
        profile_id="private-server-a",
        principal_id="private-user-a",
    )

    assert local_a != local_b
    assert server_before == server_after_workspace_switch
    rendered = repr((local_a, local_b, server_before))
    for private_value in (
        "private-profile-a",
        "private-server-a",
        "private-user-a",
        "private-a.db",
        "private-b.db",
    ):
        assert private_value not in rendered


@pytest.mark.asyncio
async def test_capture_scope_owns_library_content_evidence(tmp_path: Path) -> None:
    authority, database, _repository, service = _local_service(tmp_path)
    scope = CollectionsCaptureScopeService()

    assert await scope.get_library_user_content_evidence() is LibraryContentEvidence.UNKNOWN

    scope.activate(authority, service)
    assert await scope.get_library_user_content_evidence() is LibraryContentEvidence.EMPTY

    await scope.save_capture(
        CaptureSaveRequest(authority.key, "https://example.test/evidence")
    )
    assert (
        await scope.get_library_user_content_evidence()
        is LibraryContentEvidence.HAS_USER_CONTENT
    )
    database.close()


@pytest.mark.asyncio
async def test_local_backend_covers_capture_owned_crud_and_bounded_pages(
    tmp_path: Path,
) -> None:
    authority, database, _repository, service = _local_service(tmp_path)
    scope = CollectionsCaptureScopeService(
        resolve_media_reference=lambda _reference: ExternalReferenceAvailability(
            "available"
        ),
        resolve_note_reference=lambda _reference: ExternalReferenceAvailability(
            "available"
        ),
    )
    scope.activate(authority, service)

    for index in range(45):
        outcome = await scope.save_capture(
            CaptureSaveRequest(
                authority.key,
                f"https://example.test/{index:03d}",
                title=f"Capture {index:03d}",
                text_content=f"Body {index:03d}",
            )
        )
        assert outcome.outcome_unknown is False

    page_two = await scope.list_page(CapturePageRequest(authority.key, page=2))
    page_three = await scope.list_page(CapturePageRequest(authority.key, page=3))
    assert page_two.total == 45
    assert len(page_two.items) == 20
    assert len(page_three.items) == 5

    identity = page_two.items[0].identity
    detail = await scope.get_detail(identity)
    changed = await scope.update_capture(
        identity,
        detail.capture.revision,
        {
            "favorite": True,
            "status": "reading",
            "tags": ("Research", "AI"),
            "freeform_note": "Keep this note exactly.\n",
        },
    )
    assert changed.favorite is True
    assert changed.status == "reading"
    assert changed.tags == ("AI", "Research")
    assert changed.freeform_note == "Keep this note exactly.\n"

    archived = await scope.archive(identity, changed.revision)
    assert archived.status == "archived"
    restored = await scope.undo_archive(identity, archived.revision)
    assert restored.status == "reading"

    draft = SavedCaptureSearch(
        authority.key,
        "new",
        "Research",
        CapturePageRequest(authority.key, tags=("research",)),
        "",
        "",
        1,
    )
    saved_search = await scope.save_saved_search(draft)
    saved_page = await scope.list_saved_searches(page=1)
    assert saved_page.items == (saved_search,)

    highlight = await scope.save_highlight(
        identity,
        quote="Important sentence",
        note="Check this",
    )
    assert await scope.list_highlights(identity) == (highlight,)
    assert (await scope.delete_highlight(identity, highlight.highlight_id)).success

    link = await scope.link_note(
        identity,
        ExternalNoteReference(authority.key, "note-1"),
    )
    resolved = await scope.get_detail(identity)
    assert resolved.note_links == ((link, ExternalReferenceAvailability("available")),)
    assert (await scope.unlink_note(identity, link.link_id)).success
    database.close()


@pytest.mark.asyncio
async def test_local_sync_repository_calls_leave_the_event_loop(tmp_path: Path) -> None:
    authority, database, repository, _service = _local_service(tmp_path)
    caller_thread = threading.get_ident()
    worker_threads: list[int] = []
    original = repository.list_page

    def record_thread(request: CapturePageRequest) -> CapturePage:
        worker_threads.append(threading.get_ident())
        return original(request)

    repository.list_page = record_thread  # type: ignore[method-assign]
    service = LocalCollectionsCaptureService(authority, repository)

    await service.list_page(CapturePageRequest(authority.key))

    assert worker_threads and worker_threads[0] != caller_thread
    database.close()


@pytest.mark.asyncio
async def test_scope_switch_clears_snapshots_and_fences_late_results() -> None:
    authority_a = build_server_capture_authority("server-a", "user-a")
    authority_b = build_server_capture_authority("server-b", "user-b")
    started = asyncio.Event()
    release = asyncio.Event()

    class DelayedBackend:
        async def list_page(self, request: CapturePageRequest) -> CapturePage:
            started.set()
            await release.wait()
            return CapturePage(request, (), 0, source_revision="old")

    scope = CollectionsCaptureScopeService()
    scope.activate(authority_a, DelayedBackend())
    pending = asyncio.create_task(scope.list_page(CapturePageRequest(authority_a.key)))
    await started.wait()
    scope.activate(authority_b, DelayedBackend())
    release.set()

    with pytest.raises(CollectionsCaptureError) as caught:
        await pending

    assert caught.value.reason == "stale_authority_result"
    assert scope.active_authority == authority_b
    assert scope.page_snapshot is None
    assert scope.detail_snapshot is None
    assert scope.saved_search_snapshot is None


@pytest.mark.asyncio
async def test_scope_deactivate_clears_snapshots_and_fences_late_results() -> None:
    authority = build_server_capture_authority("server-a", "user-a")
    started = asyncio.Event()
    release = asyncio.Event()

    class DelayedBackend:
        async def list_page(self, request: CapturePageRequest) -> CapturePage:
            started.set()
            await release.wait()
            return CapturePage(request, (), 0, source_revision="old")

    scope = CollectionsCaptureScopeService()
    scope.activate(authority, DelayedBackend())
    pending = asyncio.create_task(scope.list_page(CapturePageRequest(authority.key)))
    await started.wait()
    scope.deactivate()
    release.set()

    with pytest.raises(CollectionsCaptureError) as caught:
        await pending

    assert caught.value.reason == "stale_authority_result"
    assert scope.active_authority is None
    assert scope.page_snapshot is None
    assert scope.detail_snapshot is None
    assert scope.saved_search_snapshot is None


def test_scope_rejects_backend_from_a_different_authority(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    authority_a, database_a, _repository_a, _service_a = _local_service(tmp_path / "a")
    _authority_b, database_b, _repository_b, service_b = _local_service(tmp_path / "b")

    with pytest.raises(CollectionsCaptureError) as caught:
        CollectionsCaptureScopeService().activate(authority_a, service_b)

    assert caught.value.reason == "authority_mismatch"
    database_a.close()
    database_b.close()


@pytest.mark.asyncio
async def test_local_capabilities_are_complete_and_source_owned(tmp_path: Path) -> None:
    _authority, database, _repository, service = _local_service(tmp_path)

    capabilities = await service.capabilities()

    assert isinstance(capabilities, CaptureCapabilities)
    assert set(capabilities.values) == set(CAPTURE_CAPABILITY_NAMES)
    for action in (
        "browse",
        "capture",
        "update",
        "highlights",
        "linked_notes",
        "archive",
        "hard_delete",
        "legacy_recovery",
    ):
        assert capabilities.for_action(action).state is CapabilityState.SUPPORTED
    for action in ("summarize", "listen", "offline_copy", "retry_extraction"):
        capability = capabilities.for_action(action)
        assert capability.state is CapabilityState.UNSUPPORTED
        assert capability.reason
    database.close()


@pytest.mark.asyncio
async def test_local_extraction_heartbeat_prevents_same_authority_recovery(
    tmp_path: Path,
) -> None:
    clock = MutableClock(datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc))
    extraction_started = threading.Event()
    extraction_release = threading.Event()

    def extractor(_url: str) -> dict[str, str]:
        extraction_started.set()
        assert extraction_release.wait(timeout=10)
        return {"content": "Extracted body", "title": "Extracted title"}

    authority, database, repository, service = _local_service(
        tmp_path,
        clock=clock,
        extractor=extractor,
        heartbeat_interval=0.01,
    )
    second_database = LibraryCollectionsDB(tmp_path / "collections.db")
    second_repository = CollectionsCaptureRepository(
        second_database,
        authority_key=authority.key,
        clock=clock,
        extraction_lease_seconds=300,
    )

    outcome = await service.save_capture(
        CaptureSaveRequest(authority.key, "https://example.test/heartbeat")
    )
    assert outcome.capture is not None
    assert await asyncio.to_thread(extraction_started.wait, 10)
    clock.value = datetime(2026, 9, 1, 12, 4, tzinfo=timezone.utc)
    await asyncio.sleep(0.05)
    clock.value = datetime(2026, 9, 1, 12, 6, tzinfo=timezone.utc)

    assert await asyncio.to_thread(second_repository.interrupt_stale_extractions) == 0
    extraction_release.set()
    await service.drain_extractions()
    detail = await asyncio.to_thread(repository.get_detail, outcome.capture.identity)
    assert detail is not None
    assert detail.processing_state == "ready"
    assert detail.text_content == "Extracted body"
    database.close()
    second_database.close()


@pytest.mark.asyncio
async def test_local_extraction_survives_reading_state_revision_changes(
    tmp_path: Path,
) -> None:
    authority, database, repository, service = _local_service(
        tmp_path,
        extractor=lambda _url: {"content": "Extracted after metadata update"},
    )

    outcome = await service.save_capture(
        CaptureSaveRequest(authority.key, "https://example.test/revision-race")
    )
    assert outcome.capture is not None
    repository.update_capture(
        outcome.capture.identity,
        expected_revision=outcome.capture.revision,
        changes={"favorite": True, "status": "reading"},
    )

    await service.drain_extractions()

    detail = repository.get_detail(outcome.capture.identity)
    assert detail is not None
    assert detail.processing_state == "ready"
    assert detail.favorite is True
    assert detail.status == "reading"
    assert detail.text_content == "Extracted after metadata update"
    database.close()


@pytest.mark.asyncio
async def test_reference_failures_are_bounded_without_mutating_capture(
    tmp_path: Path,
) -> None:
    authority, database, repository, service = _local_service(tmp_path)
    outcome = await service.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://example.test/reference",
            text_content="Original body",
        )
    )
    assert outcome.capture is not None
    await service.link_note(
        outcome.capture.identity,
        ExternalNoteReference(authority.key, "note-1"),
    )

    def unavailable(_reference):
        raise CollectionsCaptureError("private_transport_detail", retryable=True)

    scope = CollectionsCaptureScopeService(resolve_note_reference=unavailable)
    scope.activate(authority, service)

    resolved = await scope.get_detail(outcome.capture.identity)
    persisted = await asyncio.to_thread(repository.get_detail, outcome.capture.identity)

    assert resolved.note_links[0][1] == ExternalReferenceAvailability(
        "unavailable",
        "reference_resolution_retryable",
    )
    assert "private" not in repr(resolved)
    assert persisted is not None and persisted.text_content == "Original body"
    database.close()


@pytest.mark.asyncio
async def test_archive_receipt_survives_source_switch_for_originating_authority(
    tmp_path: Path,
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    authority_a, database_a, _repository_a, service_a = _local_service(tmp_path / "a")
    authority_b, database_b, _repository_b, service_b = _local_service(tmp_path / "b")
    scope = CollectionsCaptureScopeService()
    scope.activate(authority_a, service_a)
    outcome = await scope.save_capture(
        CaptureSaveRequest(authority_a.key, "https://example.test/archive")
    )
    assert outcome.capture is not None
    reading = await scope.update_capture(
        outcome.capture.identity,
        outcome.capture.revision,
        {"status": "reading"},
    )
    archived = await scope.archive(reading.identity, reading.revision)

    scope.activate(authority_b, service_b)
    scope.activate(authority_a, service_a)
    restored = await scope.undo_archive(archived.identity, archived.revision)

    assert restored.status == "reading"
    database_a.close()
    database_b.close()
