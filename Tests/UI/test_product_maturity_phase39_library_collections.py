"""Phase 3.9 Library Collections mounted UI regressions."""

from __future__ import annotations

import time

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_collections_service import LibraryCollectionRecord

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_library_snapshot,
    _wait_for_selector,
)


class FakeLibraryCollectionsService:
    def __init__(self, records=()):
        self.records = list(records)
        self.created = []
        self.renamed = []
        self.deleted = []
        self._counter = len(self.records) + 1
        self._timestamp_counter = 0

    def _now(self) -> str:
        self._timestamp_counter += 1
        return f"2026-05-08T04:{self._timestamp_counter:02d}:00Z"

    def list_collections(self):
        return tuple(self.records)

    def create_collection(self, name, *, description=""):
        timestamp = self._now()
        record = LibraryCollectionRecord(
            collection_id=f"collection-{self._counter}",
            name=name.strip(),
            description=description.strip(),
            item_count=0,
            source_authority="local",
            sync_status="local-only",
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._counter += 1
        self.records.append(record)
        self.created.append((name, description))
        return record

    def rename_collection(self, collection_id, name, *, description=None):
        timestamp = self._now()
        renamed = None
        for index, record in enumerate(self.records):
            if record.collection_id != collection_id:
                continue
            renamed = LibraryCollectionRecord(
                collection_id=record.collection_id,
                name=name.strip(),
                description="" if description is None else description.strip(),
                item_count=record.item_count,
                source_authority=record.source_authority,
                sync_status=record.sync_status,
                created_at=record.created_at,
                updated_at=timestamp,
            )
            self.records[index] = renamed
            break
        if renamed is None:
            raise KeyError(collection_id)
        self.renamed.append((collection_id, name, description))
        return renamed

    def delete_collection(self, collection_id):
        before = len(self.records)
        self.records = [record for record in self.records if record.collection_id != collection_id]
        self.deleted.append(collection_id)
        return len(self.records) != before


class RaisingLibraryCollectionsService:
    def list_collections(self):
        raise RuntimeError("collections database unavailable")


def _seed_library_sources(app) -> None:
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "id": "chat-1"}]
    )


async def _wait_for_text(screen, pilot, expected: str, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if expected in _visible_text(screen):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for text {expected!r}: {_visible_text(screen)}")


@pytest.mark.asyncio
async def test_library_collections_mode_mounts_panel_and_defers_scoped_actions() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_collections_service = FakeLibraryCollectionsService(
        (
            LibraryCollectionRecord(
                collection_id="collection-1",
                name="Research",
                description="Policy sources",
                item_count=2,
                source_authority="local",
                sync_status="sync-unavailable",
                created_at="2026-05-08T04:00:00Z",
                updated_at="2026-05-08T04:05:00Z",
            ),
        )
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        assert screen.query_one("#library-collections-panel").parent is screen.query_one(
            "#library-source-detail"
        )
        assert len(screen.query("#library-rag-run-query")) == 0
        assert "Sync: sync-unavailable" in _visible_text(screen)
        assert "Updated 2026-05-08 04:05 UTC" in _visible_text(screen)
        assert screen.query_one("#library-open-study", Button).disabled is True
        assert screen.query_one("#library-open-flashcards", Button).disabled is True
        assert screen.query_one("#library-open-quizzes", Button).disabled is True
        assert screen.query_one("#library-use-in-console", Button).disabled is True
        assert "Collection-scoped Study, Flashcards, Quizzes, and Console are later-stage." in (
            _visible_text(screen)
        )


@pytest.mark.asyncio
async def test_library_collections_create_rename_and_delete_workflow() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = FakeLibraryCollectionsService()
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")
        assert "Group saved Library items for Search/RAG, Study, and Console." in _visible_text(
            screen
        )

        screen.query_one("#library-collection-name-input", Input).value = "Research"
        screen.query_one("#library-collection-description-input", Input).value = "Policy sources"
        await pilot.pause()
        screen.query_one("#library-create-collection", Button).press()
        await _wait_for_text(screen, pilot, "Research")

        assert service.created == [("Research", "Policy sources")]
        assert "0 items" in _visible_text(screen)
        assert "Sync: local-only" in _visible_text(screen)
        assert "Updated 2026-05-08 04:01 UTC" in _visible_text(screen)

        screen.query_one("#library-collection-name-input", Input).value = "Briefing Queue"
        screen.query_one("#library-collection-description-input", Input).value = "Updated"
        await pilot.pause()
        screen.query_one("#library-rename-collection", Button).press()
        await _wait_for_text(screen, pilot, "Briefing Queue")

        assert service.renamed == [("collection-1", "Briefing Queue", "Updated")]
        assert "Updated 2026-05-08 04:02 UTC" in _visible_text(screen)

        screen.query_one("#library-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-confirm-delete-collection")
        assert service.deleted == []

        screen.query_one("#library-confirm-delete-collection", Button).press()
        await _wait_for_text(
            screen,
            pilot,
            "Group saved Library items for Search/RAG, Study, and Console.",
        )

    assert service.deleted == ["collection-1"]


@pytest.mark.asyncio
async def test_library_collections_service_failure_renders_recovery_copy() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_collections_service = RaisingLibraryCollectionsService()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-error")

        error_text = screen.query_one("#library-collections-error", Static).renderable
        assert "Library Collections are unavailable" in str(error_text)
        assert "collections database unavailable" not in _visible_text(screen)
