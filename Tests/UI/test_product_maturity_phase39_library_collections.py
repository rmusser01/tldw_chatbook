"""Phase 3.9 Library Collections mounted UI regressions."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_collections_service import LibraryCollectionRecord
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.runtime_policy.types import RuntimeSourceState

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


class DeleteFailsLibraryCollectionsService(FakeLibraryCollectionsService):
    def delete_collection(self, collection_id):
        self.deleted.append(collection_id)
        return False


def _activate_server_sync_scope(app) -> None:
    app.runtime_policy.state = RuntimeSourceState(
        active_source="server",
        active_server_id="server-a",
        server_configured=True,
    )
    app.workspace_registry_service.create_workspace(
        workspace_id="workspace-1",
        name="Workspace 1",
    )
    app.workspace_registry_service.set_active_workspace("workspace-1")
    app.server_context_provider = SimpleNamespace(
        get_active_context=lambda: SimpleNamespace(
            auth_token="header.eyJzdWIiOiJ1c2VyLWEifQ.signature"
        )
    )


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
async def test_library_collections_surfaces_sync_dry_run_report_without_write_sync(tmp_path) -> None:
    app = _build_test_app()
    _activate_server_sync_scope(app)
    _seed_library_sources(app)
    app.library_collections_service = FakeLibraryCollectionsService(
        (
            LibraryCollectionRecord(
                collection_id="collection-1",
                name="Research",
                description="Policy sources",
                item_count=2,
                source_authority="local",
                sync_status="local-only",
                created_at="2026-05-08T04:00:00Z",
                updated_at="2026-05-08T04:05:00Z",
            ),
        )
    )
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="jwt-sub:user-a",
        workspace_scope="workspace-1",
        domain="library_collections",
        report={
            "dry_run": True,
            "write_enabled": False,
            "mapped_count": 1,
            "actions": [
                {
                    "identity": {"local_entity_id": "collection-1"},
                    "local_present": True,
                    "remote_present": True,
                }
            ],
        },
    )
    app.sync_state_repository = repo
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        visible = _visible_text(screen)
        assert "Sync: dry-run only" in visible
        assert "Mirror: 1 mapped record" in visible
        assert "Review: required before writes" in visible
        assert "Review dry-run results before enabling writes." in visible
        assert "write sync enabled" not in visible.lower()
        inspector_text = " ".join(
            str(widget.renderable)
            for widget in screen.query("#library-source-inspector Static")
        )
        assert "Selected Collection" in inspector_text
        assert "Research" in inspector_text
        assert "What this means" in inspector_text
        assert "This is a read-only sync dry run. No server writes can run from this screen." in (
            inspector_text
        )
        assert "No source selected." not in inspector_text


@pytest.mark.asyncio
async def test_library_collections_scopes_sync_conflicts_to_selected_collection(tmp_path) -> None:
    app = _build_test_app()
    _activate_server_sync_scope(app)
    _seed_library_sources(app)
    app.library_collections_service = FakeLibraryCollectionsService(
        (
            LibraryCollectionRecord(
                collection_id="collection-ready",
                name="Ready Collection",
                description="Policy sources",
                item_count=2,
                source_authority="local",
                sync_status="local-only",
                created_at="2026-05-08T04:00:00Z",
                updated_at="2026-05-08T04:05:00Z",
            ),
            LibraryCollectionRecord(
                collection_id="collection-conflict",
                name="Conflict Collection",
                description="Review mappings",
                item_count=1,
                source_authority="local",
                sync_status="local-only",
                created_at="2026-05-08T04:00:00Z",
                updated_at="2026-05-08T04:05:00Z",
            ),
        )
    )
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="jwt-sub:user-a",
        workspace_scope="workspace-1",
        domain="library_collections",
        report={
            "dry_run": True,
            "write_enabled": False,
            "mapped_count": 1,
            "actions": [
                {
                    "identity": {"local_entity_id": "collection-ready"},
                    "local_present": True,
                    "remote_present": True,
                }
            ],
        },
    )
    repo.record_identity_mapping(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="jwt-sub:user-a",
        workspace_scope="workspace-1",
        domain="library_collections",
        entity_type="collection",
        local_entity_id="collection-conflict",
        remote_entity_id="remote-a",
        mapping_status="confirmed",
    )
    repo.record_identity_mapping(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="jwt-sub:user-a",
        workspace_scope="workspace-1",
        domain="library_collections",
        entity_type="collection",
        local_entity_id="collection-conflict",
        remote_entity_id="remote-b",
        mapping_status="confirmed",
    )
    app.sync_state_repository = repo
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        visible = _visible_text(screen)
        assert "Sync: dry-run only" in visible
        assert "Sync: conflict review required" not in visible


@pytest.mark.asyncio
async def test_library_collections_ignores_sync_state_from_other_scope(tmp_path) -> None:
    app = _build_test_app()
    _activate_server_sync_scope(app)
    _seed_library_sources(app)
    app.library_collections_service = FakeLibraryCollectionsService(
        (
            LibraryCollectionRecord(
                collection_id="collection-1",
                name="Research",
                description="Policy sources",
                item_count=2,
                source_authority="local",
                sync_status="local-only",
                created_at="2026-05-08T04:00:00Z",
                updated_at="2026-05-08T04:05:00Z",
            ),
        )
    )
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-b",
        authenticated_principal_id="user-b",
        workspace_scope="workspace-2",
        domain="library_collections",
        report={
            "dry_run": True,
            "write_enabled": False,
            "mapped_count": 1,
            "actions": [
                {
                    "identity": {"local_entity_id": "collection-1"},
                    "local_present": True,
                    "remote_present": True,
                }
            ],
        },
    )
    app.sync_state_repository = repo
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        visible = _visible_text(screen)
        assert "Mirror: 1 mapped record" not in visible
        assert "Sync: dry-run only" in visible


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
        assert "Sync: dry-run only" in _visible_text(screen)
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
async def test_library_collection_form_input_keeps_focus_and_updates_actions() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_collections_service = FakeLibraryCollectionsService()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        name_input = screen.query_one("#library-collection-name-input", Input)
        name_input.focus()
        await pilot.pause()
        name_input.value = "Research"
        await pilot.pause()

        assert screen.focused is name_input
        assert screen.query_one("#library-collections-panel").is_mounted
        assert screen.query_one("#library-create-collection", Button).disabled is False


@pytest.mark.asyncio
async def test_library_collections_delete_failure_keeps_selection_and_warns_user() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = DeleteFailsLibraryCollectionsService(
        (
            LibraryCollectionRecord(
                collection_id="collection-1",
                name="Research",
                description="Policy sources",
                item_count=0,
                source_authority="local",
                sync_status="local-only",
                created_at="2026-05-08T04:00:00Z",
                updated_at="2026-05-08T04:00:00Z",
            ),
        )
    )
    app.library_collections_service = service
    notifications = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")
        screen.query_one("#library-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-confirm-delete-collection")
        screen.query_one("#library-confirm-delete-collection", Button).press()
        await _wait_for_text(screen, pilot, "Research")

        assert service.deleted == ["collection-1"]
        assert "Research" in _visible_text(screen)
        assert notifications
        assert notifications[-1][0] == "Failed to delete Collection."
        assert notifications[-1][1]["severity"] == "warning"


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
