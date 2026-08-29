"""Phase 3.9 Library Collections mounted UI regressions."""

from __future__ import annotations

import asyncio
from itertools import count
import time
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.library_collections_service import (
    LibraryCollectionRecord,
    LibraryCollectionsServiceError,
    LocalLibraryCollectionsService,
)
from tldw_chatbook.Library.library_collections_state import (
    CollectionBrowseScope,
    LibraryCollectionDeleteReceipt,
    LibraryCollectionsPanelState,
)
from tldw_chatbook.Library.library_pager_state import build_library_pager_display
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.Widgets.Library.library_collections_panel import (
    LIBRARY_COLLECTIONS_STATUS_LINE,
    LibraryCollectionsPanel,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

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
        self.restored = []
        self._deleted_records = {}
        self._counter = len(self.records) + 1
        self._timestamp_counter = 0
        self.page_calls = []
        self.locator_calls = []

    def _now(self) -> str:
        self._timestamp_counter += 1
        return f"2026-05-08T04:{self._timestamp_counter:02d}:00Z"

    def list_collections(self):
        return tuple(self.records)

    @staticmethod
    def _summary(record: LibraryCollectionRecord) -> dict[str, object]:
        return {
            "collection_id": record.collection_id,
            "name": record.name,
            "description": record.description,
            "item_count": record.item_count,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def _ordered_records(self) -> list[LibraryCollectionRecord]:
        return sorted(
            self.records,
            key=lambda record: (
                record.created_at,
                record.name.casefold(),
                record.collection_id,
            ),
        )

    def list_library_collections(self, *, limit=20, offset=0):
        self.page_calls.append({"limit": limit, "offset": offset})
        records = self._ordered_records()
        return {
            "items": [self._summary(record) for record in records[offset : offset + limit]],
            "total": len(records),
            "limit": limit,
            "offset": offset,
        }

    def locate_library_collection_page(self, collection_id, *, limit=20):
        self.locator_calls.append((collection_id, {"limit": limit}))
        records = self._ordered_records()
        rank = next(
            (
                index
                for index, record in enumerate(records)
                if record.collection_id == collection_id
            ),
            None,
        )
        if rank is None:
            return None
        offset = (rank // limit) * limit
        return {
            "items": [self._summary(record) for record in records[offset : offset + limit]],
            "total": len(records),
            "limit": limit,
            "offset": offset,
            "page": offset // limit + 1,
            "target_id": collection_id,
            "target_rank": rank,
            "target_index": rank - offset,
        }

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
        self._deleted_records.update(
            {
                record.collection_id: record
                for record in self.records
                if record.collection_id == collection_id
            }
        )
        self.records = [
            record for record in self.records if record.collection_id != collection_id
        ]
        self.deleted.append(collection_id)
        return len(self.records) != before

    def restore_collection(self, collection_id):
        record = self._deleted_records.pop(collection_id)
        self.records.append(record)
        self.restored.append(collection_id)
        return record


class FailingCollectionFollowupService(FakeLibraryCollectionsService):
    """Fake whose durable writes succeed while selected follow-up reads fail."""

    fail_locator = False
    fail_page = False

    def list_library_collections(self, *, limit=20, offset=0):
        if self.fail_page:
            self.page_calls.append({"limit": limit, "offset": offset})
            raise LibraryCollectionsServiceError("follow-up page failed")
        return super().list_library_collections(limit=limit, offset=offset)

    def locate_library_collection_page(self, collection_id, *, limit=20):
        if self.fail_locator:
            self.locator_calls.append((collection_id, {"limit": limit}))
            raise LibraryCollectionsServiceError("follow-up locator failed")
        return super().locate_library_collection_page(collection_id, limit=limit)


class RecordingLibraryCollectionsService:
    """Record bounded reads around a real isolated SQLite service."""

    def __init__(self, delegate: LocalLibraryCollectionsService) -> None:
        self.delegate = delegate
        self.page_calls: list[dict[str, int]] = []
        self.locator_calls: list[tuple[str, dict[str, int]]] = []
        self.fail_locator_once = False

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def list_library_collections(self, *, limit=20, offset=0):
        self.page_calls.append({"limit": limit, "offset": offset})
        return self.delegate.list_library_collections(limit=limit, offset=offset)

    def locate_library_collection_page(self, collection_id, *, limit=20):
        self.locator_calls.append((collection_id, {"limit": limit}))
        if self.fail_locator_once:
            self.fail_locator_once = False
            raise LibraryCollectionsServiceError("injected follow-up failure")
        return self.delegate.locate_library_collection_page(
            collection_id,
            limit=limit,
        )


class RaisingLibraryCollectionsService:
    def list_collections(self):
        raise RuntimeError("collections database unavailable")

    def list_library_collections(self, *, limit=20, offset=0):
        raise RuntimeError("collections database unavailable")


class DeleteFailsLibraryCollectionsService(FakeLibraryCollectionsService):
    def delete_collection(self, collection_id):
        self.deleted.append(collection_id)
        return False


class DelayedRestoreLibraryCollectionsService(FakeLibraryCollectionsService):
    def __init__(self, records=()):
        super().__init__(records)
        self.restore_started = asyncio.Event()
        self.restore_release = asyncio.Event()

    async def restore_collection(self, collection_id):
        self.restore_started.set()
        await self.restore_release.wait()
        return super().restore_collection(collection_id)


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
    # Server-active policy makes decorative Library rail counts eligible to
    # call the production client. These tests exercise Collection sync state,
    # not remote study/prompt/skill discovery, so keep those unrelated seams
    # hermetic at the shared policy helper.
    app.study_scope_service = SimpleNamespace()
    app.study_quiz_scope_service = SimpleNamespace()
    app.prompt_scope_service = SimpleNamespace()
    app.skills_scope_service = SimpleNamespace()


class FakeSyncProfileSummaryService:
    def __init__(self, summary: dict[str, object]):
        self.summary = summary
        self.summary_calls = []
        self.push_calls = []
        self.pull_calls = []

    def get_sync_v2_profile_summary(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
    ):
        self.summary_calls.append(
            {
                "server_profile_id": server_profile_id,
                "authenticated_principal_id": authenticated_principal_id,
                "workspace_scope": workspace_scope,
            }
        )
        return dict(self.summary)

    def push_v2_envelopes(self, *args, **kwargs):
        self.push_calls.append((args, kwargs))
        raise AssertionError("Library status rendering must not push sync envelopes")

    def pull_v2_envelopes(self, *args, **kwargs):
        self.pull_calls.append((args, kwargs))
        raise AssertionError("Library status rendering must not pull sync envelopes")


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
    raise AssertionError(
        f"Timed out waiting for text {expected!r}: {_visible_text(screen)}"
    )


def _paged_collection_rows(count: int = 20) -> list[dict[str, object]]:
    return [
        {
            "collection_id": f"collection-{index}",
            "name": f"Collection with a deliberately complete title {index}",
            "description": f"Detail {index}",
            "item_count": index,
            "source_authority": "local",
            "sync_status": "local-only",
            "created_at": "2026-05-08T04:00:00Z",
            "updated_at": "2026-05-08T04:00:00Z",
        }
        for index in range(1, count + 1)
    ]


def _collection_records(count: int) -> tuple[LibraryCollectionRecord, ...]:
    return tuple(
        LibraryCollectionRecord(
            collection_id=f"collection-{index:02d}",
            name=f"Collection {index:02d}",
            description=f"Detail {index:02d}",
            item_count=index,
            source_authority="local",
            sync_status="local-only",
            created_at="2026-05-08T04:00:00Z",
            updated_at="2026-05-08T04:00:00Z",
        )
        for index in range(1, count + 1)
    )


def _painted_text(screen: LibraryScreen) -> str:
    """Return only text painted into the current production-shaped frame."""

    return "\n".join(
        "".join(segment.text for segment in strip)
        for strip in screen._compositor.render_strips()
    )


class _CollectionsPagerPanelApp(App):
    def __init__(
        self,
        state: LibraryCollectionsPanelState,
        *,
        pager,
        page_actions_disabled: bool = False,
    ) -> None:
        super().__init__()
        self._state = state
        self._pager = pager
        self._page_actions_disabled = page_actions_disabled

    def compose(self) -> ComposeResult:
        yield LibraryCollectionsPanel(
            self._state,
            pager=self._pager,
            page_actions_disabled=self._page_actions_disabled,
            id="library-collections-panel",
        )


@pytest.mark.asyncio
async def test_library_collections_pager_topology_and_exact_first_page_copy() -> None:
    rows = _paged_collection_rows()
    state = LibraryCollectionsPanelState.from_values(
        collections=rows,
        selected_collection_id="collection-1",
        create_name="New Collection",
        rename_name="Renamed Collection",
    )
    pager = build_library_pager_display(
        applied_page=1,
        requested_page=1,
        page_size=20,
        row_count=20,
        total=45,
        freshness="fresh",
    )

    async with _CollectionsPagerPanelApp(state, pager=pager).run_test(
        size=(100, 30)
    ) as pilot:
        panel = pilot.app.query_one("#library-collections-panel")
        row_scroll = panel.query_one("#library-collections-rows-scroll")
        previous = panel.query_one("#library-collections-previous", Button)
        next_button = panel.query_one("#library-collections-next", Button)

        assert len(panel.query(".library-collection-row")) == 20
        assert row_scroll.parent is panel.query_one("#library-collections-list")
        assert panel.query_one("#library-collections-list") in previous.ancestors
        assert row_scroll not in previous.ancestors
        assert panel.query_one("#library-collections-list") in next_button.ancestors
        assert str(panel.query_one("#library-collections-title", Static).renderable) == (
            "Collections (45)"
        )
        assert str(panel.query_one("#library-collections-range", Static).renderable) == (
            "1-20 of 45"
        )
        assert str(panel.query_one("#library-collections-page", Static).renderable) == (
            "Page 1 of 3"
        )
        assert previous.disabled is True
        assert next_button.disabled is False
        await pilot.pause()


@pytest.mark.asyncio
async def test_library_collections_stale_page_is_readable_but_inert() -> None:
    rows = _paged_collection_rows()
    state = LibraryCollectionsPanelState.from_values(
        collections=rows,
        selected_collection_id="collection-1",
        create_name="New Collection",
        rename_name="Renamed Collection",
        delete_receipt=LibraryCollectionDeleteReceipt(
            collection_id="collection-deleted",
            name="Deleted Collection",
        ),
    )
    pager = build_library_pager_display(
        applied_page=1,
        requested_page=1,
        page_size=20,
        row_count=20,
        total=None,
        freshness="stale",
        stale_copy="Collections changed; retry to load a current page.",
    )

    async with _CollectionsPagerPanelApp(
        state,
        pager=pager,
        page_actions_disabled=True,
    ).run_test() as pilot:
        panel = pilot.app.query_one("#library-collections-panel")

        assert str(panel.query_one("#library-collections-title", Static).renderable) == (
            "Collections"
        )
        assert all(button.disabled for button in panel.query(".library-collection-row"))
        for selector in (
            "#library-create-collection",
            "#library-rename-collection",
            "#library-delete-collection",
            "#library-collections-delete-undo",
            "#library-collections-previous",
            "#library-collections-next",
        ):
            assert panel.query_one(selector, Button).disabled is True
        assert panel.query_one("#library-collections-retry", Button).disabled is False
        assert str(panel.query_one("#library-collections-range", Static).renderable) == (
            "List may be out of date"
        )
        await pilot.pause()


@pytest.mark.asyncio
async def test_library_collections_first_load_failure_keeps_total_unavailable() -> None:
    state = LibraryCollectionsPanelState.from_values(
        collections=(),
        status="error",
        error_message="Database unavailable",
    )
    pager = build_library_pager_display(
        applied_page=None,
        requested_page=1,
        page_size=20,
        row_count=0,
        total=None,
        freshness="uninitialized",
        error_copy="Couldn't load Collections. Check the local Library and retry.",
    )

    async with _CollectionsPagerPanelApp(state, pager=pager).run_test() as pilot:
        panel = pilot.app.query_one("#library-collections-panel")

        assert str(panel.query_one("#library-collections-title", Static).renderable) == (
            "Collections"
        )
        assert "Total unavailable" in str(
            panel.query_one("#library-collections-range", Static).renderable
        )
        assert panel.query_one("#library-collections-retry", Button).disabled is False
        await pilot.pause()


@pytest.mark.asyncio
async def test_library_collections_page_navigation_uses_bounded_source_and_focus() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = FakeLibraryCollectionsService(_collection_records(45))
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-20 of 45")

        assert service.page_calls == [{"limit": 20, "offset": 0}]
        assert len(screen.query(".library-collection-row")) == 20
        next_button = screen.query_one("#library-collections-next", Button)
        next_button.focus()
        next_button.press()
        await _wait_for_text(screen, pilot, "21-40 of 45")

        assert service.page_calls[-1] == {"limit": 20, "offset": 20}
        assert len(screen.query(".library-collection-row")) == 20
        assert getattr(screen.focused, "id", None) == "library-collections-next"

        screen.query_one("#library-collections-previous", Button).press()
        await _wait_for_text(screen, pilot, "1-20 of 45")
        assert service.page_calls[-1] == {"limit": 20, "offset": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (170, 48)])
async def test_library_collections_production_geometry_walks_all_pages(
    size: tuple[int, int],
) -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = FakeLibraryCollectionsService(_collection_records(45))
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=size) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-20 of 45")

        list_pane = screen.query_one("#library-collections-list")
        row_scroll = screen.query_one("#library-collections-rows-scroll")
        previous = screen.query_one("#library-collections-previous", Button)
        next_button = screen.query_one("#library-collections-next", Button)
        range_line = screen.query_one("#library-collections-range")
        page_line = screen.query_one("#library-collections-page")
        assert list_pane.region.contains_region(previous.region)
        assert list_pane.region.contains_region(next_button.region)
        assert list_pane.region.contains_region(range_line.region)
        assert list_pane.region.contains_region(page_line.region)
        assert row_scroll not in previous.ancestors
        assert row_scroll not in next_button.ancestors
        first_frame = _painted_text(screen)
        assert "1-20 of 45" in first_frame
        assert "Page 1 of 3" in first_frame
        assert "Collection 01" in first_frame

        next_button.focus()
        next_button.press()
        await _wait_for_text(screen, pilot, "21-40 of 45")
        middle_frame = _painted_text(screen)
        assert "21-40 of 45" in middle_frame
        assert "Page 2 of 3" in middle_frame
        assert "Collection 21" in middle_frame

        screen.query_one("#library-collections-next", Button).press()
        await _wait_for_text(screen, pilot, "41-45 of 45")
        final_frame = _painted_text(screen)
        assert "41-45 of 45" in final_frame
        assert "Page 3 of 3" in final_frame
        assert "Collection 41" in final_frame
        assert getattr(screen.focused, "id", None) == "library-collections-previous"
        assert [call["offset"] for call in service.page_calls] == [0, 20, 40]

        name_input = screen.query_one("#library-collection-name-input", Input)
        assert not name_input.disabled
        name_input.focus()
        name_input.scroll_visible(animate=False)
        await pilot.pause()
        assert screen.focused is name_input


@pytest.mark.asyncio
async def test_library_collections_persists_only_the_applied_page() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_collections_service = FakeLibraryCollectionsService(
        _collection_records(45)
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-20 of 45")
        screen.query_one("#library-collections-next", Button).press()
        await _wait_for_text(screen, pilot, "21-40 of 45")

        assert screen.save_state()["library_collections_page"] == 2
        screen._library_collections_browse_controller.begin(
            CollectionBrowseScope(page=3)
        )
        assert screen.save_state()["library_collections_page"] == 2


@pytest.mark.parametrize("value", [True, "2", 0, -1, 2**63])
def test_library_collections_invalid_restored_pages_normalize_to_one(value) -> None:
    assert LibraryScreen._restore_library_collections_scope(
        {"library_collections_page": value}
    ) == CollectionBrowseScope()


@pytest.mark.asyncio
async def test_library_collection_create_locates_its_owning_page_without_walking() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = FakeLibraryCollectionsService(_collection_records(45))
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-20 of 45")
        screen.query_one("#library-collection-name-input", Input).value = "Created"
        await pilot.pause()
        screen.query_one("#library-create-collection", Button).press()
        await _wait_for_text(screen, pilot, "41-46 of 46")

        assert service.page_calls == [{"limit": 20, "offset": 0}]
        assert service.locator_calls == [("collection-46", {"limit": 20})]
        assert "Selected: Created" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_collection_rename_relocates_equal_time_row_by_stable_id() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = FakeLibraryCollectionsService(_collection_records(45))
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-20 of 45")
        screen.query_one("#library-collection-name-input", Input).value = "ZZZ"
        await pilot.pause()
        screen.query_one("#library-rename-collection", Button).press()
        await _wait_for_text(screen, pilot, "41-45 of 45")

        assert service.page_calls == [{"limit": 20, "offset": 0}]
        assert service.locator_calls == [("collection-01", {"limit": 20})]
        assert "Selected: ZZZ" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_collection_delete_clamps_once_and_restore_locates_receipt() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = FakeLibraryCollectionsService(_collection_records(41))
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-20 of 41")
        screen.query_one("#library-collections-next", Button).press()
        await _wait_for_text(screen, pilot, "21-40 of 41")
        screen.query_one("#library-collections-next", Button).press()
        await _wait_for_text(screen, pilot, "41-41 of 41")

        screen.query_one("#library-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-confirm-delete-collection")
        screen.query_one("#library-confirm-delete-collection", Button).press()
        await _wait_for_text(screen, pilot, "21-40 of 40")

        assert [call["offset"] for call in service.page_calls] == [0, 20, 40, 40, 20]
        assert screen.query("#library-collections-delete-receipt")

        screen.query_one("#library-collections-delete-undo", Button).press()
        await _wait_for_text(screen, pilot, "41-41 of 41")

        assert service.locator_calls == [("collection-41", {"limit": 20})]
        assert not screen.query("#library-collections-delete-receipt")
        assert "Selected: Collection 41" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_collection_create_stays_committed_when_locator_fails() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = FailingCollectionFollowupService(_collection_records(20))
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-20 of 20")
        service.fail_locator = True

        screen.query_one("#library-collection-name-input", Input).value = "Created"
        await pilot.pause()
        screen.query_one("#library-create-collection", Button).press()
        await _wait_for_text(screen, pilot, "Collections changed; retry")

        controller = screen._library_collections_browse_controller
        assert service.created == [("Created", "")]
        assert service.locator_calls == [("collection-21", {"limit": 20})]
        assert controller.freshness == "stale"
        assert controller.pager.title_count is None
        assert "Selected: Created" in _visible_text(screen)
        assert screen.query_one("#library-create-collection", Button).disabled
        assert screen.query_one("#library-rename-collection", Button).disabled
        assert screen.query_one("#library-delete-collection", Button).disabled
        assert not screen.query_one("#library-collections-retry", Button).disabled


@pytest.mark.asyncio
async def test_library_collection_restore_keeps_receipt_when_locator_fails() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = FailingCollectionFollowupService(_collection_records(1))
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-1 of 1")

        screen.query_one("#library-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-confirm-delete-collection")
        screen.query_one("#library-confirm-delete-collection", Button).press()
        await _wait_for_text(screen, pilot, "No Collections yet")
        service.fail_locator = True

        screen.query_one("#library-collections-delete-undo", Button).press()
        await _wait_for_text(screen, pilot, "Collections changed; retry")

        controller = screen._library_collections_browse_controller
        assert service.restored == ["collection-01"]
        assert service.locator_calls == [("collection-01", {"limit": 20})]
        assert controller.freshness == "stale"
        assert screen.query("#library-collections-delete-receipt")
        assert "Selected: Collection 01" in _visible_text(screen)
        assert screen.query_one("#library-create-collection", Button).disabled
        assert not screen.query_one("#library-collections-retry", Button).disabled


@pytest.mark.asyncio
async def test_library_collection_delete_stays_committed_when_page_reload_fails() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = FailingCollectionFollowupService(_collection_records(2))
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-2 of 2")
        service.fail_page = True

        screen.query_one("#library-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-confirm-delete-collection")
        screen.query_one("#library-confirm-delete-collection", Button).press()
        await _wait_for_text(screen, pilot, "Collections changed; retry")

        controller = screen._library_collections_browse_controller
        assert service.deleted == ["collection-01"]
        assert controller.freshness == "stale"
        assert screen.query("#library-collections-delete-receipt")
        assert "Collection 01" not in tuple(
            str(record.get("name", "")) for record in controller.retained_items
        )
        assert screen.query_one("#library-delete-collection", Button).disabled
        assert not screen.query_one("#library-collections-retry", Button).disabled


@pytest.mark.asyncio
async def test_library_collections_isolated_sqlite_mutation_walkthrough(tmp_path) -> None:
    identifier = count(1)
    delegate = LocalLibraryCollectionsService(
        LibraryCollectionsDB(tmp_path / "collections-live.db"),
        id_factory=lambda: f"collection-{next(identifier):02d}",
        now_factory=lambda: "2026-05-08T04:00:00Z",
    )
    for index in range(1, 46):
        delegate.create_collection(f"Collection {index:02d}")
    service = RecordingLibraryCollectionsService(delegate)
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-20 of 45")

        name_input = screen.query_one("#library-collection-name-input", Input)
        name_input.value = "Aardvark"
        await pilot.pause()
        screen.query_one("#library-create-collection", Button).press()
        await _wait_for_text(screen, pilot, "1-20 of 46")
        assert "Selected: Aardvark" in _visible_text(screen)

        screen.query_one("#library-collection-name-input", Input).value = "Zulu"
        await pilot.pause()
        screen.query_one("#library-rename-collection", Button).press()
        await _wait_for_text(screen, pilot, "41-46 of 46")
        assert "Selected: Zulu" in _visible_text(screen)

        screen.query_one("#library-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-confirm-delete-collection")
        screen.query_one("#library-confirm-delete-collection", Button).press()
        await _wait_for_text(screen, pilot, "41-45 of 45")
        assert delegate.get_collection("collection-46") is None

        screen.query_one("#library-collections-delete-undo", Button).press()
        await _wait_for_text(screen, pilot, "41-46 of 46")
        assert delegate.get_collection("collection-46") is not None
        assert "Selected: Zulu" in _visible_text(screen)

        service.fail_locator_once = True
        screen.query_one("#library-collection-name-input", Input).value = (
            "Failure Known"
        )
        await pilot.pause()
        screen.query_one("#library-create-collection", Button).press()
        await _wait_for_text(screen, pilot, "Collections changed; retry")
        assert delegate.get_collection("collection-47") is not None
        assert screen.query_one("#library-create-collection", Button).disabled

        screen.query_one("#library-collections-retry", Button).press()
        await _wait_for_text(screen, pilot, "41-47 of 47")
        assert "Selected: Failure Known" in _visible_text(screen)
        assert service.locator_calls == [
            ("collection-46", {"limit": 20}),
            ("collection-46", {"limit": 20}),
            ("collection-46", {"limit": 20}),
            ("collection-47", {"limit": 20}),
            ("collection-47", {"limit": 20}),
        ]


@pytest.mark.asyncio
async def test_library_collections_mode_mounts_panel_and_defers_scoped_actions() -> (
    None
):
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

        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "Sync: dry-run only")

        route_content = screen.query_one("#library-canvas-route-content")
        assert screen.query_one("#library-collections-panel").parent is route_content
        assert route_content.parent is screen.query_one("#library-canvas")
        assert len(screen.query("#library-rag-run-query")) == 0
        assert "Sync: dry-run only" in _visible_text(screen)
        assert "Updated 2026-05-08 04:05 UTC" in _visible_text(screen)
        # TASK-2855: the retired action-region's per-mode
        # Study/Flashcards/Quizzes/Console handoff buttons never mount for
        # a collection selection (they only live in the
        # create-study/-flashcards/-quizzes canvases now, as screen-global
        # actions rather than collection-scoped ones); the deferred-actions
        # roadmap copy ("Blocked later: ...", "Next: collection item
        # adapters...") that used to be the surviving surface for
        # "collection item actions stay blocked" was spec/roadmap
        # vocabulary and was replaced by one plain-language status line.
        assert LIBRARY_COLLECTIONS_STATUS_LINE in _visible_text(screen)
        assert (
            "Blocked later: item reader, Search/RAG, Study, Console handoff, server sync"
            not in _visible_text(screen)
        )
        assert (
            "Next: collection item adapters are required before item-level actions unlock."
            not in _visible_text(screen)
        )


@pytest.mark.asyncio
async def test_library_collections_surfaces_sync_dry_run_report_without_write_sync(
    tmp_path,
) -> None:
    app = _build_test_app()
    _activate_server_sync_scope(app)
    _seed_library_sources(app)
    # Keep this mounted UI contract hermetic.  ``None`` would trigger the
    # app's production services and, under the server-active policy below,
    # their live count/context calls.
    app.study_scope_service = SimpleNamespace()
    app.study_quiz_scope_service = SimpleNamespace()
    app.prompt_scope_service = SimpleNamespace()
    app.skills_scope_service = SimpleNamespace()
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

        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "Sync: dry-run only")

        visible = _visible_text(screen)
        assert "Sync: dry-run only" in visible
        assert "Mirror: 1 mapped record" in visible
        assert "Review: required before writes" in visible
        assert "Review dry-run results before enabling writes." in visible
        assert "write sync enabled" not in visible.lower()
        # The retired Inspector column's "Selected Collection Record" /
        # "What this means" / "read-only sync dry run" copy has no verbatim
        # successor; the merged collection-detail column (inside the
        # collections panel itself) is the surviving scoped surface that
        # carries the same "selected item, no writes without review" claim.
        detail_text = " ".join(
            str(widget.renderable)
            for widget in screen.query("#library-collection-detail Static")
        )
        assert "Selected: Research" in detail_text
        # TASK-2855: the "Write Sync Safety" heading and its help sentence
        # were spec-internal chrome and were removed; the dry-run
        # promotion data above (asserted at the top of this test) is the
        # genuinely useful part and now lives behind the Details
        # disclosure instead.
        assert "Write Sync Safety" not in detail_text
        assert (
            "Review these labels before any future server write promotion."
            not in detail_text
        )
        assert "No Collection selected." not in detail_text


@pytest.mark.asyncio
async def test_library_collections_surfaces_sync_profile_summary_without_write_sync() -> (
    None
):
    app = _build_test_app()
    _seed_library_sources(app)
    app.runtime_policy = SimpleNamespace(
        state=RuntimeSourceState(
            active_source="server",
            server_configured=True,
            active_server_id="server-a",
        )
    )
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
    sync_scope = FakeSyncProfileSummaryService(
        {
            "status": "pending",
            "profile": {
                "server_profile_id": "server-a",
                "authenticated_principal_id": None,
                "workspace_scope": None,
                "profile_mode": "local_first_sync",
                "device_id": "device-1",
                "dataset_id": "dataset-1",
                "last_error": None,
            },
            "cursor": None,
            "outbox": {"pending": 2, "dispatched": 0, "by_domain": {}},
            "identity_map": {"total": 0, "by_domain": {}},
            "conflicts": {"count": 0, "latest": []},
            "last_mirror_report": None,
        }
    )
    app.sync_scope_service = sync_scope
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-sync-profile-status")

        visible = _visible_text(screen)
        assert "Sync profile: pending local changes" in visible
        assert "2 pending local changes are waiting for the next sync pass." in visible
        assert "This view only reads sync state; it does not start sync." in visible
        assert sync_scope.summary_calls == [
            {
                "server_profile_id": "server-a",
                "authenticated_principal_id": None,
                "workspace_scope": None,
            }
        ]
        assert sync_scope.push_calls == []
        assert sync_scope.pull_calls == []


@pytest.mark.asyncio
async def test_library_collections_does_not_load_sync_profile_summary_in_local_mode() -> (
    None
):
    app = _build_test_app()
    _seed_library_sources(app)
    app.runtime_policy = SimpleNamespace(
        state=RuntimeSourceState(
            active_source="local",
            server_configured=True,
            active_server_id="server-a",
        )
    )
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
    sync_scope = FakeSyncProfileSummaryService({"status": "pending"})
    app.sync_scope_service = sync_scope
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "Sync: dry-run only")

        assert len(screen.query("#library-sync-profile-status")) == 0
        assert sync_scope.summary_calls == []
        assert sync_scope.push_calls == []
        assert sync_scope.pull_calls == []


@pytest.mark.asyncio
async def test_library_collections_validates_sync_profile_scope_before_summary_load() -> (
    None
):
    app = _build_test_app()
    _seed_library_sources(app)
    app.runtime_policy = SimpleNamespace(
        state=RuntimeSourceState(
            active_source="server",
            server_configured=True,
            active_server_id="server-a<script>",
        )
    )
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
    sync_scope = FakeSyncProfileSummaryService({"status": "pending"})
    app.sync_scope_service = sync_scope
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        assert len(screen.query("#library-sync-profile-status")) == 0
        assert sync_scope.summary_calls == []
        assert sync_scope.push_calls == []
        assert sync_scope.pull_calls == []


@pytest.mark.asyncio
async def test_library_collections_scopes_sync_conflicts_to_selected_collection(
    tmp_path,
) -> None:
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

        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_text(screen, pilot, "1-2 of 2")
        ready_row = next(
            row
            for row in screen.query(".library-collection-row")
            if getattr(row, "collection_id", "") == "collection-ready"
        )
        ready_row.press()
        await _wait_for_text(screen, pilot, "Sync: dry-run only")

        visible = _visible_text(screen)
        assert "Sync: dry-run only" in visible
        assert "Sync: conflict review required" not in visible


@pytest.mark.asyncio
async def test_library_collections_ignores_sync_state_from_other_scope(
    tmp_path,
) -> None:
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

        screen.query_one("#library-row-browse-collections", Button).press()
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

        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")
        assert "No Collections yet." in _visible_text(screen)
        assert "create one below to start" in _visible_text(screen)

        screen.query_one("#library-collection-name-input", Input).value = "Research"
        screen.query_one(
            "#library-collection-description-input", Input
        ).value = "Policy sources"
        await pilot.pause()
        screen.query_one("#library-create-collection", Button).press()
        await _wait_for_text(screen, pilot, "Research")

        assert service.created == [("Research", "Policy sources")]
        assert "0 items" in _visible_text(screen)
        assert "Sync: dry-run only" in _visible_text(screen)
        assert "Updated 2026-05-08 04:01 UTC" in _visible_text(screen)

        screen.query_one(
            "#library-collection-name-input", Input
        ).value = "Briefing Queue"
        screen.query_one(
            "#library-collection-description-input", Input
        ).value = "Updated"
        await pilot.pause()
        screen.query_one("#library-rename-collection", Button).press()
        await _wait_for_text(screen, pilot, "Briefing Queue")

        assert service.renamed == [("collection-1", "Briefing Queue", "Updated")]
        assert "Updated 2026-05-08 04:02 UTC" in _visible_text(screen)

        screen.query_one("#library-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-confirm-delete-collection")
        assert service.deleted == []
        confirm = screen.query_one("#library-confirm-delete-collection", Button)
        assert "Undo" in str(confirm.tooltip)
        assert "cannot be undone" not in str(confirm.tooltip)

        confirm.press()
        await _wait_for_selector(screen, pilot, "#library-collections-delete-receipt")
        assert "✓ deleted · Collection · Briefing Queue" in _visible_text(screen)
        assert "Collections (0)" in _visible_text(screen)

        screen.query_one("#library-collections-delete-undo", Button).press()
        await _wait_for_text(screen, pilot, "Briefing Queue")
        assert not screen.query("#library-collections-delete-receipt")
        assert "Collections (1)" in _visible_text(screen)

    assert service.deleted == ["collection-1"]
    assert service.restored == ["collection-1"]


@pytest.mark.asyncio
async def test_library_collection_undo_blocks_concurrent_create() -> None:
    record = LibraryCollectionRecord(
        collection_id="collection-1",
        name="Research",
        description="Policy sources",
        item_count=2,
        source_authority="local",
        sync_status="local-only",
        created_at="2026-05-08T04:00:00Z",
        updated_at="2026-05-08T04:00:00Z",
    )
    app = _build_test_app()
    _seed_library_sources(app)
    service = DelayedRestoreLibraryCollectionsService((record,))
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")
        screen.query_one("#library-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-confirm-delete-collection")
        screen.query_one("#library-confirm-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-delete-receipt")

        screen.query_one("#library-collections-delete-undo", Button).press()
        await asyncio.wait_for(service.restore_started.wait(), timeout=2.0)
        screen._library_collection_name_input = "New while restoring"
        create_event = SimpleNamespace(stop=lambda: None)
        await screen.create_library_collection(create_event)

        assert service.created == []
        service.restore_release.set()
        await _wait_for_text(screen, pilot, "Research")


@pytest.mark.asyncio
async def test_library_collection_delete_receipt_dismiss_keeps_record_deleted() -> None:
    record = LibraryCollectionRecord(
        collection_id="collection-1",
        name="Research",
        description="Policy sources",
        item_count=0,
        source_authority="local",
        sync_status="local-only",
        created_at="2026-05-08T04:00:00Z",
        updated_at="2026-05-08T04:00:00Z",
    )
    app = _build_test_app()
    _seed_library_sources(app)
    service = FakeLibraryCollectionsService((record,))
    app.library_collections_service = service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")
        screen.query_one("#library-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-confirm-delete-collection")
        screen.query_one("#library-confirm-delete-collection", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-delete-receipt")

        screen.query_one(
            "#library-collections-delete-receipt-dismiss", Button
        ).press()
        for _ in range(20):
            await pilot.pause()
            if not screen.query("#library-collections-delete-receipt"):
                break

        assert not screen.query("#library-collections-delete-receipt")
        assert service.restored == []
        assert service.records == []


@pytest.mark.asyncio
async def test_library_collection_form_input_keeps_focus_and_updates_actions() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_collections_service = FakeLibraryCollectionsService()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-row-browse-collections", Button).press()
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
async def test_library_collections_delete_failure_keeps_selection_and_warns_user() -> (
    None
):
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

        screen.query_one("#library-row-browse-collections", Button).press()
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

        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-error")

        error_text = screen.query_one("#library-collections-error", Static).renderable
        assert "Couldn't load Collections" in str(error_text)
        assert "collections database unavailable" not in _visible_text(screen)
