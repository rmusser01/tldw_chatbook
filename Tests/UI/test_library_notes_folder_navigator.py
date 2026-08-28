"""Library-screen orchestration tests for the Database Notes folder tree."""

from __future__ import annotations

from dataclasses import replace
from html import unescape
from types import SimpleNamespace

import pytest

from Tests.UI.test_destination_shells import StaticLibraryNotesScopeService
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
from tldw_chatbook.Notes.note_folder_models import (
    FolderCapabilityError,
    FolderCollisionError,
    FolderConflictError,
    FolderValidationError,
    NoteFolder,
    NoteFolderMembership,
    NoteFolderPage,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _page(
    *,
    folders=(),
    notes=(),
    memberships=(),
    next_folder_offset=None,
    next_note_offset=None,
    next_membership_offset=None,
    unfiled_note_ids=None,
) -> NoteFolderPage:
    return NoteFolderPage(
        folders=tuple(folders),
        memberships=tuple(memberships),
        notes=tuple(notes),
        total_folders=len(folders),
        total_notes=len(notes),
        next_offset=next_note_offset,
        next_folder_offset=next_folder_offset,
        total_memberships=len(memberships),
        next_membership_offset=next_membership_offset,
        unfiled_note_ids=unfiled_note_ids,
    )


def _folder(folder_id: str, parent_id: str | None, path: str) -> NoteFolder:
    return NoteFolder(
        folder_id=folder_id,
        parent_id=parent_id,
        name=path.rsplit("/", 1)[-1],
        path=path,
        normalized_path=path.casefold(),
        version=1,
        deleted=False,
    )


def _membership(
    membership_id: str, folder_id: str, note_id: str
) -> NoteFolderMembership:
    return NoteFolderMembership(
        membership_id=membership_id,
        folder_id=folder_id,
        note_id=note_id,
        ownership="manual",
        owner_id="",
        owner_active=True,
        version=1,
    )


class _FolderService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def load_note_folder_tree_batch(self, **kwargs):
        self.calls.append(kwargs)
        expanded = tuple(kwargs["expanded_folder_ids"])
        if not expanded:
            return _page(
                folders=(_folder("personal", None, "/Personal"),),
                notes=({"id": "loose", "title": "Loose"},),
            )
        return _page(folders=(_folder("ideas", "personal", "/Personal/Ideas"),))


def _screen_fake(service: _FolderService):
    return SimpleNamespace(
        app_instance=SimpleNamespace(
            notes_scope_service=service,
            notes_user_id="tester",
        ),
        _library_notes_tree_root_page=None,
        _library_notes_tree_expanded_page=None,
        _library_notes_tree_expanded_ids=set(),
        _library_notes_tree_generation=1,
        _library_notes_tree_loading=True,
        _library_notes_tree_error="",
        _library_notes_user_id=lambda: "tester",
        is_mounted=False,
    )


@pytest.mark.asyncio
async def test_initial_tree_load_uses_one_bounded_bulk_call_and_no_note_detail():
    service = _FolderService()
    fake = _screen_fake(service)

    await LibraryScreen._load_library_notes_tree(fake, generation=1, refresh_root=True)

    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["expanded_folder_ids"] == ()
    assert 1 <= call["folder_limit"] <= 500
    assert 1 <= call["note_limit"] <= 1000
    assert 1 <= call["membership_limit"] <= 1000
    assert fake._library_notes_tree_root_page.total_folders == 1
    assert fake._library_notes_tree_loading is False


@pytest.mark.asyncio
async def test_filter_loads_placements_for_matches_outside_expanded_branches(
    monkeypatch,
):
    parent = _folder("work", None, "/Work")
    child = _folder("project", "work", "/Work/Project")
    search_page = _page(
        folders=(parent, child),
        memberships=(_membership("m1", "project", "n1"),),
        notes=({"id": "n1", "title": "Hidden garden plan"},),
        unfiled_note_ids=(),
    )

    class _SearchService:
        def __init__(self) -> None:
            self.note_ids = ()

        async def search_notes(self, **kwargs):
            return ({"id": "n1", "title": "Hidden garden plan"},)

        async def load_note_folder_search(self, **kwargs):
            self.note_ids = kwargs["note_ids"]
            return search_page

    service = _SearchService()
    fake = _screen_fake(service)  # type: ignore[arg-type]
    fake._library_notes_filter = "garden"
    fake._library_notes_filter_records = None
    fake._library_notes_tree_search_page = None
    fake._source_record_id = lambda record: record["id"]
    fake._focus_library_notes_filter_input = lambda: None
    fake._run_library_service_call = lambda method, **kwargs: method(**kwargs)
    fake._library_notes_tree_root_page = _page(folders=(parent,))
    fake._library_notes_tree_expanded_page = _page()
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen._sync_library_canvas",
        lambda *args, **kwargs: None,
    )

    await LibraryScreen._run_library_notes_filter(fake, "garden")

    assert service.note_ids == ("n1",)
    projection = LibraryScreen._build_library_notes_tree_projection(fake)
    assert projection is not None
    assert [row.breadcrumb for row in projection.rows if row.kind == "note"] == [
        "Work / Project / Hidden garden plan"
    ]


@pytest.mark.asyncio
async def test_filter_reveals_collapsed_note_from_folder_path_match(monkeypatch):
    parent = _folder("work", None, "/Work")
    child = _folder("project", "work", "/Work/Project")
    search_page = _page(
        folders=(parent, child),
        memberships=(_membership("m1", "project", "n1"),),
        notes=({"id": "n1", "title": "Unrelated title"},),
        unfiled_note_ids=(),
    )

    class _PathSearchService:
        async def search_notes(self, **kwargs):
            return ()

        async def load_note_folder_search(self, **kwargs):
            assert kwargs["folder_query"] == "work / project"
            return search_page

    fake = _screen_fake(_PathSearchService())  # type: ignore[arg-type]
    fake._library_notes_filter = "work / project"
    fake._library_notes_filter_records = None
    fake._library_notes_tree_search_page = None
    fake._source_record_id = lambda record: record["id"]
    fake._focus_library_notes_filter_input = lambda: None
    fake._run_library_service_call = lambda method, **kwargs: method(**kwargs)
    fake._library_notes_tree_root_page = _page(folders=(parent,))
    fake._library_notes_tree_expanded_page = _page()
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen._sync_library_canvas",
        lambda *args, **kwargs: None,
    )

    await LibraryScreen._run_library_notes_filter(fake, "work / project")

    assert [record["id"] for record in fake._library_notes_filter_records] == ["n1"]
    projection = LibraryScreen._build_library_notes_tree_projection(fake)
    assert projection is not None
    assert [row.breadcrumb for row in projection.rows if row.kind == "note"] == [
        "Work / Project / Unrelated title"
    ]


@pytest.mark.asyncio
async def test_filter_without_folder_search_capability_keeps_loaded_tree(
    monkeypatch,
):
    class _LegacySearchService:
        async def search_notes(self, **kwargs):
            return ({"id": "n1", "title": "Garden plan"},)

    service = _LegacySearchService()
    fake = _screen_fake(service)  # type: ignore[arg-type]
    fake._library_notes_filter = "garden"
    fake._library_notes_filter_records = None
    fake._library_notes_tree_search_page = None
    fake._source_record_id = lambda record: record["id"]
    fake._focus_library_notes_filter_input = lambda: None
    fake._run_library_service_call = lambda method, **kwargs: method(**kwargs)
    fake._library_notes_tree_root_page = _page(
        notes=({"id": "n1", "title": "Garden plan"},),
        unfiled_note_ids=("n1",),
    )
    fake._library_notes_tree_expanded_page = _page()
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.library_screen._sync_library_canvas",
        lambda *args, **kwargs: None,
    )

    await LibraryScreen._run_library_notes_filter(fake, "garden")

    assert fake._library_notes_tree_search_page is None
    projection = LibraryScreen._build_library_notes_tree_projection(fake)
    assert projection is not None
    assert [row.note_id for row in projection.rows if row.kind == "note"] == ["n1"]


def test_submitting_new_filter_clears_previous_result_state():
    worker_calls = []

    async def _filter(query):
        return None

    def _run_worker(awaitable, **kwargs):
        worker_calls.append(kwargs)
        awaitable.close()

    fake = SimpleNamespace(
        _library_notes_filter="old",
        _library_notes_filter_records=[{"id": "old-note"}],
        _library_notes_tree_search_page=_page(notes=({"id": "old-note"},)),
        _library_notes_select_mode=True,
        _library_notes_row_selection=SimpleNamespace(clear=lambda: None),
        _run_library_notes_filter=_filter,
        _safe_text=lambda value, max_length: value[:max_length],
        run_worker=_run_worker,
    )
    event = SimpleNamespace(value="new", stop=lambda: None)

    LibraryScreen.handle_library_notes_filter(fake, event)

    assert fake._library_notes_filter_records is None
    assert fake._library_notes_tree_search_page is None
    assert worker_calls == [{"exclusive": True, "group": "library_notes_filter"}]


@pytest.mark.asyncio
async def test_expansion_reuses_root_and_issues_one_bulk_branch_call():
    service = _FolderService()
    fake = _screen_fake(service)
    await LibraryScreen._load_library_notes_tree(fake, generation=1, refresh_root=True)
    fake._library_notes_tree_expanded_ids.add("personal")
    fake._library_notes_tree_generation = 2

    await LibraryScreen._load_library_notes_tree(fake, generation=2, refresh_root=False)

    assert len(service.calls) == 2
    assert service.calls[-1]["expanded_folder_ids"] == ("personal",)
    assert fake._library_notes_tree_expanded_page.folders[0].folder_id == "ideas"


@pytest.mark.asyncio
async def test_stale_tree_result_does_not_replace_newer_state():
    service = _FolderService()
    fake = _screen_fake(service)
    fake._library_notes_tree_generation = 2

    await LibraryScreen._load_library_notes_tree(fake, generation=1, refresh_root=True)

    assert fake._library_notes_tree_root_page is None
    assert fake._library_notes_tree_loading is True


@pytest.mark.asyncio
async def test_missing_folder_capability_finishes_loading_and_repaints_status(
    monkeypatch,
):
    fake = _screen_fake(SimpleNamespace())  # type: ignore[arg-type]
    fake._status_repaints = 0
    monkeypatch.setattr(
        LibraryScreen,
        "_sync_library_notes_tree_canvas_if_present",
        lambda self: setattr(self, "_status_repaints", self._status_repaints + 1),
    )

    await LibraryScreen._load_library_notes_tree(fake, generation=1, refresh_root=True)

    assert fake._library_notes_tree_loading is False
    assert "unavailable" in fake._library_notes_tree_error.casefold()
    assert fake._status_repaints == 1


@pytest.mark.asyncio
async def test_load_more_failure_repaints_actionable_status(monkeypatch):
    class _FailingPagingService:
        async def load_note_folder_tree_batch(self, **kwargs):
            raise RuntimeError("offline")

    fake = _screen_fake(_FailingPagingService())  # type: ignore[arg-type]
    fake._library_notes_tree_root_page = _page(next_note_offset=1)
    fake._library_notes_tree_expanded_page = _page()
    fake._status_repaints = 0
    monkeypatch.setattr(
        LibraryScreen,
        "_sync_library_notes_tree_canvas_if_present",
        lambda self: setattr(self, "_status_repaints", self._status_repaints + 1),
    )

    await LibraryScreen._load_more_library_notes_tree(fake, generation=1)

    assert fake._library_notes_tree_loading is False
    assert "try again" in fake._library_notes_tree_error.casefold()
    assert fake._status_repaints == 1


class _PagingFolderService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def load_note_folder_tree_batch(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("membership_offset") == 1:
            return _page(
                notes=({"id": "n1", "title": "One"},),
                memberships=(_membership("m1b", "ideas", "n1"),),
                next_note_offset=1,
            )
        return _page(
            notes=({"id": "n2", "title": "Two"},),
            memberships=(_membership("m2", "ideas", "n2"),),
        )


@pytest.mark.asyncio
async def test_membership_cursor_finishes_current_note_page_before_advancing_notes():
    service = _PagingFolderService()
    fake = _screen_fake(service)  # type: ignore[arg-type]
    fake._library_notes_tree_root_page = _page()
    fake._library_notes_tree_expanded_ids = {"ideas"}
    fake._library_notes_tree_expanded_page = _page(
        notes=({"id": "n1", "title": "One"},),
        memberships=(_membership("m1a", "ideas", "n1"),),
        next_note_offset=1,
        next_membership_offset=1,
    )
    fake._library_notes_tree_membership_note_offset = 0

    fake._library_notes_tree_generation = 2
    await LibraryScreen._load_more_library_notes_tree(fake, generation=2)
    assert service.calls[-1]["note_offset"] == 0
    assert service.calls[-1]["load_notes"] is True
    assert service.calls[-1]["membership_offset"] == 1
    assert {
        item.membership_id
        for item in fake._library_notes_tree_expanded_page.memberships
    } == {
        "m1a",
        "m1b",
    }

    fake._library_notes_tree_generation = 3
    fake._library_notes_tree_loading = True
    await LibraryScreen._load_more_library_notes_tree(fake, generation=3)
    assert service.calls[-1]["note_offset"] == 1
    assert service.calls[-1]["membership_offset"] == 0
    assert fake._library_notes_tree_membership_note_offset == 1


@pytest.mark.asyncio
async def test_folder_only_continuation_skips_exhausted_note_queries():
    service = _PagingFolderService()
    fake = _screen_fake(service)  # type: ignore[arg-type]
    fake._library_notes_tree_root_page = _page(next_folder_offset=1)
    fake._library_notes_tree_expanded_ids = {"ideas"}
    fake._library_notes_tree_expanded_page = _page(next_folder_offset=1)
    fake._library_notes_tree_generation = 2

    await LibraryScreen._load_more_library_notes_tree(fake, generation=2)

    assert len(service.calls) == 2
    assert [call["load_notes"] for call in service.calls] == [False, False]


@pytest.mark.asyncio
async def test_paging_does_not_reopen_an_already_exhausted_independent_cursor():
    class _ReplayService:
        async def load_note_folder_tree_batch(self, **kwargs):
            return _page(
                folders=(_folder("first", None, "/First"),),
                notes=({"id": "n2", "title": "Two"},),
                next_folder_offset=1,
            )

    fake = _screen_fake(_ReplayService())  # type: ignore[arg-type]
    fake._library_notes_tree_root_page = _page(
        folders=(_folder("first", None, "/First"),),
        notes=({"id": "n1", "title": "One"},),
        next_note_offset=1,
    )
    fake._library_notes_tree_expanded_page = _page()
    fake._library_notes_tree_generation = 2

    await LibraryScreen._load_more_library_notes_tree(fake, generation=2)

    assert fake._library_notes_tree_root_page.next_offset is None
    assert fake._library_notes_tree_root_page.next_folder_offset is None


class _MutationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def create_note_folder(self, **kwargs):
        self.calls.append(("create", kwargs))
        return _folder("new", kwargs["parent_id"], "/New")

    async def attach_note_to_folder(self, **kwargs):
        self.calls.append(("attach", kwargs))
        return SimpleNamespace(membership_id="new-membership")

    async def detach_note_from_folder(self, **kwargs):
        self.calls.append(("detach", kwargs))
        return True

    async def rename_note_folder(self, **kwargs):
        self.calls.append(("rename", kwargs))
        return SimpleNamespace(
            folder=replace(_folder("ideas", None, "/Renamed"), version=2)
        )

    async def move_note_folder(self, **kwargs):
        self.calls.append(("move_folder", kwargs))
        return SimpleNamespace(
            folder=replace(_folder("ideas", "work", "/Work/Ideas"), version=2)
        )

    async def delete_note_folder(self, **kwargs):
        self.calls.append(("delete_folder", kwargs))
        return SimpleNamespace(
            folder=replace(_folder("ideas", None, "/Ideas"), version=2, deleted=True)
        )

    async def restore_note_folder(self, **kwargs):
        self.calls.append(("restore_folder", kwargs))
        return SimpleNamespace(
            folder=replace(_folder("ideas", None, "/Ideas"), version=3)
        )


class _FailingMutationService(_MutationService):
    def __init__(self, failure: Exception) -> None:
        super().__init__()
        self.failure = failure

    async def create_note_folder(self, **kwargs):
        raise self.failure


class _PartialMoveService(_MutationService):
    async def detach_note_from_folder(self, **kwargs):
        self.calls.append(("detach", kwargs))
        raise FolderConflictError("Membership changed during mutation.")


def _mutation_fake(service: _MutationService):
    fake = _screen_fake(service)  # type: ignore[arg-type]
    fake._library_note_import_execution_active = lambda: False
    fake._library_notes_mutation_in_flight = False
    fake._library_notes_notice = ""
    fake._library_notes_deleted_folder_receipt = None
    fake._library_notes_tree_selected_placement_id = ""
    fake._refreshes = []
    fake._request_library_notes_tree_refresh = lambda **kwargs: fake._refreshes.append(
        kwargs
    )
    return fake


@pytest.mark.asyncio
async def test_create_folder_mutation_uses_normalized_service_and_refreshes_tree():
    service = _MutationService()
    fake = _mutation_fake(service)

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "create_folder",
        name="New",
        parent_id="personal",
    )

    assert ok
    assert service.calls == [
        (
            "create",
            {
                "scope": "local_note",
                "name": "New",
                "parent_id": "personal",
                "user_id": "tester",
            },
        )
    ]
    assert fake._refreshes == [{"refresh_root": True}]


@pytest.mark.asyncio
async def test_move_manual_placement_attaches_before_detaching_original():
    service = _MutationService()
    fake = _mutation_fake(service)

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "move_placement",
        note_id="n1",
        destination_folder_id="reading",
        source_folder_id="ideas",
        membership_version=3,
        protected=False,
    )

    assert ok
    assert [name for name, _ in service.calls] == ["attach", "detach"]
    assert service.calls[1][1]["expected_version"] == 3


@pytest.mark.asyncio
async def test_managed_placement_mutation_is_rejected_before_service_call():
    service = _MutationService()
    fake = _mutation_fake(service)

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "move_placement",
        note_id="n1",
        destination_folder_id="reading",
        source_folder_id="ideas",
        membership_version=3,
        protected=True,
    )

    assert not ok
    assert service.calls == []
    assert "sync" in fake._library_notes_notice.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "payload", "expected_call"),
    (
        (
            "rename_folder",
            {"folder_id": "ideas", "name": "Renamed", "expected_version": 1},
            "rename",
        ),
        (
            "move_folder",
            {"folder_id": "ideas", "parent_id": "work", "expected_version": 1},
            "move_folder",
        ),
        (
            "add_placement",
            {"folder_id": "ideas", "note_id": "n1"},
            "attach",
        ),
        (
            "detach_placement",
            {"folder_id": "ideas", "note_id": "n1", "expected_version": 1},
            "detach",
        ),
    ),
)
async def test_folder_and_membership_operations_route_to_normalized_service(
    operation, payload, expected_call
):
    service = _MutationService()
    fake = _mutation_fake(service)
    assert await LibraryScreen._execute_library_notes_tree_mutation(
        fake, operation, **payload
    )
    assert service.calls[0][0] == expected_call


@pytest.mark.asyncio
async def test_folder_remove_creates_exact_restore_receipt_and_restore_consumes_it():
    service = _MutationService()
    fake = _mutation_fake(service)
    assert await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "delete_folder",
        folder_id="ideas",
        expected_version=1,
    )
    receipt = fake._library_notes_deleted_folder_receipt
    assert (receipt.folder_id, receipt.expected_version) == ("ideas", 2)

    assert await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "restore_folder",
        folder_id=receipt.folder_id,
        expected_version=receipt.expected_version,
    )
    assert fake._library_notes_deleted_folder_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_copy"),
    (
        (FolderCollisionError("collision"), "already exists"),
        (FolderConflictError("conflict"), "changed elsewhere"),
        (FolderValidationError("invalid"), "not valid"),
        (
            FolderCapabilityError(
                reason_code="unsupported", user_message="Folders unavailable here."
            ),
            "Folders unavailable here.",
        ),
    ),
)
async def test_typed_folder_failures_produce_actionable_status(failure, expected_copy):
    fake = _mutation_fake(_FailingMutationService(failure))

    assert not await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "create_folder",
        name="New",
        parent_id=None,
    )

    assert expected_copy.casefold() in fake._library_notes_notice.casefold()


@pytest.mark.asyncio
async def test_move_detach_conflict_keeps_both_placements_and_refreshes():
    service = _PartialMoveService()
    fake = _mutation_fake(service)

    ok = await LibraryScreen._execute_library_notes_tree_mutation(
        fake,
        "move_placement",
        note_id="n1",
        destination_folder_id="reading",
        source_folder_id="ideas",
        membership_version=3,
    )

    assert ok
    assert [name for name, _ in service.calls] == ["attach", "detach"]
    assert "both folders" in fake._library_notes_notice.casefold()
    assert fake._refreshes == [{"refresh_root": True}]


class _TreeCapableNotesService(StaticLibraryNotesScopeService):
    def __init__(self, notes):
        super().__init__(notes)
        self.tree_calls: list[dict[str, object]] = []

    async def load_note_folder_tree_batch(self, **kwargs):
        self.tree_calls.append(kwargs)
        expanded = tuple(kwargs["expanded_folder_ids"])
        ideas = _folder("ideas", None, "/Ideas")
        reading = _folder("reading", None, "/Reading")
        if not expanded:
            return _page(folders=(ideas, reading))
        memberships = tuple(
            NoteFolderMembership(
                membership_id=f"m-{folder_id}",
                folder_id=folder_id,
                note_id="n-1",
                ownership="managed",
                owner_id=f"sync-{folder_id}",
                owner_active=folder_id != "reading",
                version=1,
            )
            for folder_id in expanded
        )
        return _page(
            memberships=memberships,
            notes=({"id": "n-1", "title": "Q3 retro"},),
        )


async def _wait_until(pilot, predicate, *, attempts: int = 150):
    for _ in range(attempts):
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError("Library Notes folder state did not settle")


@pytest.mark.asyncio
async def test_live_host_renders_duplicate_placements_and_preserves_focus_at_60x20():
    app = _build_test_app()
    notes = _two_notes()
    _seed_conversations(app, _two_conversations(), notes=notes)
    service = _TreeCapableNotesService(notes)
    app.notes_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot, lambda: len(screen.query(".library-notes-folder-row")) == 2
        )
        for folder_id in ("ideas", "reading"):
            folder = next(
                row
                for row in screen.query(".library-notes-folder-row")
                if getattr(row, "folder_id", "") == folder_id
            )
            folder.press()
            await _wait_until(
                pilot,
                lambda folder_id=folder_id: any(
                    getattr(row, "folder_id", "") == folder_id
                    for row in screen.query(".library-notes-row")
                ),
            )

        placements = [
            row
            for row in screen.query(".library-notes-row")
            if getattr(row, "note_id", "") == "n-1"
        ]
        assert len(placements) == 2
        assert len({row.placement_id for row in placements}) == 2
        assert service.detail_calls == []

        placements[-1].focus()
        focused_placement = placements[-1].placement_id
        await _wait_until(
            pilot,
            lambda: (
                str(getattr(screen.focused, "placement_id", "")) == focused_placement
            ),
        )
        screen.refresh(recompose=True)
        await _wait_until(
            pilot,
            lambda: (
                str(getattr(screen.focused, "placement_id", "")) == focused_placement
            ),
        )

        await pilot.resize_terminal(60, 20)
        await _wait_until(pilot, lambda: screen._library_notes_compact is True)
        focused = next(
            row
            for row in screen.query(".library-notes-row")
            if row.placement_id == focused_placement
        )
        focused.scroll_visible()
        await pilot.pause()
        assert focused.region.width <= screen.query_one("#library-canvas").region.width
        screenshot = host.export_screenshot(simplify=True)
        painted = unescape(screenshot).replace("\xa0", " ")
        assert "Q3 retro" in painted
        assert any(label in painted for label in ("Ideas", "Reading"))
        assert any(
            status in painted for status in ("Synced placement", "Needs owner review")
        )
