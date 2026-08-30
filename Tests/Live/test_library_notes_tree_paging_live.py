"""Isolated production-shaped live walkthrough for Notes tree paging."""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.test_library_adaptive_reader_closeout import (
    LibraryProductionCSSHarness,
    _assert_inside_items,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_notes_tree_paging import NotesBranchKey
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.note_folder_models import FolderPlacementId
from textual.widgets import Button


SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))


@pytest.mark.asyncio
async def test_live_real_repository_large_tree_walkthrough(tmp_path) -> None:
    """Exercise real SQLite, repository, scope service, and Notes canvas."""
    db = CharactersRAGDB(tmp_path / "notes-live.db", client_id="task-18917-live")
    repository = LocalNoteFolderRepository(db)
    primary = repository.create_folder(
        name="00 Primary research with a deliberately identifying long title",
        parent_id=None,
    )
    for index in range(24):
        repository.create_folder(name=f"Root {index:02d} " + "details " * 6, parent_id=None)
    children = [
        repository.create_folder(
            name=f"Child {index:02d} " + "details " * 5,
            parent_id=primary.folder_id,
        )
        for index in range(25)
    ]
    deep = repository.create_folder(name="Deep ancestor", parent_id=children[0].folder_id)
    deepest = repository.create_folder(name="Deepest managed leaf", parent_id=deep.folder_id)

    for index in range(25):
        assert db.add_note(f"Unfiled {index:02d} " + "details " * 6, "live")

    memberships = []
    for index in range(43):
        note_id = db.add_note(f"Primary {index:02d} " + "details " * 7, "live")
        assert note_id is not None
        memberships.append(
            repository.attach_manual(folder_id=primary.folder_id, note_id=note_id)
        )
    duplicate_id = db.add_note("Duplicate placement " + "details " * 7, "live")
    assert duplicate_id is not None
    repository.attach_manual(folder_id=primary.folder_id, note_id=duplicate_id)
    repository.reconcile_managed(
        owner_id="duplicate-owner", desired=((primary.folder_id, duplicate_id),)
    )
    shadow_id = db.add_note("Shadowed managed ancestor", "live")
    assert shadow_id is not None
    repository.reconcile_managed(
        owner_id="shadow-owner",
        desired=((primary.folder_id, shadow_id), (deepest.folder_id, shadow_id)),
    )

    assert repository.page_child_folders(parent_id=None, limit=20, offset=0).total_folders == 25
    assert repository.page_child_folders(parent_id=primary.folder_id, limit=20, offset=0).total_folders == 25
    assert repository.page_note_placements(parent_id=None, limit=20, offset=0).total_placements == 25
    assert repository.page_note_placements(parent_id=primary.folder_id, limit=20, offset=0).total_placements == 45
    assert repository.page_note_placements(parent_id=deepest.folder_id, limit=20, offset=0).total_placements == 1

    interop = NotesInteropService(tmp_path, "task-18917-live", global_db_to_use=db)
    service = NotesScopeService(interop, None, folder_repository=repository)
    original_page = service.page_note_placements
    fail_once = False
    failure_entered = asyncio.Event()
    failure_release = asyncio.Event()

    async def page_with_one_shot_failure(**kwargs):
        nonlocal fail_once
        if kwargs.get("parent_id") == primary.folder_id and kwargs.get("offset") == 20 and fail_once:
            fail_once = False
            failure_entered.set()
            await failure_release.wait()
            raise RuntimeError("live one-shot page failure")
        return await original_page(**kwargs)

    service.page_note_placements = page_with_one_shot_failure
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=[])
    app.chachanotes_db = db
    app.notes_scope_service = service
    host = LibraryProductionCSSHarness(app)

    try:
        async with host.run_test(size=SIZES[0]) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-notes", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: len(screen.query(".library-notes-folder-row")) == 21,
                message="live root tree did not settle",
            )
            root_folder_pager = next(
                pager for pager in screen.query(".library-notes-tree-pager")
                if pager.content_kind == "folders" and pager.parent_folder_id is None
            )
            root_folder_pager.press()
            await _wait_for_condition(
                pilot,
                lambda: len(screen.query(".library-notes-folder-row")) == 26,
                message="live root continuation did not settle",
            )
            primary_row = next(
                row for row in screen.query(".library-notes-folder-row")
                if row.folder_id == primary.folder_id
            )
            primary_row.press()
            primary_key = NotesBranchKey(primary.folder_id, "placements")
            await _wait_for_condition(
                pilot,
                lambda: primary_key in screen._library_notes_tree_branches
                and not screen._library_notes_tree_branches[primary_key].loading
                and {
                    row.content_kind
                    for row in screen.query(".library-notes-tree-pager")
                    if row.parent_folder_id == primary.folder_id
                }
                >= {"folders", "placements"},
                message="live primary branch did not settle",
            )
            child_key = NotesBranchKey(primary.folder_id, "folders")
            child_pager = next(
                row for row in screen.query(".library-notes-tree-pager")
                if row.parent_folder_id == primary.folder_id
                and row.content_kind == "folders"
            )
            child_pager.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    len(screen._library_notes_tree_branches[child_key].items) == 25
                    and any(
                        row.parent_folder_id == primary.folder_id
                        and row.content_kind == "placements"
                        for row in screen.query(".library-notes-tree-pager")
                    )
                ),
                message="live child-folder continuation did not settle",
            )
            pager = next(
                row for row in screen.query(".library-notes-tree-pager")
                if row.parent_folder_id == primary.folder_id
                and row.content_kind == "placements"
            )
            fail_once = True
            pager.focus()
            await pilot.pause()
            pager.press()
            await _wait_for_condition(pilot, failure_entered.is_set, message="live failure did not enter")
            failure_release.set()
            await _wait_for_condition(
                pilot,
                lambda: screen.query_one(f"#{pager.id}").paging_action == "retry",
                message="live Retry did not mount",
            )
            screen.query_one(f"#{pager.id}", Button).press()
            await _wait_for_condition(
                pilot,
                lambda: len(screen._library_notes_tree_branches[primary_key].items) == 40,
                message="live Retry continuation did not settle",
            )

            screen._request_library_notes_tree_slice(
                primary_key, direction="replace", offset=0
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    len(screen._library_notes_tree_branches[primary_key].items) == 20
                    and screen._library_notes_tree_branches[primary_key].start_offset == 0
                ),
                message="live primary first page did not restore before locate",
            )

            target = memberships[30]
            assert await screen._locate_library_notes_tree_target(
                note_id=target.note_id,
                preferred_folder_id=primary.folder_id,
                preferred_membership_id=target.membership_id,
                focus=True,
            )
            assert screen._library_notes_tree_selected_placement_id == FolderPlacementId.note(
                primary.folder_id, target.note_id, target.membership_id
            )
            await _wait_for_condition(
                pilot,
                lambda: getattr(screen.focused, "membership_id", None)
                == target.membership_id,
                message="live located placement did not receive focus",
            )
            assert getattr(screen.focused, "membership_id", None) == target.membership_id
            assert screen._library_notes_tree_branches[primary_key].start_offset == 20
            earlier = next(
                row
                for row in screen.query(".library-notes-tree-pager")
                if row.parent_folder_id == primary.folder_id
                and row.paging_action == "earlier"
            )
            earlier_generation = screen._library_notes_tree_branches[
                primary_key
            ].generation
            earlier.press()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_notes_tree_branches[primary_key].generation
                    > earlier_generation
                    and not screen._library_notes_tree_branches[primary_key].loading
                    and screen._library_notes_tree_branches[primary_key].start_offset == 0
                ),
                message="live earlier page did not settle",
            )

            await service.rename_note_folder(
                scope="local_note",
                folder_id=children[1].folder_id,
                name="Child 01 updated by live mutation",
                expected_version=children[1].version,
                user_id="task-18917-live",
            )
            mutation_generation = screen._library_notes_tree_branches[
                child_key
            ].generation
            screen._request_library_notes_tree_slice(
                NotesBranchKey(primary.folder_id, "folders"), direction="replace", offset=0
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_notes_tree_branches[child_key].generation
                    > mutation_generation
                    and not screen._library_notes_tree_branches[child_key].loading
                    and any(
                        item.folder_id == children[1].folder_id
                        and item.name == "Child 01 updated by live mutation"
                        for item in screen._library_notes_tree_branches[child_key].items
                    )
                ),
                message="live mutation refresh did not settle",
            )
            primary_row = next(
                row for row in screen.query(".library-notes-folder-row")
                if row.folder_id == primary.folder_id
            )
            primary_row.press()
            await pilot.pause()
            primary_row = next(
                row for row in screen.query(".library-notes-folder-row")
                if row.folder_id == primary.folder_id
            )
            primary_row.press()
            await pilot.pause()
            assert screen._library_notes_tree_branches[primary_key].freshness == "fresh"

            observations = []
            for size in SIZES:
                await pilot.resize_terminal(*size)
                await pilot.pause()
                shell = screen.query_one("#library-notes-reader-shell")
                if not shell.effective_layout.items_open:
                    shell.items_grip.press()
                    await _wait_for_condition(
                        pilot,
                        lambda: shell.effective_layout.items_open,
                        message=f"live Items did not open at {size}",
                    )
                items = shell.items
                notes_list = screen.query_one("#library-notes-list")
                _assert_inside_items(items, notes_list)
                assert notes_list.region.right <= items.region.right
                painted = " ".join(
                    "\n".join(strip.text for strip in screen._compositor.render_strips()).split()
                )
                assert "Notes" in painted and "Library" in painted
                observations.append(
                    f"{size[0]}x{size[1]} items={items.region.width} work={shell.work.region.width}"
                )
            print("TASK-18917 LIVE:", "; ".join(observations))
    finally:
        db.close_connection()
