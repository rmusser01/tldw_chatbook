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
    duplicate_manual = repository.attach_manual(
        folder_id=primary.folder_id, note_id=duplicate_id
    )
    (duplicate_managed,) = repository.reconcile_managed(
        owner_id="duplicate-owner", desired=((primary.folder_id, duplicate_id),)
    )
    shadow_id = db.add_note("Shadowed managed ancestor", "live")
    assert shadow_id is not None
    shadow_memberships = repository.reconcile_managed(
        owner_id="shadow-owner",
        desired=((primary.folder_id, shadow_id), (deepest.folder_id, shadow_id)),
    )
    shadow_deepest = next(
        membership
        for membership in shadow_memberships
        if membership.folder_id == deepest.folder_id
    )

    assert repository.page_child_folders(parent_id=None, limit=20, offset=0).total_folders == 25
    assert repository.page_child_folders(parent_id=primary.folder_id, limit=20, offset=0).total_folders == 25
    assert repository.page_note_placements(parent_id=None, limit=20, offset=0).total_placements == 25
    primary_pages = tuple(
        repository.page_note_placements(
            parent_id=primary.folder_id, limit=20, offset=offset
        )
        for offset in (0, 20, 40)
    )
    assert {page.total_placements for page in primary_pages} == {45}
    primary_placements = tuple(
        placement for page in primary_pages for placement in page.placements
    )
    duplicate_placements = tuple(
        placement
        for placement in primary_placements
        if str(placement.note["id"]) == duplicate_id
    )
    assert {
        placement.membership.membership_id
        for placement in duplicate_placements
        if placement.membership is not None
    } == {duplicate_manual.membership_id, duplicate_managed.membership_id}
    assert all(str(placement.note["id"]) != shadow_id for placement in primary_placements)
    deepest_page = repository.page_note_placements(
        parent_id=deepest.folder_id, limit=20, offset=0
    )
    assert deepest_page.total_placements == 1
    assert len(deepest_page.placements) == 1
    assert deepest_page.placements[0].membership == shadow_deepest

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
            await pilot.pause()
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
            await pilot.pause()
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
            await pilot.pause()
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
            await pilot.pause()
            pager = next(
                row for row in screen.query(".library-notes-tree-pager")
                if row.parent_folder_id == primary.folder_id
                and row.content_kind == "placements"
            )
            fail_once = True
            for _attempt in range(3):
                pager = next(
                    row
                    for row in screen.query(".library-notes-tree-pager")
                    if row.parent_folder_id == primary.folder_id
                    and row.content_kind == "placements"
                )
                screen.set_focus(pager, scroll_visible=True)
                await pilot.pause()
                if screen.focused is pager:
                    break
            assert screen.focused is pager
            pager.press()
            await _wait_for_condition(pilot, failure_entered.is_set, message="live failure did not enter")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen.focused is not None
                    and screen.focused.id == pager.id
                    and screen.focused.disabled
                ),
                message="live loading pager did not retain focus",
            )
            failure_release.set()
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen.query_one(f"#{pager.id}").paging_action == "retry"
                    and screen.focused is screen.query_one(f"#{pager.id}")
                ),
                message=lambda: (
                    "live Retry did not mount: "
                    f"action={screen.query_one(f'#{pager.id}').paging_action!r}; "
                    f"focused={screen.focused!r}; "
                    f"focus_parent={getattr(screen.focused, 'parent', None)!r}"
                ),
            )
            retry = screen.query_one(f"#{pager.id}", Button)
            assert str(retry.label).endswith("Retry")
            retry_observation = (
                str(retry.label), str(retry.id), str(screen.focused.id)
            )
            retry.press()
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
            assert earlier.range_copy == "Notes 21–40 of 45"
            assert earlier.action_copy == "Load earlier"
            notes_list = screen.query_one("#library-notes-list")
            notes_list.scroll_to_widget(
                earlier, animate=False, force=True, immediate=True
            )
            screen.set_focus(earlier, scroll_visible=True)
            await pilot.pause()
            earlier_strips = screen._compositor.render_strips()
            assert "Notes 21–40 of 45  Load" in earlier_strips[
                earlier.region.y
            ].text
            earlier_observation = f"{earlier.range_copy} {earlier.action_copy}"
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

            assert await screen._locate_library_notes_tree_target(
                note_id=duplicate_id,
                preferred_folder_id=primary.folder_id,
                preferred_membership_id=duplicate_manual.membership_id,
                focus=True,
            )
            duplicate_placement_id = FolderPlacementId.note(
                primary.folder_id,
                duplicate_id,
                duplicate_manual.membership_id,
            )
            assert screen._library_notes_tree_selected_placement_id == duplicate_placement_id
            await _wait_for_condition(
                pilot,
                lambda: getattr(screen.focused, "membership_id", None)
                == duplicate_manual.membership_id,
                message="live exact duplicate placement did not receive focus",
            )
            duplicate_row = screen.focused
            notes_list.scroll_to_widget(
                duplicate_row, animate=False, force=True, immediate=True
            )
            await pilot.pause()
            duplicate_painted = " ".join(
                "\n".join(
                    strip.text for strip in screen._compositor.render_strips()
                ).split()
            )
            assert "Duplicate placement" in duplicate_painted

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
            await _wait_for_condition(
                pilot,
                lambda: any(
                    row.folder_id == primary.folder_id
                    for row in screen.query(".library-notes-folder-row")
                ),
                message="live Primary row did not remount after mutation refresh",
            )
            await pilot.pause()
            primary_row = next(
                row for row in screen.query(".library-notes-folder-row")
                if row.folder_id == primary.folder_id
            )
            primary_row.press()
            await _wait_for_condition(
                pilot,
                lambda: primary.folder_id
                not in screen._library_notes_tree_expanded_ids,
                message="live Primary collapse did not settle",
            )
            await _wait_for_condition(
                pilot,
                lambda: any(
                    row.folder_id == primary.folder_id
                    for row in screen.query(".library-notes-folder-row")
                ),
                message="live Primary row did not remount after collapse",
            )
            await pilot.pause()
            primary_row = next(
                row for row in screen.query(".library-notes-folder-row")
                if row.folder_id == primary.folder_id
            )
            primary_row.press()
            await _wait_for_condition(
                pilot,
                lambda: primary.folder_id in screen._library_notes_tree_expanded_ids,
                message="live Primary re-expand did not settle",
            )
            await _wait_for_condition(
                pilot,
                lambda: any(
                    row.folder_id == primary.folder_id
                    for row in screen.query(".library-notes-folder-row")
                ),
                message="live Primary row did not remount after re-expand",
            )
            await pilot.pause()
            primary_row = next(
                row for row in screen.query(".library-notes-folder-row")
                if row.folder_id == primary.folder_id
            )
            primary_row.press()
            await _wait_for_condition(
                pilot,
                lambda: primary.folder_id
                not in screen._library_notes_tree_expanded_ids,
                message="live final Primary collapse did not settle",
            )

            assert await screen._locate_library_notes_tree_target(
                note_id=shadow_id,
                preferred_folder_id=deepest.folder_id,
                preferred_membership_id=shadow_deepest.membership_id,
                focus=True,
            )
            deepest_placement_id = FolderPlacementId.note(
                deepest.folder_id, shadow_id, shadow_deepest.membership_id
            )
            assert screen._library_notes_tree_selected_placement_id == deepest_placement_id
            await _wait_for_condition(
                pilot,
                lambda: getattr(screen.focused, "membership_id", None)
                == shadow_deepest.membership_id,
                message="live deepest located placement did not receive focus",
            )
            assert {
                primary.folder_id,
                children[0].folder_id,
                deep.folder_id,
                deepest.folder_id,
            } <= screen._library_notes_tree_expanded_ids
            deepest_row = screen.focused
            notes_list.scroll_to_widget(
                deepest_row, animate=False, force=True, immediate=True
            )
            await pilot.pause()
            deepest_painted = " ".join(
                "\n".join(
                    strip.text for strip in screen._compositor.render_strips()
                ).split()
            )
            assert "Shadowed managed ancestor" in deepest_painted

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
                deepest_row = next(
                    row
                    for row in screen.query(".library-notes-tree-note-row")
                    if getattr(row, "membership_id", None)
                    == shadow_deepest.membership_id
                )
                notes_list.scroll_to_widget(
                    deepest_row, animate=False, force=True, immediate=True
                )
                await pilot.pause()
                title_painted = " ".join(
                    "\n".join(strip.text for strip in screen._compositor.render_strips()).split()
                )
                assert "Shadowed managed ancestor" in str(deepest_row.label)
                assert "Shadowed managed" in title_painted
                root_notes_pager = next(
                    row
                    for row in screen.query(".library-notes-tree-pager")
                    if row.parent_folder_id is None
                    and row.content_kind == "placements"
                    and row.paging_action == "more"
                )
                assert root_notes_pager.range_copy == "Notes 1–20 of 25"
                assert root_notes_pager.action_copy == "Load more notes"
                notes_list.scroll_to_widget(
                    root_notes_pager, animate=False, force=True, immediate=True
                )
                screen.set_focus(root_notes_pager, scroll_visible=True)
                await pilot.pause()
                pager_painted = " ".join(
                    "\n".join(
                        strip.text for strip in screen._compositor.render_strips()
                    ).split()
                )
                assert "Notes 1–20 of 25" in pager_painted
                screen.set_focus(deepest_row, scroll_visible=True)
                await pilot.pause()
                assert getattr(screen.focused, "membership_id", None) == shadow_deepest.membership_id
                for row in screen.query(
                    ".library-notes-folder-row, .library-notes-tree-note-row, "
                    ".library-notes-tree-pager"
                ):
                    assert row.region.right <= items.region.right
                observations.append(
                    f"{size[0]}x{size[1]} title-control=Shadowed managed ancestor "
                    f"title-painted=Shadowed managed "
                    f"pager-control=Notes 1–20 of 25 Load more notes "
                    f"pager-painted=Notes 1–20 of 25 "
                    f"items={items.region.width} work={shell.work.region.width}"
                )
            print(
                "TASK-18917 LIVE:",
                f"duplicate_memberships={duplicate_manual.membership_id},"
                f"{duplicate_managed.membership_id}; "
                f"duplicate_focus={duplicate_manual.membership_id}; "
                f"deepest_focus={shadow_deepest.membership_id}; "
                f"retry={retry_observation[0]} control={retry_observation[1]} "
                f"focus={retry_observation[2]}; "
                f"earlier={earlier_observation}; "
                + "; ".join(observations),
            )
    finally:
        db.close_connection()
