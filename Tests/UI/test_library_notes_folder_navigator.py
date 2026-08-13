"""Library-screen orchestration tests for the Database Notes folder tree."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Notes.note_folder_models import NoteFolder, NoteFolderPage
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _page(*, folders=(), notes=(), memberships=()) -> NoteFolderPage:
    return NoteFolderPage(
        folders=tuple(folders),
        memberships=tuple(memberships),
        notes=tuple(notes),
        total_folders=len(folders),
        total_notes=len(notes),
        next_offset=None,
        total_memberships=len(memberships),
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
        return _page(
            folders=(_folder("ideas", "personal", "/Personal/Ideas"),)
        )


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

    await LibraryScreen._load_library_notes_tree(
        fake, generation=1, refresh_root=True
    )

    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["expanded_folder_ids"] == ()
    assert 1 <= call["folder_limit"] <= 500
    assert 1 <= call["note_limit"] <= 1000
    assert 1 <= call["membership_limit"] <= 1000
    assert fake._library_notes_tree_root_page.total_folders == 1
    assert fake._library_notes_tree_loading is False


@pytest.mark.asyncio
async def test_expansion_reuses_root_and_issues_one_bulk_branch_call():
    service = _FolderService()
    fake = _screen_fake(service)
    await LibraryScreen._load_library_notes_tree(
        fake, generation=1, refresh_root=True
    )
    fake._library_notes_tree_expanded_ids.add("personal")
    fake._library_notes_tree_generation = 2

    await LibraryScreen._load_library_notes_tree(
        fake, generation=2, refresh_root=False
    )

    assert len(service.calls) == 2
    assert service.calls[-1]["expanded_folder_ids"] == ("personal",)
    assert fake._library_notes_tree_expanded_page.folders[0].folder_id == "ideas"


@pytest.mark.asyncio
async def test_stale_tree_result_does_not_replace_newer_state():
    service = _FolderService()
    fake = _screen_fake(service)
    fake._library_notes_tree_generation = 2

    await LibraryScreen._load_library_notes_tree(
        fake, generation=1, refresh_root=True
    )

    assert fake._library_notes_tree_root_page is None
    assert fake._library_notes_tree_loading is True
