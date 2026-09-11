"""Library ▸ Notes list riders: date ordering back in Database Notes.

Group `r-list` of the notes riders wave: task 32172. Wave 1 (task-32128)
removed Sort from the folder tree because the tree's order was a hard-coded
repository contract. This group plumbs the order through paging AND the
deep-link locator, so the control can come back and mean something.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.UI.test_library_notes_folder_navigator import (
    _BranchService,
    _branch_screen_fake,
    _folder_page,
    _placement_page,
)
from Tests.UI.test_library_notes_wave_list import (
    WIDE,
    _CanvasApp,
    _folder_selected_projection,
    _kwargs_fake,
    _list_state,
    assert_every_action_fits,
)
from tldw_chatbook.Library.library_notes_tree_paging import NotesBranchKey
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    NoteTreeLocation,
    NoteTreePathStep,
)
from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
    LibraryNotesController,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


# -- the control is offered again ------------------------------------------


@pytest.mark.asyncio
async def test_the_folder_tree_offers_sort_again() -> None:
    """task-32172 AC#3: the order is a parameter now, so Sort is back."""
    app = _CanvasApp(
        pane_width=143,
        list_state=_list_state(),
        tree_projection=_folder_selected_projection(),
        sort_mode="oldest",
    )
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        sort = app.query_one("#library-notes-sort", Button)
        assert not sort.disabled
        assert "Oldest" in str(sort.label)
        assert_every_action_fits(app)


@pytest.mark.asyncio
async def test_sort_is_blocked_with_a_reason_while_the_filter_window_shows() -> None:
    """task-32172: the filter window is search-ranked, so Sort cannot own it.

    Rather than making the control vanish again (wave 1's shape), it stays
    put and says what blocks it and what to do about it.
    """
    app = _CanvasApp(
        pane_width=143,
        list_state=_list_state(),
        tree_projection=_folder_selected_projection(),
        filter_value="retro",
    )
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        sort = app.query_one("#library-notes-sort", Button)
        assert sort.disabled
        assert "○" in str(sort.label), "a disabled action carries the marker"
        reason = str(sort.tooltip)
        assert "filter" in reason.lower()
        assert "clear" in reason.lower(), f"no next step offered: {reason}"


def test_the_tree_no_longer_closes_an_open_sort_chooser() -> None:
    """task-32172: Sort survives the tree, so the chooser may stay open.

    Reconciles task-32128's ``test_the_tree_taking_over_closes_the_flat_
    sort_chooser``, whose whole reason was that the tree composed no Sort.
    """
    fake = _kwargs_fake(
        tree_projection=_folder_selected_projection(), sort_choices_visible=True
    )

    values = LibraryNotesController._library_notes_canvas_kwargs(fake)

    assert values["tree_projection"] is not None
    assert fake._library_notes_sort_choices_visible is True


def test_a_filter_window_still_closes_an_open_sort_chooser() -> None:
    """task-32172: the one blocked case must close the MODE, not just hide it.

    task-32128 review round 2's defect, kept alive for the case Sort still
    cannot own: an open chooser the canvas refuses to paint left the footer
    offering "choose sort" and spent the next Escape on nothing.
    """
    fake = _kwargs_fake(
        tree_projection=_folder_selected_projection(), sort_choices_visible=True
    )
    fake._library_notes_filter = "retro"

    LibraryNotesController._library_notes_canvas_kwargs(fake)

    assert fake._library_notes_sort_choices_visible is False


# -- the order reaches the repository seams --------------------------------


class _OrderService(_BranchService):
    """A branch service that also records the order each call asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.orders: list[tuple[str, object]] = []

    async def page_note_folder_children(self, **kwargs):
        self.orders.append(("folders", kwargs.get("order", "<absent>")))
        return await super().page_note_folder_children(**kwargs)

    async def page_note_placements(self, **kwargs):
        self.orders.append(("placements", kwargs.get("order", "<absent>")))
        return await super().page_note_placements(**kwargs)

    async def locate_note_tree_placement(self, **kwargs):
        self.orders.append(("locator", kwargs.get("order", "<absent>")))
        return NoteTreeLocation(
            # `_placement_page` mints "m-<note id>" memberships, and the
            # reconciler matches the located placement id against the loaded
            # slice's item ids.
            placement_id=FolderPlacementId.note("target", "n1", "m-n1"),
            note_id="n1",
            membership_id="m-n1",
            path=(NoteTreePathStep("target", None, 0),),
            placement_offset=20,
        )


@pytest.mark.asyncio
async def test_the_tree_pages_in_the_chosen_order() -> None:
    """task-32172 AC#1: the browse pager carries the live Sort value.

    Folder children have no date to order by, so that slice must NOT be
    handed an order it would have to reject.
    """
    service = _OrderService()
    fake = _branch_screen_fake(service)
    fake._notes_state.sort = "oldest"

    await LibraryScreen._load_library_notes_tree_slice(
        fake, NotesBranchKey(None, "folders"), direction="replace", offset=0
    )
    await LibraryScreen._load_library_notes_tree_slice(
        fake, NotesBranchKey("work", "placements"), direction="replace", offset=0
    )

    assert service.orders == [("folders", "<absent>"), ("placements", "oldest")]


class _LocatorOrderService(_OrderService):
    """Serve the located branch at the offsets the location addresses."""

    async def page_note_folder_children(self, **kwargs):
        self.orders.append(("folders", kwargs.get("order", "<absent>")))
        offset = kwargs["offset"]
        return _folder_page(
            kwargs["parent_id"],
            "target",
            start=offset,
            total=offset + 1,
            previous=max(0, offset - 20) if offset else None,
        )

    async def page_note_placements(self, **kwargs):
        self.orders.append(("placements", kwargs.get("order", "<absent>")))
        offset = kwargs["offset"]
        return _placement_page(
            kwargs["parent_id"],
            "n1",
            start=offset,
            total=offset + 1,
            previous=max(0, offset - 20) if offset else None,
        )


@pytest.mark.asyncio
async def test_the_deep_link_locator_uses_the_same_order() -> None:
    """task-32172 AC#2: the locator ranks against the order it will land in."""
    service = _LocatorOrderService()
    fake = _branch_screen_fake(service)
    fake._notes_state.sort = "newest"

    located = await LibraryScreen._locate_library_notes_tree_target(
        fake, note_id="n1", focus=False
    )

    assert located
    assert ("locator", "newest") in service.orders
    assert ("placements", "newest") in service.orders
    assert ("folders", "<absent>") in service.orders


# -- changing the value actually re-pages ----------------------------------


def _sort_choice_fake(sort: str = "newest"):
    reloads: list[bool] = []
    fake = SimpleNamespace(
        _library_notes_mutation_fenced=lambda: False,
        _library_notes_sort=sort,
        _library_notes_sort_choices_visible=True,
        _library_notes_select_mode=True,
        _library_notes_row_selection=SimpleNamespace(clear=lambda: None),
        _request_library_notes_tree_initial_load=lambda: reloads.append(True),
    )
    return fake, reloads


def test_choosing_a_new_sort_re_pages_the_tree(monkeypatch) -> None:
    """task-32172 AC#1: an in-place re-sort of the loaded window is a lie.

    Only a reload can move a note across a page boundary, so the handler
    must reload rather than repaint.
    """
    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_notes_controller"
        "._sync_library_canvas",
        lambda *_args, **_kwargs: None,
    )
    fake, reloads = _sort_choice_fake(sort="newest")
    option = Button("Oldest", id="library-notes-sort-oldest")
    option.choice_value = "oldest"

    LibraryNotesController.handle_library_notes_sort_choice(
        fake, Button.Pressed(option)
    )

    assert fake._library_notes_sort == "oldest"
    assert reloads == [True]


def test_choosing_the_sort_already_in_force_does_not_re_page(monkeypatch) -> None:
    """task-32172: re-picking the active value is a no-op, not a reload."""
    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_notes_controller"
        "._sync_library_canvas",
        lambda *_args, **_kwargs: None,
    )
    fake, reloads = _sort_choice_fake(sort="newest")
    option = Button("Newest", id="library-notes-sort-newest")
    option.choice_value = "newest"

    LibraryNotesController.handle_library_notes_sort_choice(
        fake, Button.Pressed(option)
    )

    assert fake._library_notes_sort == "newest"
    assert reloads == []


def test_a_fresh_visit_reloads_the_folders_left_expanded() -> None:
    """task-32172 AC#1: a folder's placements only exist while it is open.

    Resetting the tree used to reload the two root slices only, leaving
    every expanded folder rendered with no children until the user
    collapsed and re-opened it -- under a new sort that is the whole point
    of the reload.
    """
    service = _OrderService()
    fake = _branch_screen_fake(service)
    fake._notes_state.tree_expanded_ids = {"work"}
    scheduled: list[str] = []

    def _run_worker(coro, **kwargs):
        coro.close()
        scheduled.append(str(kwargs["group"]))

    fake.run_worker = _run_worker

    LibraryScreen._request_library_notes_tree_initial_load(fake)

    assert scheduled == [
        "library_notes_tree:notes-tree:root:folders",
        "library_notes_tree:notes-tree:root:placements",
        "library_notes_tree:notes-tree:folder:work:folders",
        "library_notes_tree:notes-tree:folder:work:placements",
    ]


