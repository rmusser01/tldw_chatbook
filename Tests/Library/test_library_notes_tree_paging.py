"""Pure reducer tests for independently paged Notes tree branches."""

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Library.library_notes_tree_paging import (
    NotesBranchKey,
    NotesBranchSliceState,
    apply_notes_slice_page,
    begin_notes_slice_load,
    empty_notes_slice,
    invalidate_notes_slice,
)
from tldw_chatbook.Notes.note_folder_models import (
    FolderPlacementId,
    NoteFolder,
    NoteFolderChildPage,
    NoteFolderMembership,
    NotePlacementPage,
    NotePlacementRecord,
)


def _folder(index: int, parent_id: str | None = None) -> NoteFolder:
    folder_id = f"f{index}"
    return NoteFolder(
        folder_id, parent_id, folder_id, f"/{folder_id}", f"/{folder_id}", 1, False
    )


def _placement(
    index: int, folder_id: str | None = "f1", membership_id: str | None = None
) -> NotePlacementRecord:
    note_id = f"n{index}"
    membership = (
        NoteFolderMembership(
            membership_id or f"m{index}", folder_id, note_id, "manual", "user", True, 1
        )
        if folder_id is not None
        else None
    )
    return NotePlacementRecord({"id": note_id, "title": note_id}, folder_id, membership)


def _page(
    start: int,
    items: tuple[NotePlacementRecord, ...],
    *,
    total: int = 6,
    previous: int | None = None,
    next_: int | None = None,
) -> NotePlacementPage:
    return NotePlacementPage(items, total, start, previous, next_)


def _loaded_state() -> NotesBranchSliceState:
    state = empty_notes_slice(
        NotesBranchKey(parent_id="f1", slice_kind="placements"), topology_epoch=7
    )
    result = apply_notes_slice_page(
        begin_notes_slice_load(state, generation=2),
        _page(0, (_placement(0), _placement(1)), next_=2),
        direction="replace",
        request_generation=2,
        topology_epoch=7,
    )
    assert result.kind == "applied"
    return result.state


def test_branch_keys_support_root_folder_and_stable_pager_ids() -> None:
    root = NotesBranchKey(parent_id=None, slice_kind="folders")
    folder = NotesBranchKey(parent_id="a:b", slice_kind="placements")

    assert root.pager_id == "notes-tree:root:folders"
    assert folder.pager_id == "notes-tree:folder:a%3Ab:placements"
    assert folder.pager_id == NotesBranchKey("a:b", "placements").pager_id


def test_replace_applies_one_contiguous_immutable_tuple() -> None:
    state = empty_notes_slice(NotesBranchKey(None, "placements"), topology_epoch=7)
    incoming = _page(2, (_placement(2), _placement(3)), previous=0, next_=4)

    result = apply_notes_slice_page(
        begin_notes_slice_load(state, generation=1),
        incoming,
        direction="replace",
        request_generation=1,
        topology_epoch=7,
    )

    assert result.kind == "applied"
    assert result.state.items == incoming.placements
    assert result.state.start_offset == 2
    assert result.state.total == 6
    with pytest.raises(FrozenInstanceError):
        result.state.total = 7  # type: ignore[misc]


def test_adjacent_more_appends_and_adjacent_previous_prepends() -> None:
    current = _loaded_state()
    appended = apply_notes_slice_page(
        current,
        _page(2, (_placement(2), _placement(3)), previous=0, next_=4),
        direction="more",
        request_generation=2,
        topology_epoch=7,
    )
    target_window = apply_notes_slice_page(
        current,
        _page(2, (_placement(2), _placement(3)), previous=0, next_=4),
        direction="target",
        request_generation=2,
        topology_epoch=7,
    ).state
    prepended = apply_notes_slice_page(
        target_window,
        _page(0, (_placement(8), _placement(9)), next_=2),
        direction="previous",
        request_generation=2,
        topology_epoch=7,
    )

    assert appended.kind == "applied"
    assert [item.note["id"] for item in appended.state.items] == [
        "n0",
        "n1",
        "n2",
        "n3",
    ]
    assert prepended.kind == "applied"
    assert [item.note["id"] for item in prepended.state.items] == [
        "n8",
        "n9",
        "n2",
        "n3",
    ]
    assert prepended.state.start_offset == 0


def test_distant_target_page_replaces_instead_of_appending() -> None:
    current = _loaded_state()
    target = _page(4, (_placement(4), _placement(5)), previous=2)

    result = apply_notes_slice_page(
        current,
        target,
        direction="target",
        request_generation=2,
        topology_epoch=7,
    )

    assert result.kind == "applied"
    assert result.state.items == target.placements
    assert result.state.start_offset == 4


def test_generation_or_topology_mismatch_is_ignored() -> None:
    current = _loaded_state()
    incoming = _page(2, (_placement(2), _placement(3)), previous=0, next_=4)

    old_request = apply_notes_slice_page(
        current, incoming, direction="more", request_generation=1, topology_epoch=7
    )
    old_topology = apply_notes_slice_page(
        current, incoming, direction="more", request_generation=2, topology_epoch=6
    )

    assert old_request.kind == old_topology.kind == "ignored"
    assert old_request.state is current
    assert old_topology.state is current


@pytest.mark.parametrize(
    "incoming",
    [
        _page(2, (_placement(2), _placement(3)), total=7, previous=0, next_=4),
        _page(3, (_placement(3), _placement(4)), previous=1, next_=5),
        _page(2, (_placement(1), _placement(3)), previous=0, next_=4),
        _page(2, (_placement(2),), previous=0, next_=4),
        _page(2, (_placement(2), _placement(3)), previous=1, next_=4),
        _page(2, (_placement(2), _placement(3)), previous=0, next_=5),
    ],
)
def test_continuation_drift_requests_one_first_page_recovery(
    incoming: NotePlacementPage,
) -> None:
    current = _loaded_state()

    result = apply_notes_slice_page(
        current, incoming, direction="more", request_generation=2, topology_epoch=7
    )

    assert result.kind == "drift"
    assert result.recovery == "reset_first"
    assert result.state.recovery_attempted is True
    assert result.state.freshness == "fresh"


def test_second_recovery_failure_becomes_stale_and_withdraws_total() -> None:
    current = _loaded_state()
    first = apply_notes_slice_page(
        current,
        _page(2, (_placement(2),), previous=0, next_=4),
        direction="more",
        request_generation=2,
        topology_epoch=7,
    )
    recovering = begin_notes_slice_load(first.state, generation=3, recovering=True)

    second = apply_notes_slice_page(
        recovering,
        _page(0, (_placement(0),), next_=2),
        direction="replace",
        request_generation=3,
        topology_epoch=7,
    )

    assert second.kind == "drift"
    assert second.recovery is None
    assert second.state.freshness == "stale"
    assert second.state.total is None
    assert second.state.previous_offset is None
    assert second.state.next_offset is None


def test_folder_pages_use_folder_placement_identity_for_overlap_checks() -> None:
    key = NotesBranchKey(None, "folders")
    first = NoteFolderChildPage((_folder(1),), 2, 0, None, 1)
    current = apply_notes_slice_page(
        begin_notes_slice_load(empty_notes_slice(key, topology_epoch=4), generation=1),
        first,
        direction="replace",
        request_generation=1,
        topology_epoch=4,
    ).state
    overlap = NoteFolderChildPage((_folder(1),), 2, 1, 0, None)

    result = apply_notes_slice_page(
        current, overlap, direction="more", request_generation=1, topology_epoch=4
    )

    assert result.kind == "drift"
    assert current.item_ids == (FolderPlacementId.folder("f1"),)


def test_reducer_does_not_mutate_prior_state_or_page() -> None:
    current = _loaded_state()
    incoming = _page(2, (_placement(2), _placement(3)), previous=0, next_=4)
    before_state = current
    before_page = incoming

    result = apply_notes_slice_page(
        current, incoming, direction="more", request_generation=2, topology_epoch=7
    )

    assert result.kind == "applied"
    assert current == before_state
    assert incoming == before_page
    assert result.state is not current


def test_invalidation_clears_items_and_rejects_old_responses() -> None:
    current = _loaded_state()
    invalidated = invalidate_notes_slice(current, topology_epoch=8)

    old = apply_notes_slice_page(
        invalidated,
        _page(2, (_placement(2), _placement(3)), previous=0, next_=4),
        direction="more",
        request_generation=2,
        topology_epoch=7,
    )

    assert invalidated.items == ()
    assert invalidated.total is None
    assert invalidated.topology_epoch == 8
    assert old.kind == "ignored"
