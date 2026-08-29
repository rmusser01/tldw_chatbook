"""Pure reducer tests for independently paged Notes tree branches."""

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Library.library_notes_tree_paging import (
    NotesBranchKey,
    NotesBranchSliceState,
    apply_notes_slice_page,
    begin_notes_slice_load,
    empty_notes_slice,
    fail_notes_slice_load,
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
        begin_notes_slice_load(
            state,
            generation=2,
            direction="replace",
            requested_offset=0,
            requested_limit=2,
        ),
        _page(0, (_placement(0), _placement(1)), next_=2),
        direction="replace",
        request_generation=2,
        topology_epoch=7,
    )
    assert result.kind == "applied"
    return result.state


def _request(
    state: NotesBranchSliceState,
    *,
    generation: int,
    direction: str,
    offset: int,
    limit: int = 2,
    recovering: bool = False,
) -> NotesBranchSliceState:
    return begin_notes_slice_load(
        state,
        generation=generation,
        direction=direction,  # type: ignore[arg-type]
        requested_offset=offset,
        requested_limit=limit,
        recovering=recovering,
    )


def test_branch_keys_support_root_folder_and_stable_pager_ids() -> None:
    root = NotesBranchKey(parent_id=None, slice_kind="folders")
    folder = NotesBranchKey(parent_id="a:b", slice_kind="placements")

    assert root.pager_id == "notes-tree:root:folders"
    assert folder.pager_id == "notes-tree:folder:a%3Ab:placements"
    assert folder.pager_id == NotesBranchKey("a:b", "placements").pager_id


def test_replace_applies_one_contiguous_immutable_tuple() -> None:
    state = empty_notes_slice(NotesBranchKey("f1", "placements"), topology_epoch=7)
    incoming = _page(2, (_placement(2), _placement(3)), previous=0, next_=4)

    result = apply_notes_slice_page(
        _request(state, generation=1, direction="replace", offset=2),
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


@pytest.mark.parametrize(
    ("key", "page"),
    [
        (
            NotesBranchKey(None, "folders"),
            NoteFolderChildPage((_folder(1, "other"),), 1, 0, None, None),
        ),
        (
            NotesBranchKey("expected", "folders"),
            NoteFolderChildPage((_folder(1, None),), 1, 0, None, None),
        ),
        (
            NotesBranchKey(None, "placements"),
            NotePlacementPage((_placement(1, "other"),), 1, 0, None, None),
        ),
        (
            NotesBranchKey("expected", "placements"),
            NotePlacementPage((_placement(1, "other"),), 1, 0, None, None),
        ),
    ],
)
def test_branch_page_parent_mismatch_is_drift(
    key: NotesBranchKey,
    page: NoteFolderChildPage | NotePlacementPage,
) -> None:
    state = empty_notes_slice(key, topology_epoch=1)

    result = apply_notes_slice_page(
        _request(state, generation=1, direction="replace", offset=0, limit=1),
        page,
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    )

    assert result.kind == "drift"
    assert result.recovery == "reset_first"


@pytest.mark.parametrize(
    ("key", "page"),
    [
        (
            NotesBranchKey(None, "folders"),
            NoteFolderChildPage((_folder(1, None),), 1, 0, None, None),
        ),
        (
            NotesBranchKey("expected", "folders"),
            NoteFolderChildPage((_folder(1, "expected"),), 1, 0, None, None),
        ),
        (
            NotesBranchKey(None, "placements"),
            NotePlacementPage((_placement(1, None),), 1, 0, None, None),
        ),
        (
            NotesBranchKey("f1", "placements"),
            NotePlacementPage((_placement(1, "f1"),), 1, 0, None, None),
        ),
    ],
)
def test_branch_page_exact_parent_match_applies(
    key: NotesBranchKey,
    page: NoteFolderChildPage | NotePlacementPage,
) -> None:
    state = empty_notes_slice(key, topology_epoch=1)

    result = apply_notes_slice_page(
        _request(state, generation=1, direction="replace", offset=0, limit=1),
        page,
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    )

    assert result.kind == "applied"


def test_note_mapping_mutation_cannot_desynchronize_reducer_item_identity() -> None:
    source = {"id": "n1", "title": "One"}
    placement = NotePlacementRecord(source, None, None)
    page = NotePlacementPage((placement,), 1, 0, None, None)
    state = empty_notes_slice(NotesBranchKey(None, "placements"), topology_epoch=1)

    result = apply_notes_slice_page(
        _request(state, generation=1, direction="replace", offset=0, limit=1),
        page,
        direction="replace",
        request_generation=1,
        topology_epoch=1,
    )
    source["id"] = "changed"

    assert result.kind == "applied"
    assert result.state.item_ids == (FolderPlacementId.unfiled("n1"),)
    assert result.state.items[0].note["id"] == "n1"
    with pytest.raises(TypeError):
        result.state.items[0].note["id"] = "changed-again"  # type: ignore[index, union-attr]


def test_adjacent_more_appends_and_adjacent_previous_prepends() -> None:
    current = _loaded_state()
    appended = apply_notes_slice_page(
        _request(current, generation=3, direction="more", offset=2),
        _page(2, (_placement(2), _placement(3)), previous=0, next_=4),
        direction="more",
        request_generation=3,
        topology_epoch=7,
    )
    target_window = apply_notes_slice_page(
        _request(current, generation=3, direction="target", offset=2),
        _page(2, (_placement(2), _placement(3)), previous=0, next_=4),
        direction="target",
        request_generation=3,
        topology_epoch=7,
    ).state
    prepended = apply_notes_slice_page(
        _request(target_window, generation=4, direction="previous", offset=0),
        _page(0, (_placement(8), _placement(9)), next_=2),
        direction="previous",
        request_generation=4,
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
        _request(current, generation=3, direction="target", offset=4),
        target,
        direction="target",
        request_generation=3,
        topology_epoch=7,
    )

    assert result.kind == "applied"
    assert result.state.items == target.placements
    assert result.state.start_offset == 4


def test_generation_or_topology_mismatch_is_ignored() -> None:
    current = _loaded_state()
    incoming = _page(2, (_placement(2), _placement(3)), previous=0, next_=4)
    requested = _request(current, generation=3, direction="more", offset=2)

    old_request = apply_notes_slice_page(
        requested,
        incoming,
        direction="more",
        request_generation=2,
        topology_epoch=7,
    )
    old_topology = apply_notes_slice_page(
        requested,
        incoming,
        direction="more",
        request_generation=3,
        topology_epoch=6,
    )

    assert old_request.kind == old_topology.kind == "ignored"
    assert old_request.state is requested
    assert old_topology.state is requested


@pytest.mark.parametrize(
    "incoming",
    [
        _page(2, (_placement(2), _placement(3)), total=7, previous=0, next_=4),
        _page(2, (_placement(1), _placement(3)), previous=0, next_=4),
        _page(2, (_placement(2), _placement(3)), previous=1, next_=4),
        _page(2, (_placement(2), _placement(3)), previous=0, next_=5),
    ],
)
def test_continuation_drift_requests_one_first_page_recovery(
    incoming: NotePlacementPage,
) -> None:
    current = _loaded_state()
    requested = _request(current, generation=3, direction="more", offset=2)

    result = apply_notes_slice_page(
        requested,
        incoming,
        direction="more",
        request_generation=3,
        topology_epoch=7,
    )

    assert result.kind == "drift"
    assert result.recovery == "reset_first"
    assert result.state.recovery_attempted is True
    assert result.state.freshness == "fresh"


@pytest.mark.parametrize(
    "incoming",
    [
        _page(2, (_placement(2),), previous=0, next_=3),
        _page(
            2,
            (_placement(2), _placement(3), _placement(4)),
            previous=0,
            next_=5,
        ),
    ],
)
def test_continuation_count_must_match_requested_limit(
    incoming: NotePlacementPage,
) -> None:
    current = _loaded_state()
    requested = _request(current, generation=3, direction="more", offset=2)

    result = apply_notes_slice_page(
        requested,
        incoming,
        direction="more",
        request_generation=3,
        topology_epoch=7,
    )

    assert result.kind == "drift"
    assert result.recovery == "reset_first"


def test_returned_offset_must_match_the_requested_offset() -> None:
    current = _loaded_state()
    requested = _request(current, generation=3, direction="more", offset=2)
    incoming = _page(3, (_placement(3), _placement(4)), previous=1, next_=5)

    result = apply_notes_slice_page(
        requested,
        incoming,
        direction="more",
        request_generation=3,
        topology_epoch=7,
    )

    assert result.kind == "drift"
    assert result.reason == "response offset differs from request"


def test_out_of_range_continuation_envelope_reaches_reducer_as_drift() -> None:
    current = _loaded_state()
    requested = _request(current, generation=3, direction="more", offset=2)
    out_of_range = _page(20, (), total=3, previous=0)

    result = apply_notes_slice_page(
        requested,
        out_of_range,
        direction="more",
        request_generation=3,
        topology_epoch=7,
    )

    assert result.kind == "drift"
    assert result.recovery == "reset_first"


def test_second_recovery_failure_becomes_stale_and_withdraws_total() -> None:
    current = _loaded_state()
    first = apply_notes_slice_page(
        _request(current, generation=3, direction="more", offset=2),
        _page(2, (_placement(2),), previous=0, next_=3),
        direction="more",
        request_generation=3,
        topology_epoch=7,
    )
    recovering = _request(
        first.state,
        generation=4,
        direction="replace",
        offset=0,
        recovering=True,
    )

    second = fail_notes_slice_load(
        recovering,
        request_generation=4,
        topology_epoch=7,
        error="Recovery request failed.",
    )

    assert second.kind == "failed"
    assert second.recovery is None
    assert second.state.freshness == "stale"
    assert second.state.total is None
    assert second.state.previous_offset is None
    assert second.state.next_offset is None
    assert second.state.items == current.items


def test_folder_pages_use_folder_placement_identity_for_overlap_checks() -> None:
    key = NotesBranchKey(None, "folders")
    first = NoteFolderChildPage((_folder(1),), 2, 0, None, 1)
    current = apply_notes_slice_page(
        _request(
            empty_notes_slice(key, topology_epoch=4),
            generation=1,
            direction="replace",
            offset=0,
            limit=1,
        ),
        first,
        direction="replace",
        request_generation=1,
        topology_epoch=4,
    ).state
    overlap = NoteFolderChildPage((_folder(1),), 2, 1, 0, None)

    result = apply_notes_slice_page(
        _request(current, generation=2, direction="more", offset=1, limit=1),
        overlap,
        direction="more",
        request_generation=2,
        topology_epoch=4,
    )

    assert result.kind == "drift"
    assert current.item_ids == (FolderPlacementId.folder("f1"),)


def test_reducer_does_not_mutate_prior_state_or_page() -> None:
    current = _loaded_state()
    incoming = _page(2, (_placement(2), _placement(3)), previous=0, next_=4)
    before_state = current
    before_page = incoming

    result = apply_notes_slice_page(
        _request(current, generation=3, direction="more", offset=2),
        incoming,
        direction="more",
        request_generation=3,
        topology_epoch=7,
    )

    assert result.kind == "applied"
    assert current == before_state
    assert incoming == before_page
    assert result.state is not current


def test_loading_state_retains_an_immutable_exact_request_contract() -> None:
    current = _loaded_state()

    requested = _request(current, generation=3, direction="more", offset=2, limit=20)

    assert requested.requested_direction == "more"
    assert requested.requested_offset == 2
    assert requested.requested_limit == 20
    with pytest.raises(FrozenInstanceError):
        requested.requested_offset = 3  # type: ignore[misc]


def test_ordinary_load_failure_preserves_visible_exact_state() -> None:
    current = _loaded_state()
    requested = _request(current, generation=3, direction="more", offset=2)

    result = fail_notes_slice_load(
        requested,
        request_generation=3,
        topology_epoch=7,
        error="Page request failed.",
    )

    assert result.kind == "failed"
    assert result.state.items == current.items
    assert result.state.total == current.total
    assert result.state.freshness == "fresh"
    assert result.state.error == "Page request failed."
    assert result.state.loading is False
    assert current.error == ""


def test_obsolete_load_failure_is_ignored_without_changing_state() -> None:
    current = _loaded_state()
    requested = _request(current, generation=3, direction="more", offset=2)

    result = fail_notes_slice_load(
        requested,
        request_generation=2,
        topology_epoch=7,
        error="Old request failed.",
    )

    assert result.kind == "ignored"
    assert result.state is requested


def test_recovery_load_failure_preserves_rows_but_withdraws_exact_metadata() -> None:
    current = _loaded_state()
    drift = apply_notes_slice_page(
        _request(current, generation=3, direction="more", offset=2),
        _page(2, (_placement(2),), previous=0, next_=3),
        direction="more",
        request_generation=3,
        topology_epoch=7,
    )
    recovering = _request(
        drift.state,
        generation=4,
        direction="replace",
        offset=0,
        recovering=True,
    )

    result = fail_notes_slice_load(
        recovering,
        request_generation=4,
        topology_epoch=7,
        error="Recovery request failed.",
    )

    assert result.kind == "failed"
    assert result.state.items == current.items
    assert result.state.freshness == "stale"
    assert result.state.total is None
    assert result.state.previous_offset is None
    assert result.state.next_offset is None
    assert result.state.error == "Recovery request failed."


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
