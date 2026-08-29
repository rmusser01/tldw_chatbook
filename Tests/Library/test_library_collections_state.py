"""Library Collections pure display-state contracts."""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_collections_state import (
    COLLECTION_BROWSE_PAGE_SIZE,
    LIBRARY_COLLECTIONS_EMPTY_COPY,
    CollectionBrowseScope,
    LibraryCollectionsPanelState,
    build_collection_browse_result,
    build_collection_locator_result,
)


def _record(
    collection_id: str,
    name: str,
    *,
    description: str = "",
    item_count: int = 0,
    sync_status: str = "local-only",
    sync_mirror_report: dict[str, object] | None = None,
    sync_readiness_report: dict[str, object] | None = None,
    sync_conflicts: tuple[dict[str, object], ...] = (),
    updated_at: str = "2026-05-08T04:00:00Z",
) -> dict[str, object]:
    return {
        "collection_id": collection_id,
        "name": name,
        "description": description,
        "item_count": item_count,
        "source_authority": "local",
        "sync_status": sync_status,
        "sync_mirror_report": sync_mirror_report or {},
        "sync_readiness_report": sync_readiness_report or {},
        "sync_conflicts": sync_conflicts,
        "created_at": "2026-05-08T03:00:00Z",
        "updated_at": updated_at,
    }


def _browse_item(collection_id: str) -> dict[str, object]:
    return {
        "collection_id": collection_id,
        "name": f"Collection {collection_id}",
        "description": "Saved sources",
        "item_count": 2,
        "created_at": "2026-05-08T03:00:00Z",
        "updated_at": "2026-05-08T04:00:00Z",
    }


def test_collection_browse_scope_exposes_fixed_page_coordinates() -> None:
    scope = CollectionBrowseScope(page=3)

    assert COLLECTION_BROWSE_PAGE_SIZE == 20
    assert scope.page == 3
    assert scope.page_size == 20
    assert scope.offset == 40
    assert scope.with_page(2) == CollectionBrowseScope(page=2)
    assert scope.fingerprint == CollectionBrowseScope(page=3).fingerprint


@pytest.mark.parametrize(
    "page",
    [True, "2", 0, -1, (2**63 - 1) // 20 + 2],
)
def test_collection_browse_scope_rejects_non_integer_or_unsafe_pages(page) -> None:
    with pytest.raises(ValueError, match="page"):
        CollectionBrowseScope(page=page)


def test_collection_browse_result_validates_and_detaches_an_exact_final_page() -> None:
    source = [_browse_item(f"collection-{index}") for index in range(21, 26)]

    result = build_collection_browse_result(
        CollectionBrowseScope(page=2),
        {"items": source, "total": 25, "limit": 20, "offset": 20},
    )
    source[0]["name"] = "Changed later"

    assert result.total == 25
    assert result.last_page == 2
    assert result.out_of_range is False
    assert tuple(item["collection_id"] for item in result.items) == (
        "collection-21",
        "collection-22",
        "collection-23",
        "collection-24",
        "collection-25",
    )
    assert result.items[0]["name"] == "Collection collection-21"


def test_collection_browse_result_accepts_empty_source_and_out_of_range_probe() -> None:
    empty = build_collection_browse_result(
        CollectionBrowseScope(),
        {"items": [], "total": 0, "limit": 20, "offset": 0},
    )
    out_of_range = build_collection_browse_result(
        CollectionBrowseScope(page=3),
        {"items": [], "total": 20, "limit": 20, "offset": 40},
    )

    assert empty.last_page == 1
    assert empty.out_of_range is False
    assert out_of_range.last_page == 1
    assert out_of_range.out_of_range is True


@pytest.mark.parametrize(
    "payload, error",
    [
        ({"items": [], "total": 0, "limit": 10, "offset": 0}, "limit"),
        ({"items": [], "total": 0, "limit": 20, "offset": 20}, "offset"),
        ({"items": [], "total": True, "limit": 20, "offset": 0}, "total"),
        ({"items": [], "total": 1, "limit": 20, "offset": 0}, "count"),
        (
            {
                "items": [_browse_item("collection-1")],
                "total": 21,
                "limit": 20,
                "offset": 0,
            },
            "count",
        ),
    ],
)
def test_collection_browse_result_rejects_incoherent_coordinates(
    payload, error
) -> None:
    with pytest.raises(ValueError, match=error):
        build_collection_browse_result(CollectionBrowseScope(), payload)


def test_collection_browse_result_rejects_duplicate_or_malformed_identities() -> None:
    duplicate = [_browse_item("collection-1"), _browse_item("collection-1")]
    malformed = _browse_item("collection-1")
    malformed["item_count"] = True

    with pytest.raises(ValueError, match="unique"):
        build_collection_browse_result(
            CollectionBrowseScope(),
            {"items": duplicate, "total": 2, "limit": 20, "offset": 0},
        )
    with pytest.raises(ValueError, match="item_count"):
        build_collection_browse_result(
            CollectionBrowseScope(),
            {"items": [malformed], "total": 1, "limit": 20, "offset": 0},
        )


def _locator_payload() -> dict[str, object]:
    return {
        "items": [_browse_item(f"collection-{index}") for index in range(21, 41)],
        "total": 45,
        "limit": 20,
        "offset": 20,
        "page": 2,
        "target_id": "collection-23",
        "target_rank": 22,
        "target_index": 2,
    }


def test_collection_locator_result_accepts_aligned_owning_page() -> None:
    result = build_collection_locator_result("collection-23", _locator_payload())

    assert result.page == 2
    assert result.offset == 20
    assert result.target_rank == 22
    assert result.target_index == 2
    assert result.items[2]["collection_id"] == "collection-23"
    assert result.browse_result.scope == CollectionBrowseScope(page=2)


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("target_id", "collection-24", "target_id"),
        ("target_rank", 42, "rank"),
        ("target_index", 3, "index"),
        ("offset", 0, "offset"),
        ("page", 3, "page"),
    ],
)
def test_collection_locator_result_rejects_unaligned_target_metadata(
    field, value, error
) -> None:
    payload = _locator_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=error):
        build_collection_locator_result("collection-23", payload)


def test_collection_locator_result_rejects_absent_or_duplicate_target() -> None:
    absent = _locator_payload()
    absent["items"][2] = _browse_item("collection-missing")
    duplicate = _locator_payload()
    duplicate["items"][3] = _browse_item("collection-23")

    with pytest.raises(ValueError, match="target"):
        build_collection_locator_result("collection-23", absent)
    with pytest.raises(ValueError, match="unique"):
        build_collection_locator_result("collection-23", duplicate)


def test_empty_panel_state_explains_library_collections_scope() -> None:
    state = LibraryCollectionsPanelState.from_values(collections=(), status="ready")

    assert state.status == "empty"
    assert state.empty_copy == LIBRARY_COLLECTIONS_EMPTY_COPY
    # task-4023 AC#7: one sentence combining purpose and next action.
    assert "create one below to start" in state.empty_copy
    assert "reading and review" in state.empty_copy
    assert state.selected_collection is None
    assert state.delete_action.enabled is False
    assert (
        state.delete_action.disabled_reason == "Select a Collection before deleting it."
    )


def test_ready_state_selects_first_collection_by_default() -> None:
    state = LibraryCollectionsPanelState.from_values(
        collections=(
            _record("collection-b", "Research"),
            _record("collection-a", "Briefing Queue"),
        ),
        selected_collection_id=None,
    )

    assert state.status == "ready"
    assert state.selected_collection is not None
    assert state.selected_collection.collection_id == "collection-b"
    assert state.selected_collection.name == "Research"
    assert state.collections[0].selected is True
    assert state.collections[1].selected is False


def test_invalid_create_and_rename_inputs_disable_actions_with_reasons() -> None:
    state = LibraryCollectionsPanelState.from_values(
        collections=(_record("collection-1", "Research"),),
        selected_collection_id="collection-1",
        create_name=" ",
        rename_name="<script>alert(1)</script>",
    )

    assert state.create_action.enabled is False
    assert state.create_action.disabled_reason == "Enter a Collection name."
    assert state.rename_action.enabled is False
    assert state.rename_action.disabled_reason == "Enter a safe Collection name."


def test_sync_status_renders_local_only_and_sync_unavailable_copy() -> None:
    local_state = LibraryCollectionsPanelState.from_values(
        collections=(_record("collection-1", "Research", sync_status="local-only"),),
    )
    unavailable_state = LibraryCollectionsPanelState.from_values(
        collections=(
            _record("collection-2", "Server Queue", sync_status="sync-unavailable"),
        ),
    )

    assert local_state.selected_collection is not None
    assert local_state.selected_collection.sync_status_label == "Sync: local-only"
    assert unavailable_state.selected_collection is not None
    assert (
        unavailable_state.selected_collection.sync_status_label
        == "Sync: sync-unavailable"
    )


def test_sync_dry_run_status_summarizes_ready_conflict_orphaned_and_unsupported_states() -> (
    None
):
    ready = _record(
        "collection-ready",
        "Ready",
        sync_status="",
        sync_mirror_report={
            "dry_run": True,
            "write_enabled": False,
            "mapped_count": 2,
            "actions": [
                {"local_present": True, "remote_present": True},
                {"local_present": True, "remote_present": True},
            ],
        },
    )
    conflicted = _record(
        "collection-conflict",
        "Conflict",
        sync_status="",
        sync_mirror_report={"dry_run": True, "write_enabled": False, "mapped_count": 1},
        sync_conflicts=({"conflict_type": "duplicate_local_side"},),
    )
    orphaned = _record(
        "collection-orphaned",
        "Orphaned",
        sync_status="",
        sync_mirror_report={
            "dry_run": True,
            "write_enabled": False,
            "mapped_count": 1,
            "actions": [{"local_present": True, "remote_present": False}],
        },
    )
    unsupported = _record(
        "collection-unsupported",
        "Unsupported",
        sync_status="",
        sync_readiness_report={
            "sync_eligible": False,
            "write_enabled": False,
            "reason_codes": ("not_registered",),
        },
    )

    state = LibraryCollectionsPanelState.from_values(
        collections=(ready, conflicted, orphaned, unsupported),
        selected_collection_id="collection-ready",
    )

    assert state.collections[0].sync_status == "dry-run-ready"
    assert state.collections[0].sync_status_label == "Sync dry-run: ready"
    assert state.collections[0].sync_status_detail == (
        "Read-only mirror check: 2 mapped records. No writes will be queued."
    )
    assert state.collections[1].sync_status == "dry-run-conflict"
    assert state.collections[1].sync_status_detail == (
        "Read-only mirror check: 1 conflict needs review. No writes will be queued."
    )
    assert state.collections[2].sync_status == "dry-run-orphaned"
    assert state.collections[2].sync_status_detail == (
        "Read-only mirror check: orphaned local or remote mappings need review. No writes will be queued."
    )
    assert state.collections[3].sync_status == "dry-run-unsupported"
    assert state.collections[3].sync_status_detail == (
        "Read-only mirror check unavailable: not_registered. No writes will be queued."
    )


def test_selected_collection_detail_exposes_stable_updated_at_label() -> None:
    state = LibraryCollectionsPanelState.from_values(
        collections=(
            _record(
                "collection-1",
                "Research",
                item_count=3,
                updated_at="2026-05-08T04:05:06Z",
            ),
        ),
    )

    detail = state.selected_collection
    assert detail is not None
    assert detail.item_count_label == "3 items"
    assert detail.updated_at_label == "Updated 2026-05-08 04:05 UTC"


def test_delete_action_is_disabled_when_no_collection_is_selected() -> None:
    state = LibraryCollectionsPanelState.from_values(
        collections=(_record("collection-1", "Research"),),
        selected_collection_id="missing",
    )

    assert state.selected_collection is None
    assert state.delete_action.enabled is False
    assert (
        state.delete_action.disabled_reason == "Select a Collection before deleting it."
    )
