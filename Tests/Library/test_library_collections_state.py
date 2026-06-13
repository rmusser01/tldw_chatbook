"""Library Collections pure display-state contracts."""

from __future__ import annotations

from tldw_chatbook.Library.library_collections_state import (
    LIBRARY_COLLECTIONS_EMPTY_COPY,
    LibraryCollectionsPanelState,
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


def test_empty_panel_state_explains_library_collections_scope() -> None:
    state = LibraryCollectionsPanelState.from_values(collections=(), status="ready")

    assert state.status == "empty"
    assert state.empty_copy == LIBRARY_COLLECTIONS_EMPTY_COPY
    assert state.empty_copy == "Group saved Library items for Search/RAG, Study, and Console."
    assert state.selected_collection is None
    assert state.delete_action.enabled is False
    assert state.delete_action.disabled_reason == "Select a Collection before deleting it."


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
    assert unavailable_state.selected_collection.sync_status_label == "Sync: sync-unavailable"


def test_sync_dry_run_status_summarizes_ready_conflict_orphaned_and_unsupported_states() -> None:
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
    assert state.delete_action.disabled_reason == "Select a Collection before deleting it."
