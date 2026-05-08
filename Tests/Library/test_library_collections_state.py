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
    updated_at: str = "2026-05-08T04:00:00Z",
) -> dict[str, object]:
    return {
        "collection_id": collection_id,
        "name": name,
        "description": description,
        "item_count": item_count,
        "source_authority": "local",
        "sync_status": sync_status,
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
