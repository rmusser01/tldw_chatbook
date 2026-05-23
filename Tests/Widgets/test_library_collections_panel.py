"""Library Collections widget rendering tests."""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.Library.library_collections_state import LibraryCollectionsPanelState
from tldw_chatbook.Widgets.Library.library_collections_panel import LibraryCollectionsPanel


pytestmark = pytest.mark.asyncio


async def test_library_collections_panel_renders_read_only_sync_dry_run_detail(widget_pilot):
    state = LibraryCollectionsPanelState.from_values(
        collections=(
            {
                "collection_id": "collection-1",
                "name": "Research",
                "description": "Selected sources",
                "item_count": 2,
                "source_authority": "local",
                "sync_status": "",
                "sync_mirror_report": {
                    "dry_run": True,
                    "write_enabled": False,
                    "mapped_count": 2,
                    "actions": [
                        {"local_present": True, "remote_present": True},
                        {"local_present": True, "remote_present": True},
                    ],
                },
                "created_at": "2026-05-08T03:00:00Z",
                "updated_at": "2026-05-08T04:00:00Z",
            },
        ),
        selected_collection_id="collection-1",
    )

    async with await widget_pilot(LibraryCollectionsPanel, state=state) as pilot:
        await pilot.pause()
        assert str(pilot.app.query_one("#library-collection-sync-status", Static).renderable) == (
            "Sync dry-run: ready"
        )
        assert str(pilot.app.query_one("#library-collection-sync-detail", Static).renderable) == (
            "Read-only mirror check: 2 mapped records. No writes will be queued."
        )


async def test_library_collections_panel_renders_write_sync_promotion_labels(widget_pilot):
    state = LibraryCollectionsPanelState.from_values(
        collections=(
            {
                "collection_id": "collection-1",
                "name": "Research",
                "description": "Selected sources",
                "item_count": 2,
                "source_authority": "local",
                "sync_promotion_state": {
                    "authority_label": "Authority: local",
                    "sync_label": "Sync: dry-run only",
                    "review_label": "Review: required before writes",
                    "conflict_label": "Conflicts: none",
                    "rollback_label": "Rollback: not required",
                    "mirror_label": "Mirror: 2 mapped records",
                    "primary_recovery": "Writes stay blocked until review, conflict, and rollback gates are ready.",
                    "mutation_allowed": False,
                },
                "created_at": "2026-05-08T03:00:00Z",
                "updated_at": "2026-05-08T04:00:00Z",
            },
        ),
        selected_collection_id="collection-1",
    )

    async with await widget_pilot(LibraryCollectionsPanel, state=state) as pilot:
        await pilot.pause()
        assert str(pilot.app.query_one("#library-collection-sync-status", Static).renderable) == (
            "Sync: dry-run only"
        )
        assert str(pilot.app.query_one("#library-collection-sync-safety-heading", Static).renderable) == (
            "Write Sync Safety"
        )
        assert str(pilot.app.query_one("#library-collection-sync-safety-help", Static).renderable) == (
            "Review these labels before any future server write promotion."
        )
        assert str(pilot.app.query_one("#library-collection-sync-detail", Static).renderable) == (
            "Authority: local | Mirror: 2 mapped records | Review: required before writes | "
            "Conflicts: none | Rollback: not required | "
            "Writes stay blocked until review, conflict, and rollback gates are ready."
        )
