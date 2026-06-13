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


async def test_library_collections_panel_renders_sync_profile_status_banner(widget_pilot):
    state = LibraryCollectionsPanelState.from_values(
        collections=(
            {
                "collection_id": "collection-1",
                "name": "Research",
                "description": "Selected sources",
                "item_count": 2,
                "source_authority": "local",
                "sync_status": "local-only",
                "created_at": "2026-05-08T03:00:00Z",
                "updated_at": "2026-05-08T04:00:00Z",
            },
        ),
        selected_collection_id="collection-1",
        sync_profile_summary={
            "status": "pending",
            "profile": {
                "server_profile_id": "server-a",
                "authenticated_principal_id": "user-a",
                "workspace_scope": None,
                "profile_mode": "local_first_sync",
                "device_id": "device-1",
                "dataset_id": "dataset-1",
                "last_error": None,
            },
            "cursor": None,
            "outbox": {"pending": 2, "dispatched": 1, "by_domain": {}},
            "identity_map": {"total": 0, "by_domain": {}},
            "conflicts": {"count": 0, "latest": []},
            "last_mirror_report": None,
        },
    )

    async with await widget_pilot(LibraryCollectionsPanel, state=state) as pilot:
        await pilot.pause()
        assert str(pilot.app.query_one("#library-sync-profile-status", Static).renderable) == (
            "Sync profile: pending local changes"
        )
        assert str(pilot.app.query_one("#library-sync-profile-detail", Static).renderable) == (
            "2 pending local changes are waiting for the next sync pass."
        )
        assert str(pilot.app.query_one("#library-sync-profile-read-only", Static).renderable) == (
            "This view only reads sync state; it does not start sync."
        )
        assert pilot.app.query_one("#library-sync-profile-status", Static)._render_markup is False
        assert pilot.app.query_one("#library-sync-profile-detail", Static)._render_markup is False
        assert pilot.app.query_one("#library-sync-profile-read-only", Static)._render_markup is False
