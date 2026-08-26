"""Library Collections widget rendering tests."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Collapsible, Static

from tldw_chatbook.Library.library_collections_state import (
    LibraryCollectionDeleteReceipt,
    LibraryCollectionsPanelState,
)
from tldw_chatbook.Widgets.Library.library_collections_panel import (
    LIBRARY_COLLECTIONS_STATUS_LINE,
    LibraryCollectionsPanel,
)


pytestmark = pytest.mark.asyncio


# task-2859 item 7: canvas titles drop the "Library " prefix and match the
# sibling "Name (n)" pattern (Media/Notes/Prompts/Skills already do this;
# Collections used to render the bare, count-less "Library Collections").


async def test_library_collections_panel_title_matches_the_sibling_name_count_pattern(
    widget_pilot,
):
    state = LibraryCollectionsPanelState.from_values(
        collections=(
            {"collection_id": "collection-1", "name": "Research", "item_count": 2},
            {"collection_id": "collection-2", "name": "Reading list", "item_count": 0},
        ),
    )

    async with await widget_pilot(LibraryCollectionsPanel, state=state) as pilot:
        await pilot.pause()
        title = pilot.app.query_one("#library-collections-title", Static)
        assert str(title.renderable) == "Collections (2)"


async def test_library_collections_panel_empty_title_shows_zero_count(widget_pilot):
    state = LibraryCollectionsPanelState.from_values(collections=(), status="empty")

    async with await widget_pilot(LibraryCollectionsPanel, state=state) as pilot:
        await pilot.pause()
        title = pilot.app.query_one("#library-collections-title", Static)
        assert str(title.renderable) == "Collections (0)"


async def test_library_collections_panel_renders_receipt_in_empty_state(widget_pilot):
    state = LibraryCollectionsPanelState.from_values(
        collections=(),
        status="empty",
        delete_receipt=LibraryCollectionDeleteReceipt(
            collection_id="collection-1",
            name="Research",
        ),
    )

    async with await widget_pilot(LibraryCollectionsPanel, state=state) as pilot:
        await pilot.pause()
        copy = pilot.app.query_one("#library-collections-delete-receipt-copy", Static)
        assert str(copy.renderable) == "✓ deleted · Collection · Research"
        assert copy._render_markup is False
        assert pilot.app.query_one("#library-collections-delete-undo", Button)
        assert pilot.app.query_one("#library-collections-delete-receipt-dismiss", Button)
        assert pilot.app.query_one("#library-collections-empty-title", Static)


async def test_library_collections_panel_renders_read_only_sync_dry_run_detail(
    widget_pilot,
):
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
        assert str(
            pilot.app.query_one("#library-collection-sync-status", Static).renderable
        ) == ("Sync dry-run: ready")
        assert str(
            pilot.app.query_one("#library-collection-sync-detail", Static).renderable
        ) == ("Read-only mirror check: 2 mapped records. No writes will be queued.")


async def test_library_collections_panel_renders_write_sync_promotion_labels(
    widget_pilot,
):
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
        # TASK-2855: the "Write Sync Safety" heading and its help sentence
        # were spec-internal chrome and were removed; the underlying
        # dry-run promotion data is genuinely useful and survives inside
        # the collapsed-by-default Details disclosure.
        assert not pilot.app.query("#library-collection-sync-safety-heading")
        assert not pilot.app.query("#library-collection-sync-safety-help")
        assert str(
            pilot.app.query_one("#library-collection-sync-status", Static).renderable
        ) == ("Sync: dry-run only")
        assert str(
            pilot.app.query_one("#library-collection-sync-detail", Static).renderable
        ) == (
            "Authority: local | Mirror: 2 mapped records | Review: required before writes | "
            "Conflicts: none | Rollback: not required | "
            "Writes stay blocked until review, conflict, and rollback gates are ready."
        )


async def test_library_collections_panel_renders_sync_profile_status_banner(
    widget_pilot,
):
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
        assert str(
            pilot.app.query_one("#library-sync-profile-status", Static).renderable
        ) == ("Sync profile: pending local changes")
        assert str(
            pilot.app.query_one("#library-sync-profile-detail", Static).renderable
        ) == ("2 pending local changes are waiting for the next sync pass.")
        assert str(
            pilot.app.query_one("#library-sync-profile-read-only", Static).renderable
        ) == ("This view only reads sync state; it does not start sync.")
        assert (
            pilot.app.query_one("#library-sync-profile-status", Static)._render_markup
            is False
        )
        assert (
            pilot.app.query_one("#library-sync-profile-detail", Static)._render_markup
            is False
        )
        assert (
            pilot.app.query_one(
                "#library-sync-profile-read-only", Static
            )._render_markup
            is False
        )


async def test_library_collections_panel_replaces_spec_block_with_plain_status_line(
    widget_pilot,
):
    """TASK-2855 AC1: the per-selection spec/roadmap block ("Item reader
    readiness", "Authority: local", "Content use boundary", "Blocked
    later: ...", "Next: collection item adapters...") is replaced by one
    plain-language status line, and none of that vocabulary survives
    anywhere on the canvas."""
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
    )

    async with await widget_pilot(LibraryCollectionsPanel, state=state) as pilot:
        await pilot.pause()
        assert str(
            pilot.app.query_one(
                "#library-collection-status-line", Static
            ).renderable
        ) == LIBRARY_COLLECTIONS_STATUS_LINE

        rendered = " ".join(
            str(widget.renderable) for widget in pilot.app.query(Static)
        )
        for forbidden in (
            "Item reader readiness",
            "Authority: local",
            "Content use boundary",
            "Blocked later:",
            "Next: collection item adapters",
            "Write Sync Safety",
        ):
            assert forbidden not in rendered, forbidden

        for retired_id in (
            "#library-collection-membership-heading",
            "#library-collection-source-authority",
            "#library-collection-workspace-heading",
            "#library-collection-workspace-rule",
            "#library-collection-deferred-actions",
            "#library-collection-reader-later",
            "#library-collection-sync-safety-heading",
            "#library-collection-sync-safety-help",
        ):
            assert not pilot.app.query(retired_id), retired_id

        # Genuinely useful action-status copy survives unchanged.
        assert "Available now: create, rename, delete records" in rendered


async def test_library_collections_panel_moves_sync_detail_behind_details_disclosure(
    widget_pilot,
):
    """TASK-2855 AC2: sync-safety/internal detail (sync status, sync
    detail, item count, updated-at) moves behind a collapsed-by-default
    "Details" disclosure instead of always being on screen."""
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
        details = pilot.app.query_one("#library-collection-details", Collapsible)
        assert details.collapsed is True

        for widget_id in (
            "#library-collection-sync-status",
            "#library-collection-sync-detail",
            "#library-collection-item-count",
            "#library-collection-updated-at",
        ):
            widget = pilot.app.query_one(widget_id, Static)
            assert details in widget.ancestors, widget_id


async def test_library_collections_panel_empty_state_renders_message_once(
    widget_pilot,
):
    """The current empty-state guidance renders once, not twice."""
    state = LibraryCollectionsPanelState.from_values(collections=(), status="empty")

    async with await widget_pilot(LibraryCollectionsPanel, state=state) as pilot:
        await pilot.pause()
        empty_copy = pilot.app.query("#library-collections-empty")
        assert len(empty_copy) == 1
        assert str(empty_copy[0].renderable) == state.empty_copy
        assert not pilot.app.query("#library-collection-empty-reader")
        assert not pilot.app.query("#library-collection-empty-reader-title")


async def test_library_collections_panel_shows_single_create_guidance_when_name_invalid(
    widget_pilot,
):
    """TASK-2855 AC3: the three enable-Create helper sentences collapse
    into one, shown while the typed name is not yet valid."""
    state = LibraryCollectionsPanelState.from_values(collections=(), status="empty")

    async with await widget_pilot(LibraryCollectionsPanel, state=state) as pilot:
        await pilot.pause()
        guidance = pilot.app.query("#library-collection-form-guidance")
        assert len(guidance) == 1
        assert str(guidance[0].renderable) == "Enter a Collection name."
        assert not pilot.app.query("#library-collection-form-action-state")
        assert not pilot.app.query("#library-collection-form-action-boundary")


async def test_library_collections_panel_hides_create_guidance_once_name_is_valid(
    widget_pilot,
):
    """TASK-2855 AC3: once a valid, non-duplicate name is typed, the
    create-guidance sentence disappears."""
    state = LibraryCollectionsPanelState.from_values(
        collections=(), status="empty", create_name="Research"
    )

    async with await widget_pilot(
        LibraryCollectionsPanel, state=state, name_value="Research"
    ) as pilot:
        await pilot.pause()
        assert not pilot.app.query("#library-collection-form-guidance")
