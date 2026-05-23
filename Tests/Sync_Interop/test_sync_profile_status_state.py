"""Sync v2 profile summary display-state contracts."""

from __future__ import annotations

from tldw_chatbook.Sync_Interop.sync_profile_status_state import SyncProfileStatusDisplay


def _summary(
    *,
    status: str,
    profile_mode: str = "local_first_sync",
    pending: int = 0,
    dispatched: int = 0,
    conflicts: int = 0,
    last_error: str | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "profile": {
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": "workspace-1",
            "profile_mode": profile_mode,
            "device_id": "device-1",
            "dataset_id": "dataset-1",
            "last_error": last_error,
        },
        "cursor": {
            "remote_collection": "dataset-1",
            "remote_cursor": "remote-42",
            "profile_cursor": "profile-42",
        },
        "outbox": {
            "pending": pending,
            "dispatched": dispatched,
            "by_domain": {"notes": {"pending": pending, "dispatched": dispatched}},
        },
        "identity_map": {"total": 3, "confirmed": 3, "by_domain": {"notes": {"confirmed": 3}}},
        "conflicts": {"count": conflicts, "latest": []},
        "last_mirror_report": None,
    }


def test_not_configured_summary_maps_to_local_safe_copy() -> None:
    display = SyncProfileStatusDisplay.from_summary(
        {
            "status": "not_configured",
            "profile": None,
            "cursor": None,
            "outbox": {"pending": 0, "dispatched": 0, "by_domain": {}},
            "identity_map": {"total": 0, "by_domain": {}},
            "conflicts": {"count": 0, "latest": []},
            "last_mirror_report": None,
        }
    )

    assert display.status == "not_configured"
    assert display.severity == "neutral"
    assert display.label == "Sync profile: not configured"
    assert display.detail == (
        "No Sync v2 server profile is configured. Local Library data stays on this device."
    )
    assert display.read_only_notice == "This view only reads sync state; it does not start sync."


def test_server_frontend_summary_maps_to_live_frontend_copy() -> None:
    display = SyncProfileStatusDisplay.from_summary(
        _summary(status="server_frontend", profile_mode="server_frontend")
    )

    assert display.severity == "ready"
    assert display.label == "Sync profile: server front-end"
    assert display.detail == (
        "Using server-a as a live server front-end. Offline mirror sync is not configured."
    )
    assert display.dataset_label == "Dataset dataset-1"
    assert display.device_label == "Device device-1"


def test_pending_summary_includes_outbox_count_without_implying_write_action() -> None:
    display = SyncProfileStatusDisplay.from_summary(_summary(status="pending", pending=2))

    assert display.severity == "pending"
    assert display.label == "Sync profile: pending local changes"
    assert display.pending_count == 2
    assert display.detail == (
        "2 pending local changes are waiting for the next sync pass. No writes start from this view."
    )


def test_attention_required_summary_prioritizes_conflicts_and_safe_error_copy() -> None:
    display = SyncProfileStatusDisplay.from_summary(
        _summary(
            status="attention_required",
            pending=1,
            conflicts=1,
            last_error="<script>alert(1)</script>",
        )
    )

    assert display.severity == "attention"
    assert display.label == "Sync profile: needs attention"
    assert display.conflict_count == 1
    assert display.detail == (
        "1 sync conflict needs review. Last error is unavailable. No writes start from this view."
    )
