from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tldw_chatbook.runtime_policy.source_state import normalize_runtime_source_state
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


def test_normalize_runtime_source_state_resets_stale_auth_and_reachability():
    stale = RuntimeSourceState(
        active_source="server",
        active_server_id="server-a",
        server_configured=True,
        server_reachability="reachable",
        server_auth_state="authenticated",
        server_reachability_checked_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        server_auth_checked_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    normalized = normalize_runtime_source_state(
        stale,
        now=datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc),
        freshness_window=timedelta(minutes=5),
    )

    assert normalized.server_reachability == "unknown"
    assert normalized.server_reachability_checked_at is None
    assert normalized.server_auth_state == "unknown"
    assert normalized.server_auth_checked_at is None


def test_normalize_runtime_source_state_clears_server_probe_state_when_server_is_not_authoritative():
    state = RuntimeSourceState(
        active_source="local",
        active_server_id="server-a",
        server_configured=False,
        server_reachability="reachable",
        server_reachability_checked_at=datetime(2026, 4, 29, 11, 59, tzinfo=timezone.utc),
        server_auth_state="authenticated",
        server_auth_checked_at=datetime(2026, 4, 29, 11, 59, tzinfo=timezone.utc),
        last_known_server_label="Server A",
    )

    normalized = normalize_runtime_source_state(
        state,
        now=datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc),
        freshness_window=timedelta(minutes=5),
    )

    assert normalized.active_source == "local"
    assert normalized.active_server_id == "server-a"
    assert normalized.last_known_server_label == "Server A"
    assert normalized.server_reachability == "unknown"
    assert normalized.server_reachability_checked_at is None
    assert normalized.server_auth_state == "unknown"
    assert normalized.server_auth_checked_at is None
