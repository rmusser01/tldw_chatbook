"""Tests for task-236: get_external_servers reads the store exactly once.

Uses the REAL LocalMCPStore on a tmp file with a counting wrapper around
``load()`` — no fakes of the store contract, so a regression back to
per-profile snapshot loads (or a signature drift) fails here.
"""

from pathlib import Path

import pytest

from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore


class _LoadCountingStore(LocalMCPStore):
    """Real store that counts full state loads."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        return super().load()


def _service(store: LocalMCPStore) -> LocalMCPControlService:
    return LocalMCPControlService(store=store, manifest_provider=lambda: {})


def _seed(store: LocalMCPStore, n: int = 4) -> list[str]:
    ids = []
    for i in range(n):
        profile = LocalExternalMCPProfile.from_input_dict(
            {"profile_id": f"profile-{i}", "command": "echo", "args": ["hi"]}
        )
        store.save_profile(profile)
        ids.append(profile.profile_id)
    # Snapshots for half the profiles: the None-snapshot path must survive.
    for pid in ids[: n // 2]:
        store.save_discovery_snapshot(pid, {"tools": [{"name": f"tool-{pid}"}]})
    return ids


def test_get_external_servers_is_single_store_load(tmp_path):
    """AC#1: the full catalog read performs exactly one state load."""
    store = _LoadCountingStore(tmp_path / "mcp-store.json")
    ids = _seed(store, n=4)
    service = _service(store)

    store.load_calls = 0
    servers = service.get_external_servers()

    assert store.load_calls == 1, (
        f"expected exactly one store load for the whole catalog, got {store.load_calls}"
    )
    assert [s["profile_id"] for s in servers] == ids


def test_get_external_servers_shape_unchanged(tmp_path):
    """The joined read returns byte-identical content to the per-item APIs."""
    store = LocalMCPStore(tmp_path / "mcp-store.json")
    ids = _seed(store, n=4)
    service = _service(store)

    servers = {s["profile_id"]: s for s in service.get_external_servers()}

    for pid in ids:
        reference_snapshot = store.get_discovery_snapshot(pid)
        assert servers[pid]["discovery_snapshot"] == reference_snapshot
        assert servers[pid]["is_connected"] is False
    with_snapshot = [pid for pid in ids if servers[pid]["discovery_snapshot"]]
    without = [pid for pid in ids if servers[pid]["discovery_snapshot"] is None]
    assert with_snapshot and without, "seed must cover both snapshot states"


def test_store_catalog_matches_individual_accessors(tmp_path):
    """Store-level equivalence: get_external_catalog == list+get per item."""
    store = LocalMCPStore(tmp_path / "mcp-store.json")
    _seed(store, n=3)

    catalog = store.get_external_catalog()

    reference = [
        (p.profile_id, store.get_discovery_snapshot(p.profile_id))
        for p in store.list_profiles()
    ]
    assert [(p.profile_id, snap) for p, snap in catalog] == reference
