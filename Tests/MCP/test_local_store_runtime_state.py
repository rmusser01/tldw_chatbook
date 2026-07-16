from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore


@pytest.fixture()
def store(tmp_path: Path) -> LocalMCPStore:
    return LocalMCPStore(tmp_path / "local_mcp_store.json")


def _save_profile(store: LocalMCPStore, profile_id: str = "docs", command: str = "python"):
    return store.save_profile(
        LocalExternalMCPProfile(profile_id=profile_id, command=command, args=("-m", "demo"))
    )


def test_runtime_state_roundtrip_persists(store, tmp_path):
    _save_profile(store)
    record = {"last_attempt_at": "2026-07-14T00:00:00Z", "last_action": "connect",
              "ok": False, "last_ok_at": None, "last_error": "boom"}
    saved = store.save_profile_runtime_state("docs", record)
    assert saved == record
    reloaded = LocalMCPStore(tmp_path / "local_mcp_store.json")
    assert reloaded.get_profile_runtime_state("docs") == record
    assert reloaded.get_profile_runtime_state("missing") is None


def test_delete_profile_cascades_runtime_state(store):
    _save_profile(store)
    store.save_profile_runtime_state("docs", {"ok": True})
    assert store.delete_profile("docs") is True
    assert store.get_profile_runtime_state("docs") is None


def test_launch_config_change_pops_runtime_state(store):
    _save_profile(store, command="python")
    store.save_profile_runtime_state("docs", {"ok": False, "last_error": "old"})
    _save_profile(store, command="node")  # command changed -> launch config changed
    assert store.get_profile_runtime_state("docs") is None


def test_other_mutations_do_not_reset_runtime_state(store):
    _save_profile(store)
    store.save_profile_runtime_state("docs", {"ok": True})
    # a snapshot save is one of the other LocalMCPStoreState(...) reconstruction sites
    store.save_discovery_snapshot("docs", {"tools": [{"name": "a"}], "resources": [], "prompts": []})
    assert store.get_profile_runtime_state("docs") == {"ok": True}
