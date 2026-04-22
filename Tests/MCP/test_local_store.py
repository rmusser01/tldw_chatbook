from __future__ import annotations

import json

import pytest

from tldw_chatbook.MCP.local_store import (
    LocalExternalMCPProfile,
    LocalGovernanceRule,
    LocalMCPStore,
)


def test_local_store_profile_crud_persists_env_placeholders_and_safe_literals(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    profile = LocalExternalMCPProfile(
        profile_id="profile-a",
        command="python",
        args=("-m", "demo.server"),
        env_placeholders={"API_KEY": "${API_KEY}"},
        env_literals={
            "LOG_LEVEL": "debug",
            "FEATURE_FLAG": "enabled",
        },
    )

    saved = store.save_profile(profile)
    restored = store.get_profile("profile-a")

    assert saved.profile_id == "profile-a"
    assert list(saved.args) == ["-m", "demo.server"]
    assert saved.env == {
        "API_KEY": "${API_KEY}",
        "LOG_LEVEL": "debug",
        "FEATURE_FLAG": "enabled",
    }
    assert restored == saved
    assert store.list_profiles() == [saved]

    raw_payload = json.loads((tmp_path / "local_mcp_store.json").read_text(encoding="utf-8"))
    assert raw_payload["profiles"][0]["env_placeholders"]["API_KEY"] == "${API_KEY}"
    assert raw_payload["profiles"][0]["env_literals"]["LOG_LEVEL"] == "debug"

    deleted = store.delete_profile("profile-a")

    assert deleted is True
    assert store.get_profile("profile-a") is None
    assert store.list_profiles() == []


def test_local_store_rejects_secret_bearing_literal_env_entries(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")

    with pytest.raises(ValueError):
        store.save_profile(
            LocalExternalMCPProfile(
                profile_id="profile-a",
                command="python",
                env_literals={"API_KEY": "raw-secret-value"},
            )
        )

    assert not (tmp_path / "local_mcp_store.json").exists()


def test_local_store_rejects_raw_secret_like_literal_values_under_neutral_keys(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")

    with pytest.raises(ValueError):
        store.save_profile(
            LocalExternalMCPProfile(
                profile_id="profile-a",
                command="python",
                env_literals={"SERVICE_ENDPOINT": "sk-live-super-secret-value"},
            )
        )

    assert not (tmp_path / "local_mcp_store.json").exists()


def test_local_store_rejects_non_placeholder_secret_env_entries_even_when_declared_as_placeholders(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")

    with pytest.raises(ValueError):
        store.save_profile(
            LocalExternalMCPProfile(
                profile_id="profile-a",
                command="python",
                env_placeholders={"API_KEY": "raw-secret-value"},
            )
        )

    assert not (tmp_path / "local_mcp_store.json").exists()


def test_local_store_persists_discovery_snapshots_and_governance_updates(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    rule = LocalGovernanceRule(
        rule_id="rule-a",
        capability_id="mcp.inventory.list.local",
        decision="allow",
        notes="local inventory is permitted",
    )

    store.save_discovery_snapshot(
        "profile-a",
        {
            "server_id": "profile-a",
            "tools": [{"name": "remote_tool"}],
            "resources": [{"uri": "remote://resource"}],
            "prompts": [{"name": "remote_prompt"}],
        },
    )
    saved_rule = store.save_governance_rule(rule)
    restored = LocalMCPStore(tmp_path / "local_mcp_store.json")

    snapshot = restored.get_discovery_snapshot("profile-a")
    rules = restored.list_governance_rules()

    assert snapshot["tools"][0]["name"] == "remote_tool"
    assert snapshot["resources"][0]["uri"] == "remote://resource"
    assert saved_rule.rule_id == rule.rule_id
    assert saved_rule.capability_id == rule.capability_id
    assert rules[0].decision == "allow"

    raw_payload = json.loads((tmp_path / "local_mcp_store.json").read_text(encoding="utf-8"))
    assert raw_payload["discovery_snapshots"]["profile-a"]["prompts"][0]["name"] == "remote_prompt"
    assert raw_payload["governance_rules"][0]["capability_id"] == "mcp.inventory.list.local"
