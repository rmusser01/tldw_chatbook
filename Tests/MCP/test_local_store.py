from __future__ import annotations

import json

import pytest

from tldw_chatbook.MCP.local_store import (
    LocalExternalMCPProfile,
    LocalGovernanceRule,
    LocalMCPStoreLoadError,
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


def test_local_store_rejects_arbitrary_literal_strings_under_neutral_keys(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")

    with pytest.raises(ValueError):
        store.save_profile(
            LocalExternalMCPProfile(
                profile_id="profile-a",
                command="python",
                env_literals={"SERVICE_ENDPOINT": "example-service-prod"},
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


def test_local_store_rejects_blank_profile_writes_before_persistence(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")

    with pytest.raises(ValueError, match="profile_id"):
        store.save_profile(LocalExternalMCPProfile(profile_id="", command="python"))

    with pytest.raises(ValueError, match="command"):
        store.save_profile(LocalExternalMCPProfile(profile_id="profile-a", command=""))

    assert not (tmp_path / "local_mcp_store.json").exists()


def test_local_store_rejects_blank_governance_rule_writes_before_persistence(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")

    with pytest.raises(ValueError, match="rule_id"):
        store.save_governance_rule(
            LocalGovernanceRule(rule_id="", capability_id="mcp.inventory.list.local", decision="allow")
        )

    with pytest.raises(ValueError, match="capability_id"):
        store.save_governance_rule(
            LocalGovernanceRule(rule_id="rule-a", capability_id="", decision="allow")
        )

    with pytest.raises(ValueError, match="decision"):
        store.save_governance_rule(
            LocalGovernanceRule(rule_id="rule-a", capability_id="mcp.inventory.list.local", decision="")
        )

    assert not (tmp_path / "local_mcp_store.json").exists()


def test_local_store_rejects_blank_discovery_snapshot_profile_id_before_persistence(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")

    with pytest.raises(ValueError, match="profile_id"):
        store.save_discovery_snapshot(
            "",
            {
                "server_id": "profile-a",
                "tools": [{"name": "remote_tool"}],
            },
        )

    assert not (tmp_path / "local_mcp_store.json").exists()


def test_local_store_loads_legacy_env_payload_and_drops_unsafe_entries(tmp_path):
    path = tmp_path / "local_mcp_store.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "profile_id": "profile-a",
                        "command": "python",
                        "args": ["-m", "demo.server"],
                        "env": {
                            "SERVICE_URL": "https://api.example.com",
                            "MODEL_NAME": "gpt-4o-mini",
                            "SOCKET_PATH": "/tmp/mcp-demo.sock",
                            "API_KEY": "sk-live-super-secret-value",
                            "LOG_LEVEL": "debug",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    restored = LocalMCPStore(path).get_profile("profile-a")

    assert restored is not None
    assert restored.env["SERVICE_URL"] == "https://api.example.com"
    assert restored.env["MODEL_NAME"] == "gpt-4o-mini"
    assert restored.env["SOCKET_PATH"] == "/tmp/mcp-demo.sock"
    assert restored.env["LOG_LEVEL"] == "debug"
    assert "API_KEY" not in restored.env

    saved = LocalMCPStore(path).save_profile(restored)
    round_tripped = LocalMCPStore(path).get_profile("profile-a")

    assert saved.legacy_env_literals == {}
    assert round_tripped is not None
    assert "SERVICE_URL" not in round_tripped.env
    assert "MODEL_NAME" not in round_tripped.env
    assert "SOCKET_PATH" not in round_tripped.env
    assert "API_KEY" not in round_tripped.env


def test_local_store_save_profile_canonicalizes_prebuilt_profiles_and_drops_legacy_env_literals(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    profile = LocalExternalMCPProfile(
        profile_id="profile-b",
        command="python",
        args=("-m", "demo.server"),
        env_literals={"LOG_LEVEL": "debug"},
        legacy_env_literals={
            "SERVICE_ALIAS": "example-service-prod",
            "MODEL_NAME": "gpt-4o-mini",
            "SOCKET_PATH": "/tmp/mcp-demo.sock",
        },
    )

    saved = store.save_profile(profile)
    restored = store.get_profile("profile-b")

    assert saved.env["LOG_LEVEL"] == "debug"
    assert "SERVICE_ALIAS" not in saved.env
    assert "MODEL_NAME" not in saved.env
    assert "SOCKET_PATH" not in saved.env
    assert saved.legacy_env_literals == {}
    assert restored is not None
    assert restored.legacy_env_literals == {}


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


def test_local_store_clears_stale_discovery_snapshot_only_when_launch_config_changes(tmp_path):
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    profile = LocalExternalMCPProfile(
        profile_id="profile-a",
        command="python",
        args=("-m", "demo.server"),
        env_placeholders={"API_KEY": "${API_KEY}"},
        env_literals={"LOG_LEVEL": "debug"},
    )
    snapshot = {
        "server_id": "profile-a",
        "tools": [{"name": "remote_tool"}],
    }

    store.save_profile(profile)
    store.save_discovery_snapshot("profile-a", snapshot)

    store.save_profile(
        LocalExternalMCPProfile(
            profile_id="profile-a",
            command="python",
            args=("-m", "demo.server"),
            env_placeholders={"API_KEY": "${API_KEY}"},
            env_literals={"LOG_LEVEL": "debug"},
        )
    )

    assert store.get_discovery_snapshot("profile-a") == snapshot

    store.save_profile(
        LocalExternalMCPProfile(
            profile_id="profile-a",
            command="python",
            args=("-m", "demo.server"),
            env_placeholders={"API_KEY": "${API_KEY}"},
            env_literals={"LOG_LEVEL": "info"},
        )
    )

    assert store.get_discovery_snapshot("profile-a") is None


def test_local_store_raises_on_corrupt_payload_instead_of_treating_it_as_empty_state(tmp_path):
    path = tmp_path / "local_mcp_store.json"
    corrupt_payload = '{"profiles": [}'
    path.write_text(corrupt_payload, encoding="utf-8")
    store = LocalMCPStore(path)

    with pytest.raises(LocalMCPStoreLoadError):
        store.load()

    with pytest.raises(LocalMCPStoreLoadError):
        store.save_profile(
            LocalExternalMCPProfile(
                profile_id="profile-a",
                command="python",
                env_literals={"LOG_LEVEL": "debug"},
            )
        )

    assert path.read_text(encoding="utf-8") == corrupt_payload
