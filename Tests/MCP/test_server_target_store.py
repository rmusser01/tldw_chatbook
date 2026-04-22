from __future__ import annotations

from datetime import datetime, timezone

from pathlib import Path

from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget, TargetStatusMetadata


def test_bootstrap_from_legacy_config_only_when_registry_is_empty(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")

    imported = store.bootstrap_from_legacy_config(
        {
            "tldw_api": {
                "base_url": "https://Example.COM:8443/api/",
                "api_key": "super-secret",
            }
        }
    )

    assert imported is True

    targets = store.list_targets()
    assert len(targets) == 1

    target = targets[0]
    assert target.server_id == "https://example.com:8443/api"
    assert target.label == "example.com:8443"
    assert target.base_url == "https://example.com:8443/api"
    assert target.auth_reference == "legacy:tldw_api"
    assert target.last_known_server_label == "example.com:8443"

    raw_payload = (tmp_path / "server_targets.json").read_text(encoding="utf-8")
    assert "super-secret" not in raw_payload


def test_bootstrap_from_legacy_config_returns_false_for_malformed_url(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")

    imported = store.bootstrap_from_legacy_config(
        {
            "tldw_api": {
                "base_url": "https://example.com:bad/api/",
                "api_key": "super-secret",
            }
        }
    )

    assert imported is False
    assert store.list_targets() == []


def test_legacy_config_does_not_overwrite_existing_registry(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")
    saved_target = ConfiguredServerTarget(
        server_id="saved-target",
        label="Saved Target",
        base_url="https://saved.example/api",
        auth_reference="existing:reference",
        is_default=True,
    )
    store.save_targets([saved_target])

    imported = store.bootstrap_from_legacy_config(
        {
            "tldw_api": {
                "base_url": "https://other.example/api/",
                "api_key": "another-secret",
            }
        }
    )

    assert imported is False
    assert store.list_targets() == [saved_target]


def test_target_store_loads_safe_default_on_invalid_json(tmp_path):
    path = tmp_path / "server_targets.json"
    path.write_text("{not-json", encoding="utf-8")

    restored = ConfiguredServerTargetStore(path).load()

    assert restored == []


def test_target_store_uses_atomic_temp_file_replacement(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")

    store.save_targets(
        [
            ConfiguredServerTarget(
                server_id="server-a",
                label="Server A",
                base_url="https://server-a.example/api",
                auth_reference="legacy:tldw_api",
                last_known_server_label="server-a.example",
            )
        ]
    )

    assert (tmp_path / "server_targets.json").exists()
    assert not (tmp_path / "server_targets.json.tmp").exists()


def test_target_store_updates_status_metadata_without_overwriting_auth_reference(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")
    target = ConfiguredServerTarget(
        server_id="server-a",
        label="Server A",
        base_url="https://server-a.example/api",
        auth_reference="legacy:tldw_api",
    )
    store.save_targets([target])

    updated = store.update_target_status(
        "server-a",
        last_known_reachability="reachable",
        last_known_auth_state="authenticated",
        last_connected_at=datetime(2026, 4, 22, 10, 30, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 22, 10, 31, tzinfo=timezone.utc),
    )

    assert updated.auth_reference == "legacy:tldw_api"
    assert updated.last_known_reachability == "reachable"
    assert updated.last_known_auth_state == "authenticated"
    assert updated.last_connected_at == datetime(2026, 4, 22, 10, 30, tzinfo=timezone.utc)
    assert updated.updated_at == datetime(2026, 4, 22, 10, 31, tzinfo=timezone.utc)

    restored = store.list_targets()[0]
    assert restored.last_known_reachability == "reachable"
    assert restored.last_known_auth_state == "authenticated"
    assert restored.last_connected_at == datetime(2026, 4, 22, 10, 30, tzinfo=timezone.utc)
    assert restored.updated_at == datetime(2026, 4, 22, 10, 31, tzinfo=timezone.utc)


def test_target_store_normalizes_invalid_status_values_before_persisting(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")
    target = ConfiguredServerTarget(
        server_id="server-a",
        label="Server A",
        base_url="https://server-a.example/api",
    )
    store.save_targets([target])

    updated = store.update_target_status(
        "server-a",
        last_known_reachability="BROKEN",
        last_known_auth_state="INVALID",
    )

    assert updated.last_known_reachability is None
    assert updated.last_known_auth_state is None

    restored = store.list_targets()[0]
    assert restored.last_known_reachability is None
    assert restored.last_known_auth_state is None


def test_target_store_resolves_active_target_by_default_target_and_explicit_server_id(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")
    default_target = ConfiguredServerTarget(
        server_id="server-default",
        label="Default",
        base_url="https://default.example/api",
        is_default=True,
    )
    other_target = ConfiguredServerTarget(
        server_id="server-secondary",
        label="Secondary",
        base_url="https://secondary.example/api",
    )
    store.save_targets([default_target, other_target])

    assert store.resolve_active_target().server_id == "server-default"
    assert store.resolve_active_target("server-secondary").server_id == "server-secondary"
