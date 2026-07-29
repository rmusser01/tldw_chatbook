from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from threading import Barrier, BrokenBarrierError
from uuid import UUID

import pytest

from tldw_chatbook.MCP.server_target_store import (
    AuthorityScopeUnavailable,
    ConfiguredServerTargetStore,
)
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget

_CANONICAL_SCOPE = "123e4567-e89b-42d3-a456-426614174000"


def _assert_canonical_uuid4(value: str) -> None:
    parsed = UUID(value)
    assert parsed.version == 4
    assert str(parsed) == value


def _write_target_payload(path, targets: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"targets": targets}), encoding="utf-8")


def _legacy_target_payload(
    *,
    server_id: str = "server-a",
    authority_scope_id: object = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "server_id": server_id,
        "label": f"Label for {server_id}",
        "base_url": f"https://{server_id}.example/api",
    }
    if authority_scope_id is not None:
        payload["authority_scope_id"] = authority_scope_id
    return payload


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


def test_bootstrap_persists_a_canonical_uuid4_authority_scope(tmp_path):
    path = tmp_path / "server_targets.json"
    store = ConfiguredServerTargetStore(path)

    imported = store.bootstrap_from_legacy_config(
        {"tldw_api": {"base_url": "https://example.com/api"}}
    )

    assert imported is True
    target = store.list_targets()[0]
    assert target.authority_scope_id is not None
    _assert_canonical_uuid4(target.authority_scope_id)
    assert json.loads(path.read_text(encoding="utf-8"))["targets"][0][
        "authority_scope_id"
    ] == target.authority_scope_id


def test_authority_scope_round_trips_json_without_appearing_in_target_repr(tmp_path):
    path = tmp_path / "server_targets.json"
    target = ConfiguredServerTarget(
        server_id="server-a",
        label="Server A",
        base_url="https://server-a.example/api",
        authority_scope_id=_CANONICAL_SCOPE,
    )
    store = ConfiguredServerTargetStore(path)

    store.save_targets([target])
    restored = store.list_targets()[0]

    assert restored.authority_scope_id == _CANONICAL_SCOPE
    assert _CANONICAL_SCOPE not in repr(target)
    assert _CANONICAL_SCOPE not in repr(restored)
    assert "authority_scope_id" not in repr(restored)


@pytest.mark.parametrize("scope_payload", [{}, {"authority_scope_id": None}])
def test_plain_deserialization_keeps_legacy_missing_scope_available_for_routing(
    scope_payload,
):
    target = ConfiguredServerTarget.from_dict(
        {
            "server_id": "legacy-server",
            "label": "Legacy",
            "base_url": "https://legacy.example/api",
            **scope_payload,
        }
    )

    assert target.server_id == "legacy-server"
    assert target.authority_scope_id is None


def test_ensure_authority_scope_durably_upgrades_then_reloads_legacy_target(
    tmp_path, monkeypatch
):
    path = tmp_path / "server_targets.json"
    _write_target_payload(path, [_legacy_target_payload()])
    store = ConfiguredServerTargetStore(path)
    events: list[str] = []
    real_load = store.load
    real_save = store.save_targets

    def observed_load():
        events.append("load")
        return real_load()

    def observed_save(targets):
        events.append("save")
        real_save(targets)

    monkeypatch.setattr(store, "load", observed_load)
    monkeypatch.setattr(store, "save_targets", observed_save)

    scope = store.ensure_authority_scope_id("server-a")

    _assert_canonical_uuid4(scope)
    assert events == ["load", "save", "load"]
    assert store.get_target("server-a").authority_scope_id == scope
    assert not path.with_suffix(".json.tmp").exists()


def test_ensure_authority_scope_serializes_same_store_upgrade_calls(tmp_path):
    path = tmp_path / "server_targets.json"
    _write_target_payload(path, [_legacy_target_payload()])
    store = ConfiguredServerTargetStore(path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        scopes = list(
            executor.map(store.ensure_authority_scope_id, ["server-a"] * 16)
        )

    assert len(set(scopes)) == 1
    assert store.get_target("server-a").authority_scope_id == scopes[0]
    _assert_canonical_uuid4(scopes[0])


def test_ensure_authority_scope_serializes_same_process_store_instances(
    tmp_path, monkeypatch
):
    path = tmp_path / "server_targets.json"
    _write_target_payload(path, [_legacy_target_payload()])
    stores = [ConfiguredServerTargetStore(path), ConfiguredServerTargetStore(path)]
    first_read_barrier = Barrier(2)

    def synchronize_first_read(real_read_payload):
        is_first_read = True

        def synchronized_read():
            nonlocal is_first_read
            if not is_first_read:
                return real_read_payload()
            is_first_read = False
            payload = real_read_payload()
            try:
                first_read_barrier.wait(timeout=0.2)
            except BrokenBarrierError:
                pass
            return payload

        return synchronized_read

    for store in stores:
        monkeypatch.setattr(
            store,
            "_read_payload",
            synchronize_first_read(store._read_payload),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        scopes = list(
            executor.map(
                lambda store: store.ensure_authority_scope_id("server-a"),
                stores,
            )
        )

    assert len(set(scopes)) == 1
    assert ConfiguredServerTargetStore(path).get_target(
        "server-a"
    ).authority_scope_id == scopes[0]


def test_authority_scope_survives_mutable_target_and_status_updates(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")
    target = ConfiguredServerTarget(
        server_id="stable-routing-id",
        label="Old label",
        base_url="https://old.example/api",
        auth_mode="api_key",
        auth_reference="old:key",
        authority_scope_id=_CANONICAL_SCOPE,
    )
    store.save_targets([target])

    mutable_update = replace(
        target,
        label="New label",
        base_url="https://new.example/api",
        auth_mode="bearer",
        auth_reference="new:key",
    )
    store.save_targets([mutable_update])
    status_update = store.update_target_status(
        "stable-routing-id",
        last_known_reachability="reachable",
        last_known_auth_state="authenticated",
    )

    assert status_update.authority_scope_id == _CANONICAL_SCOPE
    assert store.get_target("stable-routing-id").authority_scope_id == _CANONICAL_SCOPE


def test_legacy_config_upsert_preserves_existing_authority_scope(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")
    store.save_targets(
        [
            ConfiguredServerTarget(
                server_id="https://example.com/api",
                label="Old label",
                base_url="https://example.com/api",
                auth_mode="api_key",
                authority_scope_id=_CANONICAL_SCOPE,
            )
        ]
    )

    synced = store.upsert_legacy_config_target(
        {
            "tldw_api": {
                "base_url": "https://example.com/api/",
                "bearer_token": "rotated-token",
            }
        }
    )

    assert synced is not None
    assert synced.auth_mode == "bearer"
    assert synced.authority_scope_id == _CANONICAL_SCOPE
    assert store.get_target(synced.server_id).authority_scope_id == _CANONICAL_SCOPE


def test_new_legacy_config_upsert_persists_a_fresh_authority_scope(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")

    synced = store.upsert_legacy_config_target(
        {"tldw_api": {"base_url": "https://new.example/api"}}
    )

    assert synced is not None
    assert synced.authority_scope_id is not None
    _assert_canonical_uuid4(synced.authority_scope_id)
    assert store.get_target(synced.server_id).authority_scope_id == (
        synced.authority_scope_id
    )


@pytest.mark.parametrize(
    "malformed_scope",
    [
        "not-a-uuid",
        "123e4567-e89b-42d3-a456-426614174000".upper(),
        "123e4567-e89b-12d3-a456-426614174000",
    ],
)
def test_malformed_scope_fails_authority_only_and_remains_routable(
    tmp_path, malformed_scope
):
    path = tmp_path / "server_targets.json"
    _write_target_payload(
        path,
        [
            _legacy_target_payload(
                server_id="malformed-server",
                authority_scope_id=malformed_scope,
            )
        ],
    )
    store = ConfiguredServerTargetStore(path)

    routed = store.resolve_active_target("malformed-server")

    assert routed is not None
    assert routed.authority_scope_id == malformed_scope
    with pytest.raises(AuthorityScopeUnavailable) as exc_info:
        store.ensure_authority_scope_id("malformed-server")
    assert str(exc_info.value) == "Configured server authority scope is unavailable."
    assert malformed_scope not in str(exc_info.value)


def test_duplicate_scopes_fail_authority_only_and_remain_routable(tmp_path):
    path = tmp_path / "server_targets.json"
    _write_target_payload(
        path,
        [
            _legacy_target_payload(
                server_id="server-a", authority_scope_id=_CANONICAL_SCOPE
            ),
            _legacy_target_payload(
                server_id="server-b", authority_scope_id=_CANONICAL_SCOPE
            ),
        ],
    )
    store = ConfiguredServerTargetStore(path)

    assert store.get_target("server-a") is not None
    assert store.get_target("server-b") is not None
    with pytest.raises(AuthorityScopeUnavailable) as exc_info:
        store.ensure_authority_scope_id("server-a")
    assert str(exc_info.value) == "Configured server authority scope is unavailable."
    assert _CANONICAL_SCOPE not in str(exc_info.value)


def test_scope_persistence_failure_never_returns_an_ephemeral_scope(
    tmp_path, monkeypatch
):
    path = tmp_path / "server_targets.json"
    _write_target_payload(path, [_legacy_target_payload()])
    store = ConfiguredServerTargetStore(path)

    def fail_save(_targets):
        raise OSError("simulated write failure")

    monkeypatch.setattr(store, "save_targets", fail_save)

    with pytest.raises(AuthorityScopeUnavailable) as exc_info:
        store.ensure_authority_scope_id("server-a")

    assert str(exc_info.value) == "Configured server authority scope is unavailable."
    assert store.get_target("server-a").authority_scope_id is None


def test_scope_reload_failure_never_returns_an_unverified_scope(
    tmp_path, monkeypatch
):
    path = tmp_path / "server_targets.json"
    _write_target_payload(path, [_legacy_target_payload()])
    store = ConfiguredServerTargetStore(path)
    real_load = store.load
    load_count = 0

    def fail_reload():
        nonlocal load_count
        load_count += 1
        if load_count == 1:
            return real_load()
        raise OSError("simulated reload failure")

    monkeypatch.setattr(store, "load", fail_reload)

    with pytest.raises(AuthorityScopeUnavailable) as exc_info:
        store.ensure_authority_scope_id("server-a")

    assert str(exc_info.value) == "Configured server authority scope is unavailable."
    persisted_scope = json.loads(path.read_text(encoding="utf-8"))["targets"][0][
        "authority_scope_id"
    ]
    _assert_canonical_uuid4(persisted_scope)


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


def test_upsert_legacy_config_target_adds_current_configured_server_as_default(
    tmp_path,
):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")
    store.save_targets(
        [
            ConfiguredServerTarget(
                server_id="https://old.example/api",
                label="Old",
                base_url="https://old.example/api",
                auth_reference="legacy:tldw_api",
                is_default=True,
            ),
            ConfiguredServerTarget(
                server_id="manual-target",
                label="Manual",
                base_url="https://manual.example/api",
                auth_reference="manual:keychain",
            ),
        ]
    )

    synced = store.upsert_legacy_config_target(
        {
            "tldw_api": {
                "base_url": "https://New.EXAMPLE:9443/api/",
                "api_key": "new-secret",
            }
        }
    )

    assert synced is not None
    assert synced.server_id == "https://new.example:9443/api"
    assert synced.is_default is True

    targets = store.list_targets()
    assert [target.server_id for target in targets] == [
        "https://old.example/api",
        "manual-target",
        "https://new.example:9443/api",
    ]
    assert [target.is_default for target in targets] == [False, False, True]
    assert store.resolve_active_target().server_id == "https://new.example:9443/api"

    raw_payload = (tmp_path / "server_targets.json").read_text(encoding="utf-8")
    assert "new-secret" not in raw_payload


def test_upsert_legacy_config_target_preserves_existing_status_metadata(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")
    connected_at = datetime(2026, 4, 22, 10, 30, tzinfo=timezone.utc)
    updated_at = datetime(2026, 4, 22, 10, 31, tzinfo=timezone.utc)
    store.save_targets(
        [
            ConfiguredServerTarget(
                server_id="https://example.com/api",
                label="Example",
                base_url="https://example.com/api",
                auth_reference="legacy:tldw_api",
                is_default=False,
                last_known_server_label="Example Server",
                last_known_reachability="reachable",
                last_known_auth_state="authenticated",
                last_connected_at=connected_at,
                updated_at=updated_at,
            )
        ]
    )

    synced = store.upsert_legacy_config_target(
        {
            "tldw_api": {
                "base_url": "https://example.com/api/",
                "bearer_token": "secret-token",
            }
        }
    )

    assert synced is not None
    assert synced.auth_mode == "bearer"
    assert synced.auth_reference == "legacy:tldw_api"
    assert synced.is_default is True
    assert synced.last_known_server_label == "Example Server"
    assert synced.last_known_reachability == "reachable"
    assert synced.last_known_auth_state == "authenticated"
    assert synced.last_connected_at == connected_at
    assert synced.updated_at == updated_at


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


def test_target_store_updates_status_metadata_without_overwriting_auth_reference(
    tmp_path,
):
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
    assert updated.last_connected_at == datetime(
        2026, 4, 22, 10, 30, tzinfo=timezone.utc
    )
    assert updated.updated_at == datetime(2026, 4, 22, 10, 31, tzinfo=timezone.utc)

    restored = store.list_targets()[0]
    assert restored.last_known_reachability == "reachable"
    assert restored.last_known_auth_state == "authenticated"
    assert restored.last_connected_at == datetime(
        2026, 4, 22, 10, 30, tzinfo=timezone.utc
    )
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


def test_target_store_normalizes_invalid_status_values_on_direct_save(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "server_targets.json")
    target = ConfiguredServerTarget(
        server_id="server-a",
        label="Server A",
        base_url="https://server-a.example/api",
        last_known_reachability="BROKEN",
        last_known_auth_state="INVALID",
    )

    store.save_targets([target])
    raw_payload = (tmp_path / "server_targets.json").read_text(encoding="utf-8")
    restored = store.list_targets()[0]

    assert "BROKEN" not in raw_payload
    assert "INVALID" not in raw_payload
    assert restored.last_known_reachability is None
    assert restored.last_known_auth_state is None


def test_target_store_resolves_active_target_by_default_target_and_explicit_server_id(
    tmp_path,
):
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
    assert (
        store.resolve_active_target("server-secondary").server_id == "server-secondary"
    )
