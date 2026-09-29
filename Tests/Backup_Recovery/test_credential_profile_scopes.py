"""Recovery must use the same credential namespaces as the application."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_credentials import inventory
from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.Backup_Recovery.archive_models import (
    ArchiveManifest,
    Payload,
    RecoveryReport,
)
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
from tldw_chatbook.Backup_Recovery.profile_paths import default_config_path
from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan, RetainedConfig
from tldw_chatbook.Backup_Recovery.staging import _credential_destinations
from tldw_chatbook.runtime_policy.server_context import RuntimeServerContextProvider
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore


def provider(store, profile):
    return RuntimeServerContextProvider(
        runtime_context=SimpleNamespace(),
        target_store=SimpleNamespace(),
        credential_store=store,
        app_config={},
        credential_profile_id=profile,
    )


def staged_targets(stage, filename="targets.json"):
    path = stage / filename
    path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "server_id": "peer",
                        "base_url": "https://example.invalid",
                        "auth_reference": "keyring:api_key",
                    }
                ]
            }
        )
    )
    path.chmod(0o600)
    return path


def credential_document(stage):
    config = stage / "config.toml"
    config.write_text('[general]\nusers_name="source"\n')
    config.chmod(0o600)
    files = tuple(
        Payload(
            logical_id=f"profile:p:{owner}",
            root_id="profile:p:root",
            parent_id="profile:p:root",
            relative_path=filename,
            owner_id=owner,
            payload=filename,
            size=(stage / filename).stat().st_size,
            sha256=hashlib.sha256((stage / filename).read_bytes()).hexdigest(),
        )
        for owner, filename in (
            ("config", "config.toml"),
            ("mcp.targets", "targets.json"),
        )
    )
    return ArchiveManifest(
        format_version=1,
        producer_version="test",
        captured_at="test",
        profile_ids=("p",),
        owners=(),
        directories=(),
        files=files,
        dependency_groups=(),
        consistency="coherent",
        exclusions=(),
        credential_policy="include",
        required_capabilities=(),
        report=RecoveryReport(version=1, lines=()),
        relocations=(),
    )


@pytest.mark.parametrize("profile", [None, "/source/config.toml"])
def test_capture_uses_application_profile_scope(tmp_path, monkeypatch, profile):
    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    app = provider(store, profile)
    app.store_scoped_credential("peer", "api_key", "owned-profile-value")
    if profile:
        provider(store, None).store_scoped_credential("peer", "api_key", "other-value")
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    targets = stage / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "server_id": "peer",
                        "base_url": "https://example.invalid",
                        "auth_reference": "keyring:api_key",
                    }
                ]
            }
        )
    )
    targets.chmod(0o600)
    issues = credentials.process_credentials(
        stage,
        inventory(targets, "mcp.targets"),
        mode="include",
        encrypted=True,
        profile_scopes={"p": profile},
    )
    assert not issues
    record = credentials._material(stage)[0]
    assert record["value"] == app._get_credential_secret("peer", "api_key")


def test_replacement_resolves_in_destination_profile_preserving_source(
    tmp_path, monkeypatch
):
    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    source = provider(store, "/source/config.toml")
    destination = provider(store, "/destination/config.toml")
    source.store_scoped_credential("peer", "api_key", "incoming-value")
    destination.store_scoped_credential("peer", "api_key", "destination-value")
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    path = stage / "targets.json"
    path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "server_id": "peer",
                        "base_url": "https://example.invalid",
                        "auth_reference": "keyring:api_key",
                    }
                ]
            }
        )
    )
    path.chmod(0o600)
    credentials.process_credentials(
        stage,
        inventory(path, "mcp.targets"),
        mode="include",
        encrypted=True,
        profile_scopes={"p": "/source/config.toml"},
    )
    scopes = credentials.plan_credential_scopes(
        stage,
        fresh=True,
        profile_scopes={"targets.json": "/destination/config.toml"},
    )
    record = credentials._material(stage)[0]
    plan = json.loads(scopes[record["id"]])
    credentials.apply_replacement_credential(record, plan)
    assert (
        destination._get_credential_secret("peer", plan["purpose"]) == "incoming-value"
    )
    assert destination._get_credential_secret("peer", "api_key") == "destination-value"
    assert source._get_credential_secret("peer", "api_key") == "incoming-value"
    rollback_record = {
        **record,
        "profile_id": "/destination/config.toml",
        "value": "destination-value",
    }
    reverse = credentials.plan_rollback_credential(rollback_record)
    credentials.verify_rollback_credential(rollback_record, reverse, apply=True)
    assert (
        destination._get_credential_secret("peer", reverse["purpose"])
        == "destination-value"
    )


def test_mixed_default_and_retargeted_inventory_captures_each_profile(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    custom_config = tmp_path / "custom" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(custom_config))
    source_inventory = Inventory(
        (
            StorageItem(
                "config",
                "profile:default:config",
                default_config_path(),
                "included",
                (),
            ),
            StorageItem(
                "config", "profile:custom:config", custom_config, "included", ()
            ),
        ),
        True,
        "source",
        (),
    )
    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    provider(store, None).store_scoped_credential("peer", "api_key", "default-value")
    provider(store, str(custom_config)).store_scoped_credential(
        "peer", "api_key", "custom-value"
    )
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    targets_inventory = Inventory(
        tuple(
            StorageItem(
                "mcp.targets",
                f"profile:{profile}:mcp.targets",
                staged_targets(stage, f"{profile}-targets.json"),
                "included",
                (),
            )
            for profile in ("default", "custom")
        ),
        True,
        "source",
        (),
    )

    issues = credentials.process_credentials(
        stage,
        targets_inventory,
        mode="include",
        encrypted=True,
        profile_scopes=credentials.profile_credential_scopes(source_inventory),
    )

    assert not issues
    assert {row["file"]: row["value"] for row in credentials._material(stage)} == {
        "default-targets.json": "default-value",
        "custom-targets.json": "custom-value",
    }


def test_scoped_source_resolves_in_default_destination(tmp_path, monkeypatch):
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    source_config = str(tmp_path / "source" / "config.toml")
    source = provider(store, source_config)
    destination = provider(store, None)
    source.store_scoped_credential("peer", "api_key", "incoming-value")
    destination.store_scoped_credential("peer", "api_key", "destination-value")
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    path = staged_targets(stage)
    assert not credentials.process_credentials(
        stage,
        inventory(path, "mcp.targets"),
        mode="include",
        encrypted=True,
        profile_scopes={"p": source_config},
    )
    restore_plan = RestorePlan(
        archive_digest="archive",
        mode="replace",
        restore=(
            ("profile:p:config", default_config_path()),
            ("profile:p:mcp.targets", tmp_path / "destination-targets.json"),
        ),
        retire=(),
        preserve=(),
        target_fingerprint="target",
    )
    destinations = _credential_destinations(credential_document(stage), restore_plan)
    scopes = credentials.plan_credential_scopes(
        stage,
        fresh=True,
        profile_scopes=destinations,
    )
    record = credentials._material(stage)[0]
    plan = json.loads(scopes[record["id"]])

    credentials.apply_replacement_credential(record, plan)

    assert (
        destination._get_credential_secret("peer", plan["purpose"]) == "incoming-value"
    )
    assert destination._get_credential_secret("peer", "api_key") == "destination-value"
    assert source._get_credential_secret("peer", "api_key") == "incoming-value"


def test_retained_config_selects_destination_credential_profile(tmp_path, monkeypatch):
    retained_config = tmp_path / "retained" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(retained_config))
    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    source_config = str(tmp_path / "source" / "config.toml")
    source = provider(store, source_config)
    destination = provider(store, str(retained_config))
    source.store_scoped_credential("peer", "api_key", "incoming-value")
    destination.store_scoped_credential("peer", "api_key", "destination-value")
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    path = staged_targets(stage)
    assert not credentials.process_credentials(
        stage,
        inventory(path, "mcp.targets"),
        mode="include",
        encrypted=True,
        profile_scopes={"p": source_config},
    )
    restore_plan = RestorePlan(
        archive_digest="archive",
        mode="replace",
        restore=(("profile:p:mcp.targets", tmp_path / "destination-targets.json"),),
        retire=(),
        preserve=(),
        target_fingerprint="target",
        retained_configs=(
            RetainedConfig(
                "profile:p:config",
                "profile:local:config",
                retained_config,
                "observed",
            ),
        ),
    )
    destinations = _credential_destinations(credential_document(stage), restore_plan)
    scopes = credentials.plan_credential_scopes(
        stage,
        fresh=True,
        profile_scopes=destinations,
    )
    record = credentials._material(stage)[0]
    plan = json.loads(scopes[record["id"]])

    credentials.apply_replacement_credential(record, plan)

    assert (
        destination._get_credential_secret("peer", plan["purpose"]) == "incoming-value"
    )
    assert destination._get_credential_secret("peer", "api_key") == "destination-value"
    assert source._get_credential_secret("peer", "api_key") == "incoming-value"
