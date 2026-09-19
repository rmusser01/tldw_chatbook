"""Final capture must use the source mappings reviewed by the user."""

import pytest

from tldw_chatbook.Backup_Recovery.capture import compare_scope
from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem


def test_manifest_accepts_payload_staged_with_windows_path_spelling(tmp_path):
    from pathlib import Path, PureWindowsPath
    from threading import Event

    from tldw_chatbook.Backup_Recovery.capture import _manifest_for
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    class WindowsSpelling(type(Path())):
        def relative_to(self, *other, **kwargs):
            return PureWindowsPath(super().relative_to(*other, **kwargs))

    source = tmp_path / "source"
    source.write_bytes(b"synthetic payload")
    source.chmod(0o600)
    (tmp_path / "payload").mkdir()
    staged = WindowsSpelling(tmp_path / "payload" / "content")
    staged.write_bytes(source.read_bytes())
    item = StorageItem("config", "profile:test:config", source, "included", ())
    result = _manifest_for(
        Inventory((item,), True, "scope", ()),
        [(item, staged)],
        {},
        {
            "root": tmp_path,
            "versions": {},
            "cancel": Event(),
            "limits": ArchiveLimits(),
            "mode": "exclude",
            "encrypted": False,
        },
        (),
    )
    assert json.loads(result)["files"][0]["payload"] == "payload/content"


def test_capacity_sums_requirements_on_same_actual_volume(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import space

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _: SimpleNamespace(free=space._MARGIN + 100)
    )
    space.require_capacity({tmp_path / "capture": 60})
    with pytest.raises(ValueError, match="insufficient_space"):
        space.require_capacity({tmp_path / "capture": 60, tmp_path / "output": 60})


def test_changed_source_mapping_invalidates_preview(tmp_path):
    old = Inventory(
        (StorageItem("db", "main", tmp_path / "old", "included", ()),),
        True,
        "old-scope",
        (),
    )
    new = Inventory(
        (StorageItem("db", "main", tmp_path / "new", "included", ()),),
        True,
        "new-scope",
        (),
    )
    assert "scope_changed" in compare_scope(old, new)


def test_growth_within_reviewed_owner_scope_does_not_change_mapping(tmp_path):
    old = Inventory((), True, "scope", ())
    grown = Inventory(
        (StorageItem("assets", "new", tmp_path / "new", "included", ()),),
        True,
        "scope",
        (),
    )
    assert compare_scope(old, grown) == ()


import json
from threading import Event


def test_real_config_capture_is_sanitized_and_source_is_unchanged(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery import capture as module
    from tldw_chatbook.Backup_Recovery import owner_registry
    from tldw_chatbook.Backup_Recovery.config_adapter import config_adapter
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries

    source = tmp_path / "config.toml"
    source.write_text('[API]\nopenai_api_key="synthetic-source-secret"\n')
    source.chmod(0o600)
    before = source.read_bytes()
    item = StorageItem("config", "profile:test:config", source, "included", ())
    final = classify_entries((item,))
    assert final.complete
    monkeypatch.setattr(
        module, "discover", lambda *args, **kwargs: final, raising=False
    )
    monkeypatch.setattr(owner_registry, "_adapters", {"config": config_adapter()})
    authority = application_authority(tmp_path, source, monkeypatch)
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        result = module._capture_under_maintenance(
            session,
            (source,),
            final.scope_digest,
            tmp_path / "new.tldw-backup.zip",
            options={"staging_parent": tmp_path},
            cancel=Event(),
        )
    assert source.read_bytes() == before
    doc = json.loads(result.manifest_bytes)
    assert len(doc["files"]) == 1
    assert (
        b"synthetic-source-secret"
        not in (result.root / doc["files"][0]["payload"]).read_bytes()
    )
    assert result.inventory.items[0].path == source


def test_real_discover_asset_growth_preserves_reviewed_scope(tmp_path):
    from Tests.Backup_Recovery.test_inventory import config_file
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.models import DiscoverySelections

    source = config_file(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (external / "first.txt").write_text("first")
    selections = DiscoverySelections(external_roots=(external,))
    before = discover((source,), selections=selections)
    (external / "second.txt").write_text("growth")
    after = discover((source,), selections=selections)
    assert len(after.items) > len(before.items)
    assert after.scope_digest == before.scope_digest


@pytest.mark.parametrize(
    "change", ["root", "alias", "exclusion", "unknown", "missing_config"]
)
def test_real_discover_authority_changes_invalidate_scope(tmp_path, change):
    from Tests.Backup_Recovery.test_inventory import config_file
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.models import DiscoverySelections

    source = config_file(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (external / "first.txt").write_text("first")
    selections = DiscoverySelections(external_roots=(external,))
    before = discover((source,), selections=selections)
    if change == "root":
        other = tmp_path / "other"
        other.mkdir()
        selections = DiscoverySelections(external_roots=(other,))
    elif change == "alias":
        (external / "second.txt").hardlink_to(external / "first.txt")
    elif change == "exclusion":
        selections = DiscoverySelections()
    elif change == "unknown":
        root = tmp_path / "data/Ada"
        root.mkdir(parents=True, exist_ok=True)
        (root / "unknown.durable").write_text("new owner")
    else:
        source.unlink()
    assert (
        discover((source,), selections=selections).scope_digest != before.scope_digest
    )


@pytest.fixture
def native_config_capture(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery import capture as module
    from tldw_chatbook.Backup_Recovery import owner_registry
    from tldw_chatbook.Backup_Recovery.config_adapter import config_adapter
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries

    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    config = source / "config.toml"
    config.write_text('[general]\nusers_name="fixture"\n')
    config.chmod(0o600)
    item = StorageItem("config", "profile:test:config", config, "included", ())
    state = [classify_entries((item,))]
    monkeypatch.setattr(module, "discover", lambda *args, **kwargs: state[0])
    monkeypatch.setattr(owner_registry, "_adapters", {"config": config_adapter()})
    authority = application_authority(tmp_path, source, monkeypatch)
    return module, authority, state, config


@pytest.mark.parametrize("failure", ["scope", "budget", "cancel", "dependency"])
def test_native_capture_refuses_changed_or_unavailable_boundary(
    tmp_path, native_config_capture, failure
):
    from dataclasses import replace

    module, authority, state, config = native_config_capture
    approved = state[0].scope_digest
    cancel = Event()
    options = {"staging_parent": tmp_path}
    if failure == "scope":
        approved = "stale"
    elif failure == "budget":
        options["byte_budget"] = 1
    elif failure == "cancel":
        cancel.set()
    else:
        state[0] = replace(state[0], complete=False, issues=("dependency_unavailable",))
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        pytest.raises((ValueError, InterruptedError)),
    ):
        module._capture_under_maintenance(
            session,
            (config,),
            approved,
            tmp_path / "new.tldw-backup.zip",
            options=options,
            cancel=cancel,
        )
    assert not list(tmp_path.glob("capture-*"))


def test_explicit_partial_keeps_coverage_incomplete(tmp_path, native_config_capture):
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries

    module, authority, state, config = native_config_capture
    unknown = StorageItem("unknown", "profile:test:unknown", None, "unsupported", ())
    state[0] = classify_entries((*state[0].items, unknown))
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        result = module._capture_under_maintenance(
            session,
            (config,),
            state[0].scope_digest,
            tmp_path / "new.tldw-backup.zip",
            options={"staging_parent": tmp_path, "allow_partial": True},
            cancel=Event(),
        )
    assert not result.inventory.complete
    doc = json.loads(result.manifest_bytes)
    assert doc["consistency"] == "partial"
    assert {"logical_id": unknown.logical_id, "reason": "unsupported"} in doc[
        "exclusions"
    ]


def test_native_shared_source_has_distinct_payloads_and_complete_group(
    tmp_path, native_config_capture, monkeypatch
):
    from dataclasses import replace

    from tldw_chatbook.Backup_Recovery.config_adapter import _Config
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries

    module, authority, state, config = native_config_capture
    first = replace(state[0].items[0], shared_group="shared-config")
    second = replace(first, logical_id="profile:other:config")
    state[0] = classify_entries((first, second))
    calls = []
    original = _Config.capture

    def capture(self, item, target, cancel):
        calls.append(item.logical_id)
        return original(self, item, target, cancel)

    monkeypatch.setattr(_Config, "capture", capture)
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        result = module._capture_under_maintenance(
            session,
            (config,),
            state[0].scope_digest,
            tmp_path / "new.tldw-backup.zip",
            options={"staging_parent": tmp_path},
            cancel=Event(),
        )
    doc = json.loads(result.manifest_bytes)
    assert calls == [first.logical_id]
    assert len({item["payload"] for item in doc["files"]}) == 2
    assert len({item["sha256"] for item in doc["files"]}) == 1
    assert any(
        set(group["members"]) == {first.logical_id, second.logical_id}
        and group["complete"]
        for group in doc["dependency_groups"]
    )


def test_native_sqlite_capture_records_observed_schema_and_preserves_source(
    tmp_path, native_config_capture, monkeypatch
):
    from Tests.Backup_Recovery.test_sqlite_validation import research_candidate
    from tldw_chatbook.Backup_Recovery import owner_registry
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries

    module, authority, state, config = native_config_capture
    owner, source = research_candidate(config.parent)
    source.chmod(0o600)
    before = source.read_bytes()
    item = StorageItem(
        owner.owner_id, "profile:test:" + owner.owner_id, source, "included", ()
    )
    state[0] = classify_entries((item,))
    monkeypatch.setattr(owner_registry, "_adapters", {owner.owner_id: owner})
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        result = module._capture_under_maintenance(
            session,
            (config,),
            state[0].scope_digest,
            tmp_path / "new.tldw-backup.zip",
            options={"staging_parent": tmp_path},
            cancel=Event(),
        )
    doc = json.loads(result.manifest_bytes)
    assert doc["owners"][0]["schema_version"] == 0
    assert source.read_bytes() == before


def test_native_encrypted_capture_material_remains_available(
    tmp_path, native_config_capture, monkeypatch
):
    from types import SimpleNamespace

    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import credentials

    store = SimpleNamespace(_keyring=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    module, authority, state, config = native_config_capture
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        result = module._capture_under_maintenance(
            session,
            (config,),
            state[0].scope_digest,
            tmp_path / "new.tldw-backup",
            options={
                "staging_parent": tmp_path,
                "credential_mode": "include",
                "encrypted": True,
            },
            cancel=Event(),
        )
    doc = json.loads(result.manifest_bytes)
    material = next(
        item for item in doc["files"] if item["owner_id"] == "recovery.credentials"
    )
    assert (result.root / material["payload"]).read_bytes() == (
        result.root / "credential-recovery.json"
    ).read_bytes()


def test_capture_options_reject_unknown_before_source_work():
    from tldw_chatbook.Backup_Recovery.capture import _capture_options

    with pytest.raises(ValueError, match="unknown_capture_option"):
        _capture_options({"unreviewed": True})


def test_capture_options_preserve_original_selection_contract(tmp_path):
    from tldw_chatbook.Backup_Recovery.capture import _capture_options
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    limits = ArchiveLimits(expanded_bytes=100)
    original = {
        "external_roots": (tmp_path,),
        "model_ids": ("model",),
        "temporary_media": True,
        "diagnostics": True,
        "allow_partial": True,
        "limits": limits,
        "byte_budget": 50,
    }
    settings, selections, actual_limits, budget = _capture_options(original)
    original["allow_partial"] = False
    assert settings["allow_partial"] is True
    assert settings["credential_mode"] == "exclude"
    assert settings["encrypted"] is False
    assert selections.external_roots == (tmp_path,)
    assert selections.model_ids == ("model",)
    assert selections.temporary_media and selections.diagnostics
    assert actual_limits is limits and budget == 50


@pytest.fixture
def discovered_sqlite_source(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.test_inventory import config_file
    from Tests.Backup_Recovery.test_sqlite_validation import research_candidate
    from tldw_chatbook.Backup_Recovery import owner_registry

    root = tmp_path / "data" / "Ada"
    root.mkdir(parents=True)
    owner, source = research_candidate(root)
    config = config_file(tmp_path, extra=f'[database]\nresearch_db_path="{source}"\n')
    monkeypatch.setattr(owner_registry, "_adapters", {owner.owner_id: owner})
    return config, source


def test_real_sqlite_wal_appearance_is_stable_excluded_scope(discovered_sqlite_source):
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery.inventory import discover

    config, source = discovered_sqlite_source
    before = discover((config,))
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("UPDATE research_runs SET query='ordinary growth'")
        connection.commit()
        after = discover((config,))
        sidecars = [item for item in after.items if item.owner == "sqlite.transient"]
        assert {str(item.path) for item in sidecars} == {
            str(source) + "-wal",
            str(source) + "-shm",
        }
        assert all(item.status == "intentionally_excluded" for item in sidecars)
        assert after.scope_digest == before.scope_digest


@pytest.mark.parametrize("kind", ["lookalike", "symlink", "hardlink", "directory"])
def test_unowned_or_unsafe_sqlite_sidecars_block_capture(
    discovered_sqlite_source, kind
):
    from tldw_chatbook.Backup_Recovery.inventory import discover

    config, source = discovered_sqlite_source
    before = discover((config,))
    path = source.with_name(source.name + "-wal")
    if kind == "lookalike":
        path = source.parent / "unowned.db-wal"
        path.write_bytes(b"unowned")
    elif kind == "symlink":
        path.symlink_to(config)
    elif kind == "hardlink":
        path.hardlink_to(config)
    else:
        path.mkdir()
    after = discover((config,))
    assert any(
        item.path == path and item.status == "unsupported" for item in after.items
    )
    assert after.scope_digest != before.scope_digest


def test_mixed_recovered_owner_does_not_claim_payload_sidecars(tmp_path):
    from tldw_chatbook.Backup_Recovery.inventory import _sqlite_sidecars
    from tldw_chatbook.Backup_Recovery.recovered_media import recovery_adapters

    payload = tmp_path / "asset.payload"
    payload.write_bytes(b"retained media")
    item = StorageItem(
        "recovered.media", "profile:test:recovered.media:asset", payload, "included", ()
    )
    assert _sqlite_sidecars((item,), recovery_adapters()) == ()
