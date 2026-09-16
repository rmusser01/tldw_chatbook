"""Current startup scaffolding never grants coverage to durable payloads."""

import hashlib
from dataclasses import replace

import pytest

from tldw_chatbook.Backup_Recovery.models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
    DiscoverySelections,
)


@pytest.fixture
def profile(tmp_path):
    root = tmp_path / "default_user"
    root.mkdir(mode=0o700)
    config = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "config.toml", "fixture"),
        "paths": {"data_dir": str(tmp_path)},
    }
    return config, root


def _discover(config, owner_id):
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters

    return next(
        owner for owner in recovery_adapters() if owner.owner_id == owner_id
    ).discover(config)


@pytest.mark.parametrize("kind", ["absent", "empty", "payload", "file", "symlink"])
def test_actor_import_staging_only_recognizes_empty_scaffolding(profile, kind):
    config, parent = profile
    root = parent / "actor_pack_imports"
    if kind in {"empty", "payload"}:
        root.mkdir(mode=0o700)
        if kind == "payload":
            (root / "pending.zip").write_bytes(b"unreviewed actor pack")
    elif kind == "file":
        root.write_bytes(b"wrong kind")
    elif kind == "symlink":
        root.symlink_to(parent, target_is_directory=True)
    items = _discover(config, "actor_packs.import_staging")
    expected = "unused" if kind in {"absent", "empty"} else "unsupported"
    assert items and all(item.status == expected for item in items)
    assert items[0].path == root


@pytest.mark.parametrize(
    "kind",
    [
        "absent",
        "empty",
        "current_empty",
        "current_lock",
        "foreign_empty",
        "foreign_lock",
        "empty_lock",
        "wrong_byte",
        "large_lock",
        "extra_payload",
        "lock_directory",
        "file",
        "symlink",
        "lock_symlink",
    ],
)
def test_collections_scaffolding_requires_current_authority_and_exact_lock(
    profile, kind
):
    from tldw_chatbook.Backup_Recovery.profile_paths import database_path

    config, parent = profile
    root = parent / "collections_archives"
    fingerprint = hashlib.sha256(
        (
            str(parent.resolve())
            + "\0"
            + str(database_path(config, "library_collections_db_path").resolve())
        ).encode()
    ).hexdigest()[:16]
    authority = root / (
        "foreign-authority" if kind.startswith("foreign") else fingerprint
    )
    if kind not in {"absent", "file", "symlink"}:
        root.mkdir(mode=0o700)
        if kind != "empty":
            authority.mkdir(mode=0o700)
        lock = authority / ".lifecycle.lock"
        if kind in {"current_lock", "foreign_lock", "extra_payload"}:
            lock.write_bytes(b"\0")
        elif kind == "empty_lock":
            lock.write_bytes(b"")
        elif kind == "wrong_byte":
            lock.write_bytes(b"x")
        elif kind == "large_lock":
            lock.write_bytes(b"\0\0")
        elif kind == "lock_directory":
            lock.mkdir(mode=0o700)
        elif kind == "lock_symlink":
            target = parent / "other-lock"
            target.write_bytes(b"\0")
            lock.symlink_to(target)
        if kind == "extra_payload":
            (authority / "offline-archive").write_bytes(b"durable collection archive")
    elif kind == "file":
        root.write_bytes(b"wrong kind")
    elif kind == "symlink":
        root.symlink_to(parent, target_is_directory=True)
    items = _discover(config, "collections.archives")
    expected = (
        "unused"
        if kind in {"absent", "empty", "current_empty", "current_lock"}
        else "unsupported"
    )
    if kind == "large_lock":
        expected = "unavailable"  # The fixed one-byte reader refuses oversized input.
    assert items and all(item.status == expected for item in items)
    assert items[0].path == root


@pytest.mark.parametrize(
    ("owner_id", "leaf"),
    [
        ("runtime.crash_forensics", "faulthandler.log"),
        ("runtime.scheduler_heartbeat", "scheduler_heartbeat.json"),
    ],
)
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("kind", ["absent", "empty", "payload", "directory", "symlink"])
def test_transient_runtime_files_use_checked_file_exclusion(
    profile, owner_id, leaf, diagnostics, kind
):
    config, parent = profile
    config[DISCOVERY_CONTEXT_KEY] = replace(
        config[DISCOVERY_CONTEXT_KEY],
        selections=DiscoverySelections(diagnostics=diagnostics),
    )
    path = parent / leaf
    if kind in {"empty", "payload"}:
        path.write_bytes(
            b"" if kind == "empty" else b"current process diagnostic or liveness state"
        )
    elif kind == "directory":
        path.mkdir(mode=0o700)
    elif kind == "symlink":
        target = parent / "other-runtime-file"
        target.write_bytes(b"do not follow")
        path.symlink_to(target)
    items = _discover(config, owner_id)
    expected = (
        "intentionally_excluded"
        if kind in {"absent", "empty", "payload"}
        else "unsupported"
    )
    assert len(items) == 1 and items[0].status == expected
    assert items[0].path == path


@pytest.mark.parametrize(
    "changed_owner", [None, "research.paste_staging", "collections.archives"]
)
def test_scaffold_discovery_holds_native_namespaces_without_payload_authority(
    tmp_path, monkeypatch, changed_owner
):
    from tldw_chatbook.Backup_Recovery import bootstrap, owner_registry
    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.profile_paths import database_path
    from tldw_chatbook.Backup_Recovery.storage_admission import (
        _preview_reads,
        _read_recovery_file,
    )

    source = tmp_path / "config" / "config.toml"
    source.parent.mkdir(mode=0o700)
    data = tmp_path / "data"
    source.write_text(f'[paths]\ndata_dir="{data}"\n')
    source.chmod(0o600)
    config = {"paths": {"data_dir": str(data)}}
    profile = data / "default_user"
    profile.mkdir(parents=True, mode=0o700)
    research = profile / "research_paste_staging" / "index.json"
    research.parent.mkdir(mode=0o700)
    research.write_bytes(b'{"operations":{},"schema_version":1}')
    fingerprint = hashlib.sha256(
        (
            str(profile.resolve())
            + "\0"
            + str(database_path(config, "library_collections_db_path").resolve())
        ).encode()
    ).hexdigest()[:16]
    lock = profile / "collections_archives" / fingerprint / ".lifecycle.lock"
    lock.parent.mkdir(parents=True, mode=0o700)
    lock.write_bytes(b"\0")
    for path in (research, lock):
        path.chmod(0o600)
    scaffolds = {"research.paste_staging": research, "collections.archives": lock}
    adapters = {
        owner.owner_id: owner
        for owner in recovery_adapters()
        if owner.owner_id in {"config", *scaffolds}
    }
    monkeypatch.setattr(owner_registry, "_adapters", adapters)
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    outside = tmp_path / "unselected.txt"
    outside.write_bytes(b"unselected")
    authority = admission_authority(root)
    with _preview_reads():
        preview = discover((source,))
    assert all(
        item.status == "unused" for item in preview.items if item.owner in scaffolds
    )
    names = _capture_names(authority, preview)
    with authority.maintenance(names, 1) as session:
        with session._discovery_reads(), pytest.raises(
            bootstrap.RecoveryRequired, match="capture_source_outside_scope"
        ):
            _read_recovery_file("research.paste_staging", outside, max_bytes=64 * 1024)
        if changed_owner is not None:
            scaffolds[changed_owner].write_bytes(b"x")
            with pytest.raises(ValueError, match="scope_changed"):
                session._discover_capture_inventory(
                    (source,), DiscoverySelections(), preview.scope_digest
                )
        else:
            current = session._discover_capture_inventory(
                (source,), DiscoverySelections(), preview.scope_digest
            )
            assert current.scope_digest == preview.scope_digest
            for path in scaffolds.values():
                with pytest.raises(
                    bootstrap.RecoveryRequired,
                    match="capture_source_binding_unverified",
                ), session.capture_scope((path,), staging):
                    pass
            with session.capture_scope((source,), staging):
                for phase in ("before", "after"):
                    if phase == "after":
                        with session._discovery_reads():
                            observed = discover((source,))
                        assert observed.scope_digest == preview.scope_digest
                    for owner, path in scaffolds.items():
                        with pytest.raises(
                            bootstrap.RecoveryRequired,
                            match="capture_source_outside_scope",
                        ):
                            _read_recovery_file(
                                owner,
                                path,
                                max_bytes=(
                                    1 if owner == "collections.archives" else 64 * 1024
                                ),
                            )
