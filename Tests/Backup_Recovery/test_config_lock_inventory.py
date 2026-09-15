"""The installed config writer's empty lock is not missing backup content."""

import os

import pytest

from tldw_chatbook.Backup_Recovery import owner_registry, profile_paths
from tldw_chatbook.Backup_Recovery.inventory import discover


def _profile(tmp_path, monkeypatch):
    monkeypatch.setattr(owner_registry, "_adapters", {})
    owner_registry.install_adapters()
    config = tmp_path / "config.toml"
    config.write_text('[general]\nusers_name = "test"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    monkeypatch.setattr(profile_paths, "default_config_path", lambda: config)
    return config, config.with_name(config.name + ".lock")


@pytest.mark.parametrize("exists", [False, True])
def test_exact_private_empty_config_lock_is_explicitly_excluded(
    tmp_path, monkeypatch, exists
):
    config, lock = _profile(tmp_path, monkeypatch)
    if exists:
        lock.touch(mode=0o600)
    result = discover((config,))
    entries = [item for item in result.items if item.path == lock]
    assert entries and all(
        item.owner != "unknown" and item.status == "intentionally_excluded"
        for item in entries
    )


@pytest.mark.parametrize("kind", ["content", "directory", "symlink", "hardlink", "public"])
def test_config_lock_lookalikes_are_not_excluded(tmp_path, monkeypatch, kind):
    config, lock = _profile(tmp_path, monkeypatch)
    if kind == "directory":
        lock.mkdir()
    elif kind in {"symlink", "hardlink"}:
        other = tmp_path / "other"
        other.touch(mode=0o600)
        if kind == "symlink":
            lock.symlink_to(other)
        else:
            os.link(other, lock)
    else:
        lock.touch(mode=0o600)
        if kind == "content":
            lock.write_bytes(b"not a config lock")
        else:
            lock.chmod(0o644)
    result = discover((config,))
    entries = [item for item in result.items if item.path == lock]
    assert entries and all(item.status != "intentionally_excluded" for item in entries)


def test_unrelated_empty_lock_remains_unknown(tmp_path, monkeypatch):
    config, _ = _profile(tmp_path, monkeypatch)
    lock = tmp_path / "unrelated.lock"
    lock.touch(mode=0o600)
    result = discover((config,))
    assert any(item.path == lock and item.owner == "unknown" for item in result.items)
