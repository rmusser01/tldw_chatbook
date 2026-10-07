"""Restored MCP startup uses one native ownership and identity representation."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.MCP import recovery_activation
from tldw_chatbook.Utils.platform_files import os


def test_private_mcp_source_is_read_with_its_native_identity(tmp_path):
    source = tmp_path / "config.toml"
    source.write_bytes(b"[general]\n")
    source.chmod(0o600)
    identity, payload = recovery_activation._read(source)
    assert payload == b"[general]\n"
    assert identity[:2] == (os.stat(source).st_dev, os.stat(source).st_ino)


def test_private_mcp_directory_uses_native_owner(tmp_path):
    info = os.stat(tmp_path)
    assert recovery_activation._directory(tmp_path) == (info.st_dev, info.st_ino)


@pytest.mark.parametrize("changed", [False, True])
def test_mcp_source_rechecks_path_using_the_descriptor_backend(
    tmp_path, monkeypatch, changed
):
    source = tmp_path / "config.toml"
    source.write_bytes(b"[general]\n")
    source.chmod(0o600)

    def projected(info, *, changed=False):
        # Native Windows volume IDs differ from pathlib's CRT projection.
        return SimpleNamespace(
            st_dev=info.st_dev + 123,
            st_ino=info.st_ino + int(changed),
            st_size=info.st_size,
            st_mtime_ns=info.st_mtime_ns,
            st_ctime_ns=info.st_ctime_ns,
            st_nlink=info.st_nlink,
            st_uid=1000,
        )

    backend = SimpleNamespace(
        geteuid=lambda: 1000,
        fstat=lambda fd: projected(os.fstat(fd)),
        stat=lambda path, **kwargs: projected(os.stat(path, **kwargs), changed=changed),
    )
    monkeypatch.setattr(recovery_activation, "os", backend)
    if changed:
        with pytest.raises(ValueError, match="mcp_recovery_source_changed"):
            recovery_activation._read(source)
    else:
        assert recovery_activation._read(source)[1] == b"[general]\n"


@pytest.mark.parametrize("owner", ["mcp.local", "mcp.permissions", "mcp.context"])
def test_unavailable_fresh_mcp_projection_preserves_originals_and_config_dependency(
    tmp_path, monkeypatch, owner
):
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
        storage_logical_id,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
    from tldw_chatbook.MCP.recovery import recovery_adapters

    context = DiscoveryContext(tmp_path / "config.toml", "selected")
    config = {
        DISCOVERY_CONTEXT_KEY: context,
        "paths": {"data_dir": str(tmp_path / "data")},
    }
    adapter = next(item for item in recovery_adapters() if item.owner_id == owner)
    source = user_data_dir(config) / adapter.leaf
    source.parent.mkdir(parents=True, mode=0o700)
    source.write_bytes(b'{"historical":true}')
    source.chmod(0o600)
    if owner == "mcp.permissions":
        backup = source.with_name(source.name + ".bak")
        backup.write_bytes(b"historical corruption evidence")
        backup.chmod(0o600)
    before = {p: p.read_bytes() for p in source.parent.iterdir()}

    def unavailable(*args):
        raise ValueError("projection_generation_unavailable")

    monkeypatch.setattr(recovery_activation, "inventory_path", unavailable)
    items = adapter.discover(config)

    assert {item.path for item in items[:-1]} == set(before)
    assert all(item.status == "included" for item in items[:-1])
    fresh = items[-1]
    assert (fresh.owner, fresh.logical_id, fresh.path, fresh.status) == (
        owner,
        storage_logical_id(context, owner, "fresh"),
        None,
        "unavailable",
    )
    assert all(
        item.dependencies == (storage_logical_id(context, "config"),) for item in items
    )
    assert before == {p: p.read_bytes() for p in source.parent.iterdir()}
