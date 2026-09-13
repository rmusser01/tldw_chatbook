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
