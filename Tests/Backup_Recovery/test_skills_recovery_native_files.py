"""Restored Skills roots use the native filesystem identity backend."""

import importlib
import stat
from contextlib import contextmanager
from types import SimpleNamespace

from tldw_chatbook.Backup_Recovery import native_files
from tldw_chatbook.Skills_Interop import recovery_activation
from tldw_chatbook.Utils import platform_files


def test_recovered_skill_source_uses_pinned_native_parent(tmp_path, monkeypatch):
    sentinel = object()
    source = tmp_path / "skill_trust_manifest.json"

    @contextmanager
    def pinned_directory(path):
        assert path == tmp_path
        yield sentinel

    def projected_stat(path, *, dir_fd=None, follow_symlinks=True):
        assert path == source.name
        assert dir_fd is sentinel
        assert follow_symlinks is False
        return SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600,
            st_nlink=1,
            st_dev=73,
            st_ino=91,
        )

    projected = SimpleNamespace(stat=projected_stat)
    original = platform_files.os
    namespace = recovery_activation.__dict__.copy()
    try:
        monkeypatch.setattr(platform_files, "os", projected)
        importlib.reload(recovery_activation)
        monkeypatch.setattr(native_files, "pinned_directory", pinned_directory)
        assert recovery_activation._identity(source) == (str(source), 73, 91)
    finally:
        monkeypatch.setattr(platform_files, "os", original)
        recovery_activation.__dict__.clear()
        recovery_activation.__dict__.update(namespace)
