from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Utils import filesystem_identity as identity_module
from tldw_chatbook.Utils.filesystem_identity import (
    DirectoryIdentity,
    DirectoryIdentityError,
    capture_directory_chain,
    directory_identity_from_stat,
)


def test_capture_directory_chain_canonicalizes_stable_alias(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    chain = capture_directory_chain(alias)

    assert chain.canonical_root == root.resolve()
    assert chain.identities[0] == directory_identity_from_stat(os.stat(root))


def test_capture_directory_chain_records_root_first_ancestors(tmp_path: Path) -> None:
    root = tmp_path / "one" / "two"
    root.mkdir(parents=True)

    chain = capture_directory_chain(root)

    expected_paths = (root.resolve(), *root.resolve().parents)
    assert chain.identities == tuple(
        directory_identity_from_stat(os.lstat(path)) for path in expected_paths
    )


def test_capture_directory_chain_rejects_symlink_at_canonical_locator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    canonical = root.resolve()
    real_lstat = identity_module.os.lstat

    monkeypatch.setattr(
        identity_module.Path,
        "resolve",
        lambda self, strict=True: canonical,
    )

    def symlinked_locator(path: Path | str):
        value = real_lstat(path)
        if Path(path) == canonical:
            return SimpleNamespace(
                st_dev=value.st_dev,
                st_ino=value.st_ino,
                st_mode=stat.S_IFLNK,
            )
        return value

    monkeypatch.setattr(identity_module.os, "lstat", symlinked_locator)

    with pytest.raises(DirectoryIdentityError, match="unsafe directory"):
        capture_directory_chain(root)


def test_directory_identity_rejects_missing_windows_file_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(identity_module, "_WINDOWS", True)
    monkeypatch.setattr(identity_module, "_REPARSE_POINT", 0x400)
    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFDIR)

    with pytest.raises(DirectoryIdentityError, match="file attributes"):
        directory_identity_from_stat(metadata)


def test_directory_identity_equality_is_stable() -> None:
    left = DirectoryIdentity(device=1, inode=2, mode=stat.S_IFDIR, reparse=False)
    right = DirectoryIdentity(device=1, inode=2, mode=stat.S_IFDIR, reparse=False)

    assert left == right
    assert hash(left) == hash(right)
