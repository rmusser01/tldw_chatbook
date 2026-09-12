"""First config-container proof recognizes only its safe native lock file."""

import os
from pathlib import Path

import pytest

from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem
from tldw_chatbook.Backup_Recovery.replacement import _first_config_container


def _selected_config(tmp_path: Path) -> Path:
    parent = tmp_path / "config"
    parent.mkdir(mode=0o700)
    selector = parent / "config.toml"
    selector.write_bytes(b'[general]\nusers_name = "test"\n')
    selector.chmod(0o600)
    return selector


def _prove(selector: Path, tmp_path: Path):
    inventory = Inventory(
        (
            StorageItem(
                "config",
                "profile:test:config",
                selector,
                "included",
                (),
            ),
        ),
        True,
        "test-scope",
        (),
    )
    return _first_config_container(
        selector,
        inventory,
        (),
        {},
        (),
        (selector,),
        (tmp_path / "recovery-control",),
    )


def _private_empty_lock(selector: Path) -> Path:
    lock = selector.with_name(selector.name + ".lock")
    lock.write_bytes(b"")
    lock.chmod(0o600)
    return lock


def test_first_config_container_includes_exact_empty_private_native_lock(
    tmp_path: Path,
) -> None:
    selector = _selected_config(tmp_path)
    lock = _private_empty_lock(selector)

    _, state = _prove(selector, tmp_path)

    assert lock in {row[0] for row in state}


@pytest.mark.parametrize(
    "unsafe_kind",
    ("nonempty", "symlink", "hardlink", "directory", "unsafe_mode", "unrelated"),
)
def test_first_config_container_refuses_unsafe_or_unrelated_lock_sibling(
    tmp_path: Path, unsafe_kind: str
) -> None:
    selector = _selected_config(tmp_path)
    lock = selector.with_name(selector.name + ".lock")
    if unsafe_kind == "nonempty":
        lock.write_bytes(b"held by something else")
        lock.chmod(0o600)
    elif unsafe_kind == "symlink":
        lock.symlink_to(selector)
    elif unsafe_kind == "hardlink":
        target = tmp_path / "hardlink-target"
        target.write_bytes(b"")
        target.chmod(0o600)
        os.link(target, lock)
    elif unsafe_kind == "directory":
        lock.mkdir(mode=0o700)
    elif unsafe_kind == "unsafe_mode":
        lock.write_bytes(b"")
        lock.chmod(0o644)
    else:
        _private_empty_lock(selector)
        unrelated = selector.parent / "unrelated.lock"
        unrelated.write_bytes(b"")
        unrelated.chmod(0o600)

    with pytest.raises(ValueError, match="^replacement_config_container_unverified$"):
        _prove(selector, tmp_path)


def test_first_config_container_state_detects_exact_lock_substitution(
    tmp_path: Path,
) -> None:
    selector = _selected_config(tmp_path)
    lock = _private_empty_lock(selector)
    first = _prove(selector, tmp_path)

    lock.unlink()
    replacement = selector.parent / "replacement-lock"
    replacement.write_bytes(b"")
    replacement.chmod(0o600)
    replacement.replace(lock)

    assert _prove(selector, tmp_path) != first
