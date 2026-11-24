"""Real file effects through qualified, exclusive regular-file handles."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess

import pytest

from tldw_chatbook.Backup_Recovery import native_files
from tldw_chatbook.Utils import windows_files
from tldw_chatbook.Utils.platform_files import os as native_os

ntfs = pytest.mark.skipif(os.name != "nt", reason="requires qualified Windows NTFS")


@pytest.fixture
def directory(tmp_path):
    root = tmp_path / "private"
    native_files.create_private_directory(root)
    with native_files.pinned_directory(root) as parent:
        yield root, parent


def test_unqualified_effects_refuse_before_creating_a_stage(directory, monkeypatch):
    root, parent = directory
    monkeypatch.setattr(
        native_files, "native_identity", lambda _: {"os": "Linux", "filesystem": "ext4"}
    )
    with pytest.raises(OSError):
        with native_files.held_regular_file(parent, "stage", create=True):
            pytest.fail("unqualified creation")
    assert list(root.iterdir()) == []


@ntfs
def test_creation_readback_rename_and_disposition_retain_the_proven_object(directory):
    root, parent = directory
    with native_files.held_regular_file(parent, "stage", create=True) as held:
        proof = held.identity
        assert held.stream.write(b"article") == 7
        held.flush_file()
        held.stream.seek(0)
        assert (
            hashlib.file_digest(held.stream, "sha256").digest()
            == hashlib.sha256(b"article").digest()
        )
        assert stat.S_IMODE(native_os.fstat(held.stream.fileno()).st_mode) == 0o600
        with pytest.raises(OSError):
            (root / "stage").read_bytes()
        with pytest.raises(OSError):
            (root / "stage").write_bytes(b"changed")
        with pytest.raises(OSError):
            (root / "stage").rename(root / "substitution")
        with pytest.raises(OSError):
            root.rename(root.with_name("moved"))
        with pytest.raises(OSError):
            root.parent.rename(root.parent.with_name("moved-ancestor"))
        held.rename_new("final")
        assert held.identity == proof
        assert native_os.listdir(parent) == ["final"]
    assert held.stream.closed
    assert (root / "final").read_bytes() == b"article"
    for identity, path in zip(proof, (root, root / "final"), strict=True):
        info = native_os.stat(path)
        assert identity[:2] == (info.st_dev, info.st_ino)
        assert identity[2] == path.stat().st_birthtime_ns // 100 + 116444736000000000
    with native_files.held_regular_file(parent, "final") as existing:
        assert existing.identity == proof
        assert not existing.stream.writable()
        existing.delete()
        assert existing.stream.closed
        assert list(root.iterdir()) == []


@ntfs
def test_existing_stage_or_final_collision_preserves_both_files(directory):
    root, parent = directory
    (root / "existing").write_bytes(b"other")
    with pytest.raises(FileExistsError):
        with native_files.held_regular_file(parent, "existing", create=True):
            pytest.fail("opened an existing stage")
    with native_files.held_regular_file(parent, "stage", create=True) as staged:
        staged.stream.write(b"ours")
        with pytest.raises(FileExistsError):
            staged.rename_new("existing")
        assert staged.identity
    assert (root / "stage").read_bytes() == b"ours"
    assert (root / "existing").read_bytes() == b"other"


@ntfs
@pytest.mark.parametrize(
    "kind", ["directory", "junction", "hardlink", "public", "case"]
)
def test_existing_leaf_requires_exact_private_single_regular_object(directory, kind):
    root, parent = directory
    leaf = root / "stage"
    if kind == "directory":
        leaf.mkdir()
        (leaf / "other").write_bytes(b"preserved")
    elif kind == "junction":
        target = root / "target"
        target.mkdir()
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(leaf), str(target)],
            check=True,
            capture_output=True,
        )
    else:
        (root / ("STAGE" if kind == "case" else "stage")).write_bytes(b"preserved")
        if kind == "hardlink":
            os.link(leaf, root / "other")
        elif kind == "public":
            subprocess.run(
                ["icacls", str(leaf), "/grant", "*S-1-1-0:(R)"],
                check=True,
                capture_output=True,
            )
    with pytest.raises(OSError):
        with native_files.held_regular_file(parent, "stage"):
            pytest.fail("opened ineligible leaf")
    if kind == "public":
        assert native_os.stat(leaf).st_mode & 0o044
    if not leaf.is_dir():
        assert leaf.read_bytes() == b"preserved"
    elif kind == "directory":
        assert (leaf / "other").read_bytes() == b"preserved"
    else:
        assert leaf.is_junction()


@ntfs
@pytest.mark.parametrize("action", ["rename", "delete"])
def test_failed_parent_barrier_keeps_actual_namespace_and_releases_file(
    directory, monkeypatch, action
):
    root, parent = directory

    def barrier_error(fd):
        raise OSError("injected parent barrier")

    with monkeypatch.context() as patch:
        patch.setattr(windows_files, "flush_directory", barrier_error)
        with pytest.raises(OSError, match="injected parent barrier"):
            with native_files.held_regular_file(parent, "stage", create=True) as held:
                held.stream.write(b"stage")
                if action == "rename":
                    held.rename_new("published")
                else:
                    held.delete()
    assert held.stream.closed
    assert not (root / "stage").exists()
    if action == "rename":
        assert (root / "published").read_bytes() == b"stage"
    native_files.flush_regular_parent(parent)


@ntfs
def test_exclusive_existing_data_handle_refuses_mutation_without_changes(directory):
    root, parent = directory
    path = root / "open"
    path.write_bytes(b"retain")
    with path.open("r+b") as other, pytest.raises(OSError):
        with native_files.held_regular_file(parent, "open"):
            pytest.fail("overlapped a preexisting data handle")
    assert other.closed
    assert path.read_bytes() == b"retain"


@ntfs
def test_effects_use_the_verified_handle_after_file_or_directory_exchange_is_denied(
    directory, monkeypatch
):
    root, parent = directory
    (root / "outsider").write_bytes(b"outsider")
    (root / "tree").mkdir()
    (root / "tree" / "child").write_bytes(b"keep")
    with native_files.held_regular_file(parent, "stage", create=True) as held:
        held.stream.write(b"issued")
        for other in ("outsider", "tree"):
            with pytest.raises(OSError):
                os.replace(root / other, root / "stage")
        engine = windows_files._native()
        original_open = engine.open_handle

        def refuse_source_reopen(name, **kwargs):
            if name in {"stage", "final"}:
                pytest.fail("reopened the checked leaf by name")
            return original_open(name, **kwargs)

        monkeypatch.setattr(engine, "open_handle", refuse_source_reopen)
        held.rename_new("final")
        assert held.identity
        held.delete()
    assert (root / "outsider").read_bytes() == b"outsider"
    assert (root / "tree" / "child").read_bytes() == b"keep"


@ntfs
def test_failed_data_flush_never_publishes_or_loses_stage(directory, monkeypatch):
    root, parent = directory

    def fail_flush(handle):
        raise OSError("injected data barrier")

    with pytest.raises(OSError, match="injected data barrier"):
        with native_files.held_regular_file(parent, "stage", create=True) as held:
            held.stream.write(b"not confirmed")
            monkeypatch.setattr(
                windows_files._native().kernel, "FlushFileBuffers", fail_flush
            )
            held.rename_new("final")
    assert held.stream.closed
    assert (root / "stage").read_bytes() == b"not confirmed"
    assert not (root / "final").exists()


@ntfs
def test_retired_or_copied_owner_cannot_act_on_a_reused_descriptor(directory):
    import copy

    root, parent = directory
    with native_files.held_regular_file(parent, "first", create=True) as first:
        for clone in (copy.copy, copy.deepcopy):
            with pytest.raises(TypeError):
                clone(first)
        first.delete()
    with native_files.held_regular_file(parent, "second", create=True) as second:
        first.delete()
        with pytest.raises(ValueError):
            first.rename_new("other")
        second.stream.write(b"live")
    with native_files.held_regular_file(parent, "third", create=True) as third:
        pass
    with native_files.held_regular_file(parent, "second") as live:
        with pytest.raises(ValueError):
            third.delete()
        assert live.stream.read() == b"live"
    assert {path.name for path in root.iterdir()} == {"second", "third"}
