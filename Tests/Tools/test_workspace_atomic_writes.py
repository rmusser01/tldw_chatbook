"""Native write regressions exercise the real isolated workspace worker."""

import hashlib
import os
from pathlib import Path

import pytest

from tldw_chatbook.Tools.local_tool_impls import LocalToolError, write_file
from tldw_chatbook.Tools.workspace_tool_executor import (
    WorkspaceToolExecutionError,
    WorkspaceToolExecutor,
)
from tldw_chatbook.Utils import platform_files


@pytest.mark.parametrize("precondition", ({}, {"expected_absent": True}))
def test_worker_creates_reads_and_cas_replaces_file(
    tmp_path: Path, precondition
) -> None:
    executor = WorkspaceToolExecutor(tmp_path)
    original = "first café\r\n"
    created = executor.execute(
        "fs_write",
        {"path": "new.txt", "content": original, **precondition},
        intent="write",
    )
    digest = hashlib.sha256(original.encode()).hexdigest()
    assert "wrote" in created and f"sha256: {digest}" in created
    assert (tmp_path / "new.txt").read_bytes() == original.encode()
    assert "first café" in executor.execute(
        "fs_read", {"path": "new.txt"}, intent="read"
    )
    replaced = executor.execute(
        "fs_write",
        {"path": "new.txt", "content": "second\n", "expected_sha256": digest},
        intent="write",
    )
    assert "wrote" in replaced
    assert (tmp_path / "new.txt").read_bytes() == b"second\n"
    for stale in ({"expected_sha256": digest}, {"expected_absent": True}):
        with pytest.raises(WorkspaceToolExecutionError, match="precondition"):
            executor.execute(
                "fs_write",
                {"path": "new.txt", "content": "lost", **stale},
                intent="write",
            )
        assert (tmp_path / "new.txt").read_bytes() == b"second\n"
    assert not list(tmp_path.glob(".chatbook-write-*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="native Windows private file policy")
def test_windows_writes_keep_native_private_file_policy(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("before")
    write_file("file.txt", "replacement", workspace_root=tmp_path)
    assert platform_files.os.stat(target).st_mode & 0o777 == 0o600
    write_file("new.txt", "created", workspace_root=tmp_path, expected_absent=True)
    assert platform_files.os.stat(tmp_path / "new.txt").st_mode & 0o777 == 0o600


def test_failed_temp_flush_keeps_target_and_cleans_temp(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "file.txt"
    target.write_bytes(b"before")
    file_os = platform_files.os if os.name == "nt" else os

    def refuse_flush(descriptor):
        raise OSError("injected flush failure")

    monkeypatch.setattr(file_os, "fsync", refuse_flush)
    with pytest.raises(LocalToolError, match="atomic write failed"):
        write_file("file.txt", "after", workspace_root=tmp_path)
    assert target.read_bytes() == b"before"
    assert not list(tmp_path.glob(".chatbook-write-*.tmp"))


def test_absent_target_race_does_not_clobber_intervening_file(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "file.txt"
    file_os = platform_files.os if os.name == "nt" else os
    original_link = file_os.link

    def race_link(source, destination, **kwargs):
        target.write_bytes(b"intervening")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(file_os, "link", race_link)
    with pytest.raises(LocalToolError, match="target is present"):
        write_file("file.txt", "after", workspace_root=tmp_path, expected_absent=True)
    assert target.read_bytes() == b"intervening"
    assert not list(tmp_path.glob(".chatbook-write-*.tmp"))


def test_cas_rejects_inode_replaced_before_commit(tmp_path: Path, monkeypatch) -> None:
    from tldw_chatbook.Tools import local_tool_impls

    target = tmp_path / "file.txt"
    target.write_bytes(b"before")
    competitor = tmp_path / "competitor.txt"
    competitor.write_bytes(b"intervening")
    original_assert = local_tool_impls._assert_expected_target_is_current
    calls = 0

    def replace_before_final_check(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            if os.name == "nt":
                platform_files.os.replace_open_target(competitor, target)
            else:
                os.replace(competitor, target)
        return original_assert(*args)

    monkeypatch.setattr(
        local_tool_impls,
        "_assert_expected_target_is_current",
        replace_before_final_check,
    )
    with pytest.raises(LocalToolError, match="digest changed"):
        write_file(
            "file.txt",
            "after",
            workspace_root=tmp_path,
            expected_sha256=hashlib.sha256(b"before").hexdigest(),
        )
    assert calls == 2
    assert target.read_bytes() == b"intervening"
    assert not list(tmp_path.glob(".chatbook-write-*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="native Windows locked-target replacement")
def test_windows_cas_retains_target_lock_through_atomic_replace(
    tmp_path: Path, monkeypatch
) -> None:
    import portalocker

    target = tmp_path / "file.txt"
    target.write_bytes(b"before")
    original_replace = platform_files.os.replace_open_target
    checked = []

    def replace_while_locked(*args, **kwargs):
        with target.open("r+b", buffering=0) as contender:
            with pytest.raises(portalocker.exceptions.LockException):
                portalocker.lock(
                    contender,
                    portalocker.LockFlags.EXCLUSIVE
                    | portalocker.LockFlags.NON_BLOCKING,
                )
        checked.append(True)
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(platform_files.os, "replace_open_target", replace_while_locked)
    write_file(
        "file.txt",
        "after",
        workspace_root=tmp_path,
        expected_sha256=hashlib.sha256(b"before").hexdigest(),
    )
    assert checked == [True]
    assert target.read_bytes() == b"after"


@pytest.mark.skipif(os.name != "nt", reason="native Windows junction refusal")
def test_windows_parent_junction_swap_is_refused_before_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    import subprocess

    from tldw_chatbook.Tools import local_tool_impls

    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    parent = workspace / "parent"
    parent.mkdir()
    (parent / "file.txt").write_bytes(b"original")
    (outside / "file.txt").write_bytes(b"outside sentinel")
    original_atomic = local_tool_impls._atomic_write_target

    def swap_parent_before_pin(target, data, **kwargs):
        parent.rename(workspace / "retained")
        subprocess.run(
            ["cmd", "/D", "/C", "mklink", "/J", str(parent), str(outside)],
            check=True,
            capture_output=True,
        )
        return original_atomic(target, data, **kwargs)

    monkeypatch.setattr(
        local_tool_impls, "_atomic_write_target", swap_parent_before_pin
    )
    with pytest.raises(LocalToolError, match="write parent changed"):
        write_file("parent/file.txt", "after", workspace_root=workspace)
    assert (outside / "file.txt").read_bytes() == b"outside sentinel"
    assert (workspace / "retained" / "file.txt").read_bytes() == b"original"
    assert not list(outside.glob(".chatbook-write-*.tmp"))
