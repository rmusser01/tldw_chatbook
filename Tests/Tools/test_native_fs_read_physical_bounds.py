"""Actual native fs_read observation and content must bound source bytes."""

from pathlib import Path

import pytest

from tldw_chatbook.Agents.fs_read_ledger import canonical_ledger_key
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Tools import local_tool_impls as local

pytestmark = pytest.mark.bootstrap_profile


def test_native_read_observation_refuses_oversize_before_body(tmp_path, monkeypatch):
    target = tmp_path / "large.txt"
    with target.open("wb") as stream:
        stream.truncate(local.MAX_READ_FILE_BYTES + 1)
    provider = LocalToolProvider(workspace_root=tmp_path, specs=[])
    original_open = local.os.open
    opened = []

    def observe_open(path, *args, **kwargs):
        if Path(path) == target:
            opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(local.os, "open", observe_open)
    with use_run_id("bounded-observation"):
        provider._record_fs_read_observation({"path": target.name}, tmp_path)
    assert not opened, "oversize observation opened the file body"
    assert (
        provider._read_ledger.stamp_for(
            "bounded-observation", canonical_ledger_key(target)
        )
        is None
    )


def test_native_content_rechecks_growth_after_path_stat(tmp_path, monkeypatch):
    target = tmp_path / "growing.txt"
    target.write_text("initial\n")
    original_stat = Path.stat
    initial = original_stat(target)
    grown = []

    def grow_after_stat(path, *args, **kwargs):
        info = original_stat(path, *args, **kwargs)
        if path == target and not grown:
            grown.append(True)
            with target.open("ab") as stream:
                stream.write(b"x" * local.MAX_READ_FILE_BYTES)
        return initial if path == target else info

    monkeypatch.setattr(Path, "stat", grow_after_stat)
    with pytest.raises(local.LocalToolError, match="large|changed"):
        local._read_relative_file(
            Path(target.name),
            workspace=tmp_path,
            offset=1,
            limit=1,
            sensitive_exclusions=(),
            content_stamps=True,
        )
    assert grown


def test_bounded_native_content_and_observation_have_same_truthful_hash(tmp_path):
    import hashlib

    target = tmp_path / "small.txt"
    target.write_text("first\nsecond\n")
    provider = LocalToolProvider(workspace_root=tmp_path, specs=[])
    with use_run_id("bounded-content"):
        provider._record_fs_read_observation({"path": target.name}, tmp_path)
    stamp = provider._read_ledger.stamp_for(
        "bounded-content", canonical_ledger_key(target)
    )
    assert stamp.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    result = local._read_relative_file(
        Path(target.name),
        workspace=tmp_path,
        offset=2,
        limit=1,
        sensitive_exclusions=(),
        content_stamps=True,
    )
    assert "second" in result and stamp.sha256 in result


def test_bounded_native_reader_rejects_directory_and_symlink(tmp_path):
    target = tmp_path / "source.txt"
    target.write_text("source")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    for path in (tmp_path, link):
        with pytest.raises(local.LocalToolError):
            local._read_bounded_bytes(path)


def test_native_directory_scan_never_materializes_uncapped_listdir(
    tmp_path, monkeypatch
):
    target = tmp_path / "directory"
    target.mkdir()
    for index in range(20):
        (target / f"entry-{index}.txt").touch()
    original = local.os.listdir
    materialized = []
    scanned = []
    closed = []
    original_scandir = local.os.scandir

    class ObservedDirectory:
        def __init__(self, path):
            self.native = original_scandir(path)

        def __enter__(self):
            self.native.__enter__()
            return self

        def __iter__(self):
            for entry in self.native:
                scanned.append(entry.name)
                yield entry

        def __exit__(self, *exc):
            result = self.native.__exit__(*exc)
            closed.append(True)
            return result

    def observe_listdir(path):
        names = original(path)
        if Path(path) == target:
            materialized.append(len(names))
        return names

    monkeypatch.setattr(local.os, "listdir", observe_listdir)
    monkeypatch.setattr(local.os, "scandir", ObservedDirectory)
    monkeypatch.setattr(local, "MAX_SCAN_ENTRIES", 3)
    result = local._list_relative_directory(
        Path("directory"),
        workspace=tmp_path,
        max_entries=2,
        sensitive_exclusions=(),
    )
    assert not materialized, (
        "native listing materialized the full directory before its scan cap"
    )
    assert "directory too large" in result
    assert len(scanned) == 4 and closed == [True]
