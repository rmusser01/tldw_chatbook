"""The file tool's bounded read takes ordinary files and still refuses changed ones.

Found on Windows (TASK-40608, 2026-11-27): ``_read_bounded_bytes`` refused
nearly every real file there, so ``fs_read`` answered "file changed while
reading" or "file changed while opening" for a file nobody had touched.

* ``os.open`` reads in text mode on Windows unless ``O_BINARY`` is given: CR LF
  arrives as LF and a Ctrl-Z byte ends the file, so the bytes read never
  matched the file's size.
* CPython reports a path's creation time in ``st_ctime_ns`` there and a
  descriptor's last change time (measured on 3.12.10), and gives a path named
  ``.cmd``/``.bat``/``.exe``/``.com`` execute bits that a descriptor never has.
  Comparing a path stat with a descriptor stat on those fields refused every
  file changed after it was created.

The first group fails on Windows without the fix and passes on POSIX either
way. The second group is the control: what the comparison exists for is still
refused on every platform.
"""

import hashlib
import os
import time
from pathlib import Path

import pytest

from tldw_chatbook.Tools import local_tool_impls as local

pytestmark = pytest.mark.bootstrap_profile


# --- ordinary files are read -------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [b"alpha\r\nbeta\r\n", b"\r\n\r\n", b"before\x1aafter\n", b"alpha\nbeta\n"],
    ids=["crlf", "only-crlf", "ctrl-z", "lf"],
)
def test_the_bytes_on_disk_are_the_bytes_returned(tmp_path, data):
    target = tmp_path / "notes.md"
    target.write_bytes(data)

    assert local._read_bounded_bytes(target) == data


def test_a_file_changed_after_it_was_created_is_read(tmp_path):
    target = tmp_path / "notes.md"
    target.write_bytes(b"one\n")
    time.sleep(0.05)  # its change time now moves away from its creation time
    target.write_bytes(b"two\n")

    assert local._read_bounded_bytes(target) == b"two\n"


@pytest.mark.parametrize("name", ["build.cmd", "setup.bat"])
def test_a_command_script_is_read(tmp_path, name):
    target = tmp_path / name
    target.write_bytes(b"@echo off\r\n")

    assert local._read_bounded_bytes(target) == b"@echo off\r\n"


def test_a_crlf_file_is_stamped_with_the_digest_of_its_bytes_on_disk(tmp_path):
    data = b"first\r\nsecond\r\n"
    (tmp_path / "notes.md").write_bytes(data)

    body = local._read_relative_file(
        Path("notes.md"),
        workspace=tmp_path,
        offset=1,
        limit=None,
        sensitive_exclusions=(),
        content_stamps=True,
    )

    assert body == (
        "1\tfirst\n2\tsecond\n"
        f"sha256: {hashlib.sha256(data).hexdigest()}\nsize: {len(data)}"
    )


# --- changed files are still refused ----------------------------------------


def test_a_file_swapped_between_the_stat_and_the_open_is_refused(tmp_path, monkeypatch):
    """Same size and modification time, another file: only its identity differs."""
    target = tmp_path / "notes.md"
    target.write_bytes(b"original\n")
    other = tmp_path / "other.md"
    other.write_bytes(b"replaced\n")
    info = target.stat()
    os.utime(other, ns=(info.st_atime_ns, info.st_mtime_ns))
    original_open = local.os.open

    def swap_then_open(path, *args, **kwargs):
        if os.fspath(path) == str(target) and other.exists():
            os.replace(other, target)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(local.os, "open", swap_then_open)

    with pytest.raises(local.LocalToolError, match="changed while opening"):
        local._read_bounded_bytes(target)
    assert not other.exists(), "the swap never ran"


def test_a_file_rewritten_in_place_while_it_is_read_is_refused(tmp_path, monkeypatch):
    """Same size, modification time put back: only the change time moved."""
    target = tmp_path / "notes.md"
    target.write_bytes(b"original\n")
    info = target.stat()
    original_read = local.os.read
    rewritten = []

    def rewrite_after_the_first_read(descriptor, count):
        chunk = original_read(descriptor, count)
        if not rewritten and os.fstat(descriptor).st_ino == info.st_ino:
            rewritten.append(True)
            time.sleep(0.05)
            with target.open("r+b") as stream:
                stream.write(b"tampered\n")
            os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
        return chunk

    monkeypatch.setattr(local.os, "read", rewrite_after_the_first_read)

    with pytest.raises(local.LocalToolError, match="changed while reading"):
        local._read_bounded_bytes(target)
    assert rewritten, "the rewrite never ran"
    after = target.stat()
    assert (after.st_size, after.st_mtime_ns) == (info.st_size, info.st_mtime_ns), (
        "the control must leave size and modification time as they were"
    )


def test_a_file_that_grew_while_it_was_read_is_refused(tmp_path, monkeypatch):
    target = tmp_path / "notes.md"
    target.write_bytes(b"original\n")
    original_read = local.os.read
    grown = []
    inode = target.stat().st_ino

    def grow_after_the_first_read(descriptor, count):
        chunk = original_read(descriptor, count)
        if not grown and os.fstat(descriptor).st_ino == inode:
            grown.append(True)
            with target.open("ab") as stream:
                stream.write(b"more\n")
        return chunk

    monkeypatch.setattr(local.os, "read", grow_after_the_first_read)

    with pytest.raises(local.LocalToolError, match="changed while reading"):
        local._read_bounded_bytes(target)
    assert grown, "the append never ran"
