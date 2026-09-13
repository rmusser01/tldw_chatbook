from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import tldw_chatbook.Utils.private_paths as private_paths
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathStatus,
    atomic_private_write_bytes,
    open_private_text_append,
)


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_atomic_private_write_creates_and_replaces_as_0600(tmp_path):
    target = tmp_path / "artifact.json"

    created = atomic_private_write_bytes(target, b'{"version":1}')
    replaced = atomic_private_write_bytes(target, b'{"version":2}')

    assert created.status is PrivatePathStatus.CREATED_PRIVATE
    assert replaced.status is PrivatePathStatus.ALREADY_PRIVATE
    assert target.read_bytes() == b'{"version":2}'
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_private_append_hardens_existing_file_and_appends(tmp_path):
    target = tmp_path / "events.jsonl"
    target.write_text("first\n", encoding="utf-8")
    target.chmod(0o644)

    with open_private_text_append(target) as stream:
        stream.write("second\n")

    assert target.read_text(encoding="utf-8") == "first\nsecond\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow contract")
@pytest.mark.parametrize("operation", ["replace", "append"])
def test_private_writes_reject_symlink_leaf(tmp_path, operation):
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    target = tmp_path / "artifact"
    target.symlink_to(outside)

    with pytest.raises(PrivatePathError) as caught:
        if operation == "replace":
            atomic_private_write_bytes(target, b"private")
        else:
            with open_private_text_append(target) as stream:
                stream.write("private")

    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR
    assert outside.read_text(encoding="utf-8") == "outside"


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_atomic_private_write_rejects_shared_writable_parent(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    target = shared / "artifact"

    with pytest.raises(PrivatePathError) as caught:
        atomic_private_write_bytes(target, b"private")

    assert caught.value.result.status is PrivatePathStatus.UNSAFE_PARENT
    assert not target.exists()


def test_atomic_private_write_reports_unverified_windows_posture(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "artifact"
    monkeypatch.setattr(private_paths, "_posix_guards_available", lambda: False)
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)

    result = atomic_private_write_bytes(target, b"private")

    assert result.status is PrivatePathStatus.UNVERIFIED_PLATFORM
    assert result.verified_private is False
    assert target.read_bytes() == b"private"


@pytest.mark.skipif(os.name != "posix", reason="POSIX append branch")
def test_private_append_writes_one_lf_byte_per_line(tmp_path):
    """The file's SIZE must equal the bytes the caller wrote.

    ``MCPExecutionLog`` caches a generation's sanitized bytes keyed on
    ``st_size`` (TASK-21134), so any newline translation between "what the
    caller encoded" and "what landed on disk" silently evicts that cache on
    every append and re-parses the whole log.
    """
    target = tmp_path / "events.jsonl"
    encoded = b'{"a":1}\n{"b":2}\n'

    with open_private_text_append(target) as stream:
        stream.write(encoded.decode("utf-8"))

    assert target.read_bytes() == encoded
    assert target.stat().st_size == len(encoded)


def test_private_append_stream_pins_lf_on_the_windows_branch(tmp_path, monkeypatch):
    """The unverified-platform branch must not inherit ``os.linesep``.

    ``Path.open("a")`` defaults to ``newline=None``, which translates every
    ``\\n`` written into ``os.linesep`` -- ``\\r\\n`` on Windows. That cannot be
    reproduced on POSIX (the translation target is a build-time constant, not
    ``os.linesep`` at runtime), so this pins the one thing that decides it:
    the keyword the branch passes.
    """
    recorded = {}
    real_open = Path.open

    def _recording_open(self, *args, **kwargs):
        recorded.update(kwargs)
        recorded["mode"] = args[0] if args else kwargs.get("mode")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(private_paths, "_posix_guards_available", lambda: False)
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)
    monkeypatch.setattr(Path, "open", _recording_open)

    stream = private_paths.open_private_text_append_stream(tmp_path / "events.jsonl")
    stream.close()

    assert recorded["mode"] == "a"
    assert recorded["newline"] == "\n"
