"""Tests for ``Utils/atomic_file_ops.py``, in particular the
``preserve_existing_mode`` parameter added by the task-851 review (finding
2): ``atomic_write_text`` used to always ``chmod`` the replacement file to
its ``mode`` parameter's default (0o644), which widened permissions on any
target file that had been deliberately tightened (e.g. a config file
holding secrets, chmod'd to 0o600). Enabling then disabling config
encryption measured 0600 -> 0644 -> (still) 0644 with plaintext keys.
"""

import errno
import os
import platform
import stat
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.Utils import atomic_file_ops, file_durability
from tldw_chatbook.Utils.atomic_file_ops import (
    atomic_copy,
    atomic_write_bytes,
    atomic_write_text,
)


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_preserve_existing_mode_keeps_restrictive_permissions(tmp_path):
    """A pre-existing 0600 file must stay 0600 after an atomic rewrite."""
    target = tmp_path / "secrets.toml"
    target.write_text("a = 1\n")
    target.chmod(0o600)

    atomic_write_text(target, "a = 2\n", mode=0o644, preserve_existing_mode=True)

    assert _mode(target) == 0o600
    assert target.read_text() == "a = 2\n"


def test_preserve_existing_mode_keeps_permissive_permissions(tmp_path):
    """The preserve path must not *tighten* an existing file either --
    it carries forward whatever mode the file already had, in either
    direction."""
    target = tmp_path / "public.toml"
    target.write_text("a = 1\n")
    target.chmod(0o644)

    atomic_write_text(target, "a = 2\n", mode=0o600, preserve_existing_mode=True)

    assert _mode(target) == 0o644


def test_preserve_existing_mode_uses_fallback_mode_for_new_file(tmp_path):
    """When the target does not exist yet, there is nothing to preserve --
    the caller-supplied ``mode`` (e.g. a restrictive default for a secrets
    file) is applied instead."""
    target = tmp_path / "new_secrets.toml"
    assert not target.exists()

    atomic_write_text(target, "a = 1\n", mode=0o600, preserve_existing_mode=True)

    assert _mode(target) == 0o600


def test_preserve_existing_mode_false_keeps_legacy_behavior(tmp_path):
    """Default (``preserve_existing_mode=False``) behavior for existing
    callers must be unchanged: the file is always chmod'd to ``mode``,
    even if it previously had a different mode."""
    target = tmp_path / "notes.md"
    target.write_text("hello\n")
    target.chmod(0o600)

    atomic_write_text(target, "hello again\n")

    assert _mode(target) == 0o644


def test_privacy_safe_failure_log_contains_only_category_and_exception_class(
    tmp_path, monkeypatch
):
    path_canary = "PATH-CANARY-22507"
    body_canary = "BODY-CANARY-22507"
    exception_canary = "EXCEPTION-CANARY-22507"
    target = tmp_path / path_canary
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")

    class PrivateWriteError(OSError):
        pass

    def fail_replace(*_args):
        raise PrivateWriteError(exception_canary)

    monkeypatch.setattr("os.replace", fail_replace)
    try:
        with pytest.raises(PrivateWriteError):
            atomic_write_text(
                target,
                body_canary,
                privacy_safe_log=True,
            )
    finally:
        logger.remove(sink)

    log_text = "".join(messages)
    assert "atomic_write_failed" in log_text
    assert "PrivateWriteError" in log_text
    assert path_canary not in log_text
    assert body_canary not in log_text
    assert exception_canary not in log_text
    assert "Traceback" not in log_text


def test_no_overwrite_refuses_an_existing_destination(tmp_path):
    target = tmp_path / "appeared.json"
    target.write_text("other writer", encoding="utf-8")

    with pytest.raises(FileExistsError):
        atomic_write_text(target, "private export", overwrite=False)

    assert target.read_text(encoding="utf-8") == "other writer"


# --- PR #3021 review: a no-clobber publish where hard links do not exist ---
# The ``overwrite=False`` branch published by ``os.link`` and nothing else.
# FAT32/exFAT sticks and many SMB/NFS mounts cannot hard-link: ``link()``
# raises ``PermissionError`` or ``OSError(ENOTSUP/EOPNOTSUPP)`` there -- not
# ``FileExistsError`` -- so every new-file write to such a folder failed,
# where a plain write used to work. These fail on 502bd89efc.


def _eperm() -> OSError:
    return PermissionError(errno.EPERM, "Operation not permitted")


def _bare_permission_error() -> OSError:
    return PermissionError("hard links are not permitted here")


def _enotsup() -> OSError:
    return OSError(errno.ENOTSUP, "Operation not supported")


def _eopnotsupp() -> OSError:
    return OSError(errno.EOPNOTSUPP, "Operation not supported on socket")


LINK_REFUSALS = [
    pytest.param(_eperm, id="EPERM"),
    pytest.param(_bare_permission_error, id="PermissionError-without-errno"),
    pytest.param(_enotsup, id="ENOTSUP"),
    pytest.param(_eopnotsupp, id="EOPNOTSUPP"),
]


def _refuse_hard_links(monkeypatch, refusal) -> list[tuple]:
    """Make ``os.link`` fail the way a filesystem without hard links does."""
    attempts: list[tuple] = []

    def refuse(*args, **_kwargs):
        attempts.append(args)
        raise refusal()

    monkeypatch.setattr(os, "link", refuse)
    return attempts


def _names(folder) -> list[str]:
    return sorted(entry.name for entry in folder.iterdir())


@pytest.fixture
def volume(tmp_path):
    """A folder of its own: the suite's profile redirect keeps ``test_data``
    in ``tmp_path``, and these tests assert that nothing is left behind."""
    folder = tmp_path / "volume"
    folder.mkdir()
    return folder


@pytest.mark.parametrize("refusal", LINK_REFUSALS)
def test_no_overwrite_writes_a_new_file_where_hard_links_are_unsupported(
    volume, monkeypatch, refusal
):
    attempts = _refuse_hard_links(monkeypatch, refusal)
    seen = _fsynced_inodes(monkeypatch)
    target = volume / "export.md"

    atomic_write_text(target, "exported\n", mode=0o640, overwrite=False)

    assert attempts, "the link publish must still be tried first"
    assert target.read_bytes() == b"exported\n"
    assert _mode(target) == 0o640, "the placeholder's mode must not survive"
    assert _names(volume) == ["export.md"], "temp file or placeholder left behind"
    assert _fsynced_parent(seen, volume), "the fallback publish was not made durable"


@pytest.mark.parametrize("refusal", LINK_REFUSALS)
def test_no_overwrite_still_refuses_an_existing_file_where_hard_links_are_unsupported(
    volume, monkeypatch, refusal
):
    _refuse_hard_links(monkeypatch, refusal)
    target = volume / "appeared.json"
    target.write_bytes(b"other writer")

    with pytest.raises(FileExistsError):
        atomic_write_text(target, "private export", overwrite=False)

    assert target.read_bytes() == b"other writer"
    assert _names(volume) == ["appeared.json"], "temp file left behind"


def test_no_overwrite_fallback_refuses_a_dangling_symlink(volume, monkeypatch):
    """The fallback reserves the name exclusively; it is not an
    ``exists()`` check, which reads a dangling link as "nothing there"."""
    _refuse_hard_links(monkeypatch, _enotsup)
    target = volume / "link.md"
    target.symlink_to(volume / "missing-target.md")

    with pytest.raises(FileExistsError):
        atomic_write_text(target, "must not land", overwrite=False)

    assert target.is_symlink()
    assert not (volume / "missing-target.md").exists(), "wrote through the link"
    assert _names(volume) == ["link.md"]


@pytest.mark.parametrize("refusal", LINK_REFUSALS)
def test_a_failed_fallback_publish_leaves_no_placeholder_and_no_temp_file(
    volume, monkeypatch, refusal
):
    _refuse_hard_links(monkeypatch, refusal)
    reserved_when_it_failed: list[bool] = []

    def fail_replace(_source, destination, *_args, **_kwargs):
        reserved_when_it_failed.append(os.path.lexists(destination))
        raise OSError(errno.EIO, "disk went away")

    monkeypatch.setattr(os, "replace", fail_replace)
    target = volume / "export.md"

    with pytest.raises(OSError) as caught:
        atomic_write_text(target, "never lands", overwrite=False)

    assert caught.value.errno == errno.EIO, "the publish's own failure is reported"
    assert reserved_when_it_failed == [True], (
        "the failure must come after the placeholder is created"
    )
    assert not os.path.lexists(target), "an empty placeholder was left behind"
    assert _names(volume) == []


def test_a_failed_fallback_publish_never_removes_another_writers_bytes(
    volume, monkeypatch
):
    """The clean-up removes OUR empty placeholder, never a file with bytes in
    it: another program may have written into the name in the meantime."""
    _refuse_hard_links(monkeypatch, _enotsup)

    def racing_replace(_source, destination):
        Path(destination).write_bytes(b"other writer")
        raise OSError(errno.EIO, "disk went away")

    monkeypatch.setattr(os, "replace", racing_replace)
    target = volume / "export.md"

    with pytest.raises(OSError):
        atomic_write_text(target, "never lands", overwrite=False)

    assert target.read_bytes() == b"other writer"
    assert _names(volume) == ["export.md"], "temp file left behind"


def test_a_real_link_failure_is_raised_not_retried(volume, monkeypatch):
    """Only "this filesystem cannot hard-link" falls back. A failing disk is
    reported as itself, with nothing reserved at the destination."""

    def _eio() -> OSError:
        return OSError(errno.EIO, "disk went away")

    _refuse_hard_links(monkeypatch, _eio)
    target = volume / "export.md"

    with pytest.raises(OSError) as caught:
        atomic_write_text(target, "never lands", overwrite=False)

    assert caught.value.errno == errno.EIO
    assert _names(volume) == []


def test_no_overwrite_reserves_nothing_where_hard_links_work(volume, monkeypatch):
    """Unchanged where links are supported: one link publish, no placeholder."""
    target = volume / "linked.md"
    opened: list[str] = []
    real_open = os.open

    def spy_open(path, flags, *args, **kwargs):
        opened.append(os.fspath(path))
        return real_open(path, flags, *args, **kwargs)

    links: list[tuple] = []
    real_link = os.link

    def spy_link(source, destination, *args, **kwargs):
        links.append((os.fspath(source), os.fspath(destination)))
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "link", spy_link)

    atomic_write_text(target, "payload\n", overwrite=False)

    assert [destination for _, destination in links] == [str(target)]
    assert str(target) not in opened, "a placeholder was created on a linking volume"
    assert target.read_bytes() == b"payload\n"
    assert target.stat().st_nlink == 1
    assert _names(volume) == ["linked.md"]


# --- task-32896: durability barriers -----------------------------------
# ``os.replace`` plus a *file* fsync is atomic but not durable. Until the
# parent directory is fsynced the rename itself can be lost on power failure.
# These tests fail on the pre-task-32896 helper, which fsynced only the file.


def _fsynced_inodes(monkeypatch) -> list[os.stat_result]:
    """Record an ``os.stat_result`` for every fd handed to ``os.fsync``."""
    seen: list[os.stat_result] = []
    real_fsync = os.fsync

    def spy(fd):
        try:
            seen.append(os.fstat(fd))
        except OSError:  # pragma: no cover - fd already closed
            pass
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    return seen


def _fsynced_parent(seen, parent) -> bool:
    parent_ino = parent.stat().st_ino
    return any(
        stat.S_ISDIR(entry.st_mode) and entry.st_ino == parent_ino for entry in seen
    )


def test_atomic_write_text_fsyncs_the_parent_directory(tmp_path, monkeypatch):
    seen = _fsynced_inodes(monkeypatch)

    atomic_write_text(tmp_path / "durable.txt", "payload\n")

    assert _fsynced_parent(seen, tmp_path), (
        "only the file was fsynced; the rename itself is still losable"
    )


def test_atomic_write_bytes_fsyncs_the_parent_directory(tmp_path, monkeypatch):
    seen = _fsynced_inodes(monkeypatch)

    atomic_write_bytes(tmp_path / "durable.bin", b"payload")

    assert _fsynced_parent(seen, tmp_path)


def test_no_overwrite_link_publication_also_fsyncs_the_parent_directory(
    tmp_path, monkeypatch
):
    """The ``overwrite=False`` branch publishes by ``os.link``, a second
    directory mutation that needs the same barrier as the rename branch."""
    seen = _fsynced_inodes(monkeypatch)

    atomic_write_text(tmp_path / "linked.txt", "payload\n", overwrite=False)

    assert _fsynced_parent(seen, tmp_path)


def test_file_barrier_goes_through_flush_file_not_bare_fsync(tmp_path, monkeypatch):
    """The file barrier must be the host-native one (Darwin ``F_FULLFSYNC``),
    not a plain ``os.fsync`` that leaves the drive write cache unflushed."""
    flushed: list[int] = []
    real = file_durability.flush_file

    def spy(fd):
        flushed.append(fd)
        return real(fd)

    monkeypatch.setattr(atomic_file_ops, "flush_file", spy)

    atomic_write_text(tmp_path / "native.txt", "payload\n")

    assert flushed, "atomic_write_text did not use the native file barrier"


@pytest.mark.skipif(platform.system() != "Darwin", reason="F_FULLFSYNC is Darwin-only")
def test_darwin_file_barrier_issues_f_fullfsync(tmp_path, monkeypatch):
    import fcntl

    commands: list[int] = []
    real = fcntl.fcntl

    def spy(fd, cmd, *args):
        commands.append(cmd)
        return real(fd, cmd, *args)

    monkeypatch.setattr(fcntl, "fcntl", spy)

    atomic_write_text(tmp_path / "fullfsync.txt", "payload\n")

    assert fcntl.F_FULLFSYNC in commands


def test_private_write_opens_exclusive_nofollow_with_mode_at_open(
    tmp_path, monkeypatch
):
    """``private=True`` must never let the file exist at a wider mode: it is
    created ``O_EXCL|O_NOFOLLOW`` at 0o600 and never chmod'd up afterwards."""
    creations: list[tuple[int, int]] = []
    real_open = os.open

    def spy_open(path, flags, mode=0o777, *args, **kwargs):
        if flags & os.O_CREAT:
            creations.append((flags, mode))
        return real_open(path, flags, mode, *args, **kwargs)

    chmods: list[tuple] = []
    real_chmod = os.chmod

    def spy_chmod(*args, **kwargs):
        chmods.append(args)
        return real_chmod(*args, **kwargs)

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "chmod", spy_chmod)

    target = tmp_path / "secret.txt"
    atomic_write_text(target, "token\n", private=True)

    assert creations, "no file was created"
    flags, mode = creations[0]
    assert flags & os.O_EXCL
    assert flags & os.O_NOFOLLOW
    assert mode == 0o600, "mode must be set at open, not by a later chmod"
    assert chmods == [], "a post-hoc chmod reopens the permission window"
    assert _mode(target) == 0o600
    assert target.read_text() == "token\n"


def test_private_write_bytes_is_owner_only(tmp_path):
    target = tmp_path / "secret.bin"
    atomic_write_bytes(target, b"token", private=True)

    assert _mode(target) == 0o600


def test_fsync_parent_directory_is_a_noop_on_windows(monkeypatch, tmp_path):
    """Windows has no directory fd to fsync; skip, never crash."""
    monkeypatch.setattr(file_durability.os, "name", "nt")
    opened: list = []
    monkeypatch.setattr(
        file_durability.os, "open", lambda *a, **k: opened.append(a) or 0
    )

    file_durability.fsync_parent_directory(tmp_path)

    assert opened == []


def test_fsync_parent_directory_tolerates_unsupported_but_raises_real_errors(
    tmp_path, monkeypatch
):
    import errno

    def unsupported(fd):
        raise OSError(errno.EINVAL, "not supported here")

    monkeypatch.setattr(file_durability, "flush_directory", unsupported)
    file_durability.fsync_parent_directory(tmp_path)  # tolerated

    def real_failure(fd):
        raise OSError(errno.EIO, "disk")

    monkeypatch.setattr(file_durability, "flush_directory", real_failure)
    with pytest.raises(OSError):
        file_durability.fsync_parent_directory(tmp_path)


def test_atomic_copy_persists_the_copy_and_the_parent_directory(tmp_path):
    """``shutil.copy2`` persists nothing on its own -- ``atomic_copy`` had no
    barrier at all before task-32896."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dst_dir = tmp_path / "out"
    dst_dir.mkdir()
    dst = dst_dir / "dst.bin"

    seen: list[os.stat_result] = []
    real_fsync = os.fsync

    def spy(fd):
        try:
            seen.append(os.fstat(fd))
        except OSError:  # pragma: no cover
            pass
        return real_fsync(fd)

    import unittest.mock

    with unittest.mock.patch.object(os, "fsync", spy):
        atomic_copy(src, dst)

    assert dst.read_bytes() == b"payload"
    assert any(stat.S_ISREG(entry.st_mode) for entry in seen), "copy never fsynced"
    assert _fsynced_parent(seen, dst_dir), "destination directory never fsynced"
