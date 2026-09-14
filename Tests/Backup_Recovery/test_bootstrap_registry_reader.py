"""Startup reads distinguish an active registry publisher from abandoned intent."""

# Used only for the fixed native ACL command on this test's temporary lock.
import subprocess  # nosec B404
import sys
import threading

import pytest

from Tests.Backup_Recovery.test_bootstrap import local_scope  # noqa: F401
from tldw_chatbook.Backup_Recovery import bootstrap


@pytest.mark.parametrize("interrupted", [False, True])
def test_startup_observes_completed_native_registry_publication(
    local_scope, monkeypatch, interrupted  # noqa: F811
):
    root, config, data, authority = local_scope
    extra = data.parent / "new-source"
    extra.mkdir(mode=0o700)
    published, release = threading.Event(), threading.Event()
    reached, lock_attempted, reader_done = (
        threading.Event(), threading.Event(), threading.Event()
    )
    errors, observed = [], []
    original = authority._write_new_record
    original_flock = bootstrap.fcntl.flock

    def hold_intent(parent, name, raw):
        original(parent, name, raw)
        if name == "registry.pending.json":
            published.set()
            assert release.wait(5)
            if interrupted:
                raise OSError("injected publication interruption")

    monkeypatch.setattr(authority, "_write_new_record", hold_intent)

    def publish():
        try:
            authority.register("backup.source.new", (extra,))
        except OSError as error:
            errors.append(error)

    def read():
        try:
            observed.append(bootstrap.startup_permission(config, root))
        finally:
            reader_done.set()
            reached.set()

    reader = threading.Thread(target=read)

    def flock(fd, mode):
        if threading.current_thread() is reader and mode == bootstrap.fcntl.LOCK_SH:
            lock_attempted.set()
            reached.set()
        return original_flock(fd, mode)

    monkeypatch.setattr(bootstrap.fcntl, "flock", flock)
    writer = threading.Thread(target=publish)
    writer.start()
    try:
        assert published.wait(5)
        reader.start()
        assert reached.wait(5)
        waited_for_publisher = lock_attempted.is_set() and not reader_done.is_set()
    finally:
        release.set()
        writer.join(5)
        if reader.ident is not None:
            reader.join(5)
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert bool(errors) is interrupted
    expected = (False, "recovery_scope_uncertain") if interrupted else (True, "startup_allowed")
    assert observed == [expected]
    assert waited_for_publisher
    assert (root / "admission" / "registry.pending.json").exists() is interrupted


def test_startup_registry_read_can_nest_existing_shared_native_lock(local_scope):  # noqa: F811
    from tldw_chatbook.Utils.platform_files import fcntl

    root, config, _, authority = local_scope
    with (
        authority._directory() as parent,
        authority._lock(parent, "registry.lock", fcntl.LOCK_SH),
    ):
        assert bootstrap.startup_permission(config, root) == (True, "startup_allowed")


@pytest.mark.parametrize("damage", ["absent", "public", "directory", "hardlink"])
def test_startup_never_repairs_missing_or_unsafe_registry_lock(local_scope, damage):  # noqa: F811
    from tldw_chatbook.Utils.platform_files import os

    root, config, _, _ = local_scope
    lock = root / "admission" / "registry.lock"
    if damage == "public":
        if sys.platform == "win32":
            subprocess.run(  # nosec B603 B607
                ["icacls", str(lock), "/grant", "*S-1-1-0:(R)"],
                check=True,
                capture_output=True,
            )
        else:
            os.chmod(lock, 0o644)
        assert os.stat(lock).st_mode & 0o044
    elif damage == "hardlink":
        os.link(lock, lock.with_name("alias.lock"))
    else:
        lock.unlink()
        if damage == "directory":
            lock.mkdir(mode=0o700)
    assert bootstrap.startup_permission(config, root) == (False, "recovery_scope_uncertain")
    if damage == "absent":
        assert not lock.exists()
    elif damage == "public":
        assert os.stat(lock).st_mode & 0o044
