"""Native projection-dependency publication and cross-process lock behavior."""

import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.native_files import create_private_directory
from tldw_chatbook.RAG_Search import generation
from tldw_chatbook.Utils.platform_files import fcntl, os


class _WindowsRoute:
    name = "nt"

    def __getattr__(self, name):
        return getattr(os, name)


@pytest.fixture
def dependencies(tmp_path, monkeypatch):
    root = tmp_path / "control"
    create_private_directory(root)
    store = SimpleNamespace(
        persist_directory=tmp_path / "vectors", collection_name="test"
    )
    lease = SimpleNamespace(execution_context=lambda _: (root, None))
    monkeypatch.setattr(generation, "os", _WindowsRoute(), raising=False)
    return root, store, lease


def test_dependency_union_retains_same_private_regular_lock(dependencies):
    root, store, lease = dependencies
    first, second = root.parent / "first.db", root.parent / "second.db"
    assert generation._persist_dependencies(store, lease, {first}) == {first}
    lock = root / "projection-dependencies" / ".publication.lock"
    before = os.stat(lock, follow_symlinks=False)
    assert stat.S_ISREG(before.st_mode) and before.st_mode & 0o777 == 0o600
    assert before.st_nlink == 1 and before.st_uid == os.geteuid()
    assert generation._persist_dependencies(store, lease, {second}) == {first, second}
    after = os.stat(lock, follow_symlinks=False)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert bootstrap._control_records(root) == ([], [], [])


def test_independent_process_refuses_contended_lock_before_publication(dependencies):
    root, store, lease = dependencies
    first, second = root.parent / "first.db", root.parent / "second.db"
    generation._persist_dependencies(store, lease, {first})
    lock = root / "projection-dependencies" / ".publication.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
    code = """
import sys
from pathlib import Path
from types import SimpleNamespace
from tldw_chatbook.RAG_Search import generation
from tldw_chatbook.Utils.platform_files import os
class WindowsRoute:
    name = 'nt'
    def __getattr__(self, name): return getattr(os, name)
generation.os = WindowsRoute()
root, destination, source = map(Path, sys.argv[1:])
store = SimpleNamespace(persist_directory=destination, collection_name='test')
lease = SimpleNamespace(execution_context=lambda _: (root, None))
try:
    generation._persist_dependencies(store, lease, {source})
except BlockingIOError:
    print('CONTENTION_REFUSED')
else:
    raise AssertionError('independent publication bypassed held lock')
"""
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(root),
                str(store.persist_directory),
                str(second),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "CONTENTION_REFUSED"
        assert generation._dependency_paths(store, lease) == {first}
    finally:
        os.close(descriptor)
    assert generation._persist_dependencies(store, lease, {second}) == {first, second}


@pytest.mark.parametrize("kind", ["directory", "hardlink"])
def test_unsafe_existing_lock_is_refused_without_publication(dependencies, kind):
    root, store, lease = dependencies
    directory = root / "projection-dependencies"
    create_private_directory(directory)
    lock = directory / ".publication.lock"
    if kind == "directory":
        create_private_directory(lock)
    else:
        source = directory / "unrelated"
        descriptor = os.open(source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        os.link(source, lock)
    with pytest.raises((OSError, ValueError)):
        generation._persist_dependencies(store, lease, {root.parent / "source.db"})
    assert not (directory / generation._record_name(store)).exists()


def test_failed_publication_releases_lock_without_removing_stable_name(
    dependencies, monkeypatch
):
    root, store, lease = dependencies
    publish = generation._publish_dependencies

    def fail(*args):
        raise ValueError("injected_publication_failure")

    monkeypatch.setattr(generation, "_publish_dependencies", fail)
    with pytest.raises(ValueError, match="injected_publication_failure"):
        generation._persist_dependencies(store, lease, {root.parent / "first.db"})
    lock = root / "projection-dependencies" / ".publication.lock"
    before = os.stat(lock, follow_symlinks=False)
    monkeypatch.setattr(generation, "_publish_dependencies", publish)
    assert generation._persist_dependencies(
        store, lease, {root.parent / "second.db"}
    ) == {root.parent / "second.db"}
    after = os.stat(lock, follow_symlinks=False)
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory-lock regression")
def test_posix_keeps_existing_directory_lock(dependencies, monkeypatch):
    root, store, lease = dependencies
    monkeypatch.setattr(generation, "os", os)
    directory = root / "projection-dependencies"
    create_private_directory(directory)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            generation._persist_dependencies(store, lease, {root.parent / "source.db"})
    finally:
        os.close(descriptor)
    assert not (directory / ".publication.lock").exists()
