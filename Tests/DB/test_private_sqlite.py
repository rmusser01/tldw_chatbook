from __future__ import annotations

import os
import sqlite3
import stat
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

import tldw_chatbook.DB.private_sqlite as private_sqlite
from tldw_chatbook.DB.private_sqlite import (
    SQLitePrivacyUnverifiedWarning,
    _build_read_only_uri,
    connect_private_sqlite,
)
from tldw_chatbook.Utils.private_paths import PrivatePathError, PrivatePathStatus


class StringPath:
    def __init__(self, value: str) -> None:
        self.value = value

    def __fspath__(self) -> str:
        return self.value


@pytest.mark.parametrize(
    "database", [":memory:", Path(":memory:"), StringPath(":memory:")]
)
def test_exact_memory_token_preserves_sqlite_memory_behavior(database):
    connection = connect_private_sqlite("db.base", database)
    try:
        connection.execute("CREATE TABLE memory_only (value TEXT)")
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'memory_only'"
        ).fetchone() == ("memory_only",)
    finally:
        connection.close()


@pytest.mark.parametrize("lookalike", [":memory: ", "./:memory:", "memory:"])
def test_memory_only_owner_rejects_lookalike_tokens(lookalike):
    with pytest.raises(ValueError, match="target kind"):
        connect_private_sqlite("notifications.client", lookalike)


def test_connection_accepts_str_path_and_pathlike_file_targets(tmp_path):
    for database in [
        str(tmp_path / "string.sqlite"),
        Path(tmp_path / "path.sqlite"),
        StringPath(str(tmp_path / "pathlike.sqlite")),
    ]:
        connection = connect_private_sqlite("db.base", database)
        connection.close()
        assert Path(os.fspath(database)).exists()


def test_connection_selection_never_resolves_path(tmp_path, monkeypatch):
    target = tmp_path / "db.sqlite"

    def fail_resolve(*args, **kwargs):
        pytest.fail("database target selection called Path.resolve()")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert target.exists()


def test_connection_rejects_unknown_owner(tmp_path):
    with pytest.raises(ValueError, match="Unknown SQLite owner"):
        connect_private_sqlite("unknown.owner", tmp_path / "db.sqlite")


def test_connection_rejects_owner_target_kind_mismatch(tmp_path):
    with pytest.raises(ValueError, match="target kind"):
        connect_private_sqlite("notifications.client", tmp_path / "db.sqlite")

    with pytest.raises(ValueError, match="read-only"):
        connect_private_sqlite("settings.integrity", tmp_path / "db.sqlite")
    with pytest.raises(ValueError, match="target kind"):
        connect_private_sqlite("db.base", tmp_path / "db.sqlite", read_only=True)
    with pytest.raises(ValueError, match="read-only memory"):
        connect_private_sqlite("db.base", Path(":memory:"), read_only=True)


def test_connection_rejects_nul_uri_override_and_arbitrary_file_uri(tmp_path):
    with pytest.raises(ValueError, match="NUL"):
        connect_private_sqlite("db.base", "bad\x00.sqlite")
    with pytest.raises(ValueError, match="uri"):
        connect_private_sqlite("db.base", tmp_path / "db.sqlite", uri=True)
    with pytest.raises(ValueError, match="file:"):
        connect_private_sqlite("db.base", "file:/tmp/db.sqlite")
    with pytest.raises(ValueError, match="file:"):
        connect_private_sqlite(
            "settings.integrity",
            "file:/tmp/db.sqlite?mode=ro&immutable=1",
            read_only=True,
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode contract")
def test_first_database_creation_is_private_under_permissive_umask(tmp_path):
    target = tmp_path / "new.sqlite"
    previous = os.umask(0)
    try:
        connection = connect_private_sqlite("db.base", target)
        connection.close()
    finally:
        os.umask(previous)

    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX ordering contract")
def test_existing_database_is_hardened_before_raw_connect(tmp_path, monkeypatch):
    target = tmp_path / "existing.sqlite"
    target.write_bytes(b"")
    target.chmod(0o644)
    observed_modes = []

    def observe_connect(database, **kwargs):
        observed_modes.append(stat.S_IMODE(target.stat().st_mode))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert observed_modes == [0o600]


@pytest.mark.skipif(os.name != "posix", reason="POSIX hardening contract")
def test_existing_0400_database_is_hardened_before_writable_raw_connect(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "existing.sqlite"
    target.write_bytes(b"")
    target.chmod(0o400)
    observed_modes = []

    def observe_connect(database, **kwargs):
        observed_modes.append(stat.S_IMODE(target.stat().st_mode))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert observed_modes == [0o600]


@pytest.mark.skipif(os.name != "posix", reason="POSIX target contract")
def test_writable_connection_fails_before_raw_connect_for_unsafe_targets(
    tmp_path,
    monkeypatch,
):
    outside = tmp_path / "outside.sqlite"
    outside.write_bytes(b"outside")
    symlink = tmp_path / "symlink.sqlite"
    symlink.symlink_to(outside)
    directory = tmp_path / "directory.sqlite"
    directory.mkdir()
    hardlink = tmp_path / "hardlink.sqlite"
    alias = tmp_path / "hardlink-alias.sqlite"
    hardlink.write_bytes(b"shared")
    os.link(hardlink, alias)
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        pytest.fail("raw SQLite connect was reached")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", fail_if_called)

    for target in [symlink, directory, hardlink]:
        with pytest.raises(PrivatePathError):
            connect_private_sqlite("db.base", target)

    assert called is False
    assert outside.read_bytes() == b"outside"


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership contract")
def test_wrong_owner_main_database_blocks_raw_connect(tmp_path, monkeypatch):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    target_identity = (target.stat().st_dev, target.stat().st_ino)
    real_fstat = private_sqlite.os.fstat

    def report_wrong_owner(fd):
        file_stat = real_fstat(fd)
        if (file_stat.st_dev, file_stat.st_ino) != target_identity:
            return file_stat
        fields = list(file_stat)
        fields[4] = os.geteuid() + 1000
        return os.stat_result(fields)

    monkeypatch.setattr(private_sqlite.os, "fstat", report_wrong_owner)
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached wrong-owner file"),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", target)

    assert caught.value.result.status is PrivatePathStatus.WRONG_OWNER


@pytest.mark.skipif(os.name != "posix", reason="POSIX parent contract")
def test_writable_connection_rejects_missing_or_shared_writable_parent(
    tmp_path,
    monkeypatch,
):
    missing = tmp_path / "missing" / "db.sqlite"
    shared_sticky = tmp_path / "shared-sticky"
    shared_sticky.mkdir()
    shared_sticky.chmod(0o1777)
    existing_in_shared_sticky = shared_sticky / "existing.sqlite"
    existing_in_shared_sticky.write_bytes(b"")
    existing_in_shared_sticky.chmod(0o600)
    shared_writable = tmp_path / "shared-writable"
    shared_writable.mkdir()
    shared_writable.chmod(0o777)
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        pytest.fail("raw SQLite connect was reached")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", fail_if_called)

    for target in [
        missing,
        shared_sticky / "db.sqlite",
        existing_in_shared_sticky,
        shared_writable / "db.sqlite",
    ]:
        with pytest.raises(PrivatePathError):
            connect_private_sqlite("db.base", target)

    assert called is False
    assert not missing.parent.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX identity contract")
def test_database_replacement_between_classification_and_open_is_rejected(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "db.sqlite"
    replacement = tmp_path / "replacement.sqlite"
    target.write_bytes(b"first")
    replacement.write_bytes(b"replacement")
    real_open = private_sqlite._open_artifact_fd
    raced = False

    def replace_then_open(parent_fd, leaf, *, writable, create):
        nonlocal raced
        if not raced and not create:
            raced = True
            replacement.replace(target)
        return real_open(parent_fd, leaf, writable=writable, create=create)

    monkeypatch.setattr(private_sqlite, "_open_artifact_fd", replace_then_open)
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached after race"),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", target)

    assert caught.value.result.reason == "private_sqlite_identity_changed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX reopen identity contract")
def test_database_replacement_between_hardening_and_writable_reopen_is_rejected(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "db.sqlite"
    replacement = tmp_path / "replacement.sqlite"
    target.write_bytes(b"first")
    target.chmod(0o400)
    replacement.write_bytes(b"replacement")
    real_open = private_sqlite._open_artifact_fd
    target_open_count = 0

    def replace_before_writable_reopen(parent_fd, leaf, *, writable, create):
        nonlocal target_open_count
        if leaf == target.name and not create:
            target_open_count += 1
            if target_open_count == 2:
                replacement.replace(target)
        return real_open(parent_fd, leaf, writable=writable, create=create)

    monkeypatch.setattr(
        private_sqlite,
        "_open_artifact_fd",
        replace_before_writable_reopen,
    )
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached raced target"),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", target)

    assert target_open_count == 2
    assert caught.value.result.reason == "private_sqlite_identity_changed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX postcondition contract")
def test_database_postcondition_failure_blocks_raw_connect(tmp_path, monkeypatch):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    monkeypatch.setattr(
        private_sqlite,
        "_artifact_postcondition_holds",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached failed target"),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", target)

    assert caught.value.result.reason == "private_sqlite_postcondition_failed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX failure residue contract")
def test_raw_connect_failure_retains_only_private_residue(tmp_path, monkeypatch):
    target = tmp_path / "db.sqlite"

    def fail_connect(*args, **kwargs):
        raise sqlite3.OperationalError("simulated")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", fail_connect)

    with pytest.raises(sqlite3.OperationalError, match="simulated"):
        connect_private_sqlite("db.base", target)

    assert target.exists()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not (tmp_path / "db.sqlite-wal").exists()
    assert not (tmp_path / "db.sqlite-shm").exists()
    assert not (tmp_path / "db.sqlite-journal").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-mutation contract")
def test_connection_does_not_create_or_chmod_parent(tmp_path, monkeypatch):
    parent = tmp_path / "custom"
    parent.mkdir(mode=0o750)
    target = parent / "db.sqlite"
    before = stat.S_IMODE(parent.stat().st_mode)
    real_fchmod = private_sqlite.os.fchmod

    def reject_parent_chmod(fd, mode):
        opened = os.fstat(fd)
        if stat.S_ISDIR(opened.st_mode):
            pytest.fail("database seam chmodded its parent")
        return real_fchmod(fd, mode)

    monkeypatch.setattr(private_sqlite.os, "fchmod", reject_parent_chmod)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert stat.S_IMODE(parent.stat().st_mode) == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_existing_sidecars_are_hardened_before_raw_connect(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"historical")
    sidecar.chmod(0o644)
    observed_modes = []

    def observe_connect(database, **kwargs):
        observed_modes.append(stat.S_IMODE(sidecar.stat().st_mode))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert observed_modes == [0o600]


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar hardening contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_existing_0400_sidecars_are_hardened_before_writable_raw_connect(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"historical")
    sidecar.chmod(0o400)
    observed_modes = []

    def observe_connect(database, **kwargs):
        observed_modes.append(stat.S_IMODE(sidecar.stat().st_mode))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert observed_modes == [0o600]


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "directory"])
def test_unsafe_existing_sidecar_blocks_raw_connect(
    tmp_path,
    monkeypatch,
    suffix,
    unsafe_kind,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    if unsafe_kind == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(b"outside")
        sidecar.symlink_to(outside)
    elif unsafe_kind == "hardlink":
        sidecar.write_bytes(b"shared")
        os.link(sidecar, tmp_path / "sidecar-alias")
    else:
        sidecar.mkdir()

    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached unsafe sidecar"),
    )

    with pytest.raises(PrivatePathError):
        connect_private_sqlite("db.base", target)


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar ownership contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_wrong_owner_existing_sidecar_blocks_raw_connect(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"historical")
    sidecar_identity = (sidecar.stat().st_dev, sidecar.stat().st_ino)
    real_fstat = private_sqlite.os.fstat

    def report_wrong_owner(fd):
        file_stat = real_fstat(fd)
        if (file_stat.st_dev, file_stat.st_ino) != sidecar_identity:
            return file_stat
        fields = list(file_stat)
        fields[4] = os.geteuid() + 1000
        return os.stat_result(fields)

    monkeypatch.setattr(private_sqlite.os, "fstat", report_wrong_owner)
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached wrong-owner sidecar"),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", target)

    assert caught.value.result.status is PrivatePathStatus.WRONG_OWNER


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar identity contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sidecar_replacement_blocks_raw_connect(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"sidecar")
    replacement = tmp_path / f"replacement{suffix}"
    replacement.write_bytes(b"replacement")
    real_open = private_sqlite._open_artifact_fd
    raced = False

    def replace_then_open(parent_fd, leaf, *, writable, create):
        nonlocal raced
        if leaf == sidecar.name and not raced and not create:
            raced = True
            replacement.replace(sidecar)
        return real_open(parent_fd, leaf, writable=writable, create=create)

    monkeypatch.setattr(private_sqlite, "_open_artifact_fd", replace_then_open)
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached raced sidecar"),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", target)

    assert caught.value.result.reason == "private_sqlite_identity_changed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar reopen contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sidecar_replacement_between_hardening_and_writable_reopen_is_rejected(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"sidecar")
    sidecar.chmod(0o400)
    replacement = tmp_path / f"replacement{suffix}"
    replacement.write_bytes(b"replacement")
    real_open = private_sqlite._open_artifact_fd
    sidecar_open_count = 0

    def replace_before_writable_reopen(parent_fd, leaf, *, writable, create):
        nonlocal sidecar_open_count
        if leaf == sidecar.name and not create:
            sidecar_open_count += 1
            if sidecar_open_count == 2:
                replacement.replace(sidecar)
        return real_open(parent_fd, leaf, writable=writable, create=create)

    monkeypatch.setattr(
        private_sqlite,
        "_open_artifact_fd",
        replace_before_writable_reopen,
    )
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached raced sidecar"),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", target)

    assert sidecar_open_count == 2
    assert caught.value.result.reason == "private_sqlite_identity_changed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar identity contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sidecar_postcondition_failure_blocks_raw_connect(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"sidecar")
    monkeypatch.setattr(
        private_sqlite,
        "_artifact_postcondition_holds",
        lambda *args, **kwargs: False if kwargs.get("selected") == sidecar else True,
    )
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached unsafe sidecar"),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", target)

    assert caught.value.result.reason == "private_sqlite_postcondition_failed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar contract")
def test_missing_sidecars_are_not_precreated(tmp_path, monkeypatch):
    target = tmp_path / "db.sqlite"
    seen = []

    def observe_connect(database, **kwargs):
        seen.extend(
            Path(f"{target}{suffix}").exists()
            for suffix in ["-wal", "-shm", "-journal"]
        )
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert seen == [False, False, False]


def _assert_private_regular_owned(path: Path) -> None:
    file_stat = path.stat()
    assert stat.S_ISREG(file_stat.st_mode)
    assert file_stat.st_uid == os.geteuid()
    assert stat.S_IMODE(file_stat.st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX SQLite artifact contract")
def test_real_wal_and_shm_are_private_under_permissive_umask(tmp_path):
    target = tmp_path / "wal.sqlite"
    previous = os.umask(0)
    try:
        connection = connect_private_sqlite("db.base", target)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE items (value TEXT)")
        connection.execute("INSERT INTO items VALUES ('private')")
        connection.commit()
        _assert_private_regular_owned(target)
        _assert_private_regular_owned(Path(f"{target}-wal"))
        _assert_private_regular_owned(Path(f"{target}-shm"))
        connection.close()
    finally:
        os.umask(previous)


@pytest.mark.skipif(os.name != "posix", reason="POSIX SQLite artifact contract")
def test_real_rollback_journal_is_private_under_permissive_umask(tmp_path):
    target = tmp_path / "delete.sqlite"
    previous = os.umask(0)
    try:
        connection = connect_private_sqlite("db.base", target)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE items (value TEXT)")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO items VALUES ('private')")
        _assert_private_regular_owned(Path(f"{target}-journal"))
        connection.rollback()
        connection.close()
    finally:
        os.umask(previous)


@pytest.mark.skipif(os.name != "posix", reason="POSIX SQLite sidecar contract")
def test_real_reopen_hardens_existing_wal_and_shm_before_use(tmp_path):
    target = tmp_path / "historical-wal.sqlite"
    first = connect_private_sqlite("db.base", target)
    first.execute("PRAGMA journal_mode=WAL")
    first.execute("CREATE TABLE items (value TEXT)")
    first.execute("INSERT INTO items VALUES ('private')")
    first.commit()
    wal = Path(f"{target}-wal")
    shm = Path(f"{target}-shm")
    wal.chmod(0o644)
    shm.chmod(0o644)

    second = connect_private_sqlite("db.base", target)
    try:
        assert second.execute("SELECT value FROM items").fetchone() == ("private",)
        _assert_private_regular_owned(wal)
        _assert_private_regular_owned(shm)
    finally:
        second.close()
        first.close()


@pytest.mark.parametrize(
    "name", ["space name.sqlite", "query?.sqlite", "hash#.sqlite", "雪.sqlite"]
)
def test_read_only_uri_preserves_special_filename_identity_and_rejects_writes(
    tmp_path,
    name,
):
    target = tmp_path / name
    source = sqlite3.connect(target)
    source.execute("CREATE TABLE items (value TEXT)")
    source.execute("INSERT INTO items VALUES ('expected')")
    source.commit()
    source.close()

    connection = connect_private_sqlite(
        "settings.integrity",
        target,
        read_only=True,
    )
    try:
        assert connection.execute("SELECT value FROM items").fetchone() == ("expected",)
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("INSERT INTO items VALUES ('forbidden')")
    finally:
        connection.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX shared-sticky contract")
def test_read_only_owned_file_is_rejected_in_shared_sticky_parent(
    tmp_path,
    monkeypatch,
):
    shared = tmp_path / "shared"
    shared.mkdir()
    target = shared / "db.sqlite"
    source = sqlite3.connect(target)
    source.execute("CREATE TABLE items (value TEXT)")
    source.close()
    target.chmod(0o600)
    shared.chmod(0o1777)
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail(
            "raw SQLite connect reached shared-sticky namespace"
        ),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite(
            "settings.integrity",
            target,
            read_only=True,
        )

    assert caught.value.result.status is PrivatePathStatus.UNSAFE_PARENT


@pytest.mark.skipif(os.name != "posix", reason="POSIX SQLite namespace contract")
def test_read_only_wal_open_in_shared_sticky_parent_never_uses_or_creates_shm(
    tmp_path,
):
    shared = tmp_path / "shared"
    shared.mkdir()
    target = shared / "wal.sqlite"
    source = sqlite3.connect(target)
    source.execute("PRAGMA journal_mode=WAL")
    source.execute("CREATE TABLE items (value TEXT)")
    source.execute("INSERT INTO items VALUES ('private')")
    source.commit()
    shm = Path(f"{target}-shm")
    public_shm = b"public replacement must remain untouched"
    try:
        shm.unlink()
        shared.chmod(0o1777)

        with pytest.raises(PrivatePathError):
            connect_private_sqlite(
                "settings.integrity",
                target,
                read_only=True,
            )

        assert not shm.exists()
        shm.write_bytes(public_shm)
        shm.chmod(0o666)
        before = shm.stat()
        try:
            with pytest.raises(PrivatePathError):
                connect_private_sqlite(
                    "settings.integrity",
                    target,
                    read_only=True,
                )
            assert shm.read_bytes() == public_shm
            assert stat.S_IMODE(shm.stat().st_mode) == 0o666
            assert (shm.stat().st_dev, shm.stat().st_ino) == (
                before.st_dev,
                before.st_ino,
            )
        finally:
            shm.unlink(missing_ok=True)
    finally:
        shared.chmod(0o700)
        source.close()
        shm.unlink(missing_ok=True)


def test_read_only_missing_and_unsafe_sources_fail_closed(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises((FileNotFoundError, PrivatePathError)):
        connect_private_sqlite("settings.integrity", missing, read_only=True)

    source = tmp_path / "source.sqlite"
    source.write_bytes(b"sqlite")
    alias = tmp_path / "alias.sqlite"
    os.link(source, alias)
    with pytest.raises(PrivatePathError):
        connect_private_sqlite("settings.integrity", source, read_only=True)


@pytest.mark.skipif(os.name != "posix", reason="POSIX read-only source contract")
@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory", "wrong_owner"])
def test_read_only_rejects_unsafe_source_kinds(tmp_path, monkeypatch, unsafe_kind):
    target = tmp_path / "source.sqlite"
    if unsafe_kind == "symlink":
        outside = tmp_path / "outside.sqlite"
        outside.write_bytes(b"outside")
        target.symlink_to(outside)
    elif unsafe_kind == "directory":
        target.mkdir()
    else:
        target.write_bytes(b"")
        target_identity = (target.stat().st_dev, target.stat().st_ino)
        real_fstat = private_sqlite.os.fstat

        def report_wrong_owner(fd):
            file_stat = real_fstat(fd)
            if (file_stat.st_dev, file_stat.st_ino) != target_identity:
                return file_stat
            fields = list(file_stat)
            fields[4] = os.geteuid() + 1000
            return os.stat_result(fields)

        monkeypatch.setattr(private_sqlite.os, "fstat", report_wrong_owner)

    with pytest.raises(PrivatePathError):
        connect_private_sqlite("settings.integrity", target, read_only=True)


def test_read_only_path_query_characters_are_percent_encoded(tmp_path, monkeypatch):
    target = tmp_path / "db.sqlite?immutable=1#fragment"
    target.write_bytes(b"")
    captured = []

    def capture_connect(database, **kwargs):
        captured.append((database, kwargs))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", capture_connect)
    connection = connect_private_sqlite(
        "settings.integrity",
        target,
        read_only=True,
    )
    connection.close()

    assert "%3Fimmutable%3D1%23fragment" in captured[0][0]
    assert captured[0][0].endswith("?mode=ro")
    assert captured[0][1]["uri"] is True


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            r"C:\Users\A B\雪?#.sqlite",
            "file:///C:/Users/A%20B/%E9%9B%AA%3F%23.sqlite?mode=ro",
        ),
        (
            r"\\server\share\A B\雪?#.sqlite",
            "file://server/share/A%20B/%E9%9B%AA%3F%23.sqlite?mode=ro",
        ),
    ],
)
def test_windows_read_only_uri_builder_percent_encodes_path(path, expected):
    assert _build_read_only_uri(path, windows=True) == expected


def test_simulated_windows_file_open_warns_but_memory_is_filesystem_free(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        private_sqlite,
        "_WARNED_UNVERIFIED_OWNER_IDS",
        set(),
    )
    monkeypatch.setattr(
        private_sqlite.private_paths,
        "_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_sqlite.private_paths, "_WINDOWS_PLATFORM", True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in ["first.sqlite", "second.sqlite"]:
            connection = connect_private_sqlite("db.base", tmp_path / name)
            connection.close()
        other_owner = connect_private_sqlite("db.evals", tmp_path / "evals.sqlite")
        other_owner.close()
        memory = connect_private_sqlite("db.base", Path(":memory:"))
        memory.close()
    privacy_warnings = [
        warning
        for warning in caught
        if warning.category is SQLitePrivacyUnverifiedWarning
    ]
    assert len(privacy_warnings) == 2
    assert all(
        "privacy is unverified" in str(warning.message) for warning in privacy_warnings
    )


def test_unverified_privacy_warning_is_thread_safe_per_owner(monkeypatch):
    monkeypatch.setattr(
        private_sqlite,
        "_WARNED_UNVERIFIED_OWNER_IDS",
        set(),
    )
    recorded = []
    monkeypatch.setattr(
        private_sqlite.warnings,
        "warn",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )
    workers = 8
    barrier = Barrier(workers)

    def warn_together():
        barrier.wait()
        private_sqlite._warn_unverified_platform("db.base")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda _: warn_together(), range(workers)))

    assert len(recorded) == 1


def test_unverified_warning_error_does_not_suppress_later_warning(monkeypatch):
    warned_owner_ids: set[str] = set()
    monkeypatch.setattr(
        private_sqlite,
        "_WARNED_UNVERIFIED_OWNER_IDS",
        warned_owner_ids,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", SQLitePrivacyUnverifiedWarning)
        with pytest.raises(SQLitePrivacyUnverifiedWarning):
            private_sqlite._warn_unverified_platform("db.base")

    assert warned_owner_ids == set()
    with pytest.warns(SQLitePrivacyUnverifiedWarning):
        private_sqlite._warn_unverified_platform("db.base")
    assert warned_owner_ids == {"db.base"}


@pytest.mark.skipif(os.name != "nt", reason="Windows functional posture")
def test_windows_exact_memory_and_read_only_functionality(tmp_path, monkeypatch):
    monkeypatch.setattr(
        private_sqlite,
        "_WARNED_UNVERIFIED_OWNER_IDS",
        set(),
    )
    memory = connect_private_sqlite("db.base", Path(":memory:"))
    memory.close()
    target = tmp_path / "source.sqlite"
    source = sqlite3.connect(target)
    source.close()
    with pytest.warns(SQLitePrivacyUnverifiedWarning):
        read_only = connect_private_sqlite(
            "settings.integrity",
            target,
            read_only=True,
        )
        read_only.close()
