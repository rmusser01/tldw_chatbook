from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import stat
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

import tldw_chatbook.DB.private_sqlite as private_sqlite
from tldw_chatbook.DB.private_sqlite import (
    SQLITE_OWNER_REGISTRY,
    SQLiteRestoreIndeterminateError,
    SQLiteRestoreBusyError,
    SQLitePrivacyUnverifiedWarning,
    SQLiteTargetKind,
    _build_read_only_uri,
    backup_connection_to_private,
    backup_open_connections_to_private,
    connect_private_sqlite,
    copy_private_sqlite,
    restore_private_sqlite,
)
from tldw_chatbook.Utils.private_paths import PrivatePathError, PrivatePathStatus


class StringPath:
    def __init__(self, value: str) -> None:
        self.value = value

    def __fspath__(self) -> str:
        return self.value


OWNER_KIND_CASES = tuple(
    (owner_id, target_kind)
    for owner_id, policy in SQLITE_OWNER_REGISTRY.items()
    for target_kind in sorted(policy.allowed_target_kinds, key=lambda kind: kind.value)
)
CONNECTION_BACKUP_OWNER_IDS = (
    "db.chachanotes.backup",
    "db.media.backup",
    "db.prompts.backup",
    "tts.profile_recovery",
)
COPY_BACKUP_OWNER_IDS = (
    "settings.bulk_backup",
    "settings.single_backup",
    "tts.profile_restore_stage",
)
OPEN_CONNECTION_BACKUP_OWNER_IDS = ("tts.profile_backup",)
RESTORE_BACKUP_OWNER_IDS = (
    "settings.pre_restore_backup",
    "settings.restore",
)


@pytest.mark.parametrize(
    ("owner_id", "target_kind"),
    OWNER_KIND_CASES,
    ids=[
        f"{owner_id}-{target_kind.value}" for owner_id, target_kind in OWNER_KIND_CASES
    ],
)
def test_every_registered_owner_executes_each_declared_target_kind(
    owner_id,
    target_kind,
    tmp_path,
):
    target = tmp_path / f"{owner_id.replace('.', '-')}.sqlite"
    if target_kind is SQLiteTargetKind.MEMORY:
        database = ":memory:"
        read_only = False
    elif target_kind is SQLiteTargetKind.PRIVATE_FILE:
        database = target
        read_only = False
    else:
        setup = connect_private_sqlite("db.base", target)
        try:
            setup.execute("CREATE TABLE owner_matrix (value INTEGER)")
            setup.execute("INSERT INTO owner_matrix VALUES (42)")
            setup.commit()
        finally:
            setup.close()
        database = target
        read_only = True

    connection = connect_private_sqlite(
        owner_id,
        database,
        read_only=read_only,
    )
    try:
        assert connection.execute("SELECT 6 * 7").fetchone() == (42,)
        if target_kind is SQLiteTargetKind.READ_ONLY_URI:
            assert connection.execute("SELECT value FROM owner_matrix").fetchone() == (
                42,
            )
            with pytest.raises(sqlite3.OperationalError):
                connection.execute("INSERT INTO owner_matrix VALUES (43)")
        else:
            connection.execute("CREATE TABLE owner_matrix (value INTEGER)")
            connection.execute("INSERT INTO owner_matrix VALUES (42)")
            assert connection.execute("SELECT value FROM owner_matrix").fetchone() == (
                42,
            )
    finally:
        connection.close()

    if target_kind is not SQLiteTargetKind.MEMORY:
        assert target.is_file()
        if os.name == "posix":
            assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_every_backup_enabled_owner_has_a_behavioral_operation() -> None:
    exercised_owner_ids = {
        *CONNECTION_BACKUP_OWNER_IDS,
        *COPY_BACKUP_OWNER_IDS,
        *OPEN_CONNECTION_BACKUP_OWNER_IDS,
        *RESTORE_BACKUP_OWNER_IDS,
    }

    assert exercised_owner_ids == {
        owner_id
        for owner_id, policy in SQLITE_OWNER_REGISTRY.items()
        if policy.centralized_backup_allowed
    }


@pytest.mark.parametrize("owner_id", CONNECTION_BACKUP_OWNER_IDS)
def test_connection_backup_owners_execute_real_transactional_backup(
    owner_id,
    tmp_path,
):
    source_path = tmp_path / f"{owner_id.replace('.', '-')}-source.sqlite"
    target = tmp_path / f"{owner_id.replace('.', '-')}.sqlite"
    source = connect_private_sqlite(owner_id, source_path)
    try:
        source.execute("CREATE TABLE owner_matrix_backup (value INTEGER)")
        source.execute("INSERT INTO owner_matrix_backup VALUES (42)")
        source.commit()

        backup_connection_to_private(owner_id, source, source_path, target)
        assert source.execute("SELECT value FROM owner_matrix_backup").fetchone() == (
            42,
        )
    finally:
        source.close()

    backup = connect_private_sqlite(owner_id, target)
    try:
        assert backup.execute("SELECT value FROM owner_matrix_backup").fetchone() == (
            42,
        )
    finally:
        backup.close()


@pytest.mark.parametrize("owner_id", COPY_BACKUP_OWNER_IDS)
def test_copy_backup_owners_execute_real_transactional_copy(owner_id, tmp_path):
    source_path = tmp_path / "source.sqlite"
    target_path = tmp_path / f"{owner_id.replace('.', '-')}.sqlite"
    source = connect_private_sqlite("db.base", source_path)
    try:
        source.execute("CREATE TABLE owner_matrix_copy (value INTEGER)")
        source.execute("INSERT INTO owner_matrix_copy VALUES (42)")
        source.commit()
    finally:
        source.close()

    copy_private_sqlite(owner_id, source_path, target_path)

    copied = connect_private_sqlite(owner_id, target_path, read_only=True)
    try:
        assert copied.execute("SELECT value FROM owner_matrix_copy").fetchone() == (42,)
    finally:
        copied.close()


@pytest.mark.parametrize("owner_id", OPEN_CONNECTION_BACKUP_OWNER_IDS)
def test_open_connection_backup_owners_execute_real_transactional_copy(
    owner_id,
    tmp_path,
):
    source_path = tmp_path / "open-source.sqlite"
    target_path = tmp_path / "open-target.sqlite"
    source = connect_private_sqlite(owner_id, source_path)
    destination = connect_private_sqlite(owner_id, target_path)
    try:
        source.execute("CREATE TABLE owner_matrix_open (value INTEGER)")
        source.execute("INSERT INTO owner_matrix_open VALUES (42)")
        source.commit()
        backup_open_connections_to_private(owner_id, source, destination)
        assert destination.execute(
            "SELECT value FROM owner_matrix_open"
        ).fetchone() == (42,)
    finally:
        destination.close()
        source.close()


def test_external_restore_source_mode_is_preserved(tmp_path):
    source_path = tmp_path / "external.sqlite"
    target_path = tmp_path / "stage.sqlite"
    source = sqlite3.connect(source_path)
    try:
        source.execute("CREATE TABLE external_restore (value INTEGER)")
        source.execute("INSERT INTO external_restore VALUES (42)")
        source.commit()
    finally:
        source.close()
    if os.name == "posix":
        source_path.chmod(0o644)
        original_mode = stat.S_IMODE(source_path.stat().st_mode)

    copy_private_sqlite(
        "tts.profile_restore_stage",
        source_path,
        target_path,
    )

    if os.name == "posix":
        assert stat.S_IMODE(source_path.stat().st_mode) == original_mode
        assert stat.S_IMODE(target_path.stat().st_mode) == 0o600


def test_restore_backup_owners_execute_real_restore_and_safety_snapshot(tmp_path):
    source_path = tmp_path / "restore-source.sqlite"
    destination_path = tmp_path / "restore-live.sqlite"
    pre_restore_path = tmp_path / "restore-safety.sqlite"

    for path, value in ((source_path, 42), (destination_path, 7)):
        connection = connect_private_sqlite("settings.restore", path)
        try:
            connection.execute("CREATE TABLE owner_matrix_restore (value INTEGER)")
            connection.execute(
                "INSERT INTO owner_matrix_restore VALUES (?)",
                (value,),
            )
            connection.commit()
        finally:
            connection.close()

    restore_private_sqlite(
        "settings.restore",
        "settings.pre_restore_backup",
        source_path,
        destination_path,
        pre_restore_path,
    )

    restored = connect_private_sqlite(
        "settings.restore",
        destination_path,
        read_only=True,
    )
    safety_snapshot = connect_private_sqlite(
        "settings.pre_restore_backup",
        pre_restore_path,
        read_only=True,
    )
    try:
        assert restored.execute(
            "SELECT value FROM owner_matrix_restore"
        ).fetchone() == (42,)
        assert safety_snapshot.execute(
            "SELECT value FROM owner_matrix_restore"
        ).fetchone() == (7,)
    finally:
        restored.close()
        safety_snapshot.close()


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
def test_safe_sidecar_replacement_at_first_open_is_fully_revalidated(
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
    replacement.chmod(0o600)
    real_open = private_sqlite._open_artifact_fd
    raced = False
    raw_connect_calls = []

    def replace_then_open(parent_fd, leaf, *, writable, create):
        nonlocal raced
        if leaf == sidecar.name and not raced and not create:
            raced = True
            replacement.replace(sidecar)
        return real_open(parent_fd, leaf, writable=writable, create=create)

    monkeypatch.setattr(private_sqlite, "_open_artifact_fd", replace_then_open)

    def observe_connect(database, **kwargs):
        raw_connect_calls.append((database, kwargs))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert raced is True
    assert raw_connect_calls
    assert sidecar.read_bytes() == b"replacement"
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar reopen contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_safe_sidecar_replacement_at_writable_reopen_is_fully_revalidated(
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
    replacement.chmod(0o600)
    real_open = private_sqlite._open_artifact_fd
    sidecar_open_count = 0
    raw_connect_calls = []

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

    def observe_connect(database, **kwargs):
        raw_connect_calls.append((database, kwargs))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert sidecar_open_count == 4
    assert raw_connect_calls
    assert sidecar.read_bytes() == b"replacement"
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


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

    assert caught.value.result.reason == "optional_sqlite_generation_churn"


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


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar unlink race contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_optional_sidecar_unlinked_after_open_is_treated_as_vanished(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"transient")
    sidecar.chmod(0o600)
    real_open = private_sqlite._open_artifact_fd
    raw_connect_calls = []

    def unlink_after_open(parent_fd, leaf, *, writable, create):
        file_fd = real_open(parent_fd, leaf, writable=writable, create=create)
        if leaf == sidecar.name:
            sidecar.unlink()
        return file_fd

    def observe_connect(database, **kwargs):
        raw_connect_calls.append((database, kwargs))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite, "_open_artifact_fd", unlink_after_open)
    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)

    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert raw_connect_calls
    assert not sidecar.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar unlink race contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_optional_replaced_sidecar_unlinked_after_initial_open_is_vanished(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"generation-a")
    sidecar.chmod(0o600)
    replacement = tmp_path / f"generation-b{suffix}"
    replacement.write_bytes(b"generation-b")
    replacement.chmod(0o600)
    real_open = private_sqlite._open_artifact_fd
    raw_connect_calls = []

    def replace_open_and_unlink(parent_fd, leaf, *, writable, create):
        if leaf == sidecar.name:
            replacement.replace(sidecar)
            file_fd = real_open(parent_fd, leaf, writable=writable, create=create)
            sidecar.unlink()
            return file_fd
        return real_open(parent_fd, leaf, writable=writable, create=create)

    def observe_connect(database, **kwargs):
        raw_connect_calls.append((database, kwargs))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(
        private_sqlite,
        "_open_artifact_fd",
        replace_open_and_unlink,
    )
    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)

    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert raw_connect_calls
    assert not sidecar.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar unlink race contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_optional_replaced_sidecar_unlinked_after_writable_reopen_is_vanished(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"generation-a")
    sidecar.chmod(0o600)
    replacement = tmp_path / f"generation-b{suffix}"
    replacement.write_bytes(b"generation-b")
    replacement.chmod(0o600)
    real_open = private_sqlite._open_artifact_fd
    sidecar_opens = 0
    raw_connect_calls = []

    def replace_open_and_unlink(parent_fd, leaf, *, writable, create):
        nonlocal sidecar_opens
        if leaf == sidecar.name:
            sidecar_opens += 1
            if sidecar_opens == 2:
                replacement.replace(sidecar)
                file_fd = real_open(
                    parent_fd,
                    leaf,
                    writable=writable,
                    create=create,
                )
                sidecar.unlink()
                return file_fd
        return real_open(parent_fd, leaf, writable=writable, create=create)

    def observe_connect(database, **kwargs):
        raw_connect_calls.append((database, kwargs))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(
        private_sqlite,
        "_open_artifact_fd",
        replace_open_and_unlink,
    )
    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)

    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert raw_connect_calls
    assert sidecar_opens == 2
    assert not sidecar.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar unlink race contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_optional_safe_sidecar_replacement_after_unlink_is_fully_revalidated(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"transient")
    sidecar.chmod(0o600)
    replacement = tmp_path / f"replacement{suffix}"
    replacement.write_bytes(b"replacement")
    replacement.chmod(0o600)
    real_open = private_sqlite._open_artifact_fd
    replaced = False
    raw_connect_calls = []

    def replace_after_open(parent_fd, leaf, *, writable, create):
        nonlocal replaced
        file_fd = real_open(parent_fd, leaf, writable=writable, create=create)
        if leaf == sidecar.name and not replaced:
            replaced = True
            sidecar.unlink()
            replacement.replace(sidecar)
        return file_fd

    monkeypatch.setattr(private_sqlite, "_open_artifact_fd", replace_after_open)

    def observe_connect(database, **kwargs):
        raw_connect_calls.append((database, kwargs))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert raw_connect_calls
    assert sidecar.read_bytes() == b"replacement"


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar generation contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_eligible_0644_sidecar_replacement_is_hardened_before_raw_connect(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"first")
    sidecar.chmod(0o600)
    replacement = tmp_path / f"replacement{suffix}"
    replacement.write_bytes(b"historical")
    replacement.chmod(0o644)
    real_open = private_sqlite._open_artifact_fd
    replaced = False
    observed_modes = []

    def replace_then_open(parent_fd, leaf, *, writable, create):
        nonlocal replaced
        if leaf == sidecar.name and not replaced:
            replaced = True
            replacement.replace(sidecar)
        return real_open(parent_fd, leaf, writable=writable, create=create)

    def observe_connect(database, **kwargs):
        observed_modes.append(stat.S_IMODE(sidecar.stat().st_mode))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite, "_open_artifact_fd", replace_then_open)
    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert observed_modes == [0o600]
    assert sidecar.read_bytes() == b"historical"


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar generation contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_optional_sidecar_disappearing_during_open_is_treated_as_absent(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"transient")
    sidecar.chmod(0o600)
    real_open = private_sqlite._open_artifact_fd
    removed = False
    raw_connect_calls = []

    def unlink_before_open(parent_fd, leaf, *, writable, create):
        nonlocal removed
        if leaf == sidecar.name and not removed:
            removed = True
            sidecar.unlink()
        return real_open(parent_fd, leaf, writable=writable, create=create)

    def observe_connect(database, **kwargs):
        raw_connect_calls.append((database, kwargs))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(private_sqlite, "_open_artifact_fd", unlink_before_open)
    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert raw_connect_calls
    assert not sidecar.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar generation contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_optional_sidecar_initial_unlinked_snapshot_revalidates_current_generation(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"current")
    sidecar.chmod(0o600)
    real_stat = private_sqlite.os.stat
    sidecar_stats = 0
    raw_connect_calls = []

    def report_unlinked_first_generation(path, *args, **kwargs):
        nonlocal sidecar_stats
        file_stat = real_stat(path, *args, **kwargs)
        if (
            path == sidecar.name
            and kwargs.get("dir_fd") is not None
            and kwargs.get("follow_symlinks") is False
        ):
            sidecar_stats += 1
            if sidecar_stats == 1:
                fields = list(file_stat)
                fields[3] = 0
                return os.stat_result(fields)
        return file_stat

    def observe_connect(database, **kwargs):
        raw_connect_calls.append((database, kwargs))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(
        private_sqlite.private_paths, "_posix_guards_available", lambda: True
    )
    monkeypatch.setattr(private_sqlite.os, "stat", report_unlinked_first_generation)
    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)

    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert raw_connect_calls
    assert sidecar_stats > 1
    assert sidecar.read_bytes() == b"current"


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar generation contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_safe_sidecar_replacement_during_postcondition_is_revalidated(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"first")
    sidecar.chmod(0o600)
    replacement = tmp_path / f"replacement{suffix}"
    replacement.write_bytes(b"replacement")
    replacement.chmod(0o600)
    real_postcondition = private_sqlite._artifact_postcondition_holds
    replaced = False
    raw_connect_calls = []

    def replace_during_postcondition(*args, **kwargs):
        nonlocal replaced
        if kwargs.get("selected") == sidecar and not replaced:
            replaced = True
            replacement.replace(sidecar)
            return False
        return real_postcondition(*args, **kwargs)

    def observe_connect(database, **kwargs):
        raw_connect_calls.append((database, kwargs))
        return sqlite3.Connection(":memory:")

    monkeypatch.setattr(
        private_sqlite,
        "_artifact_postcondition_holds",
        replace_during_postcondition,
    )
    monkeypatch.setattr(private_sqlite.sqlite3, "connect", observe_connect)
    connection = connect_private_sqlite("db.base", target)
    connection.close()

    assert raw_connect_calls
    assert sidecar.read_bytes() == b"replacement"


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar generation contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
@pytest.mark.parametrize(
    ("unsafe_kind", "expected_status"),
    [
        ("symlink", PrivatePathStatus.LINK_OR_NON_REGULAR),
        ("directory", PrivatePathStatus.LINK_OR_NON_REGULAR),
        ("hardlink", PrivatePathStatus.LINK_OR_NON_REGULAR),
        ("wrong_owner", PrivatePathStatus.WRONG_OWNER),
    ],
)
def test_unsafe_sidecar_replacement_fails_before_raw_connect(
    tmp_path,
    monkeypatch,
    suffix,
    unsafe_kind,
    expected_status,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"first")
    sidecar.chmod(0o600)
    outside = tmp_path / f"outside{suffix}"
    outside.write_bytes(b"outside")
    alias = tmp_path / f"alias{suffix}"
    real_open = private_sqlite._open_artifact_fd
    real_classify = private_sqlite.private_paths._classify_private_file_stat
    replacement_identity = None
    replaced = False

    def replace_after_open(parent_fd, leaf, *, writable, create):
        nonlocal replaced, replacement_identity
        file_fd = real_open(parent_fd, leaf, writable=writable, create=create)
        if leaf == sidecar.name and not replaced:
            replaced = True
            sidecar.unlink()
            if unsafe_kind == "symlink":
                sidecar.symlink_to(outside)
            elif unsafe_kind == "directory":
                sidecar.mkdir()
            else:
                sidecar.write_bytes(b"unsafe")
                sidecar.chmod(0o600)
                replacement_stat = sidecar.stat()
                replacement_identity = (
                    replacement_stat.st_dev,
                    replacement_stat.st_ino,
                )
                if unsafe_kind == "hardlink":
                    os.link(sidecar, alias)
        return file_fd

    def report_wrong_owner(file_stat, *, expected_uid):
        if unsafe_kind == "wrong_owner" and replacement_identity == (
            file_stat.st_dev,
            file_stat.st_ino,
        ):
            return PrivatePathStatus.WRONG_OWNER
        return real_classify(file_stat, expected_uid=expected_uid)

    monkeypatch.setattr(private_sqlite, "_open_artifact_fd", replace_after_open)
    if unsafe_kind == "wrong_owner":
        monkeypatch.setattr(
            private_sqlite.private_paths,
            "_classify_private_file_stat",
            report_wrong_owner,
        )
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail(
            "raw connect reached an unsafe sidecar replacement"
        ),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", target)

    assert caught.value.result.status is expected_status


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar generation contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_continuous_safe_sidecar_churn_exhausts_budget_and_fails_closed(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"generation-0")
    sidecar.chmod(0o600)
    real_open = private_sqlite._open_artifact_fd
    generations = 0

    def replace_every_open(parent_fd, leaf, *, writable, create):
        nonlocal generations
        file_fd = real_open(parent_fd, leaf, writable=writable, create=create)
        if leaf == sidecar.name:
            generations += 1
            sidecar.unlink()
            sidecar.write_bytes(f"generation-{generations}".encode())
            sidecar.chmod(0o600)
        return file_fd

    monkeypatch.setattr(private_sqlite, "_open_artifact_fd", replace_every_open)
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail(
            "raw connect reached continuously churning sidecar"
        ),
    )

    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", target)

    assert caught.value.result.reason == "optional_sqlite_generation_churn"
    assert generations == 4


@pytest.mark.skipif(os.name != "posix", reason="POSIX sidecar unlink race contract")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_optional_hardlinked_sidecar_cannot_be_laundered_by_unlink(
    tmp_path,
    monkeypatch,
    suffix,
):
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    sidecar = Path(f"{target}{suffix}")
    sidecar.write_bytes(b"shared")
    sidecar.chmod(0o600)
    alias = tmp_path / f"alias{suffix}"
    os.link(sidecar, alias)
    real_open = private_sqlite._open_artifact_fd

    def unlink_all_names_after_open(parent_fd, leaf, *, writable, create):
        file_fd = real_open(parent_fd, leaf, writable=writable, create=create)
        if leaf == sidecar.name:
            sidecar.unlink()
            alias.unlink()
        return file_fd

    monkeypatch.setattr(
        private_sqlite,
        "_open_artifact_fd",
        unlink_all_names_after_open,
    )
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail(
            "raw connect reached a laundered hardlinked sidecar"
        ),
    )

    with pytest.raises(PrivatePathError):
        connect_private_sqlite("db.base", target)


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


def _create_backup_fixture_database(
    path: Path,
    value: str,
    *,
    journal_mode: str = "DELETE",
) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA journal_mode={journal_mode}")
    connection.execute("CREATE TABLE backup_probe (value TEXT)")
    connection.execute("INSERT INTO backup_probe VALUES (?)", (value,))
    connection.commit()
    return connection


def _read_backup_fixture_value(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT value FROM backup_probe").fetchone()[0]
    finally:
        connection.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX backup privacy contract")
def test_backup_connection_creates_private_transactional_target_under_umask_zero(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.sqlite"
    source = _create_backup_fixture_database(source_path, "expected")
    target = tmp_path / "target.sqlite"
    previous = os.umask(0)
    try:
        backup_connection_to_private(
            "db.chachanotes.backup",
            source,
            source_path,
            target,
        )
    finally:
        os.umask(previous)
        source.close()

    assert _read_backup_fixture_value(target) == "expected"
    _assert_private_regular_owned(target)


def test_backup_connection_preserves_memory_source_and_rejects_non_backup_owner(
    tmp_path: Path,
) -> None:
    source = sqlite3.connect(":memory:")
    source.execute("CREATE TABLE backup_probe (value TEXT)")
    source.execute("INSERT INTO backup_probe VALUES ('memory')")
    source.commit()
    try:
        backup_connection_to_private(
            "db.media.backup",
            source,
            Path(":memory:"),
            tmp_path / "memory-backup.sqlite",
        )
        with pytest.raises(ValueError, match="centralized backup"):
            backup_connection_to_private(
                "db.base",
                source,
                ":memory:",
                tmp_path / "forbidden.sqlite",
            )
    finally:
        source.close()

    assert _read_backup_fixture_value(tmp_path / "memory-backup.sqlite") == "memory"
    assert not (tmp_path / "forbidden.sqlite").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX identity contract")
@pytest.mark.parametrize("alias_kind", ["same_lexical", "hardlink"])
def test_backup_connection_rejects_same_source_identity_before_destination_open(
    tmp_path: Path,
    monkeypatch,
    alias_kind: str,
) -> None:
    source_path = tmp_path / "source.sqlite"
    source = _create_backup_fixture_database(source_path, "expected")
    target = source_path
    if alias_kind == "hardlink":
        target = tmp_path / "alias.sqlite"
        os.link(source_path, target)
    raw_calls: list[tuple[object, ...]] = []

    def forbidden_connect(*args, **kwargs):
        raw_calls.append(args)
        pytest.fail("destination SQLite connection opened")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", forbidden_connect)
    try:
        with pytest.raises((PrivatePathError, ValueError)):
            backup_connection_to_private(
                "db.prompts.backup",
                source,
                source_path,
                target,
            )
    finally:
        source.close()

    assert raw_calls == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX backup privacy contract")
def test_backup_failure_retains_only_a_private_partial_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "partial.sqlite"

    class FailingSource:
        in_transaction = False

        def backup(self, destination, **kwargs):
            destination.execute("CREATE TABLE partial_state (value TEXT)")
            raise RuntimeError("injected backup failure")

    previous = os.umask(0)
    try:
        with pytest.raises(RuntimeError, match="injected backup failure"):
            backup_connection_to_private(
                "db.chachanotes.backup",
                FailingSource(),
                ":memory:",
                target,
            )
    finally:
        os.umask(previous)

    assert target.exists()
    _assert_private_regular_owned(target)


@pytest.mark.skipif(os.name != "posix", reason="POSIX WAL/source contract")
def test_copy_private_sqlite_reopens_wal_source_read_only_and_hardens_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "wal source.sqlite"
    source = _create_backup_fixture_database(
        source_path,
        "wal-visible",
        journal_mode="WAL",
    )
    target = tmp_path / "existing target.sqlite"
    old_target = _create_backup_fixture_database(target, "old")
    old_target.close()
    target.chmod(0o644)
    calls: list[tuple[str, Path, bool]] = []
    real_connect = private_sqlite._connect_registered_sqlite

    def tracking_connect(owner_id, database, *, read_only=False, **kwargs):
        calls.append((owner_id, Path(database), read_only))
        return real_connect(
            owner_id,
            database,
            read_only=read_only,
            **kwargs,
        )

    monkeypatch.setattr(
        private_sqlite,
        "_connect_registered_sqlite",
        tracking_connect,
    )
    try:
        copy_private_sqlite(
            "settings.bulk_backup",
            source_path,
            target,
        )
    finally:
        source.close()

    assert calls == [
        ("settings.bulk_backup", source_path, True),
        ("settings.bulk_backup", target, False),
    ]
    assert _read_backup_fixture_value(target) == "wal-visible"
    _assert_private_regular_owned(target)


@pytest.mark.skipif(os.name != "posix", reason="POSIX source/target contract")
def test_copy_private_sqlite_rejects_unsafe_destination_before_raw_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "source.sqlite"
    source = _create_backup_fixture_database(source_path, "expected")
    source.close()
    outside = tmp_path / "outside.sqlite"
    outside.write_bytes(b"outside")
    target = tmp_path / "target.sqlite"
    target.symlink_to(outside)
    real_raw_connect = sqlite3.connect
    raw_targets: list[object] = []

    def tracking_raw_connect(database, *args, **kwargs):
        raw_targets.append(database)
        return real_raw_connect(database, *args, **kwargs)

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", tracking_raw_connect)

    with pytest.raises(PrivatePathError):
        copy_private_sqlite(
            "settings.single_backup",
            source_path,
            target,
        )

    assert raw_targets == []
    assert outside.read_bytes() == b"outside"


def test_backup_close_failure_does_not_turn_committed_backup_into_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "source.sqlite"
    source = _create_backup_fixture_database(source_path, "expected")
    target = tmp_path / "target.sqlite"

    class FailingCloseConnection(sqlite3.Connection):
        def close(self) -> None:
            super().close()
            raise sqlite3.OperationalError("injected backup close failure")

    real_connect = private_sqlite._connect_registered_sqlite

    def instrumented_connect(owner_id, database, *, read_only=False, **kwargs):
        kwargs["factory"] = FailingCloseConnection
        return real_connect(
            owner_id,
            database,
            read_only=read_only,
            **kwargs,
        )

    monkeypatch.setattr(
        private_sqlite,
        "_connect_registered_sqlite",
        instrumented_connect,
    )
    try:
        with pytest.warns(RuntimeWarning, match="backup close failure"):
            backup_connection_to_private(
                "db.chachanotes.backup",
                source,
                source_path,
                target,
            )
        assert source.execute("SELECT 1").fetchone() == (1,)
    finally:
        source.close()

    assert _read_backup_fixture_value(target) == "expected"


def test_backup_cleanup_error_does_not_mask_operation_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "source.sqlite"
    source = _create_backup_fixture_database(source_path, "expected")
    target = tmp_path / "target.sqlite"

    class FailingCloseConnection(sqlite3.Connection):
        def close(self) -> None:
            super().close()
            raise sqlite3.OperationalError("injected cleanup failure")

    real_connect = private_sqlite._connect_registered_sqlite

    def instrumented_connect(owner_id, database, *, read_only=False, **kwargs):
        kwargs["factory"] = FailingCloseConnection
        return real_connect(
            owner_id,
            database,
            read_only=read_only,
            **kwargs,
        )

    monkeypatch.setattr(
        private_sqlite,
        "_connect_registered_sqlite",
        instrumented_connect,
    )
    monkeypatch.setattr(
        private_sqlite,
        "_backup_pages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected backup operation failure")
        ),
    )
    try:
        with pytest.warns(RuntimeWarning, match="cleanup failure"):
            with pytest.raises(RuntimeError, match="backup operation failure"):
                backup_connection_to_private(
                    "db.chachanotes.backup",
                    source,
                    source_path,
                    target,
                )
    finally:
        source.close()


def test_copy_close_failure_is_independent_and_keeps_committed_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "source.sqlite"
    source = _create_backup_fixture_database(source_path, "expected")
    source.close()
    target = tmp_path / "target.sqlite"
    close_events: list[str] = []

    class SourceConnection(sqlite3.Connection):
        def close(self) -> None:
            close_events.append("source")
            super().close()

    class FailingDestinationConnection(sqlite3.Connection):
        def close(self) -> None:
            close_events.append("destination")
            super().close()
            raise sqlite3.OperationalError("injected copy close failure")

    real_connect = private_sqlite._connect_registered_sqlite

    def instrumented_connect(owner_id, database, *, read_only=False, **kwargs):
        kwargs["factory"] = (
            SourceConnection if read_only else FailingDestinationConnection
        )
        return real_connect(
            owner_id,
            database,
            read_only=read_only,
            **kwargs,
        )

    monkeypatch.setattr(
        private_sqlite,
        "_connect_registered_sqlite",
        instrumented_connect,
    )

    with pytest.warns(RuntimeWarning, match="copy close failure"):
        copy_private_sqlite(
            "settings.single_backup",
            source_path,
            target,
        )

    assert close_events == ["destination", "source"]
    assert _read_backup_fixture_value(target) == "expected"


@pytest.mark.skipif(os.name != "posix", reason="POSIX restore contract")
@pytest.mark.parametrize("destination_journal_mode", ["DELETE", "WAL"])
@pytest.mark.parametrize("source_journal_mode", ["DELETE", "WAL"])
def test_restore_keeps_idle_connection_coherent_and_creates_pre_restore_backup(
    tmp_path: Path,
    destination_journal_mode: str,
    source_journal_mode: str,
) -> None:
    destination = tmp_path / "live.sqlite"
    current = _create_backup_fixture_database(
        destination,
        "before",
        journal_mode=destination_journal_mode,
    )
    current.close()
    source_path = tmp_path / "selected-backup.sqlite"
    selected = _create_backup_fixture_database(
        source_path,
        "after",
        journal_mode=source_journal_mode,
    )
    selected.close()
    pre_restore = tmp_path / "pre-restore.sqlite"
    idle = sqlite3.connect(destination, timeout=0)
    try:
        restore_private_sqlite(
            "settings.restore",
            "settings.pre_restore_backup",
            source_path,
            destination,
            pre_restore,
        )
        observed = idle.execute("SELECT value FROM backup_probe").fetchone()[0]
    finally:
        idle.close()

    assert observed == "after"
    assert _read_backup_fixture_value(destination) == "after"
    assert _read_backup_fixture_value(pre_restore) == "before"
    verification = sqlite3.connect(destination)
    try:
        assert (
            verification.execute("PRAGMA journal_mode").fetchone()[0].upper()
            == destination_journal_mode
        )
    finally:
        verification.close()
    _assert_private_regular_owned(destination)
    _assert_private_regular_owned(pre_restore)


@pytest.mark.skipif(os.name != "posix", reason="POSIX restore lock contract")
@pytest.mark.parametrize("journal_mode", ["DELETE", "WAL"])
@pytest.mark.parametrize("transaction_kind", ["reader", "writer"])
def test_restore_fails_promptly_and_unchanged_for_active_transactions(
    tmp_path: Path,
    journal_mode: str,
    transaction_kind: str,
) -> None:
    destination = tmp_path / "live.sqlite"
    current = _create_backup_fixture_database(
        destination,
        "before",
        journal_mode=journal_mode,
    )
    current.close()
    source_path = tmp_path / "selected-backup.sqlite"
    selected = _create_backup_fixture_database(source_path, "after")
    selected.close()
    pre_restore = tmp_path / "pre-restore.sqlite"
    active = sqlite3.connect(destination, timeout=0)
    if transaction_kind == "reader":
        active.execute("BEGIN")
        active.execute("SELECT value FROM backup_probe").fetchone()
    else:
        active.execute("BEGIN IMMEDIATE")
        active.execute("UPDATE backup_probe SET value = 'uncommitted'")

    started = time.monotonic()
    try:
        with pytest.raises(
            SQLiteRestoreBusyError,
            match="[Cc]lose.*retry|retry.*[Cc]lose",
        ):
            restore_private_sqlite(
                "settings.restore",
                "settings.pre_restore_backup",
                source_path,
                destination,
                pre_restore,
            )
    finally:
        elapsed = time.monotonic() - started
        active.rollback()
        active.close()

    assert elapsed < 1.0
    assert _read_backup_fixture_value(destination) == "before"
    assert not pre_restore.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX WAL restore contract")
def test_restore_fails_closed_for_queried_idle_wal_connection(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "live.sqlite"
    current = _create_backup_fixture_database(
        destination,
        "before",
        journal_mode="WAL",
    )
    current.close()
    source_path = tmp_path / "selected-backup.sqlite"
    selected = _create_backup_fixture_database(source_path, "after")
    selected.close()
    pre_restore = tmp_path / "pre-restore.sqlite"
    queried_idle = sqlite3.connect(destination, timeout=0)
    queried_idle.execute("SELECT value FROM backup_probe").fetchone()
    assert queried_idle.in_transaction is False
    try:
        with pytest.raises(SQLiteRestoreBusyError, match="live restore is unavailable"):
            restore_private_sqlite(
                "settings.restore",
                "settings.pre_restore_backup",
                source_path,
                destination,
                pre_restore,
            )
    finally:
        queried_idle.close()

    assert _read_backup_fixture_value(destination) == "before"
    assert not pre_restore.exists()


def test_restore_guard_rejects_unsupported_destination_journal_mode() -> None:
    class Result:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class UnsupportedJournalConnection:
        def execute(self, statement):
            if statement == "PRAGMA busy_timeout = 0":
                return Result(None)
            if statement == "PRAGMA journal_mode":
                return Result(("persist",))
            pytest.fail(f"restore continued after unsupported mode: {statement}")

    with pytest.raises(ValueError, match="does not support persist"):
        private_sqlite._guard_destination(
            UnsupportedJournalConnection(),
            restore=True,
        )


def test_restore_guard_rolls_back_wal_probe_when_exclusive_begin_fails(
    tmp_path: Path,
) -> None:
    database = tmp_path / "live.sqlite"
    connection = _create_backup_fixture_database(
        database,
        "before",
        journal_mode="WAL",
    )

    class FailExclusiveBegin:
        def execute(self, statement):
            if statement == "BEGIN EXCLUSIVE":
                raise sqlite3.OperationalError("injected exclusive failure")
            return connection.execute(statement)

        def rollback(self):
            return connection.rollback()

    try:
        with pytest.raises(SQLiteRestoreBusyError):
            private_sqlite._guard_destination(
                FailExclusiveBegin(),
                restore=True,
            )
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()


def test_restore_failure_before_final_backup_keeps_old_data_and_wal_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "live.sqlite"
    current = _create_backup_fixture_database(
        destination,
        "before",
        journal_mode="WAL",
    )
    current.close()
    source_path = tmp_path / "selected-backup.sqlite"
    selected = _create_backup_fixture_database(source_path, "after")
    selected.close()
    pre_restore = tmp_path / "pre-restore.sqlite"
    real_restore_mode = private_sqlite._restore_destination_mode
    injected = False

    def fail_once(connection, journal_mode, *, restore):
        nonlocal injected
        if restore and journal_mode == "wal" and not injected:
            injected = True
            raise sqlite3.OperationalError("injected WAL restoration failure")
        return real_restore_mode(
            connection,
            journal_mode,
            restore=restore,
        )

    monkeypatch.setattr(
        private_sqlite,
        "_restore_destination_mode",
        fail_once,
    )

    with pytest.raises(
        sqlite3.OperationalError,
        match="injected WAL restoration failure",
    ):
        restore_private_sqlite(
            "settings.restore",
            "settings.pre_restore_backup",
            source_path,
            destination,
            pre_restore,
        )

    assert _read_backup_fixture_value(destination) == "before"
    verification = sqlite3.connect(destination)
    try:
        assert verification.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        verification.close()


def test_restore_mode_failure_after_final_backup_rolls_back_data_and_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "live.sqlite"
    current = _create_backup_fixture_database(
        destination,
        "before",
        journal_mode="WAL",
    )
    current.close()
    source_path = tmp_path / "selected-backup.sqlite"
    selected = _create_backup_fixture_database(
        source_path,
        "after",
        journal_mode="DELETE",
    )
    selected.close()
    pre_restore = tmp_path / "pre-restore.sqlite"
    real_restore_mode = private_sqlite._restore_destination_mode
    restore_calls = 0

    def fail_after_final_backup(connection, journal_mode, *, restore):
        nonlocal restore_calls
        if restore:
            restore_calls += 1
            if restore_calls == 2:
                raise sqlite3.OperationalError(
                    "injected post-backup mode restoration failure"
                )
        return real_restore_mode(
            connection,
            journal_mode,
            restore=restore,
        )

    monkeypatch.setattr(
        private_sqlite,
        "_restore_destination_mode",
        fail_after_final_backup,
    )

    with pytest.raises(
        sqlite3.OperationalError,
        match="post-backup mode restoration failure",
    ):
        restore_private_sqlite(
            "settings.restore",
            "settings.pre_restore_backup",
            source_path,
            destination,
            pre_restore,
        )

    assert restore_calls == 3
    assert _read_backup_fixture_value(destination) == "before"
    assert _read_backup_fixture_value(pre_restore) == "before"
    verification = sqlite3.connect(destination)
    try:
        assert verification.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        verification.close()


def test_restore_rollback_failure_reports_indeterminate_live_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "live.sqlite"
    current = _create_backup_fixture_database(destination, "before")
    current.close()
    source_path = tmp_path / "selected-backup.sqlite"
    selected = _create_backup_fixture_database(source_path, "after")
    selected.close()
    pre_restore = tmp_path / "pre-restore.sqlite"
    real_reverify = private_sqlite._reverify_source
    real_backup_pages = private_sqlite._backup_pages
    real_restore_mode = private_sqlite._restore_destination_mode
    restore_backup_calls = 0
    final_backup_completed = False
    rollback_failed = False

    def fail_post_commit_reverification(source_pin):
        if final_backup_completed:
            raise private_sqlite._failure(
                source_path,
                PrivatePathStatus.OPERATION_FAILED,
                "injected_post_commit_source_change",
            )
        return real_reverify(source_pin)

    def fail_recovery_backup(source, target, *, restore):
        nonlocal final_backup_completed, restore_backup_calls, rollback_failed
        if restore:
            restore_backup_calls += 1
            if restore_backup_calls == 2:
                rollback_failed = True
                raise sqlite3.OperationalError("injected rollback backup failure")
        result = real_backup_pages(source, target, restore=restore)
        if restore:
            final_backup_completed = True
        return result

    def fail_cleanup_mode(connection, journal_mode, *, restore):
        if rollback_failed and restore:
            raise sqlite3.OperationalError("injected cleanup mode failure")
        return real_restore_mode(
            connection,
            journal_mode,
            restore=restore,
        )

    monkeypatch.setattr(
        private_sqlite,
        "_reverify_source",
        fail_post_commit_reverification,
    )
    monkeypatch.setattr(
        private_sqlite,
        "_backup_pages",
        fail_recovery_backup,
    )
    monkeypatch.setattr(
        private_sqlite,
        "_restore_destination_mode",
        fail_cleanup_mode,
    )

    with pytest.raises(
        SQLiteRestoreIndeterminateError,
        match="may already contain restored data",
    ) as caught:
        restore_private_sqlite(
            "settings.restore",
            "settings.pre_restore_backup",
            source_path,
            destination,
            pre_restore,
        )

    assert str(pre_restore) in str(caught.value)
    assert "Do not retry" in str(caught.value)
    assert any(
        "cleanup mode failure" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert _read_backup_fixture_value(destination) == "after"
    assert _read_backup_fixture_value(pre_restore) == "before"


def test_restore_final_backup_failure_is_transactional_and_keeps_prebackup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "live.sqlite"
    current = _create_backup_fixture_database(
        destination,
        "before",
        journal_mode="WAL",
    )
    current.close()
    source_path = tmp_path / "selected-backup.sqlite"
    selected = _create_backup_fixture_database(source_path, "after")
    selected.close()
    pre_restore = tmp_path / "pre-restore.sqlite"
    real_backup_pages = private_sqlite._backup_pages

    def fail_final(source, target, *, restore):
        if restore:
            raise RuntimeError("injected final backup failure")
        return real_backup_pages(source, target, restore=restore)

    monkeypatch.setattr(private_sqlite, "_backup_pages", fail_final)

    with pytest.raises(RuntimeError, match="injected final backup failure"):
        restore_private_sqlite(
            "settings.restore",
            "settings.pre_restore_backup",
            source_path,
            destination,
            pre_restore,
        )

    assert _read_backup_fixture_value(destination) == "before"
    assert _read_backup_fixture_value(pre_restore) == "before"
    verification = sqlite3.connect(destination)
    try:
        assert verification.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        verification.close()


def test_restore_close_failure_does_not_mask_commit_or_skip_other_connections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "live.sqlite"
    current = _create_backup_fixture_database(destination, "before")
    current.close()
    source_path = tmp_path / "selected-backup.sqlite"
    selected = _create_backup_fixture_database(source_path, "after")
    selected.close()
    pre_restore = tmp_path / "pre-restore.sqlite"
    close_events: list[str] = []
    opened: list[sqlite3.Connection] = []

    class SourceConnection(sqlite3.Connection):
        def close(self) -> None:
            close_events.append("source")
            super().close()

    class DestinationConnection(sqlite3.Connection):
        def close(self) -> None:
            close_events.append("destination")
            super().close()

    class FailingPreRestoreConnection(sqlite3.Connection):
        def close(self) -> None:
            close_events.append("pre_restore")
            raise sqlite3.OperationalError("injected pre-restore close failure")

    real_connect = private_sqlite._connect_registered_sqlite

    def instrumented_connect(
        owner_id,
        database,
        *,
        read_only=False,
        **kwargs,
    ):
        if owner_id == "settings.pre_restore_backup":
            kwargs["factory"] = FailingPreRestoreConnection
        elif read_only:
            kwargs["factory"] = SourceConnection
        else:
            kwargs["factory"] = DestinationConnection
        connection = real_connect(
            owner_id,
            database,
            read_only=read_only,
            **kwargs,
        )
        opened.append(connection)
        return connection

    monkeypatch.setattr(
        private_sqlite,
        "_connect_registered_sqlite",
        instrumented_connect,
    )
    try:
        with pytest.warns(RuntimeWarning, match="pre-restore close failure"):
            restore_private_sqlite(
                "settings.restore",
                "settings.pre_restore_backup",
                source_path,
                destination,
                pre_restore,
            )

        assert close_events == ["destination", "pre_restore", "source"]
        assert _read_backup_fixture_value(destination) == "after"
    finally:
        for connection in opened:
            with contextlib.suppress(sqlite3.Error):
                sqlite3.Connection.close(connection)


def test_restore_rejects_same_source_and_destination_before_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "same.sqlite"
    source = _create_backup_fixture_database(database, "before")
    source.close()
    raw_calls: list[object] = []

    def forbidden_connect(*args, **kwargs):
        raw_calls.append(args)
        pytest.fail("raw SQLite connection opened")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", forbidden_connect)

    with pytest.raises(ValueError, match="same"):
        restore_private_sqlite(
            "settings.restore",
            "settings.pre_restore_backup",
            database,
            database,
            tmp_path / "pre-restore.sqlite",
        )

    assert raw_calls == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory contract")
def test_sqlite_opens_under_the_unresolved_platform_temporary_directory():
    """TASK-950: `tempfile.gettempdir()` crosses `/var -> private/var` on macOS.

    `tmp_path` hides this because pytest resolves its base directory, so this
    test deliberately uses the platform temporary path exactly as the stdlib
    hands it out.
    """

    scratch = Path(tempfile.mkdtemp(prefix="tldw-950-sqlite-"))
    try:
        target = scratch / "task950.sqlite"
        connection = connect_private_sqlite("db.base", target)
        try:
            connection.execute("CREATE TABLE probe (value INTEGER)")
            connection.execute("INSERT INTO probe VALUES (?)", (950,))
            connection.commit()
            assert connection.execute("SELECT value FROM probe").fetchone() == (950,)
        finally:
            connection.close()
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
