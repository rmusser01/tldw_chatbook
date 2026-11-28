"""The files SQLite creates beside a private database stay private (TASK-42749).

SQLite makes its write-ahead log (``-wal``), the log's index (``-shm``) and its
rollback journal (``-journal``) itself. On POSIX it gives them the database
file's own mode. On Windows it creates them with whatever the folder hands
down, so in a folder other users may read they were readable by those users
until the next connect made them private. The seam now creates a missing log
and index privately before SQLite needs them; SQLite opens an empty one in
place. The rollback journal is not covered (ADR-126, 2026-11-28 note).
"""

from __future__ import annotations

import os as stdlib_os
import stat
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.Utils import private_paths
from tldw_chatbook.Utils.private_paths import PrivatePathError, PrivatePathStatus

# The file opens real databases through the seam (lessons-testing-evidence,
# TASK-34000 P1-B): keep the collection-time profile.
pytestmark = pytest.mark.bootstrap_profile

_WINDOWS = private_paths._WINDOWS_PLATFORM
# BUILTIN\Users, by SID so the test does not depend on the display language.
_BUILTIN_USERS = "*S-1-5-32-545"
windows_only = pytest.mark.skipif(not _WINDOWS, reason="Windows inherited-ACL gap")


def _mode(path) -> int | None:
    """Return a path's permission bits as the private seam reads them."""
    try:
        return stat.S_IMODE(private_paths.os.stat(path, follow_symlinks=False).st_mode)
    except FileNotFoundError:
        return None


def _logs(database: Path) -> dict[str, int | None]:
    return {suffix: _mode(f"{database}{suffix}") for suffix in ("-wal", "-shm")}


@pytest.fixture
def readable_folder(tmp_path):
    """Make a folder other local users may read and never write.

    On Windows BUILTIN\\Users gets read access that files and folders created
    inside inherit. On POSIX the folder is 0o755 and the process umask is
    cleared, so nothing but the creator's own choice of mode keeps a new file
    private.
    """
    folder = tmp_path / "readable"
    folder.mkdir()
    previous_umask = None
    if _WINDOWS:
        subprocess.run(
            ["icacls", str(folder), "/grant", f"{_BUILTIN_USERS}:(OI)(CI)RX"],
            check=True,
            capture_output=True,
        )  # nosec B603 B607 -- fixed tool and synthetic temp path
        assert _mode(folder) == 0o744
    else:
        folder.chmod(0o755)
        previous_umask = stdlib_os.umask(0)
    try:
        # The premise, checked on every run: a file created here without care
        # is one other users may read. Without it the test would pass on nothing.
        ordinary = folder / "ordinary-child"
        ordinary.write_bytes(b"created without the seam")
        assert _mode(ordinary) & 0o044
        yield folder
    finally:
        if previous_umask is not None:
            stdlib_os.umask(previous_umask)


def _closed_database(folder: Path, *, wal: bool) -> Path:
    database = folder / "store.sqlite"
    connection = connect_private_sqlite("db.base", database)
    try:
        if wal:
            connection.execute("PRAGMA journal_mode=WAL").fetchone()
        connection.execute("CREATE TABLE entry(value TEXT)")
        connection.execute("INSERT INTO entry VALUES ('first')")
        connection.commit()
    finally:
        connection.close()
    if not wal:
        # The new database got empty private logs at its first connect (see
        # test_a_new_database_that_stays_in_rollback_mode_keeps_empty_logs).
        for suffix in ("-wal", "-shm"):
            Path(f"{database}{suffix}").unlink(missing_ok=True)
    assert _logs(database) == {"-wal": None, "-shm": None}
    return database


@windows_only
def test_a_new_database_that_stays_in_rollback_mode_keeps_empty_logs(
    readable_folder,
):
    """Known cost: a new database may switch to WAL, so its logs are made at once.

    If it never does, the two empty private files stay beside it. SQLite
    treats them as absent and the next connect checks them like any others.
    """
    database = readable_folder / "store.sqlite"
    connection = connect_private_sqlite("db.base", database)
    try:
        connection.execute("CREATE TABLE entry(value TEXT)")
        connection.commit()
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    finally:
        connection.close()
    assert _logs(database) == {"-wal": 0o600, "-shm": 0o600}
    assert {
        private_paths.os.stat(f"{database}{suffix}").st_size
        for suffix in ("-wal", "-shm")
    } == {0}


def test_wal_and_shm_are_private_after_the_first_write_in_a_folder_others_may_read(
    readable_folder,
):
    """The log and its index are private as soon as SQLite has written them."""
    database = readable_folder / "store.sqlite"
    connection = connect_private_sqlite("db.base", database)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE entry(value TEXT)")
        connection.execute("INSERT INTO entry VALUES ('recent page content')")
        connection.commit()

        observed = {
            suffix or "database": _mode(f"{database}{suffix}")
            for suffix in ("", "-wal", "-shm")
        }
    finally:
        connection.close()

    assert observed == {"database": 0o600, "-wal": 0o600, "-shm": 0o600}


@windows_only
def test_an_existing_wal_database_gets_private_logs_before_sqlite_writes(
    readable_folder,
):
    """Reopening a closed WAL database: SQLite writes into the names made here."""
    database = _closed_database(readable_folder, wal=True)
    connection = connect_private_sqlite("db.base", database)
    try:
        made_here = {
            suffix: private_paths.os.stat(f"{database}{suffix}").st_ino
            for suffix in ("-wal", "-shm")
        }
        connection.execute("INSERT INTO entry VALUES ('second')")
        connection.commit()
        assert {
            suffix: private_paths.os.stat(f"{database}{suffix}").st_ino
            for suffix in ("-wal", "-shm")
        } == made_here
        assert _logs(database) == {"-wal": 0o600, "-shm": 0o600}
        assert private_paths.os.stat(f"{database}-wal").st_size > 0
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        connection.close()


@windows_only
def test_nothing_is_created_in_a_folder_that_hands_nothing_to_others(tmp_path):
    """A private folder already gives SQLite's files private access; no walk, no file."""
    folder = tmp_path / "private"
    private_paths.os.mkdir(folder, 0o700)
    assert _mode(folder) == 0o700
    database = folder / "store.sqlite"
    walked = []
    real_walk = private_paths._open_verified_parent

    def counting_walk(selected, **kwargs):
        if selected == database:
            walked.append(selected)
        return real_walk(selected, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(private_paths, "_open_verified_parent", counting_walk)
        connection = connect_private_sqlite("db.base", database)
    try:
        assert _logs(database) == {"-wal": None, "-shm": None}
        # Only the database file's own preparation walked to this folder.
        assert len(walked) == 1
    finally:
        connection.close()


@windows_only
def test_nothing_is_created_beside_a_rollback_journal_database(readable_folder):
    """A rollback-mode database never opens a log, so none is left beside it."""
    database = _closed_database(readable_folder, wal=False)
    connection = connect_private_sqlite("db.base", database)
    try:
        assert _logs(database) == {"-wal": None, "-shm": None}
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    finally:
        connection.close()


@windows_only
def test_nothing_is_created_beside_an_empty_file_a_caller_requires_to_exist(
    readable_folder,
):
    """A must_exist connect to an empty temporary (TTS backup) gets no logs."""
    database = readable_folder / "temporary.sqlite"
    descriptor = private_paths.os.open(
        str(database),
        private_paths.os.O_RDWR | private_paths.os.O_CREAT | private_paths.os.O_EXCL,
        0o600,
    )
    private_paths.os.close(descriptor)
    connection = connect_private_sqlite("db.base", database, must_exist=True)
    try:
        assert _logs(database) == {"-wal": None, "-shm": None}
    finally:
        connection.close()


@windows_only
def test_a_read_only_connect_creates_nothing(readable_folder):
    """Read-only connects never create files beside the database.

    ``notes.sync_state`` keeps the private-mode check on read-only opens, so
    only the read-only rule stops creation here.
    """
    database = _closed_database(readable_folder, wal=True)
    connection = connect_private_sqlite("notes.sync_state", database, read_only=True)
    try:
        assert _logs(database) == {"-wal": None, "-shm": None}
    finally:
        connection.close()


@windows_only
def test_a_log_sqlite_creates_first_is_still_made_private(
    readable_folder, monkeypatch
):
    """Another connection may create the log between the check and the create."""
    database = _closed_database(readable_folder, wal=True)
    real = private_paths._sqlite_may_use_wal

    def another_connection_creates_the_log(parent_fd, leaf, new_database_allowed):
        answer = real(parent_fd, leaf, new_database_allowed)
        Path(f"{database}-wal").write_bytes(b"")  # inherits the folder's read
        assert _mode(f"{database}-wal") == 0o644
        return answer

    monkeypatch.setattr(
        private_paths, "_sqlite_may_use_wal", another_connection_creates_the_log
    )
    connection = connect_private_sqlite("db.base", database)
    try:
        assert _logs(database) == {"-wal": 0o600, "-shm": 0o600}
    finally:
        connection.close()


@windows_only
def test_a_hard_linked_log_found_at_creation_refuses_the_connect(
    readable_folder, monkeypatch
):
    """A name found during creation gets the same file checks as one found before."""
    database = _closed_database(readable_folder, wal=True)
    real = private_paths._sqlite_may_use_wal

    def link_the_log(parent_fd, leaf, new_database_allowed):
        answer = real(parent_fd, leaf, new_database_allowed)
        elsewhere = readable_folder / "elsewhere"
        elsewhere.write_bytes(b"")
        stdlib_os.link(elsewhere, f"{database}-wal")
        return answer

    monkeypatch.setattr(private_paths, "_sqlite_may_use_wal", link_the_log)
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached a linked log"),
    )
    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", database)
    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR


@windows_only
def test_a_created_log_that_is_not_private_refuses_the_connect(
    readable_folder, monkeypatch
):
    """The create is checked after it is made, not trusted because it was asked."""
    database = _closed_database(readable_folder, wal=True)
    real = private_paths._private_file_postcondition_holds

    def report_failure_for_the_log(file_fd, parent_fd, leaf, **kwargs):
        if leaf.endswith("-wal"):
            return False
        return real(file_fd, parent_fd, leaf, **kwargs)

    monkeypatch.setattr(
        private_paths, "_private_file_postcondition_holds", report_failure_for_the_log
    )
    monkeypatch.setattr(
        private_sqlite.sqlite3,
        "connect",
        lambda *args, **kwargs: pytest.fail("raw connect reached an unchecked log"),
    )
    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", database)
    assert caught.value.result.reason == "private_sqlite_log_postcondition_failed"


@windows_only
def test_an_existing_unsafe_log_still_refuses_before_anything_is_created(
    readable_folder, monkeypatch
):
    """The connect's own checks of existing names still run, and run first."""
    database = _closed_database(readable_folder, wal=True)
    elsewhere = readable_folder / "elsewhere"
    elsewhere.write_bytes(b"")
    stdlib_os.link(elsewhere, f"{database}-shm")
    monkeypatch.setattr(
        private_paths,
        "create_private_sqlite_logs",
        lambda *args, **kwargs: pytest.fail("creation ran before the checks refused"),
    )
    with pytest.raises(PrivatePathError) as caught:
        connect_private_sqlite("db.base", database)
    assert caught.value.result.status is PrivatePathStatus.LINK_OR_NON_REGULAR
    assert _mode(f"{database}-wal") is None


@windows_only
def test_the_log_is_private_after_the_seam_switches_a_destination_back_to_wal(
    readable_folder,
):
    """Backup and restore switch WAL off and on again; the new log is private too."""
    database = _closed_database(readable_folder, wal=True)
    connection = connect_private_sqlite("db.base", database, timeout=0)
    try:
        mode = private_sqlite._guard_destination(connection, restore=False)
        assert mode == "wal"
        assert _mode(f"{database}-wal") is None  # the switch to DELETE removed it
        private_sqlite._restore_destination_mode(connection, mode, restore=False)
        connection.execute("INSERT INTO entry VALUES ('during a backup')")
        connection.commit()
        assert private_paths.os.stat(f"{database}-wal").st_size > 0
        assert _mode(f"{database}-wal") == 0o600
    finally:
        connection.close()


@windows_only
def test_the_switch_back_to_wal_creates_nothing_in_a_private_folder(tmp_path):
    """Without a folder result the held folder itself is checked, and gives nothing."""
    folder = tmp_path / "private"
    private_paths.os.mkdir(folder, 0o700)
    database = _closed_database(folder, wal=True)
    connection = connect_private_sqlite("db.base", database, timeout=0)
    try:
        mode = private_sqlite._guard_destination(connection, restore=False)
        private_sqlite._restore_destination_mode(connection, mode, restore=False)
        assert _mode(f"{database}-wal") is None
    finally:
        connection.close()


def test_a_rollback_journal_is_private_on_posix(readable_folder):
    """POSIX gives the journal the database's own mode; nothing here changes that."""
    if _WINDOWS:
        pytest.skip("Windows rollback journals are not covered (ADR-126, 2026-11-28)")
    database = readable_folder / "store.sqlite"
    connection = connect_private_sqlite("db.base", database, isolation_level=None)
    try:
        connection.execute("CREATE TABLE entry(value TEXT)")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO entry VALUES ('pending')")
        assert _mode(f"{database}-journal") == 0o600
        connection.execute("COMMIT")
    finally:
        connection.close()
