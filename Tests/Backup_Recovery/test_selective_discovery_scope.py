"""Nested discovery retains native read authority outside selected payloads."""

import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_capture_sqlite_materialization import (
    state,
)
from Tests.Backup_Recovery.test_capture_sqlite_materialization import (
    wal_owner as wal_owner,  # noqa: PLC0414 - actual installed WAL owner fixture
)
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
)
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.Utils.platform_files import os


@pytest.fixture
def nested_capture(tmp_path, monkeypatch):
    root = tmp_path / "bootstrap"
    sources = tmp_path / "sources"
    sources.mkdir(mode=0o700)
    selected = sources / "selected.db"
    dependency = sources / "dependency.db"
    outside = tmp_path / "outside.db"
    for path in (selected, dependency, outside):
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE fixture(value TEXT)")
            connection.execute("INSERT INTO fixture VALUES ('preserved')")
            connection.commit()
        path.chmod(0o600)
    selector = tmp_path / "selected.toml"
    selector.write_text('[general]\nusers_name="fixture"\n')
    selector.chmod(0o600)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    authority = admission_authority(root)
    authority.register("profile", (sources, selector))
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    bind_profile(root, selector, ("profile",), root / "admission")
    before = {
        path: (path.read_bytes(), path.stat().st_ino)
        for path in (selected, dependency, outside)
    }
    with (
        authority.maintenance(("profile", "bootstrap.unbound"), 2) as session,
        session.capture_scope((selected,), stage),
    ):
        yield session, dependency, outside
    assert all(
        (path.read_bytes(), path.stat().st_ino) == state
        for path, state in before.items()
    )


def test_nested_discovery_reads_held_unselected_sqlite_and_retires_escaped_handle(
    nested_capture,
):
    session, dependency, _ = nested_capture
    with session._discovery_reads():
        connection = connect_private_sqlite(
            "recovery.operations.agent_runs", dependency, read_only=True
        )
        assert connection.execute("SELECT value FROM fixture").fetchone() == (
            "preserved",
        )
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    session._check()


def test_nested_discovery_cannot_write_held_unselected_sqlite(nested_capture):
    session, dependency, _ = nested_capture
    with (
        session._discovery_reads(),
        pytest.raises(bootstrap.RecoveryRequired, match="discovery_read_only_required"),
    ):
        connect_private_sqlite("recovery.operations.agent_runs", dependency)


def test_nested_discovery_cannot_read_outside_held_native_roots(nested_capture):
    session, _, outside = nested_capture
    with (
        session._discovery_reads(),
        pytest.raises(bootstrap.RecoveryRequired, match="capture_source_outside_scope"),
    ):
        connect_private_sqlite(
            "recovery.operations.agent_runs", outside, read_only=True
        )


def test_nested_discovery_requires_installed_recovery_owner(nested_capture):
    session, dependency, _ = nested_capture
    with (
        session._discovery_reads(),
        pytest.raises(bootstrap.RecoveryRequired, match="discovery_read_only_required"),
    ):
        connect_private_sqlite("recovered.media", dependency, read_only=True)


def test_nested_discovery_does_not_expand_payload_capture_authority(nested_capture):
    session, dependency, _ = nested_capture
    for _ in range(2):
        with pytest.raises(
            bootstrap.RecoveryRequired, match="capture_path_outside_scope"
        ):
            connect_private_sqlite(
                "recovery.operations.agent_runs", dependency, read_only=True
            )
        with session._discovery_reads():
            session._check()


@pytest.mark.parametrize("clean_wal", [False, True])
def test_nested_discovery_reuses_selected_wal_image_and_retires_handle(
    wal_owner, clean_wal
):
    from tldw_chatbook.Backup_Recovery.storage_admission import _local

    source, stage, authority, adapter, item, table = wal_owner
    if clean_wal:
        with closing(sqlite3.connect(source)) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert not Path(str(source) + "-wal").exists()
    before = state(source)
    owner = (
        "recovery.domain.research"
        if adapter.owner_id == "research.local"
        else "recovery.core.chachanotes"
    )
    with authority.maintenance(("core", "bootstrap.unbound"), 2) as session:
        with session.capture_scope((source,), stage):
            adapter.capture(item, stage / "captured.db", Event())
            scope = _local.capture_scope
            target = scope.sqlite_snapshots[source][1]
            copied = scope.copied_bytes
            with session._discovery_reads():
                connection = connect_private_sqlite(owner, source, read_only=True)
                actual = Path(connection.execute("PRAGMA database_list").fetchone()[2])
                assert actual == target and actual != source
                assert connection.execute("SELECT title FROM " + table).fetchall() == [
                    ("WAL-only",)
                ]
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
            assert scope.copied_bytes == copied
            assert state(source) == before
        session._check()
    assert state(source) == before


@pytest.mark.parametrize("changed", ["main", "-wal", "-shm", "parent"])
def test_nested_discovery_still_refuses_changed_selected_source(wal_owner, changed):
    source, stage, authority, adapter, item, _ = wal_owner
    owner = (
        "recovery.domain.research"
        if adapter.owner_id == "research.local"
        else "recovery.core.chachanotes"
    )
    refused_before_open = False
    with authority.maintenance(("core", "bootstrap.unbound"), 2) as session:
        with (
            pytest.raises(ValueError, match="capture_sqlite_source_changed"),
            session.capture_scope((source,), stage),
        ):
            adapter.capture(item, stage / "captured.db", Event())
            changed_path = (
                source.parent
                if changed == "parent"
                else source
                if changed == "main"
                else Path(str(source) + changed)
            )
            info = changed_path.stat()
            os.utime(changed_path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
            with session._discovery_reads():
                try:
                    connection = connect_private_sqlite(owner, source, read_only=True)
                except ValueError as error:
                    assert str(error) == "capture_sqlite_source_changed"
                    refused_before_open = True
                else:
                    connection.close()
        session._check()
    assert refused_before_open
