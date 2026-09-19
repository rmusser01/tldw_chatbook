"""Native-held path discovery uses bounded private SQL without payload authority."""

import sqlite3
from contextlib import closing, nullcontext
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_capture_sqlite_materialization import state
from Tests.Backup_Recovery.test_capture_sqlite_materialization import (
    wal_owner as wal_owner,  # noqa: PLC0414 - real installed WAL producer fixture
)
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.storage_admission import _local
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.Utils.platform_files import os


def reader_owner(adapter):
    return (
        "recovery.domain.research"
        if adapter.owner_id == "research.local"
        else "recovery.core.chachanotes"
    )


def test_private_discovery_reads_wal_once_without_expanding_payload_scope(wal_owner):
    source, stage, authority, adapter, _, table = wal_owner
    before = state(source)
    required = sum(Path(str(source) + suffix).stat().st_size for suffix in ("", "-wal"))
    with authority.maintenance(("core", "bootstrap.unbound"), 3) as session:
        with session._capture_bound_sources((), stage, ArchiveLimits(), 1024**3):
            capture = _local.capture_scope
            with session._discovery_reads(private_sqlite=True, byte_budget=required):
                connection = connect_private_sqlite(
                    reader_owner(adapter), source, read_only=True
                )
                private = Path(connection.execute("PRAGMA database_list").fetchone()[2])
                assert private != source
                query = {
                    "notes": "SELECT title FROM notes",
                    "research_sessions": "SELECT title FROM research_sessions",
                }[table]
                assert connection.execute(query).fetchall() == [("WAL-only",)]
                with closing(
                    connect_private_sqlite(
                        reader_owner(adapter), source, read_only=True
                    )
                ) as reused:
                    assert (
                        Path(reused.execute("PRAGMA database_list").fetchone()[2])
                        == private
                    )
                assert state(source) == before
                assert capture.sources == () and not capture.sqlite_snapshots
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
            assert not private.exists()
            with pytest.raises(
                bootstrap.RecoveryRequired, match="capture_path_outside_scope"
            ):
                connect_private_sqlite(reader_owner(adapter), source, read_only=True)
        session._check()
    assert state(source) == before


@pytest.mark.parametrize("limit", ["member", "total"])
def test_private_discovery_enforces_copy_limits_without_source_mutation(
    wal_owner, limit
):
    source, _, authority, adapter, _, _ = wal_owner
    before = state(source)
    options = (
        {"limits": ArchiveLimits(member_bytes=1)}
        if limit == "member"
        else {"byte_budget": 1}
    )
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 3) as session,
        session._discovery_reads(private_sqlite=True, **options),
        pytest.raises(ValueError, match="preview_sqlite_limit"),
    ):
        connect_private_sqlite(reader_owner(adapter), source, read_only=True)
    assert state(source) == before


@pytest.mark.parametrize("refusal", ["outside", "write", "ordinary_owner"])
def test_private_discovery_keeps_original_native_admission(
    wal_owner, tmp_path, refusal
):
    source, _, authority, adapter, _, _ = wal_owner
    outside = tmp_path / "outside.db"
    outside.write_bytes(source.read_bytes())
    outside.chmod(0o600)
    before = state(source), state(outside)
    path = outside if refusal == "outside" else source
    owner = "recovered.media" if refusal == "ordinary_owner" else reader_owner(adapter)
    expected = (
        "capture_source_outside_scope"
        if refusal == "outside"
        else "discovery_read_only_required"
    )
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 3) as session,
        session._discovery_reads(private_sqlite=True),
        pytest.raises(bootstrap.RecoveryRequired, match=expected),
    ):
        connect_private_sqlite(owner, path, read_only=refusal != "write")
    assert (state(source), state(outside)) == before


@pytest.mark.parametrize("suffix", ["", "-wal"])
def test_private_discovery_rechecks_source_before_reusing_copy(wal_owner, suffix):
    source, _, authority, adapter, _, _ = wal_owner
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 3) as session,
        pytest.raises(ValueError, match="preview_sqlite_changed"),
        session._discovery_reads(private_sqlite=True),
    ):
        with closing(
            connect_private_sqlite(reader_owner(adapter), source, read_only=True)
        ):
            pass
        changed = Path(str(source) + suffix)
        info = changed.stat()
        os.utime(changed, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        with pytest.raises(ValueError, match="preview_sqlite_changed"):
            connect_private_sqlite(reader_owner(adapter), source, read_only=True)


@pytest.mark.parametrize("changed", ["main", "-wal", "-shm", "parent"])
def test_private_discovery_keeps_full_target_fingerprint_refusal(wal_owner, changed):
    """Exercise the existing fingerprint boundary, without fabricating a receipt."""
    from tldw_chatbook.Backup_Recovery.restore_plan import (
        RestorePlan,
        _fingerprint,
        recheck_targets,
    )

    source, _, authority, adapter, _, _ = wal_owner
    paths = (
        source.parent,
        *(Path(str(source) + suffix) for suffix in ("", "-wal", "-shm", "-journal")),
    )
    plan = RestorePlan(
        "a" * 64,
        "replace",
        (),
        (),
        tuple((str(i), path) for i, path in enumerate(paths)),
        _fingerprint(paths, None),
    )
    with authority.maintenance(("core", "bootstrap.unbound"), 3) as session:
        expected_exit = (
            pytest.raises(ValueError, match="preview_sqlite_changed")
            if changed in {"main", "-wal"}
            else nullcontext()
        )
        with expected_exit, session._discovery_reads(private_sqlite=True):
            with closing(
                connect_private_sqlite(reader_owner(adapter), source, read_only=True)
            ):
                pass
            recheck_targets(plan)
            path = (
                source.parent
                if changed == "parent"
                else source
                if changed == "main"
                else Path(str(source) + changed)
            )
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
            with pytest.raises(ValueError, match="target_changed"):
                recheck_targets(plan)


def test_private_discovery_cleans_up_after_owner_failure(wal_owner):
    source, _, authority, adapter, _, _ = wal_owner
    before = state(source)
    with authority.maintenance(("core", "bootstrap.unbound"), 3) as session:
        with (
            pytest.raises(RuntimeError, match="owner failure"),
            session._discovery_reads(private_sqlite=True),
        ):
            connection = connect_private_sqlite(
                reader_owner(adapter), source, read_only=True
            )
            private = Path(connection.execute("PRAGMA database_list").fetchone()[2])
            raise RuntimeError("owner failure")
        assert not private.exists()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
        session._check()
    assert state(source) == before
