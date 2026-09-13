"""First backup qualifies local discovery without changing live enrollment."""

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.inventory import discover
from tldw_chatbook.Backup_Recovery.models import DiscoverySelections


def test_unbound_capture_accepts_only_fenced_local_discovery(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import owner_registry
    from tldw_chatbook.Backup_Recovery.config_adapter import config_adapter

    monkeypatch.setattr(owner_registry, "_adapters", {"config": config_adapter()})
    root = tmp_path / "bootstrap"
    source_root = tmp_path / "source"
    source_root.mkdir(mode=0o700)
    source = source_root / "config.toml"
    source.write_text('[general]\nusers_name="fixture"\n')
    source.chmod(0o600)
    unrelated = source_root / "unselected.bin"
    unrelated.write_text("preserved")
    staging = tmp_path / "stage"
    staging.mkdir(mode=0o700)
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    authority = admission_authority(root)
    authority.register("capture.source", (source_root,))
    preview = discover((source,))
    with authority.maintenance(("bootstrap.unbound", "capture.source"), 1) as session:
        with (
            pytest.raises(
                bootstrap.RecoveryRequired, match="capture_source_binding_unverified"
            ),
            session.capture_scope((source,), staging),
        ):
            pass
        current = session._discover_capture_inventory(
            (source,), DiscoverySelections(), preview.scope_digest
        )
        assert current.scope_digest == preview.scope_digest
        with session.capture_scope((source,), staging):
            pass
        with (
            pytest.raises(
                bootstrap.RecoveryRequired, match="capture_source_binding_unverified"
            ),
            session.capture_scope((unrelated,), staging),
        ):
            pass
    assert bootstrap._records(root) == ([], [])
    with pytest.raises(
        bootstrap.RecoveryRequired, match="maintenance_session_inactive"
    ):
        session._discover_capture_inventory(
            (source,), DiscoverySelections(), preview.scope_digest
        )


def test_fenced_discovery_rejects_changed_scope(tmp_path, monkeypatch):
    root = tmp_path / "bootstrap"
    source = tmp_path / "config.toml"
    source.write_text('[general]\nusers_name="fixture"\n')
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    authority = admission_authority(root)
    authority.register("capture.source", (source,))
    preview = discover((source,))
    source.write_text('[general]\nusers_name="changed"\n')
    with (
        authority.maintenance(("bootstrap.unbound", "capture.source"), 1) as session,
        pytest.raises(ValueError, match="scope_changed"),
    ):
        session._discover_capture_inventory(
            (source,), DiscoverySelections(), preview.scope_digest
        )


def test_preview_sqlite_probes_are_read_only_and_retire_without_bootstrap(
    tmp_path, monkeypatch
):
    import sqlite3

    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    source = tmp_path / "source.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE fixture (value TEXT)")
        connection.execute("INSERT INTO fixture VALUES ('preserved')")
    source.chmod(0o600)
    before = source.read_bytes()
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    with _preview_reads():
        connection = connect_private_sqlite(
            "recovery.files.persona", source, read_only=True
        )
        assert connection.execute("SELECT value FROM fixture").fetchone() == (
            "preserved",
        )
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM fixture")
        with pytest.raises(
            bootstrap.RecoveryRequired, match="discovery_read_only_required"
        ):
            connect_private_sqlite("recovery.files.persona", source)
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    assert source.read_bytes() == before
    assert not root.exists()


@pytest.mark.parametrize("live_wal", [False, True])
def test_preview_preserves_wal_rows_without_source_side_effects(
    tmp_path, monkeypatch, live_wal
):
    import sqlite3
    from contextlib import closing
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    source_root = tmp_path / "source"
    source_root.mkdir(mode=0o700)
    source = source_root / "wal.db"
    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE fixture(value TEXT)")
    writer.execute("INSERT INTO fixture VALUES ('committed WAL row')")
    writer.commit()
    source.chmod(0o600)
    if not live_wal:
        writer.close()

    def snapshot():
        return {
            path.name: (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns)
            for path in source_root.iterdir()
        }, source_root.stat().st_mtime_ns

    before = snapshot()
    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    try:
        with _preview_reads():
            with closing(
                connect_private_sqlite("recovery.files.persona", source, read_only=True)
            ) as reader:
                assert reader.execute("SELECT value FROM fixture").fetchone() == (
                    "committed WAL row",
                )
                copied = Path(reader.execute("PRAGMA database_list").fetchone()[2])
                assert copied != source
                assert snapshot() == before
            assert snapshot() == before
        assert snapshot() == before
        assert not copied.exists()
        assert not root.exists()
    finally:
        writer.close()


@pytest.mark.parametrize("budget_kind", ["member", "total"])
def test_preview_sqlite_copy_obeys_reviewed_limits(tmp_path, budget_kind):
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    source = tmp_path / "source.db"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("CREATE TABLE fixture(value)")
    source.chmod(0o600)
    size = source.stat().st_size
    limits = (
        ArchiveLimits(member_bytes=size - 1)
        if budget_kind == "member"
        else ArchiveLimits()
    )
    budget = size - 1 if budget_kind == "total" else limits.expanded_bytes
    with (
        _preview_reads(limits=limits, byte_budget=budget),
        pytest.raises(ValueError, match="preview_sqlite_limit"),
    ):
        connect_private_sqlite("recovery.files.persona", source, read_only=True)


def test_preview_rejects_source_change_and_removes_private_copy(tmp_path, monkeypatch):
    import os
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery.storage_admission import _local, _preview_reads
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    source = tmp_path / "source.db"
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("CREATE TABLE fixture(value)")
        writer.execute("INSERT INTO fixture VALUES ('before')")
        writer.commit()
    source.chmod(0o600)
    original = os.read
    changed = False

    def change_after_read(fd, count):
        nonlocal changed
        data = original(fd, count)
        if not changed and os.fstat(fd).st_ino == source.stat().st_ino:
            changed = True
            with closing(sqlite3.connect(source)) as writer:
                writer.execute("UPDATE fixture SET value='concurrent writer'")
                writer.commit()
        return data

    monkeypatch.setattr(os, "read", change_after_read)
    with _preview_reads():
        scope = _local.preview_scope
        with pytest.raises(ValueError, match="preview_sqlite_changed"):
            connect_private_sqlite("recovery.files.persona", source, read_only=True)
        temporary = scope.directory
        assert not list(temporary.iterdir())
    assert changed
    assert not temporary.exists()


@pytest.mark.parametrize("journal_mode", ["DELETE", "WAL"])
def test_preview_reuses_verified_snapshot_across_owner_reads(tmp_path, journal_mode):
    import sqlite3
    from contextlib import closing
    from pathlib import Path

    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    source = tmp_path / "source.db"
    with closing(sqlite3.connect(source)) as writer:
        writer.execute(
            {"DELETE": "PRAGMA journal_mode=DELETE", "WAL": "PRAGMA journal_mode=WAL"}[
                journal_mode
            ]
        )
        writer.execute("CREATE TABLE fixture(value)")
        writer.execute("INSERT INTO fixture VALUES ('before')")
        writer.commit()
        source.chmod(0o600)
        with _preview_reads():
            with closing(
                connect_private_sqlite(
                    "recovery.core.chachanotes", source, read_only=True
                )
            ) as first:
                assert first.execute("SELECT value FROM fixture").fetchone() == (
                    "before",
                )
                copied = Path(first.execute("PRAGMA database_list").fetchone()[2])
            writer.execute("UPDATE fixture SET value='after'")
            writer.commit()
            with closing(
                connect_private_sqlite("recovery.files.persona", source, read_only=True)
            ) as second:
                assert second.execute("SELECT value FROM fixture").fetchone() == (
                    "before",
                )
                assert (
                    Path(second.execute("PRAGMA database_list").fetchone()[2]) == copied
                )
        assert not copied.exists()
        with _preview_reads(), closing(
            connect_private_sqlite("recovery.files.persona", source, read_only=True)
        ) as fresh:
            assert fresh.execute("SELECT value FROM fixture").fetchone() == ("after",)


def test_preview_cached_snapshot_rejects_replaced_source_path(tmp_path):
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery.storage_admission import _preview_reads
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    source = tmp_path / "source.db"
    replacement = tmp_path / "replacement.db"
    for path in (source, replacement):
        with closing(sqlite3.connect(path)) as writer:
            writer.execute("CREATE TABLE fixture(value)")
        path.chmod(0o600)
    with _preview_reads():
        with closing(
            connect_private_sqlite("recovery.core.chachanotes", source, read_only=True)
        ) as first:
            first.execute("SELECT value FROM fixture").fetchall()
        replacement.replace(source)
        with pytest.raises(ValueError, match="preview_sqlite_changed"):
            connect_private_sqlite("recovery.files.persona", source, read_only=True)
