"""Exact native-76 archive compatibility without domain-write authority."""

from contextlib import closing
import sqlite3
from threading import Event

import pytest

from Tests.DB.test_chachanotes_v77_agent_chat_starts_migration import _legacy_native76
from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.Backup_Recovery import sqlite_validation as validation
from tldw_chatbook.DB.recovery_core import core_adapters


def _owner():
    return next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")


def _dump(path):
    with closing(sqlite3.connect(path)) as connection:
        return tuple(connection.iterdump())


@pytest.mark.parametrize("dictionary", [False, True])
def test_native76_staged_upgrade_preserves_machine_receipts(tmp_path, dictionary):
    path = tmp_path / "native.sqlite"
    _legacy_native76(path, dictionary=dictionary)
    with closing(sqlite3.connect(path)) as connection:
        rows = tuple(connection.execute("SELECT * FROM console_dispatch_checkpoints"))
        messages = tuple(connection.execute("SELECT * FROM messages ORDER BY id"))
    assert validation.validate_candidate(_owner(), path, Event(), migrate=True) == ()
    assert validation.validated_schema_version(_owner(), path, Event()) == 77
    with closing(sqlite3.connect(path)) as connection:
        assert (
            tuple(connection.execute("SELECT * FROM console_dispatch_checkpoints"))
            == rows
        )
        assert (
            tuple(connection.execute("SELECT * FROM messages ORDER BY id")) == messages
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert validation.validate_candidate(_owner(), path, Event(), migrate=False) == ()


@pytest.mark.parametrize("version", [75, 76])
def test_ordinary_older_schema_validates_but_staged_ddl_remains_refused(
    tmp_path, version
):
    path = tmp_path / "ordinary.sqlite"
    with chachanotes_db_at_version(path, version):
        pass
    before = _dump(path)
    assert validation.validate_candidate(_owner(), path, Event(), migrate=False) == ()
    assert validation.validate_candidate(_owner(), path, Event(), migrate=True) == (
        "unsupported_schema_migration",
    )
    assert _dump(path) == before


@pytest.mark.parametrize(
    "alteration", ["extra_table", "wrong_stamp", "shared_subscription"]
)
def test_primary_staged_gate_rejects_hybrid_and_other_owner_catalogs(
    tmp_path, alteration
):
    path = tmp_path / "invalid.sqlite"
    _legacy_native76(path, subscriptions=alteration == "shared_subscription")
    with closing(sqlite3.connect(path)) as connection:
        if alteration == "extra_table":
            connection.execute("CREATE TABLE unqualified(value)")
        elif alteration == "wrong_stamp":
            connection.execute("UPDATE db_schema_version SET version=75")
        connection.commit()
    before = _dump(path)
    assert validation.validate_candidate(_owner(), path, Event(), migrate=True)
    assert _dump(path) == before


@pytest.mark.parametrize("failure", ["validation", "cancel"])
def test_native76_staged_failure_rolls_back_stamp_and_receipts(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "rollback.sqlite"
    _legacy_native76(path)
    before = _dump(path)
    original = validation._check
    calls = []
    cancel = Event()

    def check(connection, owner, policy, restrictions):
        result = original(connection, owner, policy, restrictions)
        calls.append(result)
        if len(calls) == 2:
            assert result == ((), 77)
            if failure == "validation":
                return (("invalid_domain_reference",), None)
            cancel.set()
        return result

    monkeypatch.setattr(validation, "_check", check)
    assert validation.validate_candidate(_owner(), path, cancel, migrate=True)
    assert len(calls) == 2
    assert _dump(path) == before


@pytest.mark.parametrize(
    "case",
    [
        "outside_window",
        "wrong_owner",
        "wrong_column",
        "domain_write",
        "ddl",
        "trigger_source",
        "other_database",
    ],
)
def test_native76_metadata_authority_is_exact(tmp_path, case):
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    path = tmp_path / "authority.sqlite"
    _legacy_native76(path)
    with open_recovery_validation(
        "db.chachanotes.primary", path, writable=True, with_restrictions=True
    ) as (connection, restrictions):
        restrictions.migrating = case != "outside_window"
        restrictions.migration_owner = (
            "db.subscriptions" if case == "wrong_owner" else "db.chachanotes.primary"
        )
        if case in {"trigger_source", "other_database"}:
            assert (
                restrictions.authorize(
                    sqlite3.SQLITE_UPDATE,
                    "db_schema_version",
                    "version",
                    "temp" if case == "other_database" else "main",
                    "untrusted_trigger" if case == "trigger_source" else None,
                )
                == sqlite3.SQLITE_DENY
            )
        else:
            sql = {
                "outside_window": "UPDATE db_schema_version SET version=77",
                "wrong_owner": "UPDATE db_schema_version SET version=77",
                "wrong_column": "UPDATE db_schema_version SET schema_name='other'",
                "domain_write": "UPDATE messages SET content='lost'",
                "ddl": "CREATE TABLE unqualified(value)",
            }[case]
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(sql)


@pytest.mark.parametrize("version", [76, 77])
@pytest.mark.parametrize("dictionary", [False, True])
def test_subscription_native_catalog_retains_exact_embedded_stamps(
    tmp_path, version, dictionary
):
    from tldw_chatbook.DB.recovery_operations import recovery_adapters

    path = tmp_path / "subscription.sqlite"
    _legacy_native76(path, subscriptions=True, dictionary=dictionary)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("UPDATE db_schema_version SET version=?", (version,))
        connection.commit()
    owner = next(a for a in recovery_adapters() if a.owner_id == "db.subscriptions")
    before = _dump(path)
    assert validation.validate_candidate(owner, path, Event(), migrate=True) == ()
    assert _dump(path) == before
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("UPDATE db_schema_version SET version=75")
        connection.commit()
    assert validation.validate_candidate(owner, path, Event(), migrate=True) == (
        "unsupported_schema_version",
    )
