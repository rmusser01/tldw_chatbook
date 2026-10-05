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


def _shipped76(path, *, dictionary=False):
    """Populate the actual shipped chain with both predecessor receipt origins."""
    with chachanotes_db_at_version(path, 76) as db:
        conversation = db.add_conversation({"title": "shipped recovery"})
        connection = db.get_connection()
        for origin in ("manual", "queued"):
            user = db.add_message(
                {"conversation_id": conversation, "sender": "user", "content": origin}
            )
            assistant = db.add_message(
                {
                    "conversation_id": conversation,
                    "sender": "assistant",
                    "content": "pending",
                }
            )
            connection.execute(
                "INSERT INTO console_dispatch_checkpoints "
                "(assistant_message_id,user_message_id,conversation_id,preparation_id,attempt_id,state,"
                "checkpoint_revision,user_message_version,assistant_message_version,origin,queue_entry_id,"
                "frozen_authority_json,resolved_destination_json,reconstructability_json,created_at,updated_at) "
                "VALUES (?,?,?,?,?,'accepted',7,1,1,?,?,?,? ,?,'2026-09-01','2026-09-02')",
                (
                    assistant,
                    user,
                    conversation,
                    "prepare-" + origin,
                    "attempt-" + origin,
                    origin,
                    "queue-old" if origin == "queued" else None,
                    '{"saved":"authority"}',
                    '{"saved":"destination"}',
                    '{"saved":"reconstructability"}',
                ),
            )
        if dictionary:
            from tldw_chatbook.DB.recovery_core_schema import (
                _CHAT_DICTIONARIES_UPDATED_TRIGGER,
            )

            connection.execute("DROP TRIGGER chat_dictionaries_au")
            connection.execute(_CHAT_DICTIONARIES_UPDATED_TRIGGER)
        connection.commit()
    return _checkpoint_state(path)


def _checkpoint_state(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        receipts = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM console_dispatch_checkpoints ORDER BY assistant_message_id"
            )
        ]
        messages = [
            dict(row)
            for row in connection.execute("SELECT * FROM messages ORDER BY id")
        ]
        indexes = tuple(
            sorted(
                (
                    tuple(row)[1:],
                    tuple(
                        tuple(column)
                        for column in connection.execute(
                            'PRAGMA index_xinfo("' + row[1].replace('"', '""') + '")'
                        )
                    ),
                )
                for row in connection.execute(
                    "PRAGMA index_list(console_dispatch_checkpoints)"
                )
            )
        )
        return receipts, messages, indexes


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


@pytest.mark.parametrize("version", [75])
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


@pytest.mark.parametrize("failure", ["late_index", "validation", "cancel"])
def test_shipped76_staged_failure_rolls_back_checkpoint_rebuild(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "source.sqlite"
    _shipped76(source)
    original_bytes = source.read_bytes()
    path = tmp_path / "candidate.sqlite"
    path.write_bytes(original_bytes)
    before = _dump(path)
    original_authorize = validation._Restrictions.authorize
    original_check = validation._check
    reached = []
    final_checks = []
    scopes = []
    cancel = Event()

    def authorize(self, action, first, second, database, trigger):
        if self.shipped_checkpoint_migration:
            reached.append((action, first))
            if (
                failure == "late_index"
                and action == sqlite3.SQLITE_CREATE_INDEX
                and first == "idx_console_dispatch_checkpoints_user_message"
            ):
                return sqlite3.SQLITE_DENY
        return original_authorize(self, action, first, second, database, trigger)

    def check(connection, owner, policy, restrictions):
        scopes.append(restrictions)
        result = original_check(connection, owner, policy, restrictions)
        if result == ((), 77):
            assert not restrictions.shipped_checkpoint_migration
            assert connection.execute(
                "SELECT COUNT(*) FROM console_dispatch_checkpoints"
            ).fetchone() == (2,)
            final_checks.append(result)
            if failure == "validation":
                return (("invalid_domain_reference",), None)
            if failure == "cancel":
                cancel.set()
        return result

    monkeypatch.setattr(validation._Restrictions, "authorize", authorize)
    monkeypatch.setattr(validation, "_check", check)
    expected = {
        "late_index": "sqlite_validation_unavailable",
        "validation": "invalid_domain_reference",
        "cancel": "cancelled",
    }[failure]
    assert validation.validate_candidate(_owner(), path, cancel, migrate=True) == (
        expected,
    )
    assert (sqlite3.SQLITE_ALTER_TABLE, "main") in reached
    assert (
        sqlite3.SQLITE_REINDEX,
        "idx_console_dispatch_checkpoint_conversation",
    ) in reached
    assert len(final_checks) == (failure != "late_index")
    assert all(
        not scope.shipped_checkpoint_migration and not scope.migrating
        for scope in scopes
    )
    assert _dump(path) == before
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE messages SET content='lost';",
        "CREATE TABLE unqualified(value);",
        "INSERT INTO console_dispatch_checkpoints SELECT * FROM console_dispatch_checkpoints;",
        "SELECT random() /* task20_installed_file_witness */;",
        "UPDATE sqlite_sequence SET name='unrelated';",
    ],
)
def test_shipped76_migration_authority_is_exact(tmp_path, monkeypatch, sql):
    from pathlib import Path

    path = tmp_path / "candidate.sqlite"
    _shipped76(path)
    before = _dump(path)
    read_text = Path.read_text
    authorize = validation._Restrictions.authorize
    reads = []
    denied = []
    reached = []

    def installed(self, *args, **kwargs):
        text = read_text(self, *args, **kwargs)
        if self.name == "chachanotes_v76_to_v77_agent_chat_starts.sql":
            assert (
                self
                == Path(validation.__file__).resolve().parents[1]
                / "DB/migrations"
                / self.name
            )
            reads.append(self)
            return text + "\n" + sql + "\n"
        return text

    def observe(self, *args):
        result = authorize(self, *args)
        if self.shipped_checkpoint_migration:
            reached.append(args)
            if result == sqlite3.SQLITE_DENY:
                denied.append(args)
        return result

    monkeypatch.setattr(Path, "read_text", installed)
    monkeypatch.setattr(validation._Restrictions, "authorize", observe)
    assert validation.validate_candidate(_owner(), path, Event(), migrate=True) == (
        "sqlite_validation_unavailable",
    )
    assert len(reads) == 1
    assert (
        sqlite3.SQLITE_REINDEX,
        "idx_console_dispatch_checkpoints_user_message",
        None,
        "main",
        None,
    ) in reached
    assert denied
    if "witness" in sql:
        assert (sqlite3.SQLITE_FUNCTION, None, "random", None, None) in denied
    assert _dump(path) == before


def test_shipped76_callback_scope_is_exact(tmp_path):
    from tldw_chatbook.DB.private_sqlite import open_recovery_validation

    path = tmp_path / "scope.sqlite"
    _shipped76(path)
    with open_recovery_validation(
        "db.chachanotes.primary", path, writable=True, with_restrictions=True
    ) as (connection, restrictions):
        allowed = [
            (
                sqlite3.SQLITE_CREATE_TABLE,
                "console_dispatch_checkpoints_v77",
                None,
                "main",
                None,
            ),
            (
                sqlite3.SQLITE_INSERT,
                "console_dispatch_checkpoints_v77",
                None,
                "main",
                None,
            ),
            (
                sqlite3.SQLITE_DROP_TABLE,
                "console_dispatch_checkpoints",
                None,
                "main",
                None,
            ),
            (
                sqlite3.SQLITE_ALTER_TABLE,
                "main",
                "console_dispatch_checkpoints_v77",
                None,
                None,
            ),
            (
                sqlite3.SQLITE_CREATE_INDEX,
                "idx_console_dispatch_checkpoint_conversation",
                "console_dispatch_checkpoints",
                "main",
                None,
            ),
            (sqlite3.SQLITE_FUNCTION, None, "sqlite_rename_table", None, None),
            (sqlite3.SQLITE_UPDATE, "sqlite_sequence", "name", "main", None),
            (sqlite3.SQLITE_UPDATE, "sqlite_temp_master", "sql", "temp", None),
        ]
        for active, owner, shipped in [
            (False, "db.chachanotes.primary", True),
            (True, "db.subscriptions", True),
            (True, "db.chachanotes.primary", False),
            (True, "db.chachanotes.primary", True),
        ]:
            restrictions.migrating = active
            restrictions.migration_owner = owner
            restrictions.shipped_checkpoint_migration = shipped
            restrictions.shipped_checkpoint_rename = shipped
            expected = (
                sqlite3.SQLITE_OK
                if active and owner == "db.chachanotes.primary" and shipped
                else sqlite3.SQLITE_DENY
            )
            for action in allowed:
                assert restrictions.authorize(*action) == expected, action
                assert (
                    restrictions.authorize(*action[:4], "untrusted_trigger")
                    == sqlite3.SQLITE_DENY
                )
                assert (
                    restrictions.authorize(*action[:3], "untrusted_db", None)
                    == sqlite3.SQLITE_DENY
                )
        restrictions.shipped_checkpoint_rename = False
        assert (
            restrictions.authorize(
                sqlite3.SQLITE_UPDATE, "sqlite_sequence", "name", "main", None
            )
            == sqlite3.SQLITE_DENY
        )
        for sql in (
            "UPDATE sqlite_sequence SET name='unrelated'",
            "CREATE TEMP TABLE unqualified(value)",
            "ATTACH ':memory:' AS other",
            "SELECT load_extension('untrusted')",
            "UPDATE sqlite_sequence SET seq=0",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(sql)


@pytest.mark.parametrize("alteration", ["trigger", "index"])
def test_shipped76_malformed_catalog_refuses_before_writing(
    tmp_path, monkeypatch, alteration
):
    path = tmp_path / "malformed.sqlite"
    _shipped76(path)
    with closing(sqlite3.connect(path)) as connection:
        if alteration == "trigger":
            connection.execute(
                "CREATE TRIGGER unqualified AFTER INSERT ON console_dispatch_checkpoints BEGIN SELECT 1; END"
            )
        else:
            connection.execute(
                "DROP INDEX idx_console_dispatch_checkpoint_conversation"
            )
            connection.execute(
                "CREATE INDEX idx_console_dispatch_checkpoint_conversation ON console_dispatch_checkpoints(attempt_id)"
            )
        connection.commit()
    before = path.read_bytes()
    reached = []
    original = validation._Restrictions.authorize

    def authorize(self, *args):
        if self.migrating or self.shipped_checkpoint_migration:
            reached.append(args)
        return original(self, *args)

    monkeypatch.setattr(validation._Restrictions, "authorize", authorize)
    assert validation.validate_candidate(_owner(), path, Event(), migrate=True) == (
        "unsupported_schema",
    )
    assert not reached
    assert path.read_bytes() == before
