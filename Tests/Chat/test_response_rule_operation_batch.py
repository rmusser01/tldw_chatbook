"""One complete rules lookup keeps its original transactions and resource owner."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import sqlite3
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chat.response_rules.models import RuleScope
from tldw_chatbook.Chat.response_rules.repository import ResponseRuleRepository
from tldw_chatbook.Chat.response_rules.store import ResponseRuleStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class CustomRuleStore(ResponseRuleStore):
    pass


@pytest.mark.parametrize(
    "custom,memory", ((False, False), (True, False), (False, True))
)
def test_scope_lookup_batches_only_stock_store_and_keeps_transactions(
    tmp_path, monkeypatch, custom, memory
):
    database = CharactersRAGDB(
        ":memory:" if memory else tmp_path / "rules.sqlite", "rules-batch"
    )
    store = (CustomRuleStore if custom else ResponseRuleStore)(
        ResponseRuleRepository(database)
    )
    admissions = []
    original = storage.acquire_storage

    def counted(path=None, **kwargs):
        admissions.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(storage, "acquire_storage", counted)
    connection = database.get_connection()
    statements = []
    connection.set_trace_callback(statements.append)
    admissions.clear()
    try:
        assert (
            store.effective_rules(
                RuleScope("chat", "chat"),
                RuleScope("workspace", "workspace"),
                RuleScope("global", "profile"),
            )
            == ()
        )
        assert len(admissions) == (0 if memory else 3 if custom else 1)
        assert (
            sum("FROM console_response_rule_bindings" in sql for sql in statements) == 3
        )
        assert sum(sql.startswith("BEGIN") for sql in statements) == 3
        assert sum(sql == "COMMIT" for sql in statements) == 3
        assert database.get_connection() is connection
        assert database.registered_connection_count() == 1
    finally:
        connection.set_trace_callback(None)
        database.close_connection()


def test_new_rule_lookup_handle_retires_before_callback_returns(tmp_path, monkeypatch):
    database = CharactersRAGDB(tmp_path / "rules.sqlite", "rules-batch")
    store = ResponseRuleStore(ResponseRuleRepository(database))
    admissions = []
    original = storage.acquire_storage

    def counted(path=None, **kwargs):
        admissions.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(storage, "acquire_storage", counted)

    def lookup():
        try:
            result = store.effective_rules(
                RuleScope("chat", "chat"),
                RuleScope("workspace", "workspace"),
                RuleScope("global", "profile"),
            )
            return result, database.registered_connection_count()
        finally:
            database.close_connection()  # Retire the old implementation in RED too.

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result, handles = pool.submit(lookup).result()
        assert result == ()
        assert handles == 1  # Original caller's handle only.
        assert len(admissions) == 2  # Operation plus independent SQLite lease.
    finally:
        database.close_connection()


def test_failed_rule_lookup_retires_new_handle_before_error_returns(tmp_path):
    database = CharactersRAGDB(tmp_path / "rules.sqlite", "rules-batch")
    store = ResponseRuleStore(ResponseRuleRepository(database))
    with database.transaction() as cursor:
        cursor.execute("DROP TABLE console_response_rule_bindings")

    def lookup():
        try:
            with pytest.raises(sqlite3.OperationalError, match="no such table"):
                store.effective_rules(
                    RuleScope("chat", "chat"), None, RuleScope("global", "profile")
                )
            return database.registered_connection_count()
        finally:
            database.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(lookup).result() == 1
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    "owner",
    (
        "store-instance",
        "store-class",
        "repository",
        "database",
        "database-class-binding",
    ),
)
def test_substituted_readers_keep_original_unbatched_admission(
    tmp_path, monkeypatch, owner
):
    database = CharactersRAGDB(tmp_path / "custom.sqlite", "custom-rules")
    repository = ResponseRuleRepository(database)
    store = ResponseRuleStore(repository)
    admissions, calls = [], []
    acquire = storage.acquire_storage

    def counted(path=None, **kwargs):
        admissions.append(path)
        return acquire(path, **kwargs)

    def bindings(*args):
        calls.append(args)
        return ()

    @contextmanager
    def transaction():
        calls.append("transaction")
        yield SimpleNamespace(execute=lambda *args: ())

    if owner == "store-instance":
        monkeypatch.setattr(store, "list_bindings", bindings)
    elif owner == "store-class":
        monkeypatch.setattr(ResponseRuleStore, "list_bindings", bindings)
    elif owner == "repository":
        monkeypatch.setattr(repository, "list_bindings", bindings)
    elif owner == "database-class-binding":
        from tldw_chatbook.DB import ChaChaNotes_DB

        monkeypatch.setattr(
            ChaChaNotes_DB, "CharactersRAGDB", type("Replacement", (), {})
        )
        monkeypatch.setattr(store, "list_bindings", bindings)
    else:
        monkeypatch.setattr(database, "transaction", transaction)
    monkeypatch.setattr(storage, "acquire_storage", counted)
    try:
        assert (
            store.effective_rules(
                RuleScope("chat", "chat"),
                RuleScope("workspace", "workspace"),
                RuleScope("global", "profile"),
            )
            == ()
        )
        assert len(calls) == 3
        assert admissions == []
    finally:
        database.close_connection()


def test_custom_rule_reader_keeps_its_new_worker_handle(tmp_path, monkeypatch):
    database = CharactersRAGDB(tmp_path / "custom.sqlite", "custom-rules")
    store = ResponseRuleStore(ResponseRuleRepository(database))

    def bindings(scope):
        database.get_connection()
        return ()

    monkeypatch.setattr(store, "list_bindings", bindings)

    def lookup():
        try:
            assert (
                store.effective_rules(
                    RuleScope("chat", "chat"), None, RuleScope("global", "profile")
                )
                == ()
            )
            return database.registered_connection_count()
        finally:
            database.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(lookup).result() == 2
    finally:
        database.close_connection()


def test_reader_changed_during_admission_cannot_publish(tmp_path, monkeypatch):
    from tldw_chatbook.Chat.response_rules.repository import RuleBindingConflict

    database = CharactersRAGDB(tmp_path / "rules.sqlite", "rules-batch")
    store = ResponseRuleStore(ResponseRuleRepository(database))
    acquire = storage.acquire_storage
    calls = []

    def changed(scope):
        calls.append(scope)
        return ()

    def replace_reader(path=None, **kwargs):
        lease = acquire(path, **kwargs)
        monkeypatch.setattr(store, "list_bindings", changed)
        return lease

    monkeypatch.setattr(storage, "acquire_storage", replace_reader)
    try:
        with pytest.raises(RuleBindingConflict, match="rule_source_changed"):
            store.effective_rules(
                RuleScope("chat", "chat"), None, RuleScope("global", "profile")
            )
        assert calls == []
        assert database.registered_connection_count() == 1
    finally:
        database.close_connection()
