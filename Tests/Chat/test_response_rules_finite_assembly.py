"""Each real rule resolution owns one finite admission, never a later resolution."""

import inspect
import sqlite3
import sys
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_participant_lifetimes as lifetimes
from Tests.Chat import response_rules_store_fixtures as fixtures
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chat.response_rules.models import RuleScope
from tldw_chatbook.Chat.response_rules.repository import ResponseRuleRepository

pytestmark = pytest.mark.bootstrap_profile
local_root = lifetimes.local_root
rule_store = fixtures.rule_store
activate = fixtures.activate


@pytest.fixture(autouse=True)
def _isolated_admission(local_root):
    # Install the existing native-root fixture before the real DB fixture runs.
    pass


@contextmanager
def _original_reads(db):
    """Observe original native owners and SQL getter order, without replacements."""
    operation = inspect.unwrap(storage._repository_operation)
    acquire = storage.acquire_storage
    getters = (
        ResponseRuleRepository.list_bindings,
        ResponseRuleRepository.get_revision,
    )
    originals = (operation, acquire, *getters)
    codes = tuple(fn.__code__ for fn in originals)
    thread = threading.get_ident()
    result = SimpleNamespace(leases=[], operations=[], holds=[], reads=[])
    tool = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
    sys.monitoring.use_tool_id(tool, "response-rules-finite-assembly")
    events = sys.monitoring.events

    def started(code, _offset):
        if threading.get_ident() != thread:
            return
        frame = sys._getframe(1)
        if frame.f_locals["self"].db is db:
            result.reads.append(code.co_name)

    def returned(_code, _offset, lease):
        if threading.get_ident() != thread:
            return
        parent = sys._getframe(1).f_back
        if (
            parent.f_code is codes[0]
            and parent.f_locals["participant"].repository() is db
        ):
            assert type(lease) is storage.StorageLease
            assert lease in storage._live_leases and lease._key is not None
            result.leases.append(lease)

    def yielded(_code, _offset, current):
        if (
            threading.get_ident() != thread
            or current.participant.repository() is not db
        ):
            return
        assert current in storage._operations
        assert current.lease in storage._live_leases
        assert current.hold is storage._holds[current.lease._key]
        assert current.hold.ready.is_set() and current.hold.thread.is_alive()
        assert current.hold.error is None and not current.hold.stop.is_set()
        if current not in result.operations:
            result.operations.append(current)
        if current.hold not in result.holds:
            result.holds.append(current.hold)

    try:
        sys.monitoring.register_callback(tool, events.PY_START, started)
        sys.monitoring.register_callback(tool, events.PY_RETURN, returned)
        sys.monitoring.register_callback(tool, events.PY_YIELD, yielded)
        sys.monitoring.set_local_events(tool, codes[0], events.PY_YIELD)
        sys.monitoring.set_local_events(tool, codes[1], events.PY_RETURN)
        for code in codes[2:]:
            sys.monitoring.set_local_events(tool, code, events.PY_START)
        yield result
    finally:
        for code in codes:
            sys.monitoring.set_local_events(tool, code, 0)
        for event in (events.PY_START, events.PY_RETURN, events.PY_YIELD):
            sys.monitoring.register_callback(tool, event, None)
        sys.monitoring.free_tool_id(tool)
        assert all(
            fn.__code__ is code for fn, code in zip(originals, codes, strict=True)
        )


def _retired(db, borrowed, observed):
    assert observed.operations and observed.leases
    assert {item.lease for item in observed.operations} == set(observed.leases)
    assert all(item not in storage._operations for item in observed.operations)
    assert all(
        item not in storage._live_leases and item._key is None
        for item in observed.leases
    )
    assert getattr(storage._operation_local, "operation", None) is None
    assert db._local.conn is borrowed
    assert borrowed.execute("SELECT 1").fetchone()[0] == 1
    assert not borrowed.in_transaction


def _close_original_owner(db, borrowed, observations):
    db.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        borrowed.execute("SELECT 1")
    assert not db._maintenance_participant.connections
    holds = {hold for observed in observations for hold in observed.holds}
    assert holds
    assert all(hold.stop.is_set() and not hold.thread.is_alive() for hold in holds)
    assert all(
        hold.error is None and hold not in storage._retiring_holds for hold in holds
    )


@pytest.mark.parametrize("inherited", [False, True], ids=["empty", "inherited"])
def test_effective_rules_has_one_native_owner_per_fresh_assembly(
    rule_store, inherited, record_property
):
    store, db, origin = rule_store
    chat = RuleScope("chat", origin.conversation_id)
    workspace = RuleScope("workspace", "rules-workspace")
    global_scope = RuleScope("global", origin.profile_id)
    if inherited:
        chat_binding = activate(store, origin)
        store.promote("rule", 1, workspace, expected_binding_revision=0)
        store.promote("rule", 1, global_scope, expected_binding_revision=0)
    borrowed = db.get_connection()
    with _original_reads(db) as first:
        before = store.effective_rules(chat, workspace, global_scope)
    assert [(rule.rule_id, rule.revision) for rule in before] == (
        [("rule", 1)] if inherited else []
    )
    assert first.reads == ["list_bindings"] * 3 + ["get_revision"] * (
        3 if inherited else 0
    )
    _retired(db, borrowed, first)
    # Actual supported writes occur after complete retirement of the first read.
    if inherited:
        store.set_binding(
            replace(chat_binding, revision=None, state="excluded"),
            expected_binding_revision=1,
        )
    else:
        activate(store, origin, global_scope)
    with _original_reads(db) as second:
        after = store.effective_rules(chat, workspace, global_scope)
    assert [(rule.rule_id, rule.revision) for rule in after] == (
        [] if inherited else [("rule", 1)]
    )
    assert second.reads == ["list_bindings"] * 3 + ["get_revision"] * (
        2 if inherited else 1
    )
    _retired(db, borrowed, second)
    assert not set(first.operations) & set(second.operations)
    assert not set(first.leases) & set(second.leases)
    _close_original_owner(db, borrowed, (first, second))
    counts = [len(first.leases), len(second.leases)]
    record_property("original_native_acquisitions", counts)
    record_property("original_repository_reads", [first.reads, second.reads])
    # Original source reaches all payload/freshness/native-retirement assertions,
    # then fails only here: empty [3, 4], inherited [6, 5].
    assert counts == [1, 1]


@pytest.mark.parametrize(
    "missing", [True, False], ids=["missing-revision", "reader-error"]
)
def test_effective_rules_keeps_custom_reader_failure_and_native_retirement(
    rule_store, monkeypatch, missing
):
    store, db, origin = rule_store
    chat = RuleScope("chat", origin.conversation_id)
    global_scope = RuleScope("global", origin.profile_id)
    activate(store, origin, global_scope)
    original = store.get_revision
    entered = []

    def read_then_fail(rule_id, revision):
        value = original(rule_id, revision)
        entered.append((value.rule_id, value.revision))
        if missing:
            raise KeyError((rule_id, revision))
        raise RuntimeError("original-reader-completed")

    monkeypatch.setattr(store, "get_revision", read_then_fail)
    borrowed = db.get_connection()
    error = (
        nullcontext()
        if missing
        else pytest.raises(RuntimeError, match="original-reader-completed")
    )
    with _original_reads(db) as observed:
        with error:
            assert store.effective_rules(chat, None, global_scope) == ()
    assert entered == [("rule", 1)]
    assert observed.reads == ["list_bindings", "list_bindings", "get_revision"]
    _retired(db, borrowed, observed)
    _close_original_owner(db, borrowed, (observed,))
