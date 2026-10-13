"""Real SQLite admission must not hold Console revocation or store locks."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
from threading import Event, Thread
from time import monotonic

import pytest

from Tests.Chat.response_rules_fixtures import inputs
from Tests.Chat.response_rules_store_fixtures import learning
from Tests.Chat.test_response_rules_runtime import native as _native_fixture
from Tests.Chat.test_response_rules_runtime import seed
from tldw_chatbook.Chat.response_rules.evaluator import (
    aggregate_checks,
    check_deterministic,
)
from tldw_chatbook.Chat.response_rules.models import RuleBinding
from tldw_chatbook.Chat.response_rules.repository import RuleBindingConflict
from tldw_chatbook.DB.base_db import operation_owned_connection

native = _native_fixture

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        "activate",
        "save_draft",
        "save_assessment",
        "set_binding",
        "promote",
        "delete_binding",
    ],
)
@pytest.mark.parametrize("revoke", ["stop", "source"])
async def test_writer_admission_keeps_revocation_responsive_and_refuses_stale_write(
    native, monkeypatch, operation, revoke
):
    _owner, rules, chats, session, _gateway, controller = native
    seed(native)
    scope = rules.scopes(session.id)[0]
    if operation in {"set_binding", "promote", "delete_binding"}:
        declared = learning(rules._source(session.id))
        rules.store.activate(
            declared.rule, declared.validation, scope, expected_binding_revision=0
        )
    source = rules._source(session.id)
    declared = learning(source)
    db = rules.store.repository.db
    borrowed = db.get_connection()
    transaction = db.transaction
    tables = [
        row[0]
        for row in borrowed.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'console_response_rule_%' ORDER BY name"
        )
    ]

    def snapshot():
        return {
            table: tuple(
                tuple(row)
                for row in borrowed.execute(
                    'SELECT * FROM "' + table + '" ORDER BY rowid'
                )
            )
            for table in tables
        }

    before = snapshot()
    held, release, entering = Event(), Event(), Event()
    errors = []

    def writer():
        try:
            with operation_owned_connection(db), transaction(immediate=True):
                held.set()
                assert release.wait(4)
        except Exception as exc:  # noqa: BLE001 - retain worker failures for the owning test.
            errors.append(exc)

    writer_thread = Thread(target=writer)
    writer_thread.start()
    assert await asyncio.to_thread(held.wait, 2)

    @contextmanager
    def observed(*, immediate=False):
        entering.set()
        with transaction(immediate=immediate) as cursor:
            yield cursor

    monkeypatch.setattr(db, "transaction", observed)
    kwargs = {"current": lambda: rules.acceptance_current(source)}
    if operation == "activate":
        args = (declared.rule, declared.validation, scope)
        kwargs.update(expected_binding_revision=0, admission_lock=rules._lock)
    elif operation == "save_draft":
        args = (scope, source, declared)
        kwargs.update(complaint="Missing evidence", admission_lock=rules._lock)
    elif operation == "save_assessment":
        case = inputs("Missing proof")
        args = (
            aggregate_checks(
                source, (check_deterministic(declared.rule, case),), inputs=case
            ),
        )
        kwargs["admission_lock"] = rules._lock
    elif operation == "set_binding":
        args = (RuleBinding(scope, "rule", 1, "disabled", 2),)
        kwargs["expected_binding_revision"] = 1
    elif operation == "promote":
        args = ("rule", 1, rules.scopes(session.id)[2])
        kwargs["expected_binding_revision"] = 0
    else:
        args = (scope, "rule")
        kwargs["expected_binding_revision"] = 1
    rules._loop = asyncio.get_running_loop()
    rules._project(session.id, phase="testing")
    task = asyncio.create_task(
        rules.run_db_call(getattr(rules.store, operation), *args, **kwargs)
    )
    timer = None
    try:
        assert await asyncio.to_thread(entering.wait, 2)

        def fail_safe_release():
            # The RED must terminate even while the app loop is blocked by a lock.
            release.wait(0.5)
            release.set()

        timer = Thread(target=fail_safe_release)
        timer.start()
        start = monotonic()
        if revoke == "stop":
            if operation in {"set_binding", "delete_binding"}:
                # Destructive intent already revoked this check before writer wait.
                assert not rules.acceptance_current(source)
                assert not controller.stop_active_run()
            else:
                assert controller.stop_active_run()
        else:
            chats.set_workspace_context(
                replace(chats.workspace_context, active_workspace_id="changed-source")
            )
        # Exercise the other synchronous consumer of the shared store lock too.
        rules.store.register_context(scope, None, rules.scopes(session.id)[2])
        elapsed = monotonic() - start
        assert elapsed < 0.1, (
            f"{operation}/{revoke} blocked app progress for {elapsed:.3f}s"
        )
        assert not rules.acceptance_current(source)
        progressed = []
        asyncio.get_running_loop().call_soon(lambda: progressed.append(True))
        await asyncio.sleep(0)
        assert progressed and not task.done() and not release.is_set()
        release.set()
        with pytest.raises(RuleBindingConflict):
            await task
        assert snapshot() == before
        assert rules.state(session.id).phase == "idle"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await rules.wait_for_cleanup()
        writer_thread.join(2)
        if timer is not None:
            timer.join(2)
        assert not writer_thread.is_alive() and not errors
        assert not rules._db_tasks
        assert db.get_connection() is borrowed
        assert borrowed.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("temporary", [False, True])
@pytest.mark.parametrize("rollback", [False, True])
async def test_activation_fences_settlement_and_publishes_memory_only_after_commit(
    native, monkeypatch, temporary, rollback
):
    _owner, rules, _chats, session, _gateway, _controller = native
    seed(native)
    source = rules._source(session.id)
    if temporary:
        source = replace(source, conversation_id=None)
    declared = learning(source)
    scope = rules.scopes(session.id)[0]
    db = rules.store.repository.db
    transaction = db.transaction
    inspected = []

    @contextmanager
    def observed(*, immediate=False):
        with transaction(immediate=immediate) as cursor:
            yield cursor
            if not immediate:
                return

            def inspect():
                locks = (rules.store._lock, rules._lock)
                acquired = [lock.acquire(blocking=False) for lock in locks]
                for lock, owned in zip(locks, acquired, strict=True):
                    if owned:
                        lock.release()
                inspected.append(acquired)

            thread = Thread(target=inspect)
            thread.start()
            thread.join(2)
            assert not thread.is_alive()
            assert inspected == [[False, False]]
            if temporary:
                assert rules.store._temporary[session.id].bindings == {}
            if rollback:
                raise RuntimeError("settlement failure")

    monkeypatch.setattr(db, "transaction", observed)
    operation = rules.run_db_call(
        rules.store.activate,
        declared.rule,
        declared.validation,
        scope,
        expected_binding_revision=0,
        admission_lock=rules._lock,
    )
    if rollback:
        with pytest.raises(RuntimeError, match="settlement failure"):
            await operation
    else:
        binding = await operation
        assert binding.revision == 1
    assert inspected == [[False, False]]
    bindings = rules.store.list_bindings(scope)
    assert len(bindings) == (0 if rollback else 1)
    if rollback:
        with pytest.raises(KeyError):
            rules.store.get_revision(declared.rule.rule_id, 1)
    await rules.wait_for_cleanup()
    assert not rules._db_tasks
    # Both locks must be released on the commit and rollback paths.
    for lock in (rules.store._lock, rules._lock):
        assert lock.acquire(blocking=False)
        lock.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("temporary", [False, True])
@pytest.mark.parametrize("action", ["disabled", "excluded", "delete"])
@pytest.mark.parametrize("admission", ["wait", "readonly"])
async def test_destructive_action_alone_revokes_before_writer_admission(
    native, monkeypatch, temporary, action, admission
):
    import sqlite3

    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Widgets.Console.response_rules_modal import ResponseRulesModal

    _owner, rules, chats, session, _gateway, _controller = native
    if temporary:
        session = chats.create_session(ephemeral=True)
        chats.append_message(
            session.id, role=ConsoleMessageRole.USER, content="Explain result"
        )
        chats.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="Missing proof"
        )
    else:
        seed(native)
    declared = learning(rules._source(session.id))
    scope = rules.scopes(session.id)[0]
    before = rules.store.activate(
        declared.rule, declared.validation, scope, expected_binding_revision=0
    )
    source = rules._source(session.id)
    assert (source.conversation_id is None) is temporary
    assert rules.acceptance_current(source)
    resolution = await _gateway.resolve_for_send(None)
    rules.begin_generation(source.branch_id, resolution)
    assessment = await rules.assess_completed(source)
    rules.release_generation(session.id, source.branch_id)
    assert assessment is not None
    assert assessment.state == "completed" and assessment.outcome == "violation"
    assert rules.current_assessment(source, assessment.assessment_id) == assessment
    modal = ResponseRulesModal(scope, rules.store, rules)
    assert modal._current_scope()
    rules._loop = asyncio.get_running_loop()
    rules._project(session.id, phase="checking")
    db = rules.store.repository.db
    borrowed = db.get_connection()
    transaction = db.transaction
    held, release, entering, refused = Event(), Event(), Event(), Event()
    errors = []

    def writer():
        try:
            with operation_owned_connection(db), transaction(immediate=True):
                held.set()
                assert release.wait(4)
        except Exception as exc:  # noqa: BLE001 - relay owned writer failure.
            errors.append(exc)

    writer_thread = Thread(target=writer)
    writer_thread.start()
    assert await asyncio.to_thread(held.wait, 2)

    @contextmanager
    def observed(*, immediate=False):
        entering.set()
        connection = db.get_connection()
        if admission == "readonly":
            # Real SQLite refuses BEGIN IMMEDIATE; no deadline or busy timeout changes.
            connection.execute("PRAGMA query_only=ON")
        try:
            with transaction(immediate=immediate) as cursor:
                yield cursor
        except sqlite3.OperationalError:
            refused.set()
            raise
        finally:
            if admission == "readonly":
                connection.execute("PRAGMA query_only=OFF")

    monkeypatch.setattr(db, "transaction", observed)
    if action == "delete":
        operation, args = rules.store.delete_binding, (scope, before.rule_id)
    else:
        operation, args = rules.store.set_binding, (replace(before, state=action),)
    task = asyncio.create_task(
        rules.run_db_call(
            operation,
            *args,
            expected_binding_revision=1,
            current=modal._current_scope,
        )
    )
    try:
        assert await asyncio.to_thread(entering.wait, 2)
        # Only Disable/Exclude/Delete has run: no Stop or Workspace mutation.
        assert not release.is_set()
        assert not rules.acceptance_current(source)
        assert rules.pinned_rules(source) == ()
        assert rules.current_assessment(source, assessment.assessment_id) is None
        assert modal._current_scope()  # User intent outlives the revoked assessment.
        start = monotonic()
        rules.store.register_context(*rules.scopes(session.id))
        assert monotonic() - start < 0.1
        await asyncio.sleep(0)
        assert rules.state(session.id).phase == "idle"
        assert await rules.assess_completed(source) is None
        if admission == "wait":
            assert not task.done()
            release.set()
            await task
            expected = (
                ()
                if action == "delete"
                else (replace(before, state=action, binding_revision=2),)
            )
            assert rules.store.list_bindings(scope) == expected
        else:
            assert await asyncio.to_thread(refused.wait, 2)
            assert not rules.acceptance_current(source)
            # Admission failed while the writer was held. Let physical handle
            # retirement checkpoint normally before awaiting the callback owner.
            release.set()
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                await task
            assert rules.store.list_bindings(scope) == (before,)
            assert not rules.acceptance_current(source)
        assert not rules._assessments
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await rules.wait_for_cleanup()
        writer_thread.join(2)
        assert not writer_thread.is_alive() and not errors and not rules._db_tasks
        assert db.get_connection() is borrowed
        assert borrowed.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["scope", "binding", "new_assessment_source"])
async def test_destructive_write_rechecks_user_owner_and_fences_work_started_during_wait(
    native, monkeypatch, change
):
    from tldw_chatbook.Widgets.Console.response_rules_modal import ResponseRulesModal

    _owner, rules, chats, session, _gateway, _controller = native
    seed(native)
    declared = learning(rules._source(session.id))
    scope = rules.scopes(session.id)[0]
    binding = rules.store.activate(
        declared.rule, declared.validation, scope, expected_binding_revision=0
    )
    source = rules._source(session.id)
    modal = ResponseRulesModal(scope, rules.store, rules)
    db = rules.store.repository.db
    transaction = db.transaction
    held, release, entering = Event(), Event(), Event()
    errors = []

    def writer():
        try:
            with operation_owned_connection(db), transaction(immediate=True) as cursor:
                held.set()
                assert release.wait(4)
                if change == "binding":
                    cursor.execute(
                        "UPDATE console_response_rule_bindings SET binding_revision=2 WHERE rule_id=?",
                        (binding.rule_id,),
                    )
        except Exception as exc:  # noqa: BLE001 - relay owned writer failure.
            errors.append(exc)

    thread = Thread(target=writer)
    thread.start()
    assert await asyncio.to_thread(held.wait, 2)

    @contextmanager
    def observed(*, immediate=False):
        entering.set()
        with transaction(immediate=immediate) as cursor:
            yield cursor

    monkeypatch.setattr(db, "transaction", observed)
    task = asyncio.create_task(
        rules.run_db_call(
            rules.store.set_binding,
            replace(binding, state="disabled"),
            expected_binding_revision=1,
            current=modal._current_scope,
        )
    )
    try:
        assert await asyncio.to_thread(entering.wait, 2)
        assert not rules.acceptance_current(source)
        if change == "scope":
            chats.create_session(title="Another Chat", ephemeral=True)
            assert not modal._current_scope()
        elif change == "new_assessment_source":
            source = rules._source(session.id)
            assert rules.acceptance_current(source)
        release.set()
        if change == "new_assessment_source":
            await task
            assert not rules.acceptance_current(source)
            assert rules.store.list_bindings(scope) == (
                replace(binding, state="disabled", binding_revision=2),
            )
        else:
            with pytest.raises(RuleBindingConflict):
                await task
            assert rules.store.list_bindings(scope) == (
                replace(binding, binding_revision=2 if change == "binding" else 1),
            )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await rules.wait_for_cleanup()
        thread.join(2)
        assert not thread.is_alive() and not errors and not rules._db_tasks
