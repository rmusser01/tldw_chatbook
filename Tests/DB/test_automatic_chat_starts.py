"""Native chat starts share finite allowance while retaining local run ownership."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.DB.test_automatic_work_budget import _automatic_work_db as db, chain  # noqa: F401
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused


def prepare(db, source_run, target, attempt, **changes):
    arguments = dict(
        attempt_id=attempt,
        source_run_id=source_run,
        target_conversation_id=target,
        target_session_id=f"session-{target}",
        target_session_incarnation=f"incarnation-{target}",
        owner_id="owner",
        draft_revision=1,
        context_epoch=0,
        request_fingerprint="a" * 64,
    )
    arguments.update(changes)
    return db.automatic_work.prepare_chat_start(**arguments)


def source_run(db, **limits):
    root = chain(db, conversation="source", **limits)
    return root, db.create_run(
        conversation_id="source", agent_kind="primary", work_chain_id=root
    )


def test_two_targets_share_the_last_generation(db):
    root, source = source_run(db, generations=1)
    first = prepare(db, source, "target-a", "attempt-a")
    assert db.automatic_work.allowance_root(first.chain_id) == root
    assert db.automatic_work.snapshot(first.chain_id).conversation_id == "target-a"
    assert db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    assert not db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    assert not db.automatic_work.abort_chat_start(first.id, owner_id="owner")
    with pytest.raises(AutomaticWorkRefused, match="generation_budget"):
        prepare(db, source, "target-b", "attempt-b")
    assert db.automatic_work.snapshot(root).used["generation"] == 1


def test_recursive_starts_use_direct_root_and_local_primary(db):
    root, source = source_run(db)
    first = prepare(db, source, "a", "first")
    target = db.create_run(
        conversation_id="a", agent_kind="primary", work_chain_id=first.chain_id
    )
    second = prepare(db, target, "b", "second")
    assert second.source_chain_id == first.chain_id
    assert db.automatic_work.allowance_root(second.chain_id) == root
    assert db.get_run(target)["parent_run_id"] is None
    with pytest.raises(ValueError, match="scope"):
        db.create_run(
            conversation_id="b",
            agent_kind="primary",
            parent_run_id=source,
            work_chain_id=second.chain_id,
        )
    with pytest.raises(ValueError, match="scope"):
        db.create_run(conversation_id="b", agent_kind="primary", work_chain_id=root)
    with db.transaction() as conn:
        with pytest.raises(sqlite3.IntegrityError, match="scope"):
            conn.execute(
                "UPDATE agent_runs SET conversation_id='b' WHERE id=?", (target,)
            )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_preparation_identity_abort_and_completion_are_exact(db):
    root, source = source_run(db)
    first = prepare(db, source, "a", "first")
    assert prepare(db, source, "a", "first") == first
    for changes in (
        {"draft_revision": 2},
        {"context_epoch": 1},
        {"request_fingerprint": "b" * 64},
        {"target_session_id": "other"},
        {"target_session_incarnation": "other"},
        {"owner_id": "other"},
    ):
        with pytest.raises(ValueError, match="identity"):
            prepare(db, source, "a", "first", **changes)
    with pytest.raises(ValueError, match="owner"):
        db.automatic_work.read_chat_start_attempt(first.id, owner_id="other")
    assert not db.automatic_work.complete_chat_start(first.id, owner_id="owner")
    assert db.automatic_work.abort_chat_start(first.id, owner_id="owner")
    assert not db.automatic_work.abort_chat_start(first.id, owner_id="owner")
    assert not db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    assert db.automatic_work.snapshot(root).available["generation"] == 3
    second = prepare(db, source, "a", "second")
    assert db.automatic_work.accept_chat_start(second.id, owner_id="owner")
    assert db.automatic_work.complete_chat_start(second.id, owner_id="owner")
    assert not db.automatic_work.complete_chat_start(second.id, owner_id="owner")
    with db.connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM automatic_wake_claims").fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT wake_delivered_at FROM agent_runs WHERE id=?", (source,)
            ).fetchone()[0]
            is None
        )


@pytest.mark.parametrize("terminal", [False, True])
def test_unavailable_source_mints_nothing(db, terminal):
    if terminal:
        root, source = source_run(db)
        db.set_status(source, "done")
    else:
        source = db.create_run(conversation_id="source", agent_kind="primary")
    with pytest.raises(AutomaticWorkRefused, match="source"):
        prepare(db, source, "a", "first")
    with db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM automatic_work_chains").fetchone()[
            0
        ] == int(terminal)
        assert (
            conn.execute("SELECT COUNT(*) FROM automatic_work_reservations").fetchone()[
                0
            ]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM automatic_chat_start_attempts"
            ).fetchone()[0]
            == 0
        )


def test_two_targets_race_for_one_generation(db):
    root, source = source_run(db, generations=1)
    ready = threading.Barrier(2)

    def compete(number):
        ready.wait(timeout=5)
        try:
            return prepare(db, source, f"target-{number}", f"attempt-{number}")
        except AutomaticWorkRefused as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(compete, [1, 2]))
    assert results.count("generation_budget") == 1
    assert db.automatic_work.snapshot(root).reserved["generation"] == 1
    with db.connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM automatic_work_chains").fetchone()[0]
            == 2
        )


@pytest.mark.parametrize("accepted", [False, True])
def test_recovery_marks_foreign_starts_and_fences_old_owner(db, accepted):
    root, source = source_run(db)
    first = prepare(db, source, "a", "first")
    sibling = prepare(db, source, "b", "second")
    if accepted:
        assert db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    assert db.automatic_work.recover(current_owner_id="replacement") == 1
    for member in (root, first.chain_id, sibling.chain_id):
        assert db.automatic_work.snapshot(member).status == "review_required"
    assert (
        db.automatic_work.read_chat_start_attempt(first.id, owner_id="owner").state
        == "review_required"
    )
    assert not db.automatic_work.abort_chat_start(first.id, owner_id="owner")
    with pytest.raises(AutomaticWorkRefused, match="runtime_owner_replaced"):
        db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    with pytest.raises(AutomaticWorkRefused, match="runtime_owner_replaced"):
        prepare(db, source, "c", "third")
    assert db.automatic_work.recover(current_owner_id="replacement") == 0


def test_membership_and_attempt_identity_cannot_be_rewritten(db):
    root, source = source_run(db)
    first = prepare(db, source, "a", "first")
    second_root = chain(db, conversation="other", submission="other")
    with db.transaction() as conn:
        for member, parent in (
            (first.chain_id, second_root),
            (root, first.chain_id),
            (root, root),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE automatic_work_chains SET allowance_root_chain_id=? WHERE id=?",
                    (parent, member),
                )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO automatic_work_chains (id, conversation_id, root_submission_id, limits_json, created_at, last_observed_at, allowance_root_chain_id) SELECT 'invalid','invalid','invalid',limits_json,created_at,last_observed_at,? FROM automatic_work_chains WHERE id=?",
                (first.chain_id, root),
            )
        for column, value in (
            ("source_run_id", "other"),
            ("source_chain_id", second_root),
            ("chain_id", root),
            ("conversation_id", "other"),
            ("session_id", "other"),
            ("session_incarnation", "other"),
            ("owner_id", "other"),
            ("draft_revision", 2),
            ("context_epoch", 1),
            ("request_fingerprint", "b" * 64),
            ("generation_reservation_id", "other"),
            ("id", "other"),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                conn.execute(
                    f"UPDATE automatic_chat_start_attempts SET {column}=? WHERE id=?",
                    (value, first.id),
                )


def test_descendant_unknown_usage_blocks_siblings(db):
    root, source = source_run(db)
    first = prepare(db, source, "a", "first")
    sibling = prepare(db, source, "b", "sibling")
    db.automatic_work.reserve(
        first.chain_id,
        reservation_id="usage",
        owner_id="owner",
        kind="tokens",
        amount=20,
    )
    db.automatic_work.commit("usage", owner_id="owner")
    db.automatic_work.settle("usage", owner_id="owner", actual_amount=None)
    for member in (root, first.chain_id, sibling.chain_id):
        assert db.automatic_work.snapshot(member).status == "review_required"
    with pytest.raises(AutomaticWorkRefused, match="usage_unknown"):
        db.automatic_work.accept_chat_start(sibling.id, owner_id="owner")


def test_descendant_overage_pauses_siblings_and_late_usage_keeps_review(db):
    root, source = source_run(db, budget_tokens=100)
    first = prepare(db, source, "a", "first")
    sibling = prepare(db, source, "b", "sibling")
    db.automatic_work.reserve(
        first.chain_id,
        reservation_id="usage",
        owner_id="owner",
        kind="tokens",
        amount=20,
    )
    db.automatic_work.commit("usage", owner_id="owner")
    db.automatic_work.settle("usage", owner_id="owner", actual_amount=101)
    for member in (root, first.chain_id, sibling.chain_id):
        assert db.automatic_work.snapshot(member).pause_reason == "tokens_budget"
    with pytest.raises(AutomaticWorkRefused, match="tokens_budget"):
        db.automatic_work.accept_chat_start(sibling.id, owner_id="owner")
    db.automatic_work.recover(current_owner_id="replacement")
    assert db.automatic_work.snapshot(root).status == "review_required"
    assert db.automatic_work.snapshot(sibling.chain_id).reserved["generation"] == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"draft_revision": True},
        {"draft_revision": -1},
        {"context_epoch": 2**63},
        {"request_fingerprint": "private body"},
        {"request_fingerprint": "G" * 64},
    ],
)
def test_malformed_native_identity_cannot_mint_allowance(db, changes):
    root, source = source_run(db)
    with pytest.raises(ValueError):
        prepare(db, source, "a", "first", **changes)
    assert db.automatic_work.snapshot(root).reserved["generation"] == 0


def test_descendant_unknown_usage_settles_late_without_restoring_authority(db):
    root, source = source_run(db)
    first = prepare(db, source, "a", "first")
    db.automatic_work.reserve(
        first.chain_id,
        reservation_id="usage",
        owner_id="owner",
        kind="tokens",
        amount=20,
    )
    db.automatic_work.commit("usage", owner_id="owner")
    assert db.automatic_work.recover(current_owner_id="replacement") == 1
    assert db.automatic_work.settle("usage", owner_id="owner", actual_amount=10)
    assert db.automatic_work.snapshot(first.chain_id).used["tokens"] == 10
    assert db.automatic_work.snapshot(root).status == "review_required"


def test_failed_native_write_rolls_back_member_and_reserved_generation(db):
    root, source = source_run(db)
    with db.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER deny_start BEFORE INSERT ON automatic_chat_start_attempts BEGIN SELECT RAISE(ABORT, 'write refused'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="write refused"):
        prepare(db, source, "a", "first")
    assert db.automatic_work.snapshot(root).reserved["generation"] == 0
    with db.connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM automatic_work_chains").fetchone()[0]
            == 1
        )
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_failed_native_acceptance_keeps_prepared_charge_and_unstarted_clock(db):
    root, source = source_run(db)
    first = prepare(db, source, "a", "first")
    with db.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER deny_accept BEFORE UPDATE OF state ON automatic_chat_start_attempts WHEN NEW.state='accepted' BEGIN SELECT RAISE(ABORT, 'write refused'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="write refused"):
        db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    snapshot = db.automatic_work.snapshot(root)
    assert snapshot.used["generation"] == 0
    assert snapshot.reserved["generation"] == 1
    assert snapshot.started_at is None
    assert (
        db.automatic_work.read_chat_start_attempt(first.id, owner_id="owner").state
        == "prepared"
    )
    assert db.automatic_work.abort_chat_start(first.id, owner_id="owner")


def test_acceptance_and_abort_race_has_one_winner(db):
    root, source = source_run(db)
    first = prepare(db, source, "a", "first")
    ready = threading.Barrier(2)

    def race(accept):
        ready.wait(timeout=5)
        method = (
            db.automatic_work.accept_chat_start
            if accept
            else db.automatic_work.abort_chat_start
        )
        return method(first.id, owner_id="owner")

    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted, aborted = list(pool.map(race, [True, False]))
    assert accepted != aborted
    snapshot = db.automatic_work.snapshot(root)
    assert snapshot.used["generation"] == int(accepted)
    assert snapshot.reserved["generation"] == 0
    assert db.automatic_work.read_chat_start_attempt(
        first.id, owner_id="owner"
    ).state == ("accepted" if accepted else "aborted")


def test_native_abort_never_refunds_a_separately_committed_reservation(db):
    root, source = source_run(db)
    first = prepare(db, source, "a", "first")
    assert db.automatic_work.commit(first.generation_reservation_id, owner_id="owner")
    assert not db.automatic_work.abort_chat_start(first.id, owner_id="owner")
    assert db.automatic_work.snapshot(root).used["generation"] == 1


def test_stale_owner_cannot_pause_a_completed_native_replacement(db):
    root, source = source_run(db)
    first = prepare(db, source, "a", "first")
    db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    db.automatic_work.complete_chat_start(first.id, owner_id="owner")
    assert db.automatic_work.recover(current_owner_id="replacement") == 0
    with pytest.raises(AutomaticWorkRefused, match="runtime_owner_replaced"):
        prepare(db, source, "b", "second")
    for member in (root, first.chain_id):
        assert db.automatic_work.snapshot(member).status == "active"
    with db.transaction() as conn:
        with pytest.raises(sqlite3.IntegrityError, match="scope"):
            conn.execute(
                "INSERT INTO agent_runs (id, conversation_id, agent_kind, task, status, steps, budget, created_at, updated_at, work_chain_id) SELECT 'invalid','other',agent_kind,task,status,steps,budget,created_at,updated_at,work_chain_id FROM agent_runs WHERE id=?",
                (source,),
            )
