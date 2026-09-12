"""Claims and acceptance are atomic; chat rows and UI marks are not authority."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.Agents.agent_models import TERMINAL_RUN_STATUSES

from Tests.DB.test_automatic_work_budget import (
    _automatic_work_db,  # noqa: F401
    chain,
)


def survivor(db, chain_id, *, conversation="conversation"):
    parent = db.create_run(
        conversation_id=conversation, agent_kind="primary", work_chain_id=chain_id
    )
    db.set_status(parent, "done", "parent")
    child = db.create_run(
        conversation_id=conversation, parent_run_id=parent, agent_kind="subagent"
    )
    db.set_status(child, "done", "saved result")
    return child


def claim(
    db, chain_id, run_ids, *, attempt="attempt", owner="owner", session="session"
):
    return db.automatic_work.claim_wake(
        chain_id,
        attempt_id=attempt,
        owner_id=owner,
        session_id=session,
        run_ids=run_ids,
    )


def test_repeated_claim_and_acceptance_grant_only_one_dispatch(db):
    chain_id = chain(db)
    run_id = survivor(db, chain_id)
    first = claim(db, chain_id, [run_id])
    assert first == claim(db, chain_id, [run_id])
    assert first.state == "prepared"
    assert db.automatic_work.accept_wake("attempt", owner_id="owner") is True
    assert db.automatic_work.accept_wake("attempt", owner_id="owner") is False
    assert not db.automatic_work.abort_wake("attempt", owner_id="owner")
    assert db.automatic_work.snapshot(chain_id).used["generation"] == 1
    assert db.get_run(run_id)["wake_delivered_at"] is None
    assert run_id not in {row["id"] for row in db.undelivered_wake_runs("conversation")}
    assert db.automatic_work.complete_wake("attempt", owner_id="owner") is True
    stamped = db.get_run(run_id)["wake_delivered_at"]
    assert stamped
    assert not db.automatic_work.complete_wake("attempt", owner_id="owner")
    assert db.get_run(run_id)["wake_delivered_at"] == stamped


@pytest.mark.parametrize("parent_status", sorted(TERMINAL_RUN_STATUSES))
def test_discovered_survivor_is_claimable_after_every_terminal_parent(
    db, parent_status
):
    chain_id = chain(db)
    parent = db.create_run(
        conversation_id="conversation", agent_kind="primary", work_chain_id=chain_id
    )
    db.set_status(parent, parent_status, "parent settled")
    child = db.create_run(
        conversation_id="conversation", parent_run_id=parent, agent_kind="subagent"
    )
    db.set_status(child, "done", "saved result")

    assert child in {row["id"] for row in db.undelivered_wake_runs("conversation")}
    assert claim(db, chain_id, [child]).state == "prepared"
    assert db.automatic_work.accept_wake("attempt", owner_id="owner")
    assert db.automatic_work.complete_wake("attempt", owner_id="owner")
    assert db.get_run(child)["wake_delivered_at"]
    assert db.automatic_work.snapshot(chain_id).status == "active"


def test_proven_preacceptance_abort_refunds_and_releases_exact_batch(db):
    chain_id = chain(db, generations=1)
    run_id = survivor(db, chain_id)
    claim(db, chain_id, [run_id])
    assert db.automatic_work.abort_wake("attempt", owner_id="owner")
    assert not db.automatic_work.abort_wake("attempt", owner_id="owner")
    assert db.automatic_work.snapshot(chain_id).available["generation"] == 1
    assert run_id in {row["id"] for row in db.undelivered_wake_runs("conversation")}
    claim(db, chain_id, [run_id], attempt="retry")
    assert db.automatic_work.accept_wake("retry", owner_id="owner")
    assert not db.automatic_work.accept_wake("attempt", owner_id="owner")


@pytest.mark.parametrize("accepted", [False, True])
def test_startup_recovery_blocks_ambiguous_work_but_handle_reopen_does_not(
    db, accepted
):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    chain_id = chain(db)
    run_id = survivor(db, chain_id)
    claim(db, chain_id, [run_id])
    if accepted:
        db.automatic_work.accept_wake("attempt", owner_id="owner")
    db.close()
    assert db.automatic_work.snapshot(chain_id).status == "active"
    assert db.automatic_work.recover(current_owner_id="owner") == 0
    assert db.automatic_work.recover(current_owner_id="replacement") == 1
    assert db.automatic_work.recover(current_owner_id="replacement") == 0
    snapshot = db.automatic_work.snapshot(chain_id)
    assert snapshot.status == "review_required"
    assert snapshot.used["generation"] + snapshot.reserved["generation"] == 1
    assert not db.automatic_work.abort_wake("attempt", owner_id="owner")
    with pytest.raises(AutomaticWorkRefused, match="runtime_owner_replaced"):
        db.automatic_work.accept_wake("attempt", owner_id="owner")
    with pytest.raises(AutomaticWorkRefused):
        claim(db, chain_id, [run_id], attempt="replay", owner="replacement")
    assert db.get_run(run_id)["result"] == "saved result"


def test_two_claimers_cannot_charge_or_dispatch_the_same_result_twice(db):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    chain_id = chain(db)
    run_id = survivor(db, chain_id)

    def compete(identity):
        try:
            claim(db, chain_id, [run_id], attempt=identity, owner=identity)
            return db.automatic_work.accept_wake(identity, owner_id=identity)
        except AutomaticWorkRefused:
            return False
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(compete, ["one", "two"])) == [False, True]
    assert db.automatic_work.snapshot(chain_id).used["generation"] == 1


def test_mixed_lineage_claim_rolls_back_entire_batch(db):
    first = chain(db)
    second = chain(db, submission="second")
    first_run = survivor(db, first)
    second_run = survivor(db, second)
    with pytest.raises(ValueError, match="scope"):
        claim(db, first, [first_run, second_run])
    assert db.automatic_work.snapshot(first).reserved["generation"] == 0
    claim(db, first, [first_run])


@pytest.mark.parametrize("transition", ["accept", "complete"])
def test_transition_failure_cannot_partially_stamp_or_consume(db, transition):
    chain_id = chain(db)
    first, second = survivor(db, chain_id), survivor(db, chain_id)
    claim(db, chain_id, [first, second])
    if transition == "complete":
        db.automatic_work.accept_wake("attempt", owner_id="owner")
    with db.transaction() as conn:
        conn.execute("""CREATE TRIGGER reject_transition BEFORE UPDATE ON automatic_wake_attempts
            BEGIN SELECT RAISE(ABORT, 'injected persistence failure'); END""")
    operation = (
        db.automatic_work.accept_wake
        if transition == "accept"
        else db.automatic_work.complete_wake
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        operation("attempt", owner_id="owner")
    snapshot = db.automatic_work.snapshot(chain_id)
    assert snapshot.used["generation"] == (1 if transition == "complete" else 0)
    assert snapshot.reserved["generation"] == (0 if transition == "complete" else 1)
    assert db.get_run(first)["wake_delivered_at"] is None
    assert db.get_run(second)["wake_delivered_at"] is None
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER reject_transition")
    assert operation("attempt", owner_id="owner")


def test_claim_cannot_use_foreign_owner_session_or_a_nonterminal_run(db):
    chain_id = chain(db)
    run_id = survivor(db, chain_id)
    claim(db, chain_id, [run_id])
    for changes in ({"owner": "foreign"}, {"session": "elsewhere"}):
        with pytest.raises(ValueError, match="conflict"):
            claim(db, chain_id, [run_id], **changes)
    with pytest.raises(ValueError, match="owner"):
        db.automatic_work.accept_wake("attempt", owner_id="foreign")
    assert db.automatic_work.abort_wake("attempt", owner_id="owner")
    live = db.create_run(
        conversation_id="conversation", agent_kind="subagent", work_chain_id=chain_id
    )
    with pytest.raises(ValueError, match="survivor"):
        claim(db, chain_id, [live], attempt="live")


def test_completed_claim_survives_reopen_without_consuming_later_result(db):
    chain_id = chain(db)
    first = survivor(db, chain_id)
    claim(db, chain_id, [first])
    db.automatic_work.accept_wake("attempt", owner_id="owner")
    later = survivor(db, chain_id)
    db.automatic_work.complete_wake("attempt", owner_id="owner")
    db.close()
    assert db.automatic_work.recover(current_owner_id="new-owner") == 0
    assert [row["id"] for row in db.undelivered_wake_runs("conversation")] == [later]
    assert db.automatic_work.snapshot(chain_id).used["generation"] == 1
