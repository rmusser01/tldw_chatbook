"""Automatic dispatch authority comes from durable attempts, including helpers."""

import asyncio

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _controller_rig,
    _drain,
    _settle,
    _survivor,
    _terminal_subagent_run,
)


@pytest.fixture
def rig(tmp_path):
    return _controller_rig(tmp_path)


async def close_rig(rig):
    chacha, _, db, _, _, _, _, controller = rig
    controller._disposed = True
    for task in tuple(controller.fleet_wake._delivery_tasks):
        task.cancel()
    await asyncio.gather(*controller.fleet_wake._delivery_tasks, return_exceptions=True)
    db.close()
    chacha.close_connection()


def result_for(rig, chain_id=None):
    _, _, db, _, session, _, _, _ = rig
    _, child = _terminal_subagent_run(db, session.id, work_chain_id=chain_id)
    return child


def queue_result(rig, run_id):
    session, controller = rig[4], rig[7]
    controller.fleet_wake.capture_loop_if_running()
    controller.fleet_wake.on_fleet_drained(
        _drain(session.id, _survivor(run_id, session_id=session.id))
    )


def attempts(db):
    with db.connection() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM automatic_wake_attempts ORDER BY created_at"
            )
        ]


@pytest.mark.asyncio
async def test_fourth_automatic_generation_pauses_without_consuming_result(rig):
    _, _, db, _, session, gateway, _, wake_controller = rig
    chain_id = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    try:
        for index in range(3):
            child = result_for(rig, chain_id)
            queue_result(rig, child)
            assert await _settle(
                lambda child=child: db.get_run(child)["wake_delivered_at"] is not None
            )
            assert len(gateway.payloads) == index + 1
        fourth = result_for(rig, chain_id)
        queue_result(rig, fourth)
        assert await _settle(
            lambda: (
                db.automatic_work.snapshot(chain_id).pause_reason == "generation_budget"
            )
        )
        assert len(gateway.payloads) == 3
        assert db.get_run(fourth)["wake_delivered_at"] is None
        assert wake_controller.fleet_wake.has_pending(session.id)
        assert wake_controller.fleet_wake._retry_timer is None
        assert [row["state"] for row in attempts(db)] == ["completed"] * 3
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
async def test_acceptance_failure_prevents_substitution_and_provider_dispatch(rig):
    _, _, db, _, session, gateway, _, controller = rig
    chain_id = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    child = result_for(rig, chain_id)
    seen = []
    substitute = controller._apply_skill_substitution

    async def observed_substitution(*args, **kwargs):
        seen.append("helper")
        return await substitute(*args, **kwargs)

    controller._apply_skill_substitution = observed_substitution
    with db.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER deny_accept BEFORE UPDATE OF state ON automatic_wake_attempts WHEN NEW.state='accepted' BEGIN SELECT RAISE(ABORT, 'accept refused'); END"
        )
    try:
        queue_result(rig, child)
        assert await _settle(lambda: attempts(db) or gateway.payloads)
        await asyncio.sleep(0.1)
        assert seen == []
        assert gateway.payloads == []
        assert db.get_run(child)["wake_delivered_at"] is None
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
async def test_substitution_observes_committed_acceptance_before_any_helper(rig):
    _, _, db, _, session, _, _, controller = rig
    chain_id = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    child = result_for(rig, chain_id)
    observed = []
    substitute = controller._apply_skill_substitution

    async def observed_substitution(*args, **kwargs):
        observed.append([row["state"] for row in attempts(db)])
        return await substitute(*args, **kwargs)

    controller._apply_skill_substitution = observed_substitution
    try:
        queue_result(rig, child)
        assert await _settle(
            lambda child=child: db.get_run(child)["wake_delivered_at"] is not None
        )
        assert observed == [["accepted"]]
        assert db.automatic_work.snapshot(chain_id).used["generation"] == 1
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
async def test_completion_write_failure_retains_claim_and_never_replays(rig):
    _, _, db, _, session, gateway, _, controller = rig
    chain_id = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    child = result_for(rig, chain_id)
    with db.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER deny_complete BEFORE UPDATE OF state ON automatic_wake_attempts WHEN NEW.state='completed' BEGIN SELECT RAISE(ABORT, 'completion refused'); END"
        )
    try:
        queue_result(rig, child)
        assert await _settle(lambda: gateway.payloads)
        assert await _settle(lambda: not controller.fleet_wake._delivery_tasks)
        queue_result(rig, child)
        controller.fleet_wake.retry_soon()
        await asyncio.sleep(1.2)
        assert len(gateway.payloads) == 1
        assert db.get_run(child)["wake_delivered_at"] is None
        assert attempts(db)[0]["state"] == "accepted"
        assert db.automatic_work.snapshot(chain_id).status == "review_required"
        assert controller.fleet_wake.has_pending(session.id)
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
async def test_readiness_refusal_refunds_only_unaccepted_attempt(rig):
    _, _, db, _, session, gateway, _, controller = rig
    chain_id = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    child = result_for(rig, chain_id)
    gateway.ready = False
    try:
        queue_result(rig, child)
        assert await _settle(
            lambda: attempts(db) and attempts(db)[0]["state"] == "aborted"
        )
        snapshot = db.automatic_work.snapshot(chain_id)
        assert snapshot.used["generation"] == snapshot.reserved["generation"] == 0
        gateway.ready = True
        controller.fleet_wake.retry_soon()
        assert await _settle(
            lambda child=child: db.get_run(child)["wake_delivered_at"] is not None
        )
        assert len(gateway.payloads) == 1
        assert db.automatic_work.snapshot(chain_id).used["generation"] == 1
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
async def test_legacy_results_remain_saved_without_fresh_automatic_allowance(rig):
    _, _, db, _, session, gateway, _, controller = rig
    child = result_for(rig)
    try:
        queue_result(rig, child)
        await asyncio.sleep(0.4)
        assert gateway.payloads == []
        assert controller.fleet_wake.has_pending(session.id)
        assert db.get_run(child)["wake_delivered_at"] is None
        assert attempts(db) == []
    finally:
        await close_rig(rig)
