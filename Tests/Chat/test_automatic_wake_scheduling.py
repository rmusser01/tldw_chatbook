"""Wake scheduling preserves manual capacity and finite coalescing windows."""

import asyncio

import pytest

from Tests.Chat.test_automatic_wake_budget import close_rig, queue_result, result_for
from Tests.Chat.test_console_fleet_wake import _controller_rig, _settle


@pytest.fixture
def rig(tmp_path):
    return _controller_rig(tmp_path)


def additional_conversation(rig):
    items = list(rig)
    items[4] = rig[3].create_session(title="Another conversation")
    return tuple(items)


def pending_result(rig, name):
    chain = rig[2].automatic_work.create_chain(rig[4].id, root_submission_id=name)
    child = result_for(rig, chain)
    queue_result(rig, child)
    return child


@pytest.mark.asyncio
async def test_two_conversations_wake_while_one_primary_slot_remains_manual(rig):
    _, _, _, _, first, gateway, _, controller = rig
    second_rig = additional_conversation(rig)
    manual_rig = additional_conversation(rig)
    gate = asyncio.Event()
    gateway.stream_gate = gate
    manual = None
    try:
        pending_result(rig, "first")
        pending_result(second_rig, "second")
        assert await _settle(lambda: len(gateway.payloads) == 2)
        assert set(controller.fleet_wake.delivering_session_ids()) == {
            first.id,
            second_rig[4].id,
        }
        assert controller.send_refusal_copy(manual_rig[4].id) is None
        manual = asyncio.create_task(
            controller.submit_draft("Manual request", session_id=manual_rig[4].id)
        )
        assert await _settle(lambda: len(gateway.payloads) == 3)
    finally:
        gate.set()
        if manual is not None:
            await manual
        await close_rig(rig)


@pytest.mark.asyncio
async def test_one_primary_slot_disables_automatic_admission_until_limit_increases(
    rig, monkeypatch
):
    controller, gateway = rig[7], rig[5]
    limit = [1]
    monkeypatch.setattr(
        type(controller), "max_parallel_runs", property(lambda _: limit[0])
    )
    try:
        child = pending_result(rig, "first")
        await asyncio.sleep(0.35)
        assert gateway.payloads == []
        assert controller.fleet_wake.has_pending(rig[4].id)
        limit[0] = 3
        controller.fleet_wake.retry_soon()
        assert await _settle(
            lambda: rig[2].get_run(child)["wake_delivered_at"] is not None
        )
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
async def test_new_results_do_not_extend_the_fixed_coalescing_window(rig):
    db, session, gateway = rig[2], rig[4], rig[5]
    chain = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    try:
        queue_result(rig, result_for(rig, chain))
        await asyncio.sleep(0.18)
        queue_result(rig, result_for(rig, chain))
        assert await _settle(lambda: gateway.payloads, seconds=0.16)
        assert len(gateway.payloads) == 1
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
async def test_accepted_cancellation_keeps_generation_and_blocks_replay(rig):
    controller, db, session = rig[7], rig[2], rig[4]
    entered = asyncio.Event()
    gate = asyncio.Event()
    substitute = controller._apply_skill_substitution

    async def held_substitution(*args, **kwargs):
        entered.set()
        await gate.wait()
        return await substitute(*args, **kwargs)

    controller._apply_skill_substitution = held_substitution
    try:
        child = pending_result(rig, "first")
        assert await _settle(entered.is_set)
        for task in tuple(controller.fleet_wake._delivery_tasks):
            task.cancel()
        assert await _settle(lambda: not controller.fleet_wake._delivery_tasks)
        chain = db.get_run(child)["work_chain_id"]
        assert db.automatic_work.snapshot(chain).used["generation"] == 1
        assert db.automatic_work.snapshot(chain).status == "review_required"
        assert controller.fleet_wake.delivering_session_ids() == ()
        assert controller.in_flight_run_count() == 0
        assert controller.send_refusal_copy(session.id) is None
        assert controller.fleet_wake.has_pending(session.id)
    finally:
        gate.set()
        await close_rig(rig)


@pytest.mark.asyncio
async def test_waiting_conversation_precedes_a_busy_producers_next_wake(rig):
    second = additional_conversation(rig)
    third = additional_conversation(rig)
    gateway = rig[5]
    gate = asyncio.Event()
    gateway.stream_gate = gate
    try:
        first_child = pending_result(rig, "first")
        pending_result(second, "second")
        pending_result(third, "third")
        assert await _settle(lambda: len(gateway.payloads) == 2)
        chain = rig[2].get_run(first_child)["work_chain_id"]
        hot_child = result_for(rig, chain)
        queue_result(rig, hot_child)
        gate.set()
        assert await _settle(lambda: len(gateway.payloads) == 4)
        assert hot_child not in gateway.payloads[2][-1]["content"]
        assert hot_child in gateway.payloads[3][-1]["content"]
    finally:
        gate.set()
        await close_rig(rig)
