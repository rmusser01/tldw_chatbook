"""Application and runtime limits, fairness and retained lifetime tickets."""

import asyncio

import pytest

from tldw_chatbook.Agents.hooks_v2.budgets import BudgetExceeded, HookBudgetOwner


@pytest.mark.asyncio
async def test_execution_limits_and_suspension_keep_lifetime():
    owner = HookBudgetOwner()
    tickets = [owner.reserve(r, False) for r in (["a"] * 5 + ["b"] * 5)]
    waits = [asyncio.create_task(t.acquire()) for t in tickets]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert owner.snapshot()["execution"] == 8
    assert owner.snapshot("a")["execution"] == 4
    assert owner.snapshot()["tickets"] == 10
    tickets[0].suspend()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert tickets[4].active
    assert owner.snapshot()["tickets"] == 10
    for t in tickets:
        t.release()
    await asyncio.gather(*waits, return_exceptions=True)
    assert owner.snapshot()["tickets"] == 0


@pytest.mark.asyncio
async def test_lifetime_and_observation_limits_are_separate():
    owner = HookBudgetOwner()
    tickets = [owner.reserve(str(i // 16), False) for i in range(64)]
    with pytest.raises(BudgetExceeded):
        owner.reserve("fresh", False)
    with pytest.raises(BudgetExceeded):
        owner.reserve("0", False)
    observers = [owner.reserve(str(i // 64), True) for i in range(128)]
    with pytest.raises(BudgetExceeded):
        owner.reserve("fresh", True)
    assert owner.snapshot() == {
        "execution": 0,
        "tickets": 64,
        "observations": 128,
        "workers": 0,
    }
    for t in tickets + observers:
        t.release()
    assert all(n == 0 for n in owner.snapshot().values())


@pytest.mark.asyncio
async def test_round_robin_runtimes_and_fifo_resume():
    owner = HookBudgetOwner()
    holders = [owner.reserve(str(i // 4), False) for i in range(8)]
    await asyncio.gather(*(t.acquire() for t in holders))
    order = []
    queued = [owner.reserve(r, False) for r in ("a", "a", "b", "b")]

    async def wait(t, label):
        await t.acquire()
        order.append(label)

    tasks = [
        asyncio.create_task(wait(t, s))
        for t, s in zip(queued, ("a1", "a2", "b1", "b2"))
    ]
    await asyncio.sleep(0)
    for t in holders[:4]:
        t.release()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert order == ["a1", "b1", "a2", "b2"]
    for t in holders + queued:
        t.release()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_observation_workers_are_separate_and_cancelled_queue_releases():
    owner = HookBudgetOwner()
    obs = [owner.reserve(r, True) for r in ["a"] * 5 + ["b"] * 5]
    tasks = [asyncio.create_task(t.acquire()) for t in obs]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert owner.snapshot()["workers"] == 8
    assert owner.snapshot()["observations"] == 2
    control = owner.reserve("a", False)
    await control.acquire()
    assert owner.snapshot()["execution"] == 1
    tasks[4].cancel()
    await asyncio.gather(tasks[4], return_exceptions=True)
    assert owner.snapshot()["observations"] == 1
    for t in obs + [control]:
        t.release()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("observation", [False, True])
async def test_rejected_runtime_ids_do_not_grow_retained_budget_inventory(observation):
    owner = HookBudgetOwner()
    runtime_cap, app_cap = (64, 128) if observation else (16, 64)
    tickets = [
        owner.reserve(str(i // runtime_cap), observation) for i in range(app_cap)
    ]
    active_ids = set(owner._counts)
    try:
        for i in range(256):
            with pytest.raises(BudgetExceeded, match="delivery_capacity"):
                owner.reserve(f"rejected-{i}", observation)
        assert set(owner._counts) == active_ids
    finally:
        for ticket in tickets:
            ticket.release()
    assert not owner._counts
    fresh = owner.reserve("fresh", observation)
    fresh.release()
    assert not owner._counts
