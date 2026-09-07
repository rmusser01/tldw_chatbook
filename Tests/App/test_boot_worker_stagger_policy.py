"""Boot worker stagger/priority policy (TASK-22215).

TASK-22222's census pins WHICH workers may start during boot; this pins WHEN
and HOW MANY AT ONCE. The properties under test:

* the staggered fleet starts in the declared order, prefetches (which a
  surface can otherwise pay for inline, on the event loop) ahead of the
  resumable FTS backfills that nothing waits on;
* at most ``MAX_CONCURRENT_STAGGERED_BOOT_WORKERS`` run at a time, and each
  completion admits the next -- including the completion of a worker that
  never started (so a failing starter cannot strand the queue);
* the FTS backfills no longer ride ``on_mount`` ahead of first paint: on a
  real mounted app every staggered body observes ``_ui_ready`` already True;
* a quit inside the staggered window closes the gate rather than starting
  more work, and nothing pending is lost (each member is either re-run by the
  surface that gates on it or resumes from a frontier in its own database).

Each policy row is also cross-checked against the census allowlist, so a
worker cannot be staggered under one identity and censused under another.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from Tests.Performance.test_boot_worker_census import ALLOWED_BOOT_WORKERS
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Utils.boot_worker_policy import (
    BOOT_WORKER_KEY_BY_IDENTITY,
    BOOT_WORKER_POLICY,
    MAX_CONCURRENT_STAGGERED_BOOT_WORKERS,
    STAGGERED_BOOT_WORKER_KEYS,
    BootWorkerTier,
    StaggeredBootWorkerGate,
)


# --------------------------------------------------------------------------
# The policy itself
# --------------------------------------------------------------------------


def test_every_policy_row_is_on_the_boot_worker_census_allowlist():
    """A staggered worker must be the same worker the census pins.

    Guards the drift that would make both files individually green and
    jointly meaningless: a rename on one side only.
    """
    unknown = {
        (spec.name, spec.group)
        for spec in BOOT_WORKER_POLICY
        if (spec.name, spec.group) not in ALLOWED_BOOT_WORKERS
    }
    assert not unknown, (
        f"boot worker policy rows missing from the TASK-22222 census "
        f"allowlist: {sorted(unknown)}"
    )


def test_staggered_order_runs_prefetches_before_the_resumable_backfills():
    """Order is the whole policy -- pin it literally.

    The two actor-pack workers are prefetches for surfaces that gate on the
    same once-lock, so a late start means the surface pays the work on the
    event loop. The two FTS backfills have nothing waiting on them and the
    ChaChaNotes one can run for tens of seconds on a first post-upgrade boot,
    so it must never hold a slot ahead of a prefetch.
    """
    assert STAGGERED_BOOT_WORKER_KEYS == (
        "actor_pack_recovery",
        "actor_pack_staging_sweep",
        "chachanotes_fts_backfill",
        "subscriptions_fts_backfill",
    )


def test_the_two_fts_backfills_are_staggered_not_immediate():
    """Neither whole-table re-tokenization may ride the pre-first-paint tier."""
    tiers = {
        spec.key: spec.tier
        for spec in BOOT_WORKER_POLICY
        if spec.key.endswith("_fts_backfill")
    }
    assert tiers == {
        "chachanotes_fts_backfill": BootWorkerTier.STAGGERED,
        "subscriptions_fts_backfill": BootWorkerTier.STAGGERED,
    }


def test_the_concurrency_cap_is_below_the_staggered_fleet_size():
    """A cap that admits everything is not a cap."""
    assert 1 <= MAX_CONCURRENT_STAGGERED_BOOT_WORKERS < len(STAGGERED_BOOT_WORKER_KEYS)


def test_every_policy_row_records_what_it_unblocks():
    """The starvation check a reorder has to pass is a required field."""
    assert all(spec.unblocks.strip() for spec in BOOT_WORKER_POLICY)


# --------------------------------------------------------------------------
# The admission gate
# --------------------------------------------------------------------------


def test_gate_admits_only_up_to_the_cap():
    gate = StaggeredBootWorkerGate(("a", "b", "c", "d"), limit=2)
    assert gate.admit() == ("a", "b")
    assert gate.admit() == ()
    assert gate.in_flight == ("a", "b")
    assert gate.pending == ("c", "d")


def test_completion_admits_the_next_worker_in_policy_order():
    gate = StaggeredBootWorkerGate(("a", "b", "c", "d"), limit=2)
    gate.admit()
    assert gate.complete("a") is True
    assert gate.admit() == ("c",)
    assert gate.complete("b") is True
    assert gate.admit() == ("d",)
    assert gate.complete("c") and gate.complete("d")
    assert gate.is_drained


def test_completing_an_unknown_or_repeated_key_releases_nothing():
    """The same terminal transition reaches the gate from two call sites."""
    gate = StaggeredBootWorkerGate(("a", "b"), limit=1)
    gate.admit()
    assert gate.complete("a") is True
    assert gate.complete("a") is False
    assert gate.complete("nonsense") is False
    assert gate.admit() == ("b",)


def test_closing_the_gate_drops_pending_and_never_admits_again():
    gate = StaggeredBootWorkerGate(("a", "b", "c"), limit=1)
    gate.admit()
    assert gate.close() == ("b", "c")
    assert gate.admit() == ()
    assert gate.pending == ()
    assert gate.is_closed


def test_gate_rejects_a_zero_limit_and_duplicate_keys():
    with pytest.raises(ValueError):
        StaggeredBootWorkerGate(("a",), limit=0)
    with pytest.raises(ValueError):
        StaggeredBootWorkerGate(("a", "a"), limit=2)


# --------------------------------------------------------------------------
# The app wiring
# --------------------------------------------------------------------------


def test_app_can_start_every_staggered_key():
    """Every policy key resolves to a real starter on the app."""
    app = _build_test_app()
    starters = app.boot_worker_starters()
    assert set(starters) == set(STAGGERED_BOOT_WORKER_KEYS)
    assert all(callable(starter) for starter in starters.values())


def _fake_worker(spec_key: str) -> SimpleNamespace:
    """A stand-in for the Textual worker a starter would return."""
    from tldw_chatbook.Utils.boot_worker_policy import BOOT_WORKER_POLICY

    spec = next(row for row in BOOT_WORKER_POLICY if row.key == spec_key)
    return SimpleNamespace(
        name=spec.name, group=spec.group, is_finished=False, is_cancelled=False
    )


def test_deferred_startup_starts_the_cap_then_advances_on_completion():
    """The app starts up to the cap, then one more per terminal transition."""
    cap = MAX_CONCURRENT_STAGGERED_BOOT_WORKERS
    app = _build_test_app()
    started: list[str] = []
    workers: dict[str, SimpleNamespace] = {}

    def record(key: str):
        started.append(key)
        workers[key] = _fake_worker(key)
        return workers[key]

    app._start_boot_worker = record

    app._start_staggered_boot_workers()
    assert started == list(STAGGERED_BOOT_WORKER_KEYS[:cap])

    # Completing the oldest in-flight worker admits exactly one more, in
    # policy order, until the whole fleet has run.
    for index in range(len(STAGGERED_BOOT_WORKER_KEYS) - cap):
        app._release_boot_worker_slot(workers[STAGGERED_BOOT_WORKER_KEYS[index]])
        assert started == list(STAGGERED_BOOT_WORKER_KEYS[: cap + index + 1])

    assert started == list(STAGGERED_BOOT_WORKER_KEYS)


def test_a_starter_that_starts_nothing_still_advances_the_queue():
    """A skipped or failing start must not hold its slot forever."""
    app = _build_test_app()
    started: list[str] = []

    def record(key: str):
        started.append(key)
        if key == STAGGERED_BOOT_WORKER_KEYS[0]:
            raise RuntimeError("starter blew up")
        return None  # every other starter declines to start anything

    app._start_boot_worker = record

    app._start_staggered_boot_workers()
    assert started == list(STAGGERED_BOOT_WORKER_KEYS)


def test_shutdown_closes_the_gate_instead_of_starting_more_work():
    """A quit inside the staggered window starts nothing further."""
    app = _build_test_app()
    started: list[str] = []
    app._start_boot_worker = lambda key: (started.append(key), _fake_worker(key))[1]

    cap = MAX_CONCURRENT_STAGGERED_BOOT_WORKERS
    app._start_staggered_boot_workers()
    assert started == list(STAGGERED_BOOT_WORKER_KEYS[:cap])

    app._shutting_down = True
    app._release_boot_worker_slot(_fake_worker(STAGGERED_BOOT_WORKER_KEYS[0]))

    assert started == list(STAGGERED_BOOT_WORKER_KEYS[:cap])
    assert app._boot_worker_gate.is_closed
    assert app._boot_worker_gate.pending == ()


def test_the_worker_identity_map_covers_the_whole_policy():
    """`Worker.StateChanged` is mapped back onto the row that owns the slot."""
    assert set(BOOT_WORKER_KEY_BY_IDENTITY.values()) == {
        spec.key for spec in BOOT_WORKER_POLICY
    }


@pytest.mark.asyncio
async def test_every_staggered_body_runs_after_the_ui_is_ready():
    """On a real mounted app, no staggered body runs before first paint.

    Red on the pre-fix tree: both FTS backfills were started from
    ``on_mount``, so they observed ``_ui_ready`` False. Also the
    anti-starvation pin -- all four must actually run, not merely be queued.
    """
    app = _build_test_app()
    observed: dict[str, bool] = {}
    done = threading.Event()

    def recorder(key: str):
        def body() -> None:
            observed[key] = bool(getattr(app, "_ui_ready", False))
            if len(observed) == len(STAGGERED_BOOT_WORKER_KEYS):
                done.set()

        return body

    app.ensure_actor_pack_recovery = recorder("actor_pack_recovery")
    app.ensure_actor_pack_staging_sweep = recorder("actor_pack_staging_sweep")
    app._backfill_chachanotes_messages_fts = recorder("chachanotes_fts_backfill")
    app._backfill_subscription_items_fts = recorder("subscriptions_fts_backfill")

    async with app.run_test() as pilot:
        for _ in range(120):
            if done.is_set():
                break
            await pilot.pause(0.05)

    assert set(observed) == set(STAGGERED_BOOT_WORKER_KEYS), (
        f"staggered boot workers that never ran: "
        f"{sorted(set(STAGGERED_BOOT_WORKER_KEYS) - set(observed))}"
    )
    early = sorted(key for key, ready in observed.items() if not ready)
    assert not early, f"staggered boot workers that ran before _ui_ready: {early}"
