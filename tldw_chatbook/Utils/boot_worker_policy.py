"""Explicit start policy for the boot-time background worker fleet (TASK-22215).

The 2026-08-24 holistic perf review counted the boot-time concurrent worker
fleet growing 4 -> 7. Each member was individually justified; the aggregate is
what the user feels, because under the GIL those threads plus the Textual
message pump share one interpreter during the first seconds after mount --
worst on the first post-upgrade boot, when the ChaChaNotes FTS backfill runs to
completion alongside the screen pre-importer.

``Tests/Performance/test_boot_worker_census.py`` (TASK-22222) pins *which*
workers may start during boot. This module is the other half: *when* and *how
many at once*. It is deliberately pure -- specs, an order, a cap, and an
admission gate with no Textual, no app and no I/O -- so the policy can be
asserted directly and ``app.py`` is left holding only the wiring.

What the policy actually decides
--------------------------------

Two tiers:

``IMMEDIATE``
    Started during ``on_mount``, before first paint, because something a user
    can reach at once is degraded until they finish (or because they are
    long-lived loops that spend their life awaiting and cost the interpreter
    nothing).

``STAGGERED``
    Started after ``_ui_ready``, in the declared order, at most
    :data:`MAX_CONCURRENT_STAGGERED_BOOT_WORKERS` at a time; each completion
    admits the next. Every member here is either a *prefetch* for a surface
    that re-runs the same work itself behind a once-gate, or a resumable
    background rebuild that no surface waits on. That is the property that
    makes delaying one safe, and it is recorded per row in ``unblocks``.

The honest lever is ORDER and CONCURRENCY, not more sleeping: a sleep cannot
subdivide one ``import_module`` or one ``BEGIN IMMEDIATE`` chunk (TASK-21113,
TASK-22214). Pacing *within* a worker is that worker's own business -- see
``DB/chachanotes_fts_backfill.py`` (TASK-22200) and the screen pre-importer's
proportional yield (``app.py``'s ``SCREEN_PREIMPORT_*``).

Deliberately NOT gated here
---------------------------

* The screen pre-importer. It is a daemon thread, not a Textual worker; it
  already paces itself proportionally to each import's cost (TASK-22214), and
  it protects the FIRST click to Library/Settings -- queueing it behind a
  minutes-long FTS backfill would trade away the exact thing it exists for.
* ``on_mount``'s coroutine workers (the scheduler loop and the research
  startup reconciles). They are await-shaped: their thread time is bounded
  ``asyncio.to_thread`` hops, and the scheduler loop is time-sensitive.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

__all__ = [
    "BootWorkerSpec",
    "BootWorkerTier",
    "BOOT_WORKER_POLICY",
    "IMMEDIATE_BOOT_WORKERS",
    "STAGGERED_BOOT_WORKERS",
    "STAGGERED_BOOT_WORKER_KEYS",
    "MAX_CONCURRENT_STAGGERED_BOOT_WORKERS",
    "BOOT_WORKER_KEY_BY_IDENTITY",
    "StaggeredBootWorkerGate",
]


class BootWorkerTier(str, Enum):
    """When a boot worker is allowed to start."""

    #: Started in ``on_mount``, before first paint.
    IMMEDIATE = "immediate"
    #: Started after ``_ui_ready``, in policy order, under the concurrency cap.
    STAGGERED = "staggered"


@dataclass(frozen=True)
class BootWorkerSpec:
    """One boot worker's start policy.

    Attributes:
        key: Stable policy identifier, used by ``app.py``'s starter table and
            by the admission gate. Independent of the worker's Textual name so
            a rename cannot silently detach a row from its policy.
        name: The Textual worker ``name`` -- must match the (name, group) pair
            censused by ``Tests/Performance/test_boot_worker_census.py``.
        group: The Textual worker ``group``.
        tier: :class:`BootWorkerTier`.
        unblocks: The user-visible surface that is degraded until this worker
            finishes, or the explicit statement that nothing waits on it. This
            is the starvation check a reorder has to pass, so it is a required
            field rather than a comment.
    """

    key: str
    name: str
    group: str
    tier: BootWorkerTier
    unblocks: str

    def __post_init__(self) -> None:
        if not self.key or not self.name or not self.group:
            raise ValueError("boot worker specs need a key, a name and a group")
        if not self.unblocks:
            raise ValueError(
                f"boot worker {self.key!r} must record what it unblocks -- a "
                "reorder cannot be reviewed without it"
            )


#: The fleet, in start order. Staggered rows are admitted top-down.
#:
#: Priority reasoning, recorded because it is the reviewable part:
#:
#: 1. ``actor_pack_recovery`` and ``actor_pack_staging_sweep`` are prefetches
#:    for surfaces that gate on the SAME once-lock (Personas' first library
#:    read / ``create_persona``; the importer's ``inspect_archive``). If the
#:    worker has not run, the surface runs the work itself -- on the event
#:    loop. They are short one-shots, so they go first: finishing them frees
#:    the slots quickly and removes the inline-cost risk.
#: 2. The two FTS backfills go last, and behind each other (the cap is 1, so
#:    the order IS the schedule). Nothing waits on either -- search fills in
#:    progressively and both are resumable across kills -- and on a first
#:    post-upgrade boot the ChaChaNotes one can run for tens of seconds, so it
#:    is exactly the member that must not sit in front of a worker a surface
#:    can block on.
BOOT_WORKER_POLICY: tuple[BootWorkerSpec, ...] = (
    # -- IMMEDIATE: before first paint --
    BootWorkerSpec(
        key="scheduler_loop",
        name="run",
        group="scheduling",
        tier=BootWorkerTier.IMMEDIATE,
        unblocks=(
            "Reminders and scheduled watchlist checks that are already overdue "
            "at launch. A coroutine worker that spends its life awaiting, so "
            "starting it early costs the interpreter nothing; it must also stay "
            "on the app's one event loop (the watchlists in-flight guard is "
            "lock-free on that invariant)."
        ),
    ),
    BootWorkerSpec(
        key="ingest_restore",
        name="restore_ingest_jobs",
        group="ingest_restore",
        tier=BootWorkerTier.IMMEDIATE,
        unblocks=(
            "The Library ingest job history. Until it lands the in-memory "
            "registry is empty, so a user opening Library ingest sees no "
            "history at all -- a wrong answer, not a slow one. Bounded work "
            "against one small store."
        ),
    ),
    # -- STAGGERED: after `_ui_ready`, capped, in this order --
    BootWorkerSpec(
        key="actor_pack_recovery",
        name="deferred_actor_pack_recovery",
        group="actor_pack_recovery",
        tier=BootWorkerTier.STAGGERED,
        unblocks=(
            "Nothing hard: the Personas screen's first library read and every "
            "create_persona gate on the coordinator's own once-lock. This "
            "worker is the prefetch that keeps that gate from running SQLite "
            "recovery on the event loop -- so it is first among the staggered."
        ),
    ),
    BootWorkerSpec(
        key="actor_pack_staging_sweep",
        name="deferred_actor_pack_staging_sweep",
        group="actor_pack_staging_sweep",
        tier=BootWorkerTier.STAGGERED,
        unblocks=(
            "Nothing hard: ActorPackImportService.inspect_archive gates on the "
            "same once-lock, so a first import sweeps for itself. Prefetch of a "
            "filesystem walk, second for the same reason as recovery."
        ),
    ),
    BootWorkerSpec(
        key="chachanotes_fts_backfill",
        name="_backfill_chachanotes_messages_fts",
        group="chachanotes-fts-backfill",
        tier=BootWorkerTier.STAGGERED,
        unblocks=(
            "Nothing. Message-content search over pre-upgrade history fills in "
            "progressively and the frontier lives in the database "
            "(messages_fts_docsize), so a run that never starts, is cut short "
            "or is cancelled simply resumes next boot. The longest-running "
            "member of the fleet on a first post-upgrade boot."
        ),
    ),
    BootWorkerSpec(
        key="subscriptions_fts_backfill",
        name="_backfill_subscription_items_fts",
        group="subscriptions-fts-backfill",
        tier=BootWorkerTier.STAGGERED,
        unblocks=(
            "Nothing. Search over subscription items scraped before the FTS "
            "index existed; resumable exactly like the ChaChaNotes backfill "
            "(subscription_items_fts_docsize). Last, so it runs behind that "
            "one rather than beside it -- two whole-table re-tokenizations at "
            "once is the shape this policy exists to prevent."
        ),
    ),
)

#: How many staggered boot workers may run at once. One: the fleet is strictly
#: serial in policy order, which is the whole point of the finding -- the
#: aggregate, not any individual member, is what the user feels.
#:
#: Why not two (the first cut). With a cap of two and this order, the two
#: prefetches finish quickly and then BOTH FTS backfills are admitted -- so
#: the worst shape in the fleet, two whole-table re-tokenizations running at
#: once, is exactly what the cap would still permit. Serializing costs the
#: prefetches nothing measurable (they are ahead of the backfills, and their
#: own durations are milliseconds on a healthy profile), and nothing else in
#: the tier has a waiter.
#:
#: What still runs alongside: the screen pre-importer's daemon thread, the
#: immediate tier, and the event loop itself. This cap bounds the staggered
#: fleet, not the process.
MAX_CONCURRENT_STAGGERED_BOOT_WORKERS = 1

IMMEDIATE_BOOT_WORKERS: tuple[BootWorkerSpec, ...] = tuple(
    spec for spec in BOOT_WORKER_POLICY if spec.tier is BootWorkerTier.IMMEDIATE
)

STAGGERED_BOOT_WORKERS: tuple[BootWorkerSpec, ...] = tuple(
    spec for spec in BOOT_WORKER_POLICY if spec.tier is BootWorkerTier.STAGGERED
)

STAGGERED_BOOT_WORKER_KEYS: tuple[str, ...] = tuple(
    spec.key for spec in STAGGERED_BOOT_WORKERS
)

#: (worker name, worker group) -> policy key, for mapping a Textual
#: ``Worker.StateChanged`` back onto the row that owns its slot.
BOOT_WORKER_KEY_BY_IDENTITY: dict[tuple[str, str], str] = {
    (spec.name, spec.group): spec.key for spec in BOOT_WORKER_POLICY
}

if len({spec.key for spec in BOOT_WORKER_POLICY}) != len(BOOT_WORKER_POLICY):
    raise RuntimeError("duplicate key in BOOT_WORKER_POLICY")
if len(BOOT_WORKER_KEY_BY_IDENTITY) != len(BOOT_WORKER_POLICY):
    raise RuntimeError("duplicate (name, group) in BOOT_WORKER_POLICY")


class StaggeredBootWorkerGate:
    """Admits staggered boot workers in policy order, ``limit`` at a time.

    Pure bookkeeping: it never starts anything. The caller starts whatever
    :meth:`admit` hands back and calls :meth:`complete` when that worker
    reaches a terminal state (or failed to start at all), which frees the slot
    and lets the next admission through.

    Not thread-safe by design: every call site is the Textual event loop
    (deferred-startup scheduling, the ``Worker.StateChanged`` hook, and the
    reconcile timer), and adding a lock would invite off-loop use.
    """

    def __init__(self, keys: Iterable[str], limit: int) -> None:
        """Build a gate over ``keys`` in order.

        Args:
            keys: Policy keys, in admission order.
            limit: Maximum simultaneously in-flight workers (>= 1).

        Raises:
            ValueError: If ``limit`` < 1 or ``keys`` contains duplicates.
        """
        if limit < 1:
            raise ValueError("a boot worker gate needs room for at least one worker")
        ordered = list(keys)
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"duplicate boot worker key in {ordered!r}")
        self._pending: deque[str] = deque(ordered)
        self._in_flight: list[str] = []
        self._limit = limit
        self._closed = False

    @property
    def limit(self) -> int:
        """Maximum simultaneously in-flight workers."""
        return self._limit

    @property
    def pending(self) -> tuple[str, ...]:
        """Keys not yet admitted, in admission order."""
        return tuple(self._pending)

    @property
    def in_flight(self) -> tuple[str, ...]:
        """Keys admitted and not yet completed."""
        return tuple(self._in_flight)

    @property
    def is_closed(self) -> bool:
        """Whether the gate has been closed (shutdown); nothing more admits."""
        return self._closed

    @property
    def is_drained(self) -> bool:
        """Whether nothing is pending and nothing is in flight."""
        return not self._pending and not self._in_flight

    def admit(self) -> tuple[str, ...]:
        """Take as many keys as the cap allows, marking them in flight.

        Returns:
            The keys the caller must now start, in policy order. Empty when
            the gate is closed, drained, or already at its cap.
        """
        if self._closed:
            return ()
        admitted: list[str] = []
        while self._pending and len(self._in_flight) < self._limit:
            key = self._pending.popleft()
            self._in_flight.append(key)
            admitted.append(key)
        return tuple(admitted)

    def complete(self, key: str) -> bool:
        """Release ``key``'s slot.

        Args:
            key: A previously admitted key. Unknown or already-completed keys
                are ignored -- the same terminal transition can reach the gate
                from both the worker hook and the reconcile timer.

        Returns:
            True if a slot was actually released.
        """
        if key not in self._in_flight:
            return False
        self._in_flight.remove(key)
        return True

    def close(self) -> tuple[str, ...]:
        """Stop admitting (shutdown) and drop whatever never started.

        Returns:
            The keys that were still pending, in order -- for the caller's log.
            Their work is not lost: every staggered member is either re-run by
            the surface that gates on it or resumes from a frontier in its own
            database on the next boot.
        """
        self._closed = True
        dropped = tuple(self._pending)
        self._pending.clear()
        return dropped


def describe_policy(specs: Sequence[BootWorkerSpec] = BOOT_WORKER_POLICY) -> str:
    """Render the policy as a readable table (diagnostics, docs, review).

    Args:
        specs: Rows to render; defaults to the whole policy.

    Returns:
        One line per row: tier, key, (name, group).
    """
    return "\n".join(
        f"{spec.tier.value:<10} {spec.key:<28} ({spec.name}, {spec.group})"
        for spec in specs
    )
