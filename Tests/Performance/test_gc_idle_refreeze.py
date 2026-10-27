"""The idle heap re-freeze and its leak bound (TASK-33620.15.3, ADR-198 amendment).

Live profiling (2026-10-10, Anthropic haiku, 160x45) found ~70 ms of main-thread
generation-2 collection in every Console send window: only the boot heap was
frozen, so each pass rescanned everything the Console built after boot. The
fix re-freezes at idle moments, and because ``gc.freeze()`` never collects a
frozen object, it bounds what a re-freeze can leak with a compaction.

The scheduling tests use a fake clock and a fake ``set_timer`` owner. The leak
tests use the real collector and real Textual widgets, whose parent/child and
style cycles are exactly what a re-freeze could strand.
"""

from __future__ import annotations

import gc
import logging
import threading
import weakref
from collections.abc import Callable

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Utils import text_selection_crash_guard, ui_responsiveness
from tldw_chatbook.Utils.text_selection_crash_guard import TextualAppGuards
from tldw_chatbook.Utils.ui_responsiveness import (
    COMPACTION_GROWTH_RATIO,
    IdleHeapRefreezer,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


class Clock:
    """A settable monotonic clock."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Owner:
    """Records ``set_timer`` calls; the test fires them by hand."""

    def __init__(self) -> None:
        self.timers: list[tuple[float, Callable[[], None]]] = []
        self.busy = False

    def set_timer(self, delay: float, callback: Callable[[], None]) -> None:
        self.timers.append((delay, callback))

    def fire_last(self) -> None:
        self.timers[-1][1]()


def owner_busy(owner: Owner) -> bool:
    return owner.busy


class Ops:
    """Fake collector operations with a permanent generation of ``frozen`` objects.

    ``freed`` of each re-freeze's ``added`` objects are freed by reference
    counting right after it, as replaced render strips are. ``uncounted``
    more are frozen without being counted, as objects other threads allocate
    between a re-freeze's count and its freeze are.
    """

    def __init__(
        self,
        frozen: int = 1_000_000,
        added: int = 10_000,
        freed: int = 0,
        uncounted: int = 0,
    ) -> None:
        self.frozen = frozen
        self.added = added
        self.freed = freed
        self.uncounted = uncounted
        self.calls: list[str] = []
        self.walks = 0

    def refreeze(self, reason: str) -> int:
        self.calls.append(f"refreeze:{reason}")
        self.frozen += self.added + self.uncounted - self.freed
        return self.added

    def compact(self, reason: str) -> int:
        self.calls.append(f"compact:{reason}")
        return self.frozen

    def freeze_count(self) -> int:
        self.walks += 1
        return self.frozen


def make(clock: Clock, ops: Ops, **kwargs) -> IdleHeapRefreezer:
    return IdleHeapRefreezer(
        clock=clock,
        enabled=lambda: True,
        refreeze=ops.refreeze,
        compact=ops.compact,
        freeze_count=ops.freeze_count,
        **kwargs,
    )


def counting_collector(monkeypatch: pytest.MonkeyPatch, calls: list[tuple]) -> None:
    """Fake ``gc.collect``; each call runs a pass on the calling thread.

    The re-freeze and the compaction freeze only after a full pass really ran
    on their own thread. A real pass announces itself through
    ``gc.callbacks`` on the thread that runs it, so the fake does the same.

    Args:
        monkeypatch: Installs the fakes.
        calls: Receives ``("collect", generation)`` per call.
    """
    def collect(generation: int = 2) -> int:
        calls.append(("collect", generation))
        info = {"generation": generation, "collected": 0, "uncollectable": 0}
        for phase in ("start", "stop"):
            for callback in list(gc.callbacks):
                callback(phase, info)
        return 0

    monkeypatch.setattr(ui_responsiveness.gc, "collect", collect)


@pytest.mark.parametrize("debug", [False, True], ids=["info", "debug"])
def test_the_idle_refreeze_collects_every_unfrozen_generation_then_freezes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, debug: bool
) -> None:
    """A full collection first (reclaims old-generation garbage), then the freeze.

    The old generation is listed only for the DEBUG line: building that list
    walks every object the re-freeze is about to freeze, so it is skipped
    when nobody reads the line.

    Args:
        monkeypatch: Records the collector calls instead of running them.
        caplog: Sets the module logger's level and captures the DEBUG line.
        debug: Whether the module logger has DEBUG enabled.
    """
    caplog.set_level(
        logging.DEBUG if debug else logging.INFO, logger=ui_responsiveness.__name__
    )
    calls: list[tuple] = []
    counting_collector(monkeypatch, calls)
    monkeypatch.setattr(
        ui_responsiveness.gc,
        "get_objects",
        lambda generation=None: calls.append(("get_objects", generation)) or [1, 2],
    )
    monkeypatch.setattr(
        ui_responsiveness.gc, "freeze", lambda: calls.append(("freeze",))
    )

    ui_responsiveness.refreeze_long_lived_heap("unit-test")
    if debug:
        assert calls == [("collect", 2), ("get_objects", 2), ("freeze",)]
        assert "gc heap re-frozen (unit-test): +2 objects" in caplog.messages
    else:
        assert calls == [("collect", 2), ("freeze",)], "walked the heap for no reader"


def test_a_compaction_unfreezes_then_collects_everything_then_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole heap, frozen set included, is collected before the re-freeze.

    Args:
        monkeypatch: Records the collector calls instead of running them.
    """
    calls: list[tuple] = []
    monkeypatch.setattr(
        ui_responsiveness.gc, "unfreeze", lambda: calls.append(("unfreeze",))
    )
    counting_collector(monkeypatch, calls)
    monkeypatch.setattr(
        ui_responsiveness.gc, "freeze", lambda: calls.append(("freeze",))
    )
    monkeypatch.setattr(ui_responsiveness.gc, "get_freeze_count", lambda: 1234)

    assert ui_responsiveness.compact_long_lived_heap("unit-test") == 1234
    assert calls == [("unfreeze",), ("collect", 2), ("freeze",)]


class Cycle:
    """The smallest object reference counting cannot free: it points at itself."""

    def __init__(self) -> None:
        self.me = self


def old_garbage() -> weakref.ref:
    """A cycle promoted into generation 2 while live, then dropped."""
    cycle = Cycle()
    ref = weakref.ref(cycle)
    gc.collect()  # it survives, so it now lives in the oldest generation
    del cycle
    assert ref() is not None, "reference counting freed a cycle"
    return ref


def test_a_refreeze_reclaims_old_generation_cycles_instead_of_freezing_them() -> None:
    """The full collection before each re-freeze is what keeps old garbage out.

    Negative control: the boot points' young collection plus freeze strands
    a cycle already in generation 2 (only the unfreeze reclaims it); the
    re-freeze's full collection frees it before anything is frozen.
    """
    gc.unfreeze()
    gc.collect()
    try:
        stranded = old_garbage()
        ui_responsiveness.freeze_long_lived_heap("unit-test")
        gc.collect()
        assert stranded() is not None, "a young collection reclaimed old garbage"
        gc.unfreeze()
        gc.collect()
        assert stranded() is None

        reclaimed = old_garbage()
        ui_responsiveness.refreeze_long_lived_heap("unit-test")
        assert reclaimed() is None, "the re-freeze froze old-generation garbage"
    finally:
        gc.unfreeze()


def during_another_collection(action: Callable[[], object]) -> list[object]:
    """Run ``action`` while a collection is in progress, as a worker's pass can be.

    CPython skips ``gc.collect()`` while another pass is in progress: it
    returns 0 at once, having collected nothing. A worker thread's pass can
    hand the GIL to the UI loop from its finalizers or ``gc.callbacks``. A
    start hook is the deterministic stand-in: the outer pass has begun and
    collected nothing yet.

    Args:
        action: What the UI loop runs at that moment.

    Returns:
        What ``action`` returned, or the exception it raised.
    """
    results: list[object] = []

    def during(phase: str, info: dict) -> None:
        if phase == "start" and not results:
            try:
                results.append(action())
            except Exception as error:  # noqa: BLE001 - surfaced to the test
                results.append(error)

    gc.callbacks.append(during)
    try:
        gc.collect(0)
    finally:
        gc.callbacks.remove(during)
    return results


def test_a_refreeze_freezes_nothing_while_another_collection_runs() -> None:
    """A skipped collection is never followed by a freeze (live, 2026-10-19).

    In the round-5 memory run, a compaction ran while the probe's own
    collection was in progress on another thread. CPython skipped its
    collection, so it froze the heap again, garbage included, and the
    refreezer booked it as done. A re-freeze in that state reports
    ``busy`` and freezes nothing, so the garbage stays collectable.
    """
    gc.unfreeze()
    gc.collect()
    try:
        garbage = old_garbage()
        refreezer = IdleHeapRefreezer(enabled=lambda: True)
        # Not 0: CPython 3.12 parks immortal objects in the permanent
        # generation during every collection.
        frozen = gc.get_freeze_count()
        outcome = during_another_collection(lambda: refreezer.maintain("run_settled"))
        assert outcome == ["busy"], outcome
        assert gc.get_freeze_count() == frozen, (
            "a skipped collection was followed by a freeze"
        )
        gc.collect()
        assert garbage() is None
    finally:
        gc.unfreeze()


def test_a_compaction_another_collection_blocks_refreezes_and_stays_owed() -> None:
    """A blocked compaction restores the freeze and is retried, not booked.

    Negative control for the bookkeeping: on the blocked attempt the frozen
    garbage survives and a compaction is still due; the next maintenance
    compacts and frees it. The heap is frozen again meanwhile, so the
    automatic passes do not rescan it.
    """
    gc.unfreeze()
    gc.collect()
    try:
        cycle = Cycle()
        garbage = weakref.ref(cycle)
        gc.freeze()  # frozen while live, as a shown transcript is
        del cycle
        frozen = gc.get_freeze_count()
        refreezer = IdleHeapRefreezer(enabled=lambda: True)
        refreezer.note_stranded()

        outcome = during_another_collection(
            lambda: refreezer.maintain("conversation_painted")
        )
        assert outcome == ["busy"], outcome
        assert gc.get_freeze_count() >= frozen, (
            "the blocked compaction left the heap unfrozen"
        )
        assert garbage() is not None
        assert refreezer.compaction_due(), "a blocked compaction was booked as done"

        assert refreezer.maintain("conversation_painted") == "compact"
        assert garbage() is None
    finally:
        gc.unfreeze()


class HeldWorkerPass:
    """A worker thread's full collection, held open at its start.

    A worker's pass can hand the GIL to the UI loop from its finalizers or
    ``gc.callbacks``. A start hook that waits on an event is the
    deterministic stand-in: while it waits, CPython skips every other
    ``gc.collect()``; once released, the pass runs to the end and the
    generation-2 pass count moves.

    Args:
        collect: The real ``gc.collect`` (the test replaces the module's).
    """

    def __init__(self, collect: Callable[[int], int]) -> None:
        self._collect = collect
        self._started = threading.Event()
        self._release = threading.Event()
        self._thread = threading.Thread(target=lambda: self._collect(2), daemon=True)

    def _hold(self, phase: str, info: dict) -> None:
        if phase == "start" and threading.get_ident() == self._thread.ident:
            self._started.set()
            self._release.wait(10)

    def __enter__(self) -> HeldWorkerPass:
        gc.callbacks.append(self._hold)
        self._thread.start()
        assert self._started.wait(10), "the worker's pass never started"
        return self

    def finish(self) -> None:
        """Let the worker's pass run to the end, and wait for it."""
        self._release.set()
        self._thread.join(10)
        assert not self._thread.is_alive(), "the worker's pass never finished"

    def __exit__(self, *exc_info: object) -> None:
        self.finish()
        gc.callbacks.remove(self._hold)


def a_worker_pass_ends_inside_every_collect(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Every ``gc.collect`` call meets a worker's full pass that ends inside it.

    Args:
        monkeypatch: Replaces ``gc.collect`` for the test.

    Returns:
        Per call, whether the generation-2 pass count moved across it: what
        the pass-count check alone reads as "this call collected".
    """
    real_collect = gc.collect
    moved: list[bool] = []

    def collect(generation: int = 2) -> int:
        with HeldWorkerPass(real_collect) as worker:
            before = gc.get_stats()[2]["collections"]
            result = real_collect(generation)  # skipped: the worker's pass runs
            worker.finish()
            moved.append(gc.get_stats()[2]["collections"] != before)
        return result

    monkeypatch.setattr(ui_responsiveness.gc, "collect", collect)
    return moved


@pytest.mark.parametrize("owed", ["refreeze", "compaction"])
def test_a_worker_threads_pass_ending_inside_a_skipped_call_is_not_booked(
    monkeypatch: pytest.MonkeyPatch, owed: str
) -> None:
    """Only a full pass that ran on the calling thread books a freeze (TASK-33620.15.11).

    The check read ``gc.get_stats()[2]["collections"]`` before and after the
    call. A worker thread's pass that finished in between moved it although
    this call's own collection was skipped: a re-freeze then froze the heap
    with no collection before it, garbage included, and a compaction was
    booked as done. Every try here meets such a pass, so the maintenance
    reports ``busy``, and the collector stays enabled at its thresholds.

    Negative control: the pass count moved across every one of those calls,
    so the pass-count check alone would have booked the first.

    Args:
        monkeypatch: Wraps ``gc.collect`` so each call meets a worker's pass.
        owed: Whether the maintenance owes a re-freeze or a compaction.
    """
    gc.unfreeze()
    gc.collect()
    thresholds, enabled = gc.get_threshold(), gc.isenabled()
    try:
        refreezer = IdleHeapRefreezer(enabled=lambda: True)
        if owed == "compaction":
            refreezer.note_stranded()
        frozen = gc.get_freeze_count()
        moved = a_worker_pass_ends_inside_every_collect(monkeypatch)
        outcome = refreezer.maintain("run_settled")
        monkeypatch.undo()

        assert moved and all(moved), f"negative control: pass count moved {moved}"
        assert outcome == "busy", f"{outcome} booked over {len(moved)} skipped call(s)"
        if owed == "compaction":
            assert refreezer.compaction_due(), "a compaction that never ran was booked"
        else:
            assert gc.get_freeze_count() == frozen, "a skipped call was followed by a freeze"
        assert (gc.get_threshold(), gc.isenabled()) == (thresholds, enabled)
    finally:
        gc.unfreeze()


def test_a_request_runs_once_its_owner_is_idle_and_the_user_quiet() -> None:
    """The re-freeze waits the quiet period after the request, then runs once."""
    clock, ops, owner = Clock(), Ops(), Owner()
    refreezer = make(clock, ops, quiet_seconds=2.0)

    refreezer.request("run_settled", owner, owner_busy)
    assert [delay for delay, _ in owner.timers] == [2.0]
    assert ops.calls == []

    clock.now += 2.0
    owner.fire_last()
    assert ops.calls == ["refreeze:run_settled"]
    assert list(refreezer.actions) == [("refreeze", "run_settled")]
    owner.fire_last()  # a stale timer does nothing
    assert ops.calls == ["refreeze:run_settled"]


def test_input_inside_the_quiet_period_defers_the_refreeze() -> None:
    """A key 1.5 s after the request pushes the re-freeze to 2 s after the key."""
    clock, ops, owner = Clock(), Ops(), Owner()
    refreezer = make(clock, ops, quiet_seconds=2.0)
    refreezer.request("conversation_painted", owner, owner_busy)

    clock.now += 1.5
    refreezer.note_input()
    clock.now += 0.5
    owner.fire_last()
    assert ops.calls == []
    assert owner.timers[-1][0] == pytest.approx(1.5)

    clock.now += 1.5
    owner.fire_last()
    assert ops.calls == ["refreeze:conversation_painted"]


def test_a_busy_owner_drops_the_request() -> None:
    """While a send is in flight nothing runs, and nothing is re-armed."""
    clock, ops, owner = Clock(), Ops(), Owner()
    refreezer = make(clock, ops)
    refreezer.request("conversation_painted", owner, owner_busy)
    owner.busy = True
    clock.now += 10
    owner.fire_last()
    assert ops.calls == []
    assert len(owner.timers) == 1

    owner.busy = False
    owner.fire_last()  # the dropped request stays dropped
    assert ops.calls == []


def test_a_departed_owner_is_not_pinned_and_its_request_does_nothing() -> None:
    """Only a weak reference is kept: a dead owner's timer finds nothing to do."""
    clock, ops, owner = Clock(), Ops(), Owner()
    refreezer = make(clock, ops)
    refreezer.request("run_settled", owner, owner_busy)
    callback = owner.timers[-1][1]
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is None
    clock.now += 10
    callback()
    assert ops.calls == []


def test_a_failing_busy_check_never_raises_into_the_timer() -> None:
    """A Textual timer callback that raises ends the app: errors drop the request."""
    clock, ops, owner = Clock(), Ops(), Owner()
    refreezer = make(clock, ops)

    def broken(_owner: Owner) -> bool:
        raise RuntimeError("busy check failed")

    refreezer.request("run_settled", owner, broken)
    clock.now += 10
    owner.fire_last()
    assert ops.calls == []


def test_requests_are_ignored_under_pytest_unless_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off by default in the suite, like the screen pre-import; the env var wins.

    Args:
        monkeypatch: Sets ``TLDW_IDLE_HEAP_REFREEZE``.
    """
    owner = Owner()
    refreezer = IdleHeapRefreezer()
    monkeypatch.delenv("TLDW_IDLE_HEAP_REFREEZE", raising=False)
    refreezer.request("run_settled", owner, owner_busy)
    assert owner.timers == []

    monkeypatch.setenv("TLDW_IDLE_HEAP_REFREEZE", "1")
    refreezer.request("run_settled", owner, owner_busy)
    assert len(owner.timers) == 1


def test_a_compaction_runs_once_refreezes_add_a_quarter_and_waits_longer() -> None:
    """Growth of 25% of the frozen baseline makes the next idle moment compact."""
    clock, ops, owner = Clock(), Ops(frozen=600_000, added=40_000), Owner()
    refreezer = make(
        clock, ops, quiet_seconds=2.0, compaction_quiet_seconds=5.0, min_growth=1_000
    )
    assert COMPACTION_GROWTH_RATIO == 0.25
    for _ in range(3):
        refreezer.request("run_settled", owner, owner_busy)
        clock.now += 2.0
        owner.fire_last()
    assert not refreezer.compaction_due()  # 120,000 < 0.25 * 600,000
    refreezer.request("run_settled", owner, owner_busy)
    clock.now += 2.0
    owner.fire_last()
    assert refreezer.compaction_due()  # 160,000 >= 150,000
    assert not any(call.startswith("compact:") for call in ops.calls)

    refreezer.request("run_settled", owner, owner_busy)
    clock.now += 2.0
    owner.fire_last()
    assert ops.calls[-1] == "refreeze:run_settled"  # no re-freeze while one is due
    assert owner.timers[-1][0] == pytest.approx(3.0)
    assert len(ops.calls) == 4

    clock.now += 3.0
    owner.fire_last()
    assert ops.calls[-1] == "compact:compact:run_settled"
    assert refreezer.actions[-1] == ("compact", "run_settled")
    assert not refreezer.compaction_due()


def test_only_real_growth_of_the_frozen_set_makes_a_compaction_due() -> None:
    """Objects frozen then freed by reference counting never trigger a compaction.

    Live (2026-10-10), each send's re-freeze froze ~70K objects for ~20K of
    real growth: most were render strips the next paint replaced and
    reference counting freed. Budgeting on what re-freezes ADD compacted
    every third send, so the walk of the permanent generation decides.
    """
    clock = Clock()
    ops, owner = Ops(frozen=800_000, added=70_000, freed=66_000), Owner()
    refreezer = make(clock, ops, quiet_seconds=2.0, min_growth=1_000)
    for _ in range(20):
        refreezer.request("run_settled", owner, owner_busy)
        clock.now += 2.0
        owner.fire_last()
    assert not any(call.startswith("compact:") for call in ops.calls)
    assert not refreezer.compaction_due()  # 20 x 4,000 = 80,000 < 200,000
    assert ops.walks >= 20, ops.walks  # measured after every re-freeze

    ops.added, ops.freed = 200_000, 0
    refreezer.maintain("conversation_painted")
    assert refreezer.compaction_due()


def test_growth_a_refreeze_did_not_count_still_makes_a_compaction_due() -> None:
    """The frozen set is measured, never estimated from what re-freezes counted.

    Each re-freeze counts the old generation after its collection, and other
    threads can allocate between that count and the freeze, so a running
    total of those counts is no upper bound: a refreezer that walked only
    once the total reached the budget could miss a due compaction. (The live
    peaks once blamed on this, 372K against a 244K budget, recurred after the
    walk was added and are the bound's own last-re-freeze term; ADR-198.)
    """
    clock = Clock()
    ops, owner = Ops(frozen=800_000, added=5_000, uncounted=40_000), Owner()
    refreezer = make(clock, ops, quiet_seconds=2.0, min_growth=1_000)
    for _ in range(5):  # 5 x 45,000 = 225,000 >= 0.25 x 800,000
        refreezer.request("run_settled", owner, owner_busy)
        clock.now += 2.0
        owner.fire_last()
    assert refreezer.compaction_due(), (
        f"the frozen set grew {ops.frozen - 800_000} against a 200,000 budget"
    )


def test_stranded_garbage_makes_the_next_idle_moment_compact() -> None:
    """A frozen transcript switched away compacts at once, whatever the budget.

    Live (2026-10-19, 24 chat switches): left to the 25% budget, switched-away
    transcripts piled up in the permanent generation and one compaction
    collected 212-373K objects for 2.1-2.8 s of CPU on the UI loop, against
    329 ms for dev's worst generation-2 pass. A stranded note bounds the
    garbage a compaction meets to about one transcript.
    """
    clock, ops, owner = Clock(), Ops(frozen=1_000_000, added=50_000), Owner()
    refreezer = make(
        clock,
        ops,
        quiet_seconds=2.0,
        compaction_quiet_seconds=5.0,
        min_growth=10_000_000,  # the budget alone never compacts here
    )
    refreezer.request("conversation_painted", owner, owner_busy)
    clock.now += 2.0
    owner.fire_last()
    assert ops.calls == ["refreeze:conversation_painted"]
    assert not refreezer.compaction_due()

    refreezer.note_stranded()
    assert refreezer.compaction_due()
    refreezer.request("conversation_painted", owner, owner_busy)
    clock.now += 2.0
    owner.fire_last()
    assert len(ops.calls) == 1, "compacted before the compaction's quiet period"
    assert owner.timers[-1][0] == pytest.approx(3.0)
    clock.now += 3.0
    owner.fire_last()
    assert ops.calls[-1] == "compact:compact:conversation_painted"
    assert not refreezer.compaction_due()

    refreezer.request("run_settled", owner, owner_busy)
    clock.now += 2.0
    owner.fire_last()
    assert ops.calls[-1] == "refreeze:run_settled"


def test_a_stranded_note_outlives_a_request_dropped_while_busy() -> None:
    """The switch's request can be dropped by a send; the owed compaction is not."""
    clock, ops, owner = Clock(), Ops(), Owner()
    refreezer = make(clock, ops, min_growth=10_000_000)
    refreezer.note_stranded()
    refreezer.request("conversation_painted", owner, owner_busy)
    owner.busy = True
    clock.now += 10
    owner.fire_last()
    assert ops.calls == []

    owner.busy = False
    refreezer.request("run_settled", owner, owner_busy)
    clock.now += 10
    owner.fire_last()
    assert ops.calls == ["compact:compact:run_settled"]


def test_every_freeze_advances_the_freeze_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boot freezes, re-freezes and compactions all count as a freeze.

    The Console compares epochs to learn whether the transcript it replaces
    was frozen while it was shown.

    Args:
        monkeypatch: Stubs the collector so nothing is really frozen.
    """
    for name in ("freeze", "unfreeze"):
        monkeypatch.setattr(ui_responsiveness.gc, name, lambda *args: 0)
    counting_collector(monkeypatch, [])
    monkeypatch.setattr(ui_responsiveness.gc, "get_objects", lambda generation=None: [])
    monkeypatch.setattr(ui_responsiveness.gc, "get_freeze_count", lambda: 0)
    start = ui_responsiveness.heap_freeze_epoch()
    ui_responsiveness.freeze_long_lived_heap("unit-test")
    ui_responsiveness.refreeze_long_lived_heap("unit-test")
    ui_responsiveness.compact_long_lived_heap("unit-test")
    assert ui_responsiveness.heap_freeze_epoch() == start + 3


def boot_points_freeze_into(
    monkeypatch: pytest.MonkeyPatch,
    refreezer: IdleHeapRefreezer,
    ops: Ops,
    freeze: Callable[[], None],
) -> None:
    """Route the real boot-point helper to ``refreezer`` and the fake collector.

    ``freeze_long_lived_heap`` (``_ui_ready``, the screen and first-send
    pre-imports; ADR-198) serves the process's refreezer, which here is
    ``refreezer``; its young collection does nothing and its freeze runs
    ``freeze`` against ``ops``'s permanent generation.

    Args:
        monkeypatch: Installs the fakes.
        refreezer: Stands in for the process's refreezer.
        ops: The fake permanent generation the walks read.
        freeze: Runs in place of ``gc.freeze()``.
    """
    monkeypatch.setattr(ui_responsiveness, "_HEAP_REFREEZER", refreezer)
    monkeypatch.setattr(ui_responsiveness.gc, "collect", lambda generation=2: 0)
    monkeypatch.setattr(ui_responsiveness.gc, "freeze", freeze)
    monkeypatch.setattr(ui_responsiveness.gc, "get_freeze_count", ops.freeze_count)


def adds(ops: Ops, objects: int) -> Callable[[], None]:
    """A fake ``gc.freeze()`` that moves ``objects`` more into the frozen set."""

    def freeze() -> None:
        ops.frozen += objects

    return freeze


def test_a_boot_freeze_after_the_first_maintenance_is_measured_by_the_next_budget_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The budget re-measures the frozen set after a boot point froze (TASK-33620.15.12).

    The budget reuses its last walk until a re-freeze marks it stale. A boot
    point's freeze after the refreezer's first maintenance did not, so the
    next check read the frozen set's size from before it. A boot freeze
    before the first maintenance is measured by the first look itself and
    costs no extra walk.

    Negative control: with nothing frozen since the last walk, the check
    reuses it.

    Args:
        monkeypatch: Routes the boot points to the fake collector.
    """
    ops = Ops(frozen=1_000_000, added=100_000)
    refreezer = make(Clock(), ops, min_growth=1_000)
    boot_points_freeze_into(monkeypatch, refreezer, ops, adds(ops, 40_000))

    ui_responsiveness.freeze_long_lived_heap("ui_ready")
    assert ops.walks == 1, "a freeze before the first look walked more than once"
    refreezer.maintain("conversation_painted")  # first look: 1,040,000, then +100K
    assert not refreezer.compaction_due()  # 100,000 < 0.25 x 1,040,000
    walks = ops.walks
    assert not refreezer.compaction_due()
    assert ops.walks == walks, "negative control: an unchanged frozen set was walked"

    ui_responsiveness.freeze_long_lived_heap("screen_preimport")
    walks = ops.walks
    assert not refreezer.compaction_due()
    assert ops.walks == walks + 1, (
        "the budget check read the frozen set's size from before the boot freeze"
    )


@pytest.mark.parametrize("frozen_by", ["screen_preimport", "refreeze"])
def test_objects_a_boot_point_freezes_never_count_as_growth(
    monkeypatch: pytest.MonkeyPatch, frozen_by: str
) -> None:
    """Only what re-freezes add counts toward a compaction (TASK-33620.15.12).

    A pre-import pass that froze 300K long-lived objects after the first
    maintenance counted them as growth against the 250K budget, and brought
    a compaction forward that only re-freezes' growth should call for.

    Negative control: the same 300K frozen by a re-freeze makes one due.

    Args:
        monkeypatch: Routes the boot points to the fake collector.
        frozen_by: Which freeze adds the 300K objects.
    """
    ops = Ops(frozen=1_000_000, added=0)
    refreezer = make(Clock(), ops, min_growth=1_000)
    boot_points_freeze_into(monkeypatch, refreezer, ops, adds(ops, 300_000))
    assert refreezer.maintain("conversation_painted") == "refreeze"  # first look

    if frozen_by == "refreeze":
        ops.added = 300_000
        assert refreezer.maintain("run_settled") == "refreeze"
        ops.added = 0
        assert refreezer.maintain("run_settled") == "compact"
        return
    ui_responsiveness.freeze_long_lived_heap(frozen_by)  # before the re-freeze's walk
    assert refreezer.maintain("run_settled") == "refreeze", (
        "the boot point's objects counted as growth"
    )
    ui_responsiveness.freeze_long_lived_heap("first_send_preimport")  # after a walk
    assert not refreezer.compaction_due(), "the boot point's objects counted as growth"
    assert refreezer.maintain("run_settled") == "refreeze"


def test_a_maintenance_waits_for_a_boot_freeze_on_another_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A boot point's freeze and its accounting are one step (TASK-33620.15.12).

    The screen and first-send pre-imports freeze from their own threads. A
    maintenance that ran between that freeze and its accounting would read
    the boot point's objects as growth, so it waits for the boot freeze.

    Args:
        monkeypatch: Routes the boot points to the fake collector.
    """
    ops = Ops(frozen=1_000_000, added=0)
    refreezer = make(Clock(), ops, min_growth=1_000)
    refreezer.maintain("conversation_painted")  # first look
    ran = threading.Event()
    outcomes: list[str] = []
    seen: list[str] = []

    def maintain_on_the_ui_loop() -> None:
        outcomes.append(refreezer.maintain("run_settled"))
        ran.set()

    ui_loop = threading.Thread(target=maintain_on_the_ui_loop, daemon=True)

    def freeze() -> None:
        ops.frozen += 300_000
        ui_loop.start()
        seen.append("ran" if ran.wait(0.5) else "waited")

    boot_points_freeze_into(monkeypatch, refreezer, ops, freeze)
    preimport = threading.Thread(
        target=ui_responsiveness.freeze_long_lived_heap, args=("screen_preimport",)
    )
    preimport.start()
    preimport.join(10)
    ui_loop.join(10)
    assert seen == ["waited"], "a maintenance ran inside a boot point's freeze"
    assert outcomes == ["refreeze"], "the boot point's objects counted as growth"


class TranscriptApp(TextualAppGuards, App[None]):
    """Swaps whole "transcripts" (rows of nested widgets), as a chat switch does."""

    def compose(self) -> ComposeResult:
        yield Vertical(id="host")

    async def show(self, rows: int = 30) -> list[weakref.ref]:
        """Replace the shown transcript; returns weakrefs to the new rows.

        The old transcript's widgets stay in layout caches until the next one
        is laid out, so a switch is "remove, then mount the next".
        """
        host = self.query_one("#host")
        await host.remove_children()
        transcript = Vertical(
            *[
                Vertical(
                    Static(f"header {index}"),
                    Static(f"body {index} " * 8),
                    Static("footer"),
                )
                for index in range(rows)
            ]
        )
        await host.mount(transcript)
        await self.workers.wait_for_complete()
        return [weakref.ref(row) for row in transcript.children]


def real_refreezer(**kwargs) -> IdleHeapRefreezer:
    return IdleHeapRefreezer(enabled=lambda: True, **kwargs)


def alive(refs: list[weakref.ref]) -> int:
    return sum(ref() is not None for ref in refs)


async def test_widgets_frozen_live_and_switched_away_are_reclaimed_only_by_compaction() -> (
    None
):
    """AC#6: a re-freeze strands what it froze live; the compaction reclaims it.

    Two controls: without a freeze a switched-away transcript is collected
    (so nothing else holds it), and after a re-freeze it is not, because
    Textual widgets are cyclic and frozen cycles are never collected.
    """
    gc.unfreeze()
    gc.collect()
    app = TranscriptApp()
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            unfrozen = await app.show()
            await app.show()
            await pilot.pause()
            gc.collect()
            assert alive(unfrozen) == 0, "a switched-away transcript is held elsewhere"

            refreezer = real_refreezer(growth_ratio=0.0, min_growth=1)
            frozen = await app.show()
            assert refreezer.maintain("conversation_painted") == "refreeze"
            await app.show()
            await pilot.pause()
            gc.collect()
            assert alive(frozen) == len(frozen), (
                "frozen rows were collected without a compaction: the leak "
                "this policy bounds does not exist"
            )

            assert refreezer.maintain("run_settled") == "compact"
            assert alive(frozen) == 0
    finally:
        gc.unfreeze()


async def test_twenty_switches_keep_the_frozen_generation_bounded() -> None:
    """AC#3: re-freezing after every switch never grows the frozen set unboundedly.

    Every switch shows a fresh 30-row transcript and re-freezes. Above the
    size each compaction leaves, the permanent generation never grows by more
    than the compaction budget plus one re-freeze, and each compaction
    reclaims every row switched away before it. (Measured 2026-10-10: ~12.9K
    objects per switch; the post-compaction size rises while Textual's caches
    warm up, then holds at ~115K from the eighth compaction on, over 40
    switches.)
    """
    gc.unfreeze()
    gc.collect()
    thresholds = gc.get_threshold()
    app = TranscriptApp()
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            budget = 20_000
            refreezer = real_refreezer(growth_ratio=0.0, min_growth=budget)
            dropped: list[weakref.ref] = []
            floor: int | None = None
            excess: list[int] = []
            steps: list[int] = []
            shown = await app.show()
            for _ in range(24):
                before = gc.get_freeze_count()
                action = refreezer.maintain("conversation_painted")
                size = gc.get_freeze_count()
                if action == "compact":
                    floor = size
                    assert alive(dropped) == 0, (
                        "a compaction left switched-away rows alive"
                    )
                else:
                    steps.append(size - before)
                    if floor is not None:
                        excess.append(size - floor)
                dropped.extend(shown)
                shown = await app.show()
                await pilot.pause()
            compactions = [
                action for action, _ in refreezer.actions if action == "compact"
            ]
            assert len(compactions) >= 4, list(refreezer.actions)
            assert max(excess) <= budget + max(steps), (excess, steps)
            assert (
                alive(dropped) > 0
            )  # frozen since the last compaction: the bounded leak

            final = real_refreezer(growth_ratio=0.0, min_growth=0)
            assert final.maintain("run_settled") == "compact"
            assert alive(dropped) == 0
            assert alive(shown) == len(shown)
            # ADR-198: neither path disables the collector or tunes it.
            assert gc.isenabled()
            assert gc.get_threshold() == thresholds
    finally:
        gc.unfreeze()


class ConsoleTranscriptApp(TextualAppGuards, App[None]):
    """The real Console transcript, showing one session's conversation at a time."""

    CSS = "ConsoleTranscript { height: 20; }"

    def __init__(self) -> None:
        super().__init__()
        # Pruning would drop rows on its own; keep every row mounted.
        self.app_config = {
            "chat_defaults": {
                "prune_high_watermark": 1_000_000,
                "prune_low_watermark": 900_000,
            }
        }

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")

    async def show(self, session: str, turns: int = 12) -> set[int]:
        """Show ``session``'s conversation, as a chat switch does.

        Returns:
            The ``id()`` of every widget the transcript holds afterwards.
        """
        transcript = self.query_one(ConsoleTranscript)
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=role,
                    content=f"{session} {role.value} {index}: " + "words " * 12,
                    id=f"{session}-{role.value}-{index}",
                    status="completed",
                )
                for index in range(turns)
                for role in (ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT)
            ],
            session_id=session,
        )
        await transcript.refresh_messages()
        await self.workers.wait_for_complete()
        return {id(widget) for widget in transcript.query("*")}

    def rows(self) -> list[weakref.ref]:
        """Weak references to every widget the transcript holds now."""
        return [
            weakref.ref(widget)
            for widget in self.query_one(ConsoleTranscript).query("*")
        ]


async def switch_away(
    app: ConsoleTranscriptApp, pilot, rows: list[weakref.ref], session: str
) -> list[weakref.ref]:
    """Switch to ``session``; returns the references to rows it unmounted.

    A row the collector already freed during the switch counts as unmounted
    (its reference is dead), so an automatic collection mid-switch cannot
    shrink the sample.
    """
    kept = await app.show(session)
    await pilot.pause()
    dropped = []
    for ref in rows:
        widget = ref()
        if widget is None or id(widget) not in kept:
            dropped.append(ref)
        del widget
    return dropped


@pytest.mark.bootstrap_profile  # the transcript reads [console] settings
async def test_real_console_transcript_rows_frozen_live_are_freed_by_the_compaction() -> (
    None
):
    """AC#6 on the real ``ConsoleTranscript``: switched-away rows die after compaction.

    Controls: with nothing frozen, the rows a chat switch unmounts are
    collected (so the transcript keeps no reference to them); after a
    re-freeze they survive a full collection (the leak the compaction
    bounds is real for these widgets); the compaction then frees them.
    """
    gc.unfreeze()
    gc.collect()
    app = ConsoleTranscriptApp()
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await app.show("alpha")
            unfrozen = await switch_away(app, pilot, app.rows(), "bravo")
            assert len(unfrozen) > 20, "the switch unmounted too few rows to test"
            gc.collect()
            assert alive(unfrozen) == 0, "the transcript still holds switched-away rows"

            refreezer = real_refreezer(growth_ratio=0.0, min_growth=1)
            await app.show("charlie")
            assert refreezer.maintain("conversation_painted") == "refreeze"
            frozen = await switch_away(app, pilot, app.rows(), "delta")
            assert len(frozen) > 20
            gc.collect()
            assert alive(frozen) == len(frozen), (
                "frozen transcript rows were collected without a compaction"
            )

            assert refreezer.maintain("run_settled") == "compact"
            assert alive(frozen) == 0, "the compaction left switched-away rows alive"
    finally:
        gc.unfreeze()


@pytest.mark.bootstrap_profile  # the transcript reads [console] settings
async def test_a_stranded_note_frees_switched_away_transcripts_at_the_next_maintenance() -> (
    None
):
    """The switch policy on the real transcript: frozen rows die at the next idle run.

    Negative control: with the budget out of reach and no stranded note, the
    maintenance after a switch only re-freezes, and the switched-away rows
    (frozen while shown) survive a full collection. With the note, the next
    idle request -- quiet periods and a real Textual timer -- compacts, and
    every switched-away row is freed.
    """
    gc.unfreeze()
    gc.collect()
    app = ConsoleTranscriptApp()
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            refreezer = real_refreezer(
                min_growth=10_000_000,
                quiet_seconds=0.05,
                compaction_quiet_seconds=0.1,
            )
            await app.show("alpha")
            assert refreezer.maintain("conversation_painted") == "refreeze"
            alpha = await switch_away(app, pilot, app.rows(), "bravo")
            assert len(alpha) > 20, "the switch unmounted too few rows to test"
            assert refreezer.maintain("conversation_painted") == "refreeze"
            gc.collect()
            assert alive(alpha) == len(alpha), (
                "rows frozen while shown were collected without a compaction"
            )

            bravo = await switch_away(app, pilot, app.rows(), "charlie")
            refreezer.note_stranded()
            # Through the idle path: a request, the quiet periods, a real timer.
            refreezer.request("conversation_painted", app, lambda _owner: False)
            for _ in range(100):
                if refreezer.actions[-1][0] == "compact":
                    break
                await pilot.pause(0.05)
            assert list(refreezer.actions)[-1] == ("compact", "conversation_painted")
            assert alive(alpha) == 0, "the compaction left stranded rows alive"
            assert alive(bravo) == 0, "the compaction left stranded rows alive"
    finally:
        gc.unfreeze()


async def test_the_app_guards_stamp_user_input_for_the_quiet_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key pressed in the app defers a pending re-freeze (the input tap).

    Args:
        monkeypatch: Installs a refreezer with a fake clock as the process's.
    """
    clock, ops, owner = Clock(), Ops(), Owner()
    refreezer = make(clock, ops, quiet_seconds=2.0)
    monkeypatch.setattr(ui_responsiveness, "_HEAP_REFREEZER", refreezer)
    app = TranscriptApp()
    async with app.run_test() as pilot:
        refreezer.request("run_settled", owner, owner_busy)
        clock.now += 1.5
        await pilot.press("a")
        clock.now += 0.5
        owner.fire_last()
    assert ops.calls == []
    assert owner.timers[-1][0] == pytest.approx(1.5)


@pytest.mark.parametrize("stamp_on_post", [True, False], ids=["tap", "control"])
async def test_input_posted_but_not_yet_dispatched_holds_off_a_maintenance(
    monkeypatch: pytest.MonkeyPatch, stamp_on_post: bool
) -> None:
    """A key already posted to the app counts as input before the app handles it.

    TASK-33620.15.10: the driver posts a key to the app's queue, and the app
    handles it later; a refreezer timer firing in between must see it. The
    key is posted here without yielding to the loop, so it is still queued
    when the timer fires. Negative control: with the stamp only at dispatch
    (the posting hook removed), the same timer starts a maintenance.

    Args:
        monkeypatch: Installs a refreezer with a fake clock as the process's.
        stamp_on_post: Whether the app stamps input when it is posted.
    """
    clock, ops, owner = Clock(), Ops(), Owner()
    refreezer = make(clock, ops, quiet_seconds=2.0)
    monkeypatch.setattr(ui_responsiveness, "_HEAP_REFREEZER", refreezer)
    if not stamp_on_post:
        monkeypatch.delattr(text_selection_crash_guard.UserInputTap, "post_message")
    app = TranscriptApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        refreezer.request("run_settled", owner, owner_busy)
        clock.now += 10.0
        assert app.post_message(events.Key("a", "a"))
        owner.fire_last()  # before the loop has handled the key
        if stamp_on_post:
            assert ops.calls == []
            assert owner.timers[-1][0] == pytest.approx(2.0)
        else:
            assert ops.calls == ["refreeze:run_settled"]
        await pilot.pause()
