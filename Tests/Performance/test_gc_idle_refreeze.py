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
import weakref
from collections.abc import Callable

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tldw_chatbook.Utils import ui_responsiveness
from tldw_chatbook.Utils.text_selection_crash_guard import TextualAppGuards
from tldw_chatbook.Utils.ui_responsiveness import (
    COMPACTION_GROWTH_RATIO,
    IdleHeapRefreezer,
    refreeze_long_lived_heap,
)


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
    """Fake collector operations with a permanent generation of ``frozen`` objects."""

    def __init__(self, frozen: int = 1_000_000, added: int = 10_000) -> None:
        self.frozen = frozen
        self.added = added
        self.calls: list[str] = []

    def refreeze(self, reason: str) -> int:
        self.calls.append(f"refreeze:{reason}")
        if reason.startswith("compact:"):
            return self.frozen
        self.frozen += self.added
        return self.added

    def unfreeze(self) -> None:
        self.calls.append("unfreeze")

    def freeze_count(self) -> int:
        return self.frozen


def make(clock: Clock, ops: Ops, **kwargs) -> IdleHeapRefreezer:
    return IdleHeapRefreezer(
        clock=clock,
        enabled=lambda: True,
        refreeze=ops.refreeze,
        unfreeze=ops.unfreeze,
        freeze_count=ops.freeze_count,
        **kwargs,
    )


def test_refreeze_collects_every_unfrozen_generation_then_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full collection first (reclaims old-generation garbage), then the freeze.

    Args:
        monkeypatch: Records the collector calls instead of running them.
    """
    calls: list[tuple] = []
    monkeypatch.setattr(
        ui_responsiveness.gc, "collect", lambda generation=2: calls.append(("collect", generation)) or 0
    )
    monkeypatch.setattr(
        ui_responsiveness.gc,
        "get_objects",
        lambda generation=None: calls.append(("get_objects", generation)) or [1, 2, 3],
    )
    monkeypatch.setattr(ui_responsiveness.gc, "freeze", lambda: calls.append(("freeze",)))

    assert refreeze_long_lived_heap("unit-test") == 3
    assert calls == [("collect", 2), ("get_objects", 2), ("freeze",)]


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
    assert ops.calls.count("unfreeze") == 0

    refreezer.request("run_settled", owner, owner_busy)
    clock.now += 2.0
    owner.fire_last()
    assert ops.calls[-1] == "refreeze:run_settled"  # no re-freeze while one is due
    assert owner.timers[-1][0] == pytest.approx(3.0)
    assert len(ops.calls) == 4

    clock.now += 3.0
    owner.fire_last()
    assert ops.calls[-2:] == ["unfreeze", "refreeze:compact:run_settled"]
    assert refreezer.actions[-1] == ("compact", "run_settled")
    assert not refreezer.compaction_due()


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
                Vertical(Static(f"header {index}"), Static(f"body {index} " * 8), Static("footer"))
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


async def test_widgets_frozen_live_and_switched_away_are_reclaimed_only_by_compaction() -> None:
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
                    assert alive(dropped) == 0, "a compaction left switched-away rows alive"
                else:
                    steps.append(size - before)
                    if floor is not None:
                        excess.append(size - floor)
                dropped.extend(shown)
                shown = await app.show()
                await pilot.pause()
            compactions = [action for action, _ in refreezer.actions if action == "compact"]
            assert len(compactions) >= 4, list(refreezer.actions)
            assert max(excess) <= budget + max(steps), (excess, steps)
            assert alive(dropped) > 0  # frozen since the last compaction: the bounded leak

            final = real_refreezer(growth_ratio=0.0, min_growth=0)
            assert final.maintain("run_settled") == "compact"
            assert alive(dropped) == 0
            assert alive(shown) == len(shown)
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
