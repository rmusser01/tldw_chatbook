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

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Utils import ui_responsiveness
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
    counting right after it, as replaced render strips are.
    """

    def __init__(
        self, frozen: int = 1_000_000, added: int = 10_000, freed: int = 0
    ) -> None:
        self.frozen = frozen
        self.added = added
        self.freed = freed
        self.calls: list[str] = []
        self.walks = 0

    def refreeze(self, reason: str) -> int:
        self.calls.append(f"refreeze:{reason}")
        self.frozen += self.added - self.freed
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


def test_the_idle_refreeze_collects_every_unfrozen_generation_then_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full collection first (reclaims old-generation garbage), then the freeze.

    Args:
        monkeypatch: Records the collector calls instead of running them.
    """
    calls: list[tuple] = []
    monkeypatch.setattr(
        ui_responsiveness.gc,
        "collect",
        lambda generation=2: calls.append(("collect", generation)) or 0,
    )
    monkeypatch.setattr(
        ui_responsiveness.gc,
        "get_objects",
        lambda generation=None: calls.append(("get_objects", generation)) or [1, 2],
    )
    monkeypatch.setattr(
        ui_responsiveness.gc, "freeze", lambda: calls.append(("freeze",))
    )

    assert ui_responsiveness.refreeze_long_lived_heap("unit-test") == 2
    assert calls == [("collect", 2), ("get_objects", 2), ("freeze",)]


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
    monkeypatch.setattr(
        ui_responsiveness.gc,
        "collect",
        lambda generation=2: calls.append(("collect", generation)) or 0,
    )
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
    every third send, so the upper bound only decides when to walk the
    permanent generation, and the walk decides.
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
    assert 1 < ops.walks <= 8, ops.walks  # a walk per ~3 re-freezes, not per one

    ops.added, ops.freed = 200_000, 0
    refreezer.maintain("conversation_painted")
    assert refreezer.compaction_due()


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
    for name in ("collect", "freeze", "unfreeze"):
        monkeypatch.setattr(ui_responsiveness.gc, name, lambda *args: 0)
    monkeypatch.setattr(ui_responsiveness.gc, "get_objects", lambda generation=None: [])
    monkeypatch.setattr(ui_responsiveness.gc, "get_freeze_count", lambda: 0)
    start = ui_responsiveness.heap_freeze_epoch()
    ui_responsiveness.freeze_long_lived_heap("unit-test")
    ui_responsiveness.refreeze_long_lived_heap("unit-test")
    ui_responsiveness.compact_long_lived_heap("unit-test")
    assert ui_responsiveness.heap_freeze_epoch() == start + 3


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
    (frozen while shown) survive a full collection. With the note, that
    maintenance compacts and every switched-away row is freed.
    """
    gc.unfreeze()
    gc.collect()
    app = ConsoleTranscriptApp()
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            refreezer = real_refreezer(min_growth=10_000_000)
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
            assert refreezer.maintain("conversation_painted") == "compact"
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
