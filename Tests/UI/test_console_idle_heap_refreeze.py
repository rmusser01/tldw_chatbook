"""The Console re-freezes its long-lived heap only at idle points (TASK-33620.15.3).

Live profiling (2026-10-10) found ~70 ms of main-thread generation-2
collection in every send window: ADR-198 froze only the boot heap, so each
pass rescanned everything the Console built after boot. The Console now asks
the process's ``IdleHeapRefreezer`` for a re-freeze when its transcript shows
a different conversation and when a run settles (the transcript poll's stop
edge), and the request is dropped while a send is in flight.

This drives the real ChatScreen send path (the send-acknowledgement harness:
driver-delivered Enter, an in-memory ChaChaNotes store, a provider gateway
that holds validation open) with a recording refreezer in place of the real
collector calls.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.screen import Screen

from Tests.UI.test_console_send_acknowledgement import (
    REPLY,
    HeldAdmission,
    _painted_lines,
    build,
    eager_tasks,
    press,
    ready_console,
    until,
)
from tldw_chatbook.UI.Console_Modules import transcript as console_transcript
from tldw_chatbook.Utils import ui_responsiveness
from tldw_chatbook.Utils.ui_responsiveness import IdleHeapRefreezer

pytestmark = pytest.mark.bootstrap_profile


class BusySampler:
    """Reads the Console's busy check on every loop turn until stopped.

    Two probes can miss a gap between hand-offs (press to admission,
    admission to custody, custody to the transcript poll); a re-freeze that
    found one would land inside the send window.
    """

    def __init__(self, console) -> None:
        self.samples: list[bool] = []
        self._console = console
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while True:
            self.samples.append(console_transcript._console_heap_busy(self._console))
            await asyncio.sleep(0)

    def stop(self) -> None:
        self._task.cancel()


def sending(console) -> bool:
    """Whether the Console has a visible send in flight (its Enter was handled)."""
    flight = getattr(console, "_console_send_flight", None)
    return flight is not None and bool(flight.tasks)


def recording_refreezer(timeline) -> tuple[IdleHeapRefreezer, list[str]]:
    """A refreezer that marks the harness timeline instead of freezing."""
    reasons: list[str] = []

    def refreeze(reason: str) -> int:
        reasons.append(reason)
        timeline.mark(f"refreeze:{reason}")
        return 0

    refreezer = IdleHeapRefreezer(
        quiet_seconds=0.05,
        compaction_quiet_seconds=0.05,
        enabled=lambda: True,
        refreeze=refreeze,
        compact=lambda reason: refreeze(reason) or 1_000_000,
        freeze_count=lambda: 1_000_000,
    )
    return refreezer, reasons


async def test_console_refreezes_after_painting_a_conversation_and_after_a_run_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC#4/#5: re-freezes at the paint and the settle, never from Enter to the provider.

    Args:
        monkeypatch: Installs the recording refreezer as the process's.
    """
    host, gateway, timeline = build()
    refreezer, reasons = recording_refreezer(timeline)
    monkeypatch.setattr(ui_responsiveness, "_HEAP_REFREEZER", refreezer)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            await until(lambda: "conversation_painted" in reasons)

            held = HeldAdmission(console)
            timeline.mark("enter")
            press(host, "enter", "\r")
            await until(lambda: sending(console))
            sampler = BusySampler(console)
            # Before acceptance: only the send's own flight marks it.
            await until(held.entered.is_set)
            busy_in_admission = console_transcript._console_heap_busy(console)
            console_transcript.request_console_heap_refreeze(console, "admission_probe")
            await asyncio.sleep(0.3)  # well past the quiet period
            held.release.set()
            # Accepted, provider not yet called: the turn's custody marks it.
            await until(gateway.validation_started.is_set)
            busy_mid_send = console_transcript._console_heap_busy(console)
            console_transcript.request_console_heap_refreeze(console, "mid_send_probe")
            await asyncio.sleep(0.3)
            gateway.validation_release.set()
            await until(lambda: gateway.stream_calls == 1)
            sampler.stop()
            await until(lambda: REPLY in "\n".join(_painted_lines(host)))
            await until(lambda: "run_settled" in reasons, timeout=20)
            idle_after_the_run = console_transcript._console_heap_busy(console)

            await host.push_screen(Screen())
            await pilot.pause()
            busy_under_another_screen = console_transcript._console_heap_busy(console)

    assert busy_in_admission, "a send being admitted did not hold the re-freeze off"
    assert busy_mid_send, "an accepted send did not hold the re-freeze off"
    assert len(sampler.samples) > 20, sampler.samples
    gaps = [index for index, busy in enumerate(sampler.samples) if not busy]
    assert gaps == [], (
        f"the Console read idle on {len(gaps)} of {len(sampler.samples)} loop "
        "turns between Enter and the provider call"
    )
    assert "admission_probe" not in reasons
    assert "mid_send_probe" not in reasons
    assert not idle_after_the_run
    assert busy_under_another_screen
    enter, provider = timeline.index("enter"), timeline.index("provider")
    inside = [
        value
        for index, (kind, value) in enumerate(timeline.entries)
        if kind == "event"
        and str(value).startswith("refreeze:")
        and enter < index < provider
    ]
    assert inside == []
    assert timeline.index("refreeze:run_settled") > provider


class BusyScreen:
    """Only what the busy check reads, every part idle until a case sets one."""

    def __init__(self) -> None:
        self._closing = False
        self._closed = False
        self.app_instance = None
        self.app = SimpleNamespace(screen=self)
        self._console_send_flight = SimpleNamespace(tasks=set(), deferred=None)
        self._console_pending_send = None
        self._console_transcript_sync_timer = None
        self.poll_needed = False

    def _console_transcript_poll_needed(self) -> bool:
        return self.poll_needed


def _closing(screen: BusyScreen) -> None:
    screen._closing = True


def _covered(screen: BusyScreen) -> None:
    screen.app.screen = object()  # a modal or another screen on top


def _pressed(screen: BusyScreen) -> None:
    screen._console_send_flight.tasks.add(object())


def _deferred(screen: BusyScreen) -> None:
    screen._console_send_flight.deferred = object()


def _pending(screen: BusyScreen) -> None:
    screen._console_pending_send = object()


def _polling(screen: BusyScreen) -> None:
    screen._console_transcript_sync_timer = object()


def _live_before_the_poll(screen: BusyScreen) -> None:
    screen.poll_needed = True


@pytest.mark.parametrize(
    "hold",
    [
        _closing,
        _covered,
        _pressed,
        _deferred,
        _pending,
        _polling,
        _live_before_the_poll,
    ],
    ids=lambda hold: hold.__name__.strip("_"),
)
def test_each_part_of_the_send_window_alone_holds_the_refreeze_off(hold) -> None:
    """AC#4: every state from Enter to a settled run reads busy on its own.

    The mounted test above walks one send, where several parts overlap, so a
    part that only one path relies on (a run still live before its transcript
    poll starts, a fleet wake delivering, a change review pending) could be
    dropped with that test still green. Each case sets one part alone.

    Args:
        hold: Sets one part of the busy state on an otherwise idle screen.
    """
    idle = BusyScreen()
    assert not console_transcript._console_heap_busy(idle), "the idle screen read busy"
    screen = BusyScreen()
    hold(screen)
    assert console_transcript._console_heap_busy(screen)


class PaintedScreen:
    """Only what the paint policy reads: the (session, conversation) shown."""

    def __init__(self) -> None:
        self._console_chat_store = SimpleNamespace(active_session_id="s1")
        self.conversation = "c1"
        self.timers: list[tuple[float, object]] = []

    def _current_console_conversation_id(self) -> str:
        return self.conversation

    def set_timer(self, delay: float, callback) -> None:
        self.timers.append((delay, callback))


def test_switching_away_from_a_transcript_frozen_while_shown_owes_a_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Console notes stranded garbage only when the outgoing rows were frozen.

    The transcript is rebuilt on every switch, so rows a freeze caught while
    shown become garbage no collection reaches. Rows replaced before any
    freeze are ordinary garbage, which the re-freeze's own full collection
    reclaims.

    Args:
        monkeypatch: Installs a recording refreezer and a settable freeze epoch.
    """
    epoch = [0]
    monkeypatch.setattr(console_transcript, "heap_freeze_epoch", lambda: epoch[0])
    refreezer = IdleHeapRefreezer(
        enabled=lambda: True,
        refreeze=lambda reason: 0,
        compact=lambda reason: 1_000_000,
        freeze_count=lambda: 1_000_000,
        min_growth=10_000_000,
    )
    monkeypatch.setattr(ui_responsiveness, "_HEAP_REFREEZER", refreezer)
    screen = PaintedScreen()

    console_transcript.note_console_conversation_painted(screen)
    assert len(screen.timers) == 1, "the first paint asked for no re-freeze"
    console_transcript.note_console_conversation_painted(screen)
    assert len(screen.timers) == 1, "a repaint of the same conversation asked again"

    screen.conversation = "c2"  # switched before anything froze c1's rows
    console_transcript.note_console_conversation_painted(screen)
    assert len(screen.timers) == 2
    assert not refreezer.compaction_due(), "unfrozen rows were reported stranded"

    epoch[0] += 1  # a re-freeze ran while c2 was shown
    screen._console_chat_store.active_session_id = "s2"  # another tab
    console_transcript.note_console_conversation_painted(screen)
    assert len(screen.timers) == 3
    assert refreezer.compaction_due(), "rows frozen while shown were left stranded"


def test_a_new_chat_gaining_its_conversation_id_is_not_a_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first send that persists a new chat keeps its rows: nothing is stranded.

    The (session, conversation) key goes from ``(s1, None)`` to ``(s1, id)``
    when a new chat's first send persists it, but the transcript matches rows
    by message id, so no row is replaced. Noting that as stranded garbage
    made the next idle moment pay a full-heap compaction for nothing. The
    freeze epoch recorded while the new chat was shown carries over, so a
    later real switch still owes the compaction for rows frozen back then.

    Args:
        monkeypatch: Installs a recording refreezer and a settable freeze epoch.
    """
    epoch = [0]
    monkeypatch.setattr(console_transcript, "heap_freeze_epoch", lambda: epoch[0])
    refreezer = IdleHeapRefreezer(
        enabled=lambda: True,
        refreeze=lambda reason: 0,
        compact=lambda reason: 1_000_000,
        freeze_count=lambda: 1_000_000,
        min_growth=10_000_000,
    )
    monkeypatch.setattr(ui_responsiveness, "_HEAP_REFREEZER", refreezer)
    screen = PaintedScreen()
    screen.conversation = None  # a new chat, not yet persisted

    console_transcript.note_console_conversation_painted(screen)
    epoch[0] += 1  # a re-freeze ran while the new chat was shown
    screen.conversation = "c1"  # its first send persisted it
    console_transcript.note_console_conversation_painted(screen)
    assert not refreezer.compaction_due(), "persisting a new chat stranded its rows"

    screen.conversation = "c2"  # a real switch in the same tab
    console_transcript.note_console_conversation_painted(screen)
    assert refreezer.compaction_due(), (
        "the rows frozen while the new chat was shown were left stranded"
    )
