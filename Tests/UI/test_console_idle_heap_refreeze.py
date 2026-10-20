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
