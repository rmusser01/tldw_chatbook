from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest
from textual.app import App, ComposeResult
from textual.command import CommandList, CommandPalette, Hit, Provider
from textual.pilot import Pilot
from textual.widgets import Static


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float = 1.0) -> None:
        self.now += seconds


class PaletteProbe:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.release_batch = asyncio.Event()
        self.release_late = asyncio.Event()
        self.batch_waiting = asyncio.Event()
        self.late_waiting = asyncio.Event()
        self.cancelled = asyncio.Event()

    def callback(self, name: str) -> Callable[[], None]:
        return lambda: self.calls.append(name)


PROBE: PaletteProbe


async def wait_event(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=1.0)


async def wait_until(
    pilot: Pilot[None],
    predicate: Callable[[], bool],
    *,
    state: Callable[[], object],
    attempts: int = 50,
) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()
    pytest.fail(
        f"mounted palette condition was not reached after {attempts} pauses; "
        f"state={state()!r}"
    )


class ControlledProvider(Provider):
    async def search(self, query: str) -> AsyncIterator[Hit]:
        if query != "logs":
            return
        try:
            yield Hit(0.90, "first", PROBE.callback("first"))
            yield Hit(0.80, "second", PROBE.callback("second"))
            PROBE.batch_waiting.set()
            await wait_event(PROBE.release_batch)
            yield Hit(0.70, "batch", PROBE.callback("batch"))
            PROBE.late_waiting.set()
            await wait_event(PROBE.release_late)
            yield Hit(0.60, "late", PROBE.callback("late"))
        except asyncio.CancelledError:
            PROBE.cancelled.set()
            raise


class PaletteHarness(App[None]):
    COMMANDS = {ControlledProvider}

    def __init__(self, palette_type: type[CommandPalette]) -> None:
        super().__init__()
        self.palette_type = palette_type

    def compose(self) -> ComposeResult:
        yield Static("calling screen", id="calling-screen")

    def on_mount(self) -> None:
        self.push_screen(self.palette_type(id="--command-palette"))


@pytest.mark.asyncio
async def test_stock_palette_refresh_resets_a_navigated_highlight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(CommandPalette)
    async with app.run_test() as pilot:
        try:
            await pilot.press("l", "o", "g", "s")
            await wait_event(PROBE.batch_waiting)
            clock.advance()
            PROBE.release_batch.set()

            command_list = app.screen.query_one(CommandList)
            await wait_until(
                pilot,
                lambda: command_list.option_count == 3,
                state=lambda: {
                    "option_count": command_list.option_count,
                    "highlighted": command_list.highlighted,
                    "late_waiting": PROBE.late_waiting.is_set(),
                    "calls": PROBE.calls,
                },
            )
            assert PROBE.late_waiting.is_set()

            await pilot.press("down")
            assert command_list.highlighted == 1
            assert PROBE.calls == []

            clock.advance()
            PROBE.release_late.set()
            await wait_until(
                pilot,
                lambda: command_list.option_count == 4,
                state=lambda: {
                    "option_count": command_list.option_count,
                    "highlighted": command_list.highlighted,
                    "calls": PROBE.calls,
                },
            )
            assert command_list.highlighted == 0

            await pilot.press("enter")
            await wait_until(
                pilot,
                lambda: not CommandPalette.is_open(app),
                state=lambda: {
                    "screen": type(app.screen).__name__,
                    "calls": PROBE.calls,
                },
            )
            await wait_until(
                pilot,
                lambda: PROBE.calls == ["first"],
                state=lambda: {"calls": PROBE.calls},
            )
            assert PROBE.calls == ["first"]
        finally:
            PROBE.release_batch.set()
            PROBE.release_late.set()
