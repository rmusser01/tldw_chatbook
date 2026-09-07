from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from importlib.util import find_spec
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.command import CommandList, CommandPalette, Hit, Provider
from textual.pilot import Pilot
from textual.widgets import Static
from textual.widgets.option_list import Option

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.stable_command_palette import StableCommandPalette


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


def test_stable_palette_api_exists() -> None:
    assert find_spec("tldw_chatbook.UI.stable_command_palette") is not None


def test_tldw_cli_constructs_the_stable_palette(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = MagicMock()
    app.use_command_palette = True
    monkeypatch.setattr(StableCommandPalette, "is_open", lambda _app: False)

    assert "action_command_palette" in TldwCli.__dict__
    TldwCli.action_command_palette(app)

    palette = app.push_screen.call_args.args[0]
    assert type(palette) is StableCommandPalette
    assert palette.id == "--command-palette"


@pytest.mark.parametrize("enabled, already_open", [(False, False), (True, True)])
def test_tldw_cli_does_not_open_a_duplicate_or_disabled_palette(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    already_open: bool,
) -> None:
    app = MagicMock()
    app.use_command_palette = enabled
    monkeypatch.setattr(
        StableCommandPalette,
        "is_open",
        lambda _app: already_open,
    )

    TldwCli.action_command_palette(app)

    app.push_screen.assert_not_called()


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


@pytest.mark.asyncio
async def test_stable_palette_runs_the_navigated_command_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(StableCommandPalette)
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
            clock.advance()
            PROBE.release_late.set()
            await wait_until(
                pilot,
                lambda: PROBE.cancelled.is_set() or command_list.option_count == 4,
                state=lambda: {
                    "cancelled": PROBE.cancelled.is_set(),
                    "option_count": command_list.option_count,
                    "highlighted": command_list.highlighted,
                    "calls": PROBE.calls,
                },
            )
            assert PROBE.cancelled.is_set()
            assert command_list.option_count == 3
            assert command_list.highlighted == 1

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
                lambda: PROBE.calls == ["second"],
                state=lambda: {"calls": PROBE.calls},
            )
            assert PROBE.calls == ["second"]

            await pilot.pause()
            assert PROBE.calls == ["second"]
            assert not CommandPalette.is_open(app)
        finally:
            PROBE.release_batch.set()
            PROBE.release_late.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key", ["down", "up", "pageup", "pagedown", "ctrl+home", "ctrl+end"]
)
async def test_navigation_before_first_result_does_not_cancel_gathering(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(StableCommandPalette)
    async with app.run_test() as pilot:
        try:
            await pilot.press("l", "o", "g", "s")
            await wait_event(PROBE.batch_waiting)
            command_list = app.screen.query_one(CommandList)
            assert command_list.option_count == 0

            await pilot.press(key)
            assert not PROBE.cancelled.is_set()
            clock.advance()
            PROBE.release_batch.set()
            await wait_until(
                pilot,
                lambda: command_list.option_count == 3,
                state=lambda: {
                    "key": key,
                    "option_count": command_list.option_count,
                    "cancelled": PROBE.cancelled.is_set(),
                },
            )
            assert not PROBE.cancelled.is_set()
        finally:
            PROBE.release_batch.set()
            PROBE.release_late.set()


@pytest.mark.asyncio
async def test_navigation_during_stale_no_matches_does_not_cancel_new_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(StableCommandPalette)
    async with app.run_test() as pilot:
        try:
            palette = app.screen
            assert isinstance(palette, StableCommandPalette)
            command_list = palette.query_one(CommandList)
            command_list.clear_options().add_option(
                Option("No matches found", disabled=True, id=palette._NO_MATCHES)
            )
            palette._list_visible = True

            replacement_worker = palette._gather_commands("logs")
            palette._action_command_list("cursor_up")
            assert not replacement_worker.is_cancelled

            await wait_event(PROBE.batch_waiting)
            clock.advance()
            PROBE.release_batch.set()
            await wait_until(
                pilot,
                lambda: command_list.option_count == 3,
                state=lambda: {
                    "option_count": command_list.option_count,
                    "cancelled": PROBE.cancelled.is_set(),
                    "worker_cancelled": replacement_worker.is_cancelled,
                },
            )
            assert not PROBE.cancelled.is_set()
        finally:
            PROBE.release_batch.set()
            PROBE.release_late.set()


@pytest.mark.asyncio
async def test_settled_multi_hit_selection_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(StableCommandPalette)
    async with app.run_test() as pilot:
        try:
            await pilot.press("l", "o", "g", "s")
            await wait_event(PROBE.batch_waiting)
            clock.advance()
            PROBE.release_batch.set()
            await wait_event(PROBE.late_waiting)
            clock.advance()
            PROBE.release_late.set()

            command_list = app.screen.query_one(CommandList)
            await wait_until(
                pilot,
                lambda: command_list.option_count == 4,
                state=lambda: {
                    "option_count": command_list.option_count,
                    "calls": PROBE.calls,
                },
            )

            await pilot.press("down", "enter")
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
                lambda: PROBE.calls == ["second"],
                state=lambda: {"calls": PROBE.calls},
            )
            await pilot.pause()
            assert PROBE.calls == ["second"]
        finally:
            PROBE.release_batch.set()
            PROBE.release_late.set()


@pytest.mark.asyncio
async def test_escape_closes_without_running_a_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(StableCommandPalette)
    async with app.run_test() as pilot:
        try:
            await pilot.press("l", "o", "g", "s")
            await wait_event(PROBE.batch_waiting)
            clock.advance()
            PROBE.release_batch.set()
            await wait_until(
                pilot,
                lambda: app.screen.query_one(CommandList).option_count == 3,
                state=lambda: {
                    "option_count": app.screen.query_one(CommandList).option_count,
                    "calls": PROBE.calls,
                },
            )

            await pilot.press("escape")
            await wait_until(
                pilot,
                lambda: not CommandPalette.is_open(app),
                state=lambda: {
                    "screen": type(app.screen).__name__,
                    "calls": PROBE.calls,
                },
            )
            await pilot.pause()
            assert PROBE.calls == []
        finally:
            PROBE.release_batch.set()
            PROBE.release_late.set()
