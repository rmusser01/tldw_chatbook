"""task-22220 item 3: `InlineLoader`'s 0.5 s dots tick must die with the
loading state, not with the widget.

The pre-fix widget armed `set_interval(0.5, ...)` on mount, discarded the
handle, and `set_success`/`set_error`/`reset` never touched it -- one
immortal timer per mounted indicator, ticking for the widget's whole
mounted lifetime after the load finished. (The perf finding names the class
`InlineLoadingIndicator`; the cited lines -- `Widgets/loading_states.py`
233-266 -- are `InlineLoader`. The CCP `InlineLoadingIndicator` runs a
`@work` loop it already stops.)

Aliveness is asserted through the message pump's own timer set
(`widget._timers`, a stopped Timer has `_task is None` on Textual 8.2.8),
so the probes were expressible against the pre-fix tree: born red with one
alive timer surviving `set_success()`.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Widgets.loading_states import InlineLoader

pytestmark = pytest.mark.asyncio


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield InlineLoader()


def _alive_timers(widget) -> list:
    """Timers on the widget's pump whose asyncio task still exists."""
    return [timer for timer in widget._timers if timer._task is not None]


async def test_mount_in_loading_state_arms_exactly_one_tick():
    app = _Host()
    async with app.run_test() as pilot:
        loader = app.query_one(InlineLoader)
        assert len(_alive_timers(loader)) == 1, (
            "a loading indicator must animate while loading"
        )


async def test_set_success_stops_the_dots_tick():
    app = _Host()
    async with app.run_test() as pilot:
        loader = app.query_one(InlineLoader)
        assert _alive_timers(loader), "test premise: the tick armed on mount"

        loader.set_success()

        assert _alive_timers(loader) == [], (
            "the 0.5 s tick must stop when the loader reaches success"
        )


async def test_set_error_stops_the_dots_tick():
    app = _Host()
    async with app.run_test() as pilot:
        loader = app.query_one(InlineLoader)
        assert _alive_timers(loader), "test premise: the tick armed on mount"

        loader.set_error("boom")

        assert _alive_timers(loader) == [], (
            "the 0.5 s tick must stop when the loader reaches error"
        )


async def test_reset_rearms_the_tick_and_the_dots_animate_again():
    app = _Host()
    async with app.run_test() as pilot:
        loader = app.query_one(InlineLoader)
        loader.set_success()
        assert _alive_timers(loader) == []

        loader.reset()

        assert len(_alive_timers(loader)) == 1, (
            "reset back to loading must re-arm the tick"
        )
        # And the re-armed timer really drives the animation: poll (bounded)
        # until a dot lands rather than trusting a single fixed sleep.
        for _ in range(40):
            await pilot.pause(0.1)
            if str(loader.content) != loader.loading_text:
                break
        assert str(loader.content).startswith(f"{loader.loading_text}."), (
            f"the re-armed tick never animated; content stayed "
            f"{str(loader.content)!r}"
        )


async def test_repeated_cycles_do_not_accumulate_dead_timers():
    """A loading/success loop must not grow the pump's timer set -- a
    long-lived screen cycling an indicator would otherwise pile up stopped
    Timer objects for its whole lifetime."""
    app = _Host()
    async with app.run_test() as pilot:
        loader = app.query_one(InlineLoader)
        for _ in range(10):
            loader.set_success()
            loader.reset()
        assert len(loader._timers) <= 1, (
            f"{len(loader._timers)} Timer objects retained after 10 cycles"
        )
        assert len(_alive_timers(loader)) == 1


async def test_terminal_state_after_unmount_is_inert():
    """task-22220 teardown walk: a load finishing after its indicator was
    removed must neither raise nor arm anything on the dead pump."""
    app = _Host()
    async with app.run_test() as pilot:
        loader = app.query_one(InlineLoader)
        await loader.remove()

        loader.set_success()
        loader.reset()

        assert _alive_timers(loader) == [], (
            "no timer may run on an unmounted indicator"
        )
