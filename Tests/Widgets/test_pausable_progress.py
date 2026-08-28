# test_pausable_progress.py
# Description: Behavior coverage for PausableProgressBar/PausableLoadingIndicator (TASK-23022)
"""
TASK-23022: a hidden or off-screen indeterminate progress widget must not run
a timer. Textual's stock ``ProgressBar(total=None)`` arms a 15 Hz
``auto_refresh`` on its ``Bar`` plus an unconditional 1 Hz ETA sampler, and
``LoadingIndicator`` arms 16 Hz at mount; ``display = False`` stops none of
them (textual gates only the repaint on ``is_on_screen``, never the timer).
Measured on the Lab screen: 960 of 1018 timer fires in 15 s changed zero
pixels -- 88% of that screen's idle CPU.

These tests hold the house replacements to the full contract:

* hidden => ZERO timer fires (primary evidence is the fire count, matching
  the review's methodology; CPU is load-sensitive, fires are deterministic);
* shown  => the indeterminate animation still runs (fires occur AND the
  rendered highlight actually moves -- a paused-but-armed clock would pass a
  pure ``auto_refresh`` probe);
* hide/show cycling pauses and resumes;
* reactives changing while hidden (total -> None re-arms the Bar's clock in
  stock textual) cannot leak a running timer;
* a widget removed while hidden leaves no timer task behind, and app exit is
  clean -- pausing/resuming on visibility is exactly the lifecycle shape
  that has broken quit in this repo before.

Fire counting wraps ``Timer._tick`` and filters to timers owned by the
widgets under test, so unrelated app clocks cannot contaminate a count.
A stock-widget control asserts the harness measures (hidden stock bar MUST
fire), so a hidden-fires==0 assertion can never pass vacuously.
"""

from __future__ import annotations

import asyncio
from typing import Iterable

import pytest
from textual.app import App, ComposeResult
from textual.timer import Timer
from textual.widgets import LoadingIndicator, ProgressBar
from textual.widgets._progress_bar import Bar

from tldw_chatbook.Widgets.ModelArtifacts.install_progress import ModelInstallProgress
from tldw_chatbook.Widgets.pausable_progress import (
    PausableBar,
    PausableLoadingIndicator,
    PausableProgressBar,
)

#: Idle window long enough for a live 15-16 Hz clock to fire several times
#: even on a loaded machine, short enough to keep the suite quick.
WINDOW = 0.5

#: Minimum fires we accept as proof a 15-16 Hz clock is actually running
#: over WINDOW (nominal ~7; generous margin for scheduler jitter).
MIN_LIVE_FIRES = 3


class FireLog:
    """Counts Timer._tick invocations per target widget."""

    def __init__(self) -> None:
        self.fires: list[object] = []

    def count_for(self, *targets: object) -> int:
        return sum(1 for target in self.fires if target in targets)

    def count_for_types(self, *types: type) -> int:
        return sum(1 for target in self.fires if isinstance(target, types))


@pytest.fixture
def fire_log(monkeypatch: pytest.MonkeyPatch) -> FireLog:
    log = FireLog()
    original_tick = Timer._tick

    async def counting_tick(self: Timer, *, next_timer: float, count: int):
        try:
            target = self.target
        except Exception:  # pragma: no cover - target already gone
            target = None
        log.fires.append(target)
        return await original_tick(self, next_timer=next_timer, count=count)

    monkeypatch.setattr(Timer, "_tick", counting_tick)
    return log


def _live_timers(widgets: Iterable[object]) -> list[Timer]:
    """Every not-yet-stopped Timer owned by the given widgets."""
    return [
        timer
        for widget in widgets
        for timer in list(widget._timers)
        if timer._task is not None
    ]


def _all_paused(widgets: Iterable[object]) -> bool:
    timers = _live_timers(widgets)
    return bool(timers) and all(not t._active.is_set() for t in timers)


class _SingleWidgetApp(App[None]):
    def __init__(self, widget) -> None:
        super().__init__()
        self._widget = widget
        self.animation_level = "full"

    def compose(self) -> ComposeResult:
        yield self._widget


# ---------------------------------------------------------------------------
# harness control: the stock widget MUST fire while hidden, or fire counting
# proves nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_control_hidden_stock_progress_bar_fires(fire_log: FireLog) -> None:
    """Stock hidden indeterminate bar fires -- proves the harness measures."""
    bar = ProgressBar(total=None, show_eta=False)
    bar.display = False
    app = _SingleWidgetApp(bar)
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(WINDOW)
        inner = app.query_one(Bar)
        assert fire_log.count_for(bar, inner) >= MIN_LIVE_FIRES, (
            "the stock control stopped firing while hidden; if textual now "
            "pauses these clocks itself, the pausable shape may be obsolete -- "
            "re-evaluate TASK-23022 rather than deleting this control"
        )


# ---------------------------------------------------------------------------
# hidden => no fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hidden_indeterminate_pausable_bar_never_fires(
    fire_log: FireLog,
) -> None:
    bar = PausableProgressBar(total=None, show_eta=False)
    bar.display = False
    app = _SingleWidgetApp(bar)
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(WINDOW)
        inner = app.query_one(PausableBar)
        assert fire_log.count_for(bar, inner) == 0
        assert _all_paused([bar, inner])


@pytest.mark.asyncio
async def test_hidden_pausable_loading_indicator_never_fires(
    fire_log: FireLog,
) -> None:
    indicator = PausableLoadingIndicator()
    indicator.display = False
    app = _SingleWidgetApp(indicator)
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(WINDOW)
        assert fire_log.count_for(indicator) == 0
        assert _all_paused([indicator])


@pytest.mark.asyncio
async def test_indicator_hidden_by_ancestor_display_never_fires(
    fire_log: FireLog,
) -> None:
    """Hiding an ANCESTOR must silence the clock too (the CCP overlay shape)."""
    from textual.containers import Container

    class _AncestorApp(App[None]):
        def compose(self) -> ComposeResult:
            container = Container(PausableLoadingIndicator(id="inner"))
            container.display = False
            yield container

    app = _AncestorApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(WINDOW)
        indicator = app.query_one("#inner", PausableLoadingIndicator)
        assert fire_log.count_for(indicator) == 0
        assert _all_paused([indicator])


# ---------------------------------------------------------------------------
# shown => still animates (do not break the working case)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_visible_indeterminate_bar_fires_and_animates(
    fire_log: FireLog,
) -> None:
    bar = PausableProgressBar(total=None, show_eta=False)
    app = _SingleWidgetApp(bar)
    async with app.run_test() as pilot:
        await pilot.pause()
        inner = app.query_one(PausableBar)
        assert inner.auto_refresh == pytest.approx(1 / 15)
        first = inner.render_indeterminate().highlight_range
        await asyncio.sleep(WINDOW)
        second = inner.render_indeterminate().highlight_range
        assert fire_log.count_for(bar, inner) >= MIN_LIVE_FIRES
        # The highlight travels 30 cells/s; over WINDOW it must have moved.
        assert first != second


@pytest.mark.asyncio
async def test_visible_pausable_loading_indicator_fires(fire_log: FireLog) -> None:
    indicator = PausableLoadingIndicator()
    app = _SingleWidgetApp(indicator)
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(WINDOW)
        assert fire_log.count_for(indicator) >= MIN_LIVE_FIRES


# ---------------------------------------------------------------------------
# hide/show cycling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hide_stops_fires_and_show_resumes_them(fire_log: FireLog) -> None:
    bar = PausableProgressBar(total=None, show_eta=False)
    app = _SingleWidgetApp(bar)
    async with app.run_test() as pilot:
        await pilot.pause()
        inner = app.query_one(PausableBar)

        bar.display = False
        await pilot.pause()
        fire_log.fires.clear()
        await asyncio.sleep(WINDOW)
        assert fire_log.count_for(bar, inner) == 0

        bar.display = True
        await pilot.pause()
        fire_log.fires.clear()
        await asyncio.sleep(WINDOW)
        assert fire_log.count_for(bar, inner) >= MIN_LIVE_FIRES


@pytest.mark.asyncio
async def test_total_flips_while_hidden_cannot_leak_a_running_timer(
    fire_log: FireLog,
) -> None:
    """Stock Bar.watch_percentage re-arms 15 Hz on total=None -- even hidden.

    The re-armed timer goes through the intercepted ``set_interval``, so it
    must be born paused.
    """
    bar = PausableProgressBar(total=None, show_eta=False)
    bar.display = False
    app = _SingleWidgetApp(bar)
    async with app.run_test() as pilot:
        await pilot.pause()
        inner = app.query_one(PausableBar)
        bar.update(total=100, progress=25)  # determinate: Bar clock stops
        bar.update(total=None)  # indeterminate again: stock would re-arm LIVE
        await pilot.pause()
        fire_log.fires.clear()
        await asyncio.sleep(WINDOW)
        assert fire_log.count_for(bar, inner) == 0
        assert _all_paused([bar, inner])


# ---------------------------------------------------------------------------
# lifecycle: unmount while hidden, and app exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_while_hidden_leaves_no_timer_behind(fire_log: FireLog) -> None:
    bar = PausableProgressBar(total=None, show_eta=False)
    bar.display = False
    app = _SingleWidgetApp(bar)
    async with app.run_test() as pilot:
        await pilot.pause()
        inner = app.query_one(PausableBar)
        timers = [t for w in (bar, inner) for t in list(w._timers)]
        assert timers, "expected the framework to have armed clocks"
        await bar.remove()
        await pilot.pause()
        for timer in timers:
            assert timer._task is None or timer._task.done(), (
                f"timer {timer.name!r} still has a live task after remove()"
            )
        fire_log.fires.clear()
        await asyncio.sleep(0.2)
        assert fire_log.count_for(bar, inner) == 0


@pytest.mark.asyncio
async def test_app_exit_with_hidden_paused_widget_is_clean(
    fire_log: FireLog,
) -> None:
    bar = PausableProgressBar(total=None, show_eta=False)
    bar.display = False
    app = _SingleWidgetApp(bar)
    async with app.run_test() as pilot:
        await pilot.pause()
        inner = app.query_one(PausableBar)
        timers = [t for w in (bar, inner) for t in list(w._timers)]
        assert timers
    # run_test's context exit shuts the app down; every timer task must be
    # stopped, not left pending ("Task was destroyed but it is pending!").
    for timer in timers:
        assert timer._task is None or timer._task.done(), (
            f"timer {timer.name!r} survived app shutdown"
        )
    fire_log.fires.clear()
    await asyncio.sleep(0.2)
    assert fire_log.count_for(bar, inner) == 0


# ---------------------------------------------------------------------------
# integration: the shipped hidden-bar composite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_install_progress_is_silent_at_idle(fire_log: FireLog) -> None:
    """The six ModelInstallProgress embeds idle with ZERO progress-clock fires."""
    widget = ModelInstallProgress()
    app = _SingleWidgetApp(widget)
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(WINDOW)
        assert (
            fire_log.count_for_types(ProgressBar, Bar, LoadingIndicator) == 0
        )
        bar = app.query_one("#model-install-progress-bar", ProgressBar)
        assert isinstance(bar, PausableProgressBar)
        assert not bar.display


# ---------------------------------------------------------------------------
# structural parity with the stock widget (pins the copied compose against
# textual upstream drift)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"show_eta": False},
        {"show_eta": False, "show_percentage": False},
        {"show_bar": False},
    ],
)
async def test_pausable_progress_bar_matches_stock_structure(kwargs: dict) -> None:
    stock = ProgressBar(total=None, **kwargs)
    pausable = PausableProgressBar(total=None, **kwargs)

    class _PairApp(App[None]):
        def compose(self) -> ComposeResult:
            yield stock
            yield pausable

    app = _PairApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        stock_children = list(stock.children)
        pausable_children = list(pausable.children)
        assert len(stock_children) == len(pausable_children)
        for stock_child, pausable_child in zip(stock_children, pausable_children):
            assert isinstance(pausable_child, type(stock_child)), (
                f"PausableProgressBar composed {type(pausable_child).__name__} "
                f"where stock composes {type(stock_child).__name__}; "
                "ProgressBar.compose upstream has drifted -- update "
                "PausableProgressBar.compose to match"
            )
            assert pausable_child.id == stock_child.id
