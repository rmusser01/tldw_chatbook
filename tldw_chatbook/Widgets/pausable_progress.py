"""Progress indicators whose framework clocks stop while the widget is hidden.

TASK-23022. Textual's stock progress widgets arm repeating clocks that
``display = False`` does not stop:

* ``ProgressBar(total=None)`` makes the composed ``Bar`` indeterminate, which
  arms ``auto_refresh = 1/15`` -- a 15 Hz ``set_interval`` that runs forever.
  ``textual/dom.py`` gates only the *repaint* on ``is_on_screen`` (itself a
  ``find_widget`` raise/catch per fire), never the timer.
* ``ProgressBar.on_mount`` also arms an unconditional ``set_interval(1,
  self.update)`` (the ETA sampler), even with ``show_eta=False``.
* ``LoadingIndicator._on_mount`` arms ``auto_refresh = 1/16``.

Measured on the Lab screen: 960 of 1018 timer fires in 15 s changed zero
pixels -- 88% of that screen's idle CPU (see the 2026-08-27 holistic perf
review).

The classes here are drop-in replacements that pause every repeating clock
armed on the widget while it is outside the screen's layout map (``display``
false on itself or any ancestor), and resume them when it is shown again. A
*visible* indeterminate bar still animates. Scrolled-out-but-displayed
widgets keep their clocks, matching stock semantics.

Mechanism: ``set_interval`` is intercepted, so every clock the framework arms
-- including ones armed by base-class ``on_mount`` handlers that a subclass
cannot suppress (Textual dispatches naming-convention handlers for every
class in the MRO) -- is tracked and started paused until the widget's first
``Show`` event. A paused ``Timer``'s task blocks on its ``Event.wait()``:
zero wakeups, zero fires. ``Timer._skip`` (default) fast-forwards the
schedule on resume, so a long pause does not produce a catch-up burst, and
``Timer.stop()`` works from the paused state, so unmount/quit teardown
(``MessagePump._close_messages`` -> ``Timer._stop_all``) is unaffected.

A guard (``Tests/Architecture/test_progress_widget_clock_guard.py``) keeps
new code pointed here: constructing or subclassing the stock classes anywhere
else in the package fails it.
"""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.timer import Timer, TimerCallback
from textual.widgets import LoadingIndicator, ProgressBar
from textual.widgets._progress_bar import Bar, ETAStatus, PercentageStatus

__all__ = [
    "HiddenClocksPausedMixin",
    "PausableBar",
    "PausableLoadingIndicator",
    "PausableProgressBar",
]


class HiddenClocksPausedMixin:
    """Pause this widget's repeating clocks while it is not displayed.

    Mix in ahead of a ``Widget`` subclass. Every timer armed through
    ``set_interval`` (by any class in the MRO, including watchers and the
    ``auto_refresh`` setter) runs only while the widget is in the screen's
    layout map: timers are created paused until the first ``Show`` event,
    paused again on ``Hide``, and resumed on ``Show``.

    Contract: this mixin owns the run/pause state of every non-``pause=True``
    interval timer on the widget. A timer armed with ``pause=True`` is left
    alone entirely -- the caller keeps ownership of it.
    """

    #: Class-level default: clocks are considered stopped until the first
    #: ``Show`` event arrives (a widget mounted hidden never receives one).
    _clocks_running: bool = False

    def _tracked_clocks(self) -> list[Timer]:
        # STRONG references, deliberately. A paused Timer whose creator
        # discarded the return value (ProgressBar.on_mount does) is otherwise
        # only alive through the task<->timer reference cycle: the paused task
        # is blocked on the timer's own Event, so no event-loop root holds it,
        # cycle GC destroys it mid-pause ("Task was destroyed but it is
        # pending!"), and the clock would silently never resume on Show.
        # Observed with a WeakSet during TASK-23022 development.
        clocks = getattr(self, "_hidden_paused_clocks", None)
        if clocks is None:
            clocks = []
            self._hidden_paused_clocks = clocks
        return clocks

    def set_interval(
        self,
        interval: float,
        callback: TimerCallback | None = None,
        *,
        name: str | None = None,
        repeat: int = 0,
        pause: bool = False,
    ) -> Timer:
        """Arm an interval timer that runs only while the widget is shown."""
        timer = super().set_interval(  # type: ignore[misc]
            interval, callback, name=name, repeat=repeat, pause=pause
        )
        if not pause:
            self._tracked_clocks().append(timer)
            if not self._clocks_running:
                # Synchronous with creation: the timer's task cannot run
                # before control returns to the event loop, so a widget
                # mounted hidden never fires even once.
                timer.pause()
        return timer

    def _prune_stopped_clocks(self) -> list[Timer]:
        """Drop timers that were stopped (e.g. by the auto_refresh setter)."""
        clocks = self._tracked_clocks()
        clocks[:] = [timer for timer in clocks if timer._task is not None]
        return clocks

    def _on_show(self, event: events.Show) -> None:
        self._clocks_running = True
        for timer in self._prune_stopped_clocks():
            timer.resume()

    def _on_hide(self, event: events.Hide) -> None:
        self._clocks_running = False
        for timer in self._prune_stopped_clocks():
            timer.pause()


class PausableBar(HiddenClocksPausedMixin, Bar):
    """``Bar`` whose 15 Hz indeterminate clock stops while hidden.

    ``Bar.watch_percentage`` re-arms ``auto_refresh = 1/15`` whenever the
    bound percentage becomes ``None`` -- including while hidden. The arm goes
    through ``set_interval``, so the mixin intercepts it and the new timer is
    born paused; it resumes on the next ``Show``.
    """


class PausableProgressBar(HiddenClocksPausedMixin, ProgressBar):
    """Drop-in ``ProgressBar`` that is silent while hidden.

    While the widget (or an ancestor) is ``display: none`` neither the
    indeterminate 15 Hz refresh nor the 1 Hz ETA sampler fires. A visible
    indeterminate bar animates exactly like the stock widget.
    """

    def compose(self) -> ComposeResult:
        # Mirror of ProgressBar.compose (textual 8.2.8) with Bar swapped for
        # PausableBar. Textual offers no seam to substitute the Bar class.
        # Tests/Widgets/test_pausable_progress.py pins structural parity with
        # the stock widget so upstream drift fails loudly on a bump.
        if self.show_bar:
            yield (
                PausableBar(
                    id="bar", clock=self._clock, bar_renderable=self.BAR_RENDERABLE
                )
                .data_bind(ProgressBar.percentage)
                .data_bind(ProgressBar.gradient)
            )
        if self.show_percentage:
            yield PercentageStatus(id="percentage").data_bind(ProgressBar.percentage)
        if self.show_eta:
            yield ETAStatus(id="eta").data_bind(eta=ProgressBar._display_eta)


class PausableLoadingIndicator(HiddenClocksPausedMixin, LoadingIndicator):
    """Drop-in ``LoadingIndicator`` whose 16 Hz clock stops while hidden.

    The base arms ``auto_refresh = 1/16`` in ``_on_mount``; the mixin's
    ``set_interval`` interception pauses that timer at creation when the
    indicator is mounted hidden, and Show/Hide toggle it thereafter. The
    animation position derives from wall-clock time, so a resumed indicator
    continues mid-cycle rather than restarting.
    """
