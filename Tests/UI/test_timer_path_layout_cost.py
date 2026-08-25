"""Layout cost of repeating-clock repaints (TASK-21595).

``Static.update(content)`` ends in ``self.refresh(layout=layout)`` with
``layout: bool = True``, so a repaint on a timer arms a whole
``Screen._refresh_layout`` / ``Compositor.reflow`` every tick unless the caller
opts out. TASK-21692 (Console composer cursor blink) and TASK-21134 item 7
(media-viewer match-nav) each found one instance; this module covers the two
animation-rate instances the TASK-21595 census turned up:

* ``SplashScreen._update_animation`` -- 10-100 fps during startup.
* ``PersonaBuddyWidget._paint_frame`` -- the pet's own frame clock, >= 10 fps
  for as long as an animated buddy is on screen.

Two kinds of assertion per site:

1. **Cost** -- driven ticks must add no screen layout passes over a *measured*
   idle floor (not a bare ``== 0``, which an unrelated timer could turn into a
   flake), plus a paint assertion so a repaint that stopped happening cannot
   score zero and pass.
2. **Geometry equivalence** -- painting a given content with ``layout=False``
   must land the *same* geometry that painting it with ``layout=True`` does.
   That is an A/B against the real layout engine, not an inspection of the
   stylesheet; per TASK-21692, asserting ``outer_size`` alone is not enough, so
   the witness also carries the container size, the region, every sibling's
   region, and the painted per-row cell widths.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.geometry import Size
from textual.widget import Widget
from textual.widgets import Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddySnapshot
from tldw_chatbook.Persona_Buddy.preferences import PersonaBuddyPreferences
from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
    PersonaBuddyWidget,
)
from tldw_chatbook.Widgets.splash_screen import SplashScreen

VIEWPORT = (120, 40)


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------


class _LayoutCounter:
    """Count real layout work performed by the screen.

    Counts ``Screen._refresh_layout`` (the call that reflows the whole
    compositor) and ``Compositor.reflow``, rather than asserting on the
    arguments a caller passed to ``refresh`` -- the claim under test is about
    work performed, not about how it was requested.
    """

    def __init__(self, screen) -> None:
        self._screen = screen
        self.layout = 0
        self.reflow = 0
        self.seconds = 0.0
        self._undo: list[Callable[[], None]] = []

    def __enter__(self) -> "_LayoutCounter":
        screen = self._screen
        real_layout = screen._refresh_layout
        compositor = screen._compositor
        real_reflow = compositor.reflow

        def counting_layout(*args, **kwargs):
            self.layout += 1
            started = time.perf_counter()
            try:
                return real_layout(*args, **kwargs)
            finally:
                self.seconds += time.perf_counter() - started

        def counting_reflow(*args, **kwargs):
            self.reflow += 1
            return real_reflow(*args, **kwargs)

        screen._refresh_layout = counting_layout
        compositor.reflow = counting_reflow
        self._undo = [
            lambda: delattr(screen, "_refresh_layout"),
            lambda: delattr(compositor, "reflow"),
        ]
        return self

    def __exit__(self, *exc: Any) -> bool:
        for undo in self._undo:
            undo()
        return False

    def __repr__(self) -> str:  # pragma: no cover - only on failure
        return (
            f"layout={self.layout} reflow={self.reflow} "
            f"ms={self.seconds * 1000:.2f}"
        )


async def _settle(pilot) -> None:
    """Let every scheduled layout/refresh message drain."""
    await pilot.pause()
    await pilot.pause()


async def count_layout_passes(
    pilot, screen, rounds: int, tick: Callable[[], None] | None
) -> _LayoutCounter:
    """Drive ``rounds`` settles, calling ``tick`` before each when given."""
    with _LayoutCounter(screen) as counter:
        for _ in range(rounds):
            if tick is not None:
                tick()
            await _settle(pilot)
    return counter


def _strips(widget: Widget):
    try:
        return widget.render_lines(widget.region.reset_offset)
    except Exception:  # pragma: no cover - unmounted / zero-size
        return []


def _painted_rows(widget: Widget) -> list[int]:
    """Per-row painted cell widths for ``widget``, from the real compositor."""
    return [strip.cell_length for strip in _strips(widget)]


def _painted_text(widget: Widget) -> str:
    """What the compositor actually paints, used to prove a repaint landed."""
    return "\n".join(strip.text for strip in _strips(widget))


def _witness(widget: Widget, witnesses: Sequence[Widget]) -> tuple:
    """Everything about the layout that a repaint must not be able to move."""
    return (
        widget.outer_size,
        widget.container_size,
        widget.content_size,
        widget.region,
        widget.scrollable_content_region,
        tuple(w.region for w in witnesses),
        tuple(w.outer_size for w in witnesses),
        tuple(_painted_rows(widget)),
    )


# A content that is deliberately unlike every probe shape, used to wipe the
# geometry the forced-layout arm produced. Without it the ``layout=False`` arm
# would inherit the correct geometry from the arm before it and the test would
# pass vacuously.
_SCRUB = Text("\n".join("scrub" * 40 for _ in range(30)))


async def assert_geometry_is_content_independent(
    pilot,
    target: Static,
    contents: dict[str, Any],
    witnesses: Sequence[Widget],
) -> None:
    """Painting with ``layout=False`` must land the same geometry as with True.

    For each content: paint it with a forced layout and record the witness,
    scrub the geometry with a deliberately different shape (also with a forced
    layout), then paint the same content with ``layout=False`` and record
    again. Equal witnesses mean the layout pass was doing no work *for this
    widget's box* -- which is exactly the claim ``layout=False`` makes.
    """
    failures: list[str] = []
    for name, content in contents.items():
        target.update(content, layout=True)
        await _settle(pilot)
        with_layout = _witness(target, witnesses)
        painted_with_layout = _painted_text(target)

        target.update(_SCRUB, layout=True)
        await _settle(pilot)
        painted_scrub = _painted_text(target)

        target.update(content, layout=False)
        await _settle(pilot)
        without_layout = _witness(target, witnesses)
        painted_without_layout = _painted_text(target)

        if with_layout != without_layout:
            failures.append(
                f"{name!r}: layout=False geometry differs\n"
                f"    layout=True  -> {with_layout}\n"
                f"    layout=False -> {without_layout}"
            )
        # Sanity 1: the scrub has to actually displace the content, otherwise
        # the `layout=False` arm inherits the first arm's state and the
        # equality above is vacuous.
        if painted_scrub == painted_with_layout:
            failures.append(
                f"{name!r}: the scrub painted the same thing as the content -- "
                "this comparison is not discriminating"
            )
        # Sanity 2: the `layout=False` write has to reach the surface. A
        # `Static.update` that silently no-op'd would satisfy the geometry
        # equality trivially.
        if painted_without_layout != painted_with_layout:
            failures.append(
                f"{name!r}: layout=False painted different content than "
                f"layout=True -- the repaint half is broken"
            )
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------
# SplashScreen -- the animation frame repaint
# --------------------------------------------------------------------------


class _SplashHost(ConsolidatedCSSApp):
    """Production stylesheet stack.

    A bare ``App`` loads none of it, and even the app bundle alone is not
    enough for every widget: ``#persona-buddy-frame`` (below) has **zero**
    rules in ``tldw_cli_modular.tcss`` -- its geometry lives in the generated
    ``widget_defaults_self.tcss``, which only ``ConsolidatedCSSApp`` registers.
    A first draft of this module used the app bundle alone and a mutation that
    made the buddy frame content-sized SURVIVED, because the widget was
    mounting unstyled. Geometry conclusions need the whole stack.
    """

    CSS_PATH = BUNDLED_STYLESHEET

    def __init__(self, card: str) -> None:
        super().__init__()
        self._card = card

    def compose(self) -> ComposeResult:
        yield SplashScreen(
            card_name=self._card,
            duration=0.0,
            show_progress=True,
            skip_on_keypress=False,
        )


# One card per rendering shape the effect registry produces: a full-viewport
# character field, a glitch overlay of the static art, and a sparse starfield.
SPLASH_CARDS = ("matrix", "glitch", "starfield")


@pytest.mark.asyncio
@pytest.mark.parametrize("card", SPLASH_CARDS)
async def test_splash_animation_tick_arms_no_layout_pass(card: str) -> None:
    """An animation frame costs no more layout work than an idle settle.

    TASK-21595. The shipped cards run ``animation_speed`` between 0.01 s and
    0.1 s, so before the fix this was 10-100 whole-screen layout passes per
    second for the entire splash, during startup.
    """
    host = _SplashHost(card)
    async with host.run_test(size=VIEWPORT) as pilot:
        await pilot.pause()
        splash = host.query_one(SplashScreen)
        display = splash.query_one("#splash-display", Static)
        assert splash.effect_handler is not None, f"{card} did not start an effect"
        # Own the clock: the free-running timer must not add uncounted ticks
        # in the middle of a measurement.
        assert splash.animation_timer is not None
        splash.animation_timer.pause()
        await _settle(pilot)

        rounds = 6
        idle = await count_layout_passes(pilot, host.screen, rounds, None)
        ticking = await count_layout_passes(
            pilot, host.screen, rounds, splash._update_animation
        )

        assert ticking.layout <= idle.layout, (
            f"{card}: {rounds} animation ticks cost "
            f"{ticking.layout - idle.layout} extra screen layout passes "
            f"(idle floor {idle.layout}); ticking={ticking!r} idle={idle!r}"
        )
        assert ticking.reflow <= idle.reflow, (
            f"{card}: {rounds} animation ticks cost "
            f"{ticking.reflow - idle.reflow} extra compositor reflows "
            f"(idle floor {idle.reflow})"
        )

        # The tick must still REPAINT -- an animation that stopped advancing
        # would also score zero layout passes.
        before = display.render_lines(display.region.reset_offset)
        moved = False
        for _ in range(12):
            splash._update_animation()
            await _settle(pilot)
            if display.render_lines(display.region.reset_offset) != before:
                moved = True
                break
        assert moved, f"{card}: the animation stopped painting new frames"


@pytest.mark.asyncio
async def test_splash_display_geometry_is_content_independent() -> None:
    """``#splash-display`` cannot be sized by its content, so skipping the
    layout pass cannot change where anything lands.

    Both sheets that select it pin ``width: 100%; height: 100%``
    (``features/_splash.tcss``, ``components/_settings_splash_theme.tcss``),
    and the progress bar and progress label share the splash's vertical
    budget -- so they are carried as witnesses too.
    """
    host = _SplashHost("matrix")
    async with host.run_test(size=VIEWPORT) as pilot:
        await pilot.pause()
        splash = host.query_one(SplashScreen)
        display = splash.query_one("#splash-display", Static)
        if splash.animation_timer is not None:
            splash.animation_timer.pause()
        await _settle(pilot)

        width = display.size.width
        height = display.size.height
        assert width > 0 and height > 0
        witnesses = [splash, *splash.children]

        await assert_geometry_is_content_independent(
            pilot,
            display,
            {
                "empty": "",
                "one-row": "frame",
                "exactly-viewport-width": "x" * width,
                "one-past-viewport-width": "x" * (width + 1),
                "full-field": "\n".join("." * width for _ in range(height)),
                "taller-than-viewport": "\n".join(
                    "." * width for _ in range(height * 2)
                ),
                "cjk-double-width": "漢" * width,
                "rich-text": Text("\n".join("styled" for _ in range(height))),
            },
            witnesses,
        )


# --------------------------------------------------------------------------
# PersonaBuddyWidget -- the pet frame repaint
# --------------------------------------------------------------------------


class _StubBuddyController:
    """Minimal stand-in carrying the real preferences dataclass.

    ``_paint_frame`` reads widget state (classes, ``_snapshot``,
    ``_accepted_render``) rather than the controller; the controller is only
    consulted by ``refresh_from_controller``, which is equality-gated by
    ``_painted_authority`` and therefore is not the tick that repaints. The
    frame clock (``advance_frame``) calls ``_paint_frame`` directly, and that
    is the path measured here.
    """

    def __init__(self) -> None:
        self._preferences = PersonaBuddyPreferences(enabled=True)
        self._snapshot = PersonaBuddySnapshot(
            generation=1, selection=None, state="idle", enabled=True
        )

    def snapshot(self) -> PersonaBuddySnapshot:
        return self._snapshot

    def current_preferences(self) -> PersonaBuddyPreferences:
        return self._preferences


class _BuddyHost(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET

    def compose(self) -> ComposeResult:
        yield PersonaBuddyWidget(
            controller=_StubBuddyController(),
            view_generation=1,
            reconcile=lambda: None,
            is_current=lambda _widget: True,
        )


@pytest.mark.asyncio
async def test_persona_buddy_frame_tick_arms_no_layout_pass() -> None:
    """A pet frame repaint costs no more layout work than an idle settle.

    TASK-21595. ``advance_frame`` re-arms itself at the visual's own frame
    duration (>= 10 fps by default), and unlike ``refresh_from_controller`` it
    is not equality-gated -- advancing the frame index is the point -- so every
    frame armed a full screen layout pass while a buddy was visible.
    """
    host = _BuddyHost()
    async with host.run_test(size=VIEWPORT) as pilot:
        await pilot.pause()
        buddy = host.query_one(PersonaBuddyWidget)
        buddy._stop_frame_timer()
        if buddy._poll_timer is not None:
            buddy._poll_timer.pause()
        await _settle(pilot)

        rounds = 6
        idle = await count_layout_passes(pilot, host.screen, rounds, None)
        ticking = await count_layout_passes(
            pilot, host.screen, rounds, buddy._paint_frame
        )

        assert ticking.layout <= idle.layout, (
            f"{rounds} pet frame repaints cost {ticking.layout - idle.layout} "
            f"extra screen layout passes (idle floor {idle.layout}); "
            f"ticking={ticking!r} idle={idle!r}"
        )

        # The repaint must still reach the surface -- a `_paint_frame` that
        # silently stopped writing would also score zero layout passes.
        target = buddy.query_one("#persona-buddy-frame", Static)
        target.update("stale", layout=False)
        await _settle(pilot)
        buddy._paint_frame()
        await _settle(pilot)
        assert "stale" not in str(target.renderable), (
            "_paint_frame no longer repaints the frame surface"
        )


@pytest.mark.asyncio
async def test_persona_buddy_frame_geometry_is_content_independent() -> None:
    """``#persona-buddy-frame`` is pinned 100%/100% on its own layer, so no
    frame content can resize it -- empty, alert copy, or a multi-row pet."""
    host = _BuddyHost()
    async with host.run_test(size=VIEWPORT) as pilot:
        await pilot.pause()
        buddy = host.query_one(PersonaBuddyWidget)
        buddy._stop_frame_timer()
        if buddy._poll_timer is not None:
            buddy._poll_timer.pause()
        await _settle(pilot)

        target = buddy.query_one("#persona-buddy-frame", Static)
        width = max(target.size.width, 1)
        height = max(target.size.height, 1)
        witnesses = [buddy, *buddy.children]

        await assert_geometry_is_content_independent(
            pilot,
            target,
            {
                "empty": "",
                "alert-copy": "Persona unavailable",
                "one-row-at-width": "x" * width,
                "one-past-width": "x" * (width + 1),
                "full-box": "\n".join("." * width for _ in range(height)),
                "taller-than-box": "\n".join(
                    "." * width for _ in range(height * 3)
                ),
                "cjk-double-width": "漢" * width,
                "rich-text": Text("\n".join("pet" for _ in range(height))),
            },
            witnesses,
        )


# --------------------------------------------------------------------------
# the probe itself must be able to fail
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_geometry_equivalence_probe_catches_a_content_sized_widget() -> None:
    """A control: on a ``height: auto`` widget the probe must go red.

    Without this, a probe that silently measured nothing would make every
    ``layout=False`` above look justified. The splash display is pinned; this
    mounts an intentionally content-sized sibling and shows the same helper
    rejects it.
    """

    class _AutoHost(App):
        CSS = """
        #auto-sized { width: auto; height: auto; }
        #below { height: 3; }
        """

        def compose(self) -> ComposeResult:
            yield Static("seed", id="auto-sized")
            yield Static("below", id="below")

    host = _AutoHost()
    async with host.run_test(size=VIEWPORT) as pilot:
        await pilot.pause()
        target = host.query_one("#auto-sized", Static)
        below = host.query_one("#below", Static)

        with pytest.raises(AssertionError):
            await assert_geometry_is_content_independent(
                pilot,
                target,
                {
                    "one-row": "x",
                    "three-rows": "a\nb\nc",
                },
                [below],
            )
        assert isinstance(target.outer_size, Size)
