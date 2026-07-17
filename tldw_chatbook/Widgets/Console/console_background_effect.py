"""Pure Textual Console background effect renderer."""

from __future__ import annotations

import random
from dataclasses import dataclass

from rich.segment import Segment
from rich.style import Style
from textual.app import ComposeResult
from textual.containers import Container
from textual.css.query import NoMatches
from textual.strip import Strip
from textual.timer import Timer
from textual.widget import Widget

from tldw_chatbook.Utils.console_background_effects import (
    ConsoleBackgroundEffectSettings,
    MAX_CONSOLE_BACKGROUND_FPS,
    MIN_CONSOLE_BACKGROUND_FPS,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


_INTENSITY_DENSITY = {"low": 0.035, "medium": 0.07, "high": 0.11}
_EFFECT_CHARS = {
    "snow": ".*+",
    "rain": ".:|",
    "matrix": "abcdefghijklmnopqrstuvwxyz0123456789",
}
_EFFECT_STYLES = {
    "snow": Style(color="#d7e8f7", dim=True),
    "rain": Style(color="#70a7d8", dim=True),
    "matrix": Style(color="#4ade80", dim=True),
}


@dataclass
class _Particle:
    x: int
    y: float
    speed: float
    char: str


class ConsoleBackgroundEffect(Widget):
    """Non-focusable animated text background effect for the Console transcript."""

    can_focus = False

    def __init__(
        self,
        settings: ConsoleBackgroundEffectSettings,
        *,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.settings = settings
        self._random = random.Random(seed)
        self._particles: list[_Particle] = []
        self._timer: Timer | None = None
        # task-261: `render_line` used to call `frame_text()` (a full W×H
        # grid build) once PER LINE, making each repaint O(W·H²). The frame
        # is instead computed once per (frame serial, width, height) and the
        # split lines reused for every `render_line` call of that repaint.
        # The serial bumps on every particle step (`_advance_frame`) and on
        # `update_settings`; width/height live in the key, so a resize
        # invalidates naturally. The cache lives on the widget instance.
        self._frame_serial: int = 0
        self._frame_cache: tuple[int, int, int, list[str]] | None = None

    @property
    def is_effect_active(self) -> bool:
        """Return whether the current settings should render an effect."""
        return self.settings.active

    @property
    def frame_rate(self) -> int:
        """Return renderer-clamped frames per second."""
        return max(
            MIN_CONSOLE_BACKGROUND_FPS,
            min(MAX_CONSOLE_BACKGROUND_FPS, int(self.settings.fps)),
        )

    def on_mount(self) -> None:
        """Start frame updates once the widget is mounted and active."""
        self._sync_timer()

    def on_unmount(self) -> None:
        """Stop frame updates when the widget leaves the DOM."""
        self._stop_timer()

    def update_settings(self, settings: ConsoleBackgroundEffectSettings) -> None:
        """Apply new settings, reset particles, and refresh the renderer.

        Args:
            settings: The new background-effect settings to render with.
        """
        self.settings = settings
        self._particles.clear()
        self._invalidate_frame_cache()
        self._sync_timer()
        self.refresh(layout=False)

    def frame_text(self, width: int, height: int) -> str:
        """Return a plain text frame for the requested dimensions."""
        width = max(0, width)
        height = max(0, height)
        if not self.is_effect_active or width == 0 or height == 0:
            return "\n".join(" " * width for _ in range(height))

        self._ensure_particle_count(width, height)
        grid = [[" " for _ in range(width)] for _ in range(height)]
        for particle in self._particles:
            y = int(particle.y)
            if 0 <= particle.x < width and 0 <= y < height:
                grid[y][particle.x] = particle.char
        return "\n".join("".join(row) for row in grid)

    def render_line(self, y: int) -> Strip:
        """Render a single background frame line.

        Args:
            y: Zero-based line index within the widget to render.

        Returns:
            The rendered strip for line ``y`` of the current frame.
        """
        width = self.size.width
        height = self.size.height
        if width <= 0 or height <= 0:
            return Strip.blank(max(0, width))
        lines = self._frame_lines(width, height)
        text = lines[y] if 0 <= y < len(lines) else " " * width
        style = _EFFECT_STYLES.get(self.settings.effect, Style(dim=True))
        return Strip([Segment(text, style)], width)

    def _frame_lines(self, width: int, height: int) -> list[str]:
        """Return the current frame's lines, computing them at most once per frame.

        The full-grid ``frame_text()`` build runs only when the cached frame's
        (serial, width, height) key no longer matches — i.e. after a particle
        step, a settings change, or a resize — so a repaint's H ``render_line``
        calls share one grid computation instead of building H grids.

        Args:
            width: Frame width in cells.
            height: Frame height in cells.

        Returns:
            The frame's rows, one string of length ``width`` per row.
        """
        cached = self._frame_cache
        if cached is not None:
            serial, cached_width, cached_height, lines = cached
            if serial == self._frame_serial and cached_width == width and cached_height == height:
                return lines
        lines = self.frame_text(width, height).splitlines()
        self._frame_cache = (self._frame_serial, width, height, lines)
        return lines

    def _invalidate_frame_cache(self) -> None:
        """Drop the cached frame so the next render recomputes the grid."""
        self._frame_serial += 1
        self._frame_cache = None

    def _sync_timer(self) -> None:
        self._stop_timer()
        if not self.is_effect_active or not self.is_mounted:
            return
        interval = 1 / self.frame_rate
        self._timer = self.set_interval(interval, self._advance_frame)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _advance_frame(self) -> None:
        width = self.size.width
        height = self.size.height
        if not self.is_effect_active or width <= 0 or height <= 0:
            self.refresh(layout=False)
            return
        self._step_particles(width, height)
        self._invalidate_frame_cache()
        self.refresh(layout=False)

    def _target_particle_count(self, width: int, height: int) -> int:
        if not self.is_effect_active or width <= 0 or height <= 0:
            return 0
        density = _INTENSITY_DENSITY.get(self.settings.intensity, _INTENSITY_DENSITY["low"])
        return max(1, min(width * height, int(width * height * density)))

    def _new_particle(self, width: int, height: int) -> _Particle:
        chars = _EFFECT_CHARS.get(self.settings.effect, _EFFECT_CHARS["snow"])
        speed_ranges = {
            "snow": (0.35, 1.0),
            "rain": (0.8, 1.8),
            "matrix": (0.6, 1.4),
        }
        min_speed, max_speed = speed_ranges.get(self.settings.effect, (0.5, 1.2))
        return _Particle(
            x=self._random.randrange(max(1, width)),
            y=float(self._random.randrange(max(1, height))),
            speed=self._random.uniform(min_speed, max_speed),
            char=self._random.choice(chars),
        )

    def _ensure_particle_count(self, width: int, height: int) -> None:
        target = self._target_particle_count(width, height)
        while len(self._particles) < target:
            self._particles.append(self._new_particle(width, height))
        if len(self._particles) > target:
            del self._particles[target:]

    def _step_particles(self, width: int, height: int) -> None:
        self._particles = [
            particle
            for particle in self._particles
            if 0 <= particle.x < width and particle.y + particle.speed < height
        ]
        for particle in self._particles:
            particle.y += particle.speed
            if self.settings.effect == "matrix":
                chars = _EFFECT_CHARS["matrix"]
                particle.char = self._random.choice(chars)
        self._ensure_particle_count(width, height)


class ConsoleTranscriptSurface(Container):
    """Container for a background effect plus the native Console transcript."""

    can_focus = False
    DEFAULT_CSS = """
    ConsoleTranscriptSurface {
        layers: background content;
    }

    ConsoleTranscriptSurface > ConsoleBackgroundEffect {
        layer: background;
        dock: top;
        width: 100%;
        height: 100%;
    }

    ConsoleTranscriptSurface > ConsoleTranscript {
        layer: content;
        dock: top;
        width: 100%;
        height: 100%;
    }
    """

    def __init__(
        self,
        settings: ConsoleBackgroundEffectSettings,
        *,
        transcript: ConsoleTranscript | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.settings = settings
        self.transcript = transcript or ConsoleTranscript(id="console-native-transcript")

    def compose(self) -> ComposeResult:
        yield ConsoleBackgroundEffect(
            self.settings,
            id="console-transcript-background-effect",
            classes="console-background-effect",
        )
        yield self.transcript

    def update_settings(self, settings: ConsoleBackgroundEffectSettings) -> None:
        """Update the mounted background effect child when available."""
        self.settings = settings
        if not self.is_mounted:
            return
        try:
            effect = self.query_one(
                "#console-transcript-background-effect",
                ConsoleBackgroundEffect,
            )
        except NoMatches:
            return
        effect.update_settings(settings)
