"""Terminal graphics capability detection and the ASCII fallback renderer
(task-3401.10, ADR-044 decision 4).

Fallback order (AC4): kitty (TGP) -> sixel -> halfcell -> ascii. Detection
is env-based and conservative -- a wrong "yes" renders garbage on a
terminal that cannot honor the protocol, so only well-known signals count;
everything else degrades one rung. The ascii mode needs no protocol at
all: a TermTube-style grayscale ramp that renders anywhere.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Mapping

RenderMode = Literal["kitty", "sixel", "halfcell", "ascii"]

#: Terminals with native kitty-graphics (TGP) support.
_KITTY_TERM_PROGRAMS = frozenset({"kitty", "wezterm", "ghostty"})
#: Terminals with reliable sixel support (xterm only when compiled in --
#: most aren't, so plain xterm does NOT count).
_SIXEL_TERM_PROGRAMS = frozenset({"foot", "contour", "mlterm", "yaft"})
_SIXEL_TERM_FRAGMENTS = ("sixel", "mlterm", "foot")
#: Terms that signal "no graphics at all" -- straight to ascii.
_DUMB_TERMS = frozenset({"dumb", "linux", "cons25"})

#: TermTube's default density ramp, dark to bright.
ASCII_RAMP = " .:-=+*#%@"


def detect_render_mode(env: Mapping[str, str] | None = None) -> RenderMode:
    """Pick the richest render mode the terminal is known to support.

    Args:
        env: Environment mapping to inspect (``os.environ`` when omitted;
            tests inject a fake).

    Returns:
        ``"kitty"`` when the terminal speaks the kitty graphics protocol,
        else ``"sixel"`` for known sixel terminals, ``"halfcell"`` for
        everything else with a real display, or ``"ascii"`` for dumb/no
        terminals.
    """
    env = os.environ if env is None else env
    term = env.get("TERM", "").strip().lower()
    term_program = env.get("TERM_PROGRAM", "").strip().lower()

    if term in _DUMB_TERMS or (not term and not term_program):
        return "ascii"

    if (
        env.get("KITTY_WINDOW_ID")
        or "kitty" in term
        or term_program in _KITTY_TERM_PROGRAMS
    ):
        return "kitty"

    if (
        any(fragment in term for fragment in _SIXEL_TERM_FRAGMENTS)
        or term_program in _SIXEL_TERM_PROGRAMS
    ):
        return "sixel"

    return "halfcell"


def frame_to_ascii(image: Any, *, cols: int = 80) -> str:
    """Render one PIL frame as a grayscale ASCII ramp (TermTube-style).

    Args:
        image: PIL image (any mode; converted to grayscale).
        cols: Output width in terminal columns. Rows are derived from the
            aspect ratio with the ~2:1 cell height correction.

    Returns:
        Newline-joined rows of :data:`ASCII_RAMP` characters.
    """
    if cols < 1:
        cols = 1
    width, height = image.size
    if width <= 0 or height <= 0:
        return ""
    rows = max(1, round(cols * (height / width) * 0.5))
    gray = image.convert("L").resize((cols, rows))
    ramp_last = len(ASCII_RAMP) - 1
    lines: list[str] = []
    pixels = gray.load()
    for y in range(rows):
        lines.append(
            "".join(ASCII_RAMP[pixels[x, y] * ramp_last // 255] for x in range(cols))
        )
    return "\n".join(lines)
