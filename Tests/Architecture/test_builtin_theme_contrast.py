"""Textual's own built-in themes must clear AA too.

TASK-32901 (tier-2 S17 P2): ``ensure_readable_text_hues`` is applied to all 70
shipped themes and to every user theme, but never to ``textual.theme
.BUILTIN_THEMES`` -- which ``App.__init__`` registers itself and which
``app.py:1147-1160`` offers by name (``textual-dark``, ``textual-light``, plus
everything in ``available_themes``). Five built-ins failed AA, including
``textual-light``'s ``text-accent`` at 2.80:1 -- the one light theme in the
app's own hard-coded list. ``$ds-value-fg: $text-accent`` is live in
``screen_feature_scheduling.tcss`` and ``screen_feature_evals.tcss``.

The existing gate in ``Tests/UI/test_theme_contrast.py`` states the
requirement but parametrizes over ``ALL_THEMES``, a corpus that excludes
exactly the failing themes. This file lives outside ``Tests/UI`` so it runs
without the ADR-126 recovery participants.
"""

import re
from pathlib import Path

import pytest
from textual.color import Color
from textual.theme import BUILTIN_THEMES, Theme

import tldw_chatbook.css.Themes.themes  # noqa: F401 - applies the guard

CORE_VARIABLES = Path(__file__).resolve().parents[2] / (
    "tldw_chatbook/css/core/_variables.tcss"
)

AA = 4.5


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _ratio(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    lo, hi = min(la, lb), max(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def test_core_variables_do_not_freeze_readable_tokens_to_literals():
    """task-31264 root cause: `$name: value` in a tcss source shadows the
    theme's variables dict for that source (per-source variable scope), so a
    dark-tuned hex literal here freezes the token for every theme — the slate
    focused-button / unreadable light-theme error text symptom. The readable
    and focus tokens must therefore be *references* to Textual's generated,
    polarity-aware variables, never hex literals."""
    text = CORE_VARIABLES.read_text(encoding="utf-8")
    for token in (
        "ds-focus-bg",
        "ds-status-error-readable",
        "ds-text-placeholder",
        "ds-text-disabled-readable",
        # TASK-31429: Console rail grammar (active = primary hue, value =
        # accent hue) must follow the theme the same way.
        "ds-active-fg",
        "ds-value-fg",
    ):
        match = re.search(rf"^\${re.escape(token)}:\s*([^;]+);", text, re.M)
        assert match, f"{token} not defined in _variables.tcss"
        value = match.group(1).strip()
        assert value.startswith("$"), (
            f"{token} is frozen to literal {value!r}; use a $-reference "
            f"to a generated theme variable instead"
        )


def _resolved_variables(theme) -> dict:
    """Mirror runtime resolution: theme dict entries win over generated."""
    return {**theme.to_color_system().generate(), **(theme.variables or {})}


def _over(base: Color, color: Color) -> Color:
    """Composite a possibly-translucent color over a base surface."""
    return base.blend(Color(color.r, color.g, color.b), color.a)


def _resolve_color(value: str, base: Color) -> Color:
    """Parse a variable value ('#hex', '#hexAA', or 'auto NN%') over a base."""
    if value.startswith("auto"):
        percent = float(value.split()[1].rstrip("%")) / 100
        pole = Color(0, 0, 0) if base.brightness > 0.5 else Color(255, 255, 255)
        return base.blend(pole, percent)
    return _over(base, Color.parse(value))


def _measurable(theme: Theme) -> bool:
    """Whether the theme resolves to hex surfaces at all (ANSI palettes do not)."""
    try:
        resolved = _resolved_variables(theme)
        return all(Color.parse(resolved[key]).a == 1 for key in ("surface", "panel"))
    except Exception:
        return False


@pytest.mark.parametrize("theme", list(BUILTIN_THEMES.values()), ids=lambda t: t.name)
def test_textual_builtin_themes_clear_aa_on_readable_text_hues(theme: Theme) -> None:
    if not _measurable(theme):
        pytest.skip(f"{theme.name} has no resolvable hex surfaces")
    resolved = _resolved_variables(theme)
    surfaces = [Color.parse(resolved[key]) for key in ("surface", "panel")]
    for token in ("text-primary", "text-accent"):
        for surface in surfaces:
            blended = _resolve_color(resolved[token], surface)
            ratio = _ratio(blended.hex, surface.hex)
            assert ratio >= AA, (
                f"{theme.name}: resolved {token} {blended.hex} is {ratio:.2f}:1 "
                f"against {surface.hex} (needs {AA}:1)"
            )
