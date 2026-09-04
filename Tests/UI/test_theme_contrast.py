# Readable-color gates for every registered theme, at the values that
# actually paint: theme `variables` dict entries win only over generated
# names no tcss source defines (see the mechanism note atop themes.py).
import re
from pathlib import Path

import pytest
from textual.color import Color
from textual.theme import Theme

from tldw_chatbook.css.Themes.themes import ALL_THEMES

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


@pytest.mark.parametrize("theme", ALL_THEMES, ids=lambda t: t.name)
def test_resolved_readable_tokens_clear_aa_on_every_theme(theme: Theme) -> None:
    """The values `$text-error` / `$text-muted` resolve to at runtime (theme
    variables dict over Textual's generated set) must clear AA on the theme's
    own surfaces — these feed the ds readable tokens since task-31264;
    task-31283 extended the gate from the Orb 12 to every registered theme."""
    resolved = _resolved_variables(theme)
    surfaces = [Color.parse(resolved[k]) for k in ("surface", "panel")]
    for token in ("text-error", "text-muted"):
        for surface in surfaces:
            blended = _resolve_color(resolved[token], surface)
            ratio = _ratio(blended.hex, surface.hex)
            assert ratio >= AA, (
                f"{theme.name}: resolved {token} {blended.hex} is {ratio:.2f}:1 "
                f"against {surface.hex} (needs {AA}:1)"
            )


# task-31284: the non-obscuring focus contract needs a *visible* background
# shift (TASK-345); primary-at-30% nullified it on themes whose primary sits
# near their surface. Floor chosen at 1.25x — well clear of the measured
# 1.02–1.08x failures, achievable with readable text on both polarities.
FOCUS_SHIFT_FLOOR = 1.25


@pytest.mark.parametrize("theme", ALL_THEMES, ids=lambda t: t.name)
def test_resolved_focus_tint_is_visible_and_readable_on_every_theme(
    theme: Theme,
) -> None:
    """The resolved focus tint must visibly shift the surface and keep text
    readable on the composite (task-31284; floors documented above)."""
    resolved = _resolved_variables(theme)
    surface = Color.parse(resolved["surface"])
    text = _resolve_color(resolved["text"], surface)
    tint = Color.parse(resolved["block-cursor-blurred-background"])
    composite = _over(surface, tint)
    shift = _ratio(composite.hex, surface.hex)
    assert shift >= FOCUS_SHIFT_FLOOR, (
        f"{theme.name}: focus tint {tint.hex} shifts the surface only "
        f"{shift:.2f}x (needs {FOCUS_SHIFT_FLOOR}x)"
    )
    readable = _ratio(text.hex, composite.hex)
    assert readable >= AA, (
        f"{theme.name}: text on the focus tint is {readable:.2f}:1 "
        f"(needs {AA}:1)"
    )
