# Pins PR #2374's review fixes: muted/placeholder/error-readable text in any
# theme that defines them must clear WCAG AA (4.5:1) on the surfaces it sits on.
import re
from pathlib import Path

import pytest
from textual.color import Color

from tldw_chatbook.css.Themes.themes import ALL_THEMES

CORE_VARIABLES = Path(__file__).resolve().parents[2] / (
    "tldw_chatbook/css/core/_variables.tcss"
)
# The Orb-ported themes (PR #2374) — the resolved-token floor below is pinned
# for these; older themes carry pre-existing palette debt and are not gated.
ORB_THEMES = {
    "apricot",
    "camono",
    "christmas",
    "frutiger_aero",
    "halloween",
    "litestep",
    "litestep_dark",
    "night_city",
    "orb_dark",
    "orb_ocean",
    "parchment",
    "vintage_wood",
}

AA = 4.5
TEXT_TOKENS = (
    "ds-text-muted",
    "text-muted",
    "ds-text-placeholder",
    "ds-text-disabled-readable",
    "ds-status-error-readable",
)
SURFACE_TOKENS = ("ds-surface-panel", "ds-surface-raised", "ds-surface-inspector")


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


@pytest.mark.parametrize(
    "theme", [t for t in ALL_THEMES if t.variables], ids=lambda t: t.name
)
def test_readable_text_tokens_clear_aa_on_theme_surfaces(theme):
    variables = theme.variables
    surfaces = [variables[k] for k in SURFACE_TOKENS if k in variables]
    if not surfaces:
        pytest.skip("theme defines no ds surfaces")
    for token in TEXT_TOKENS:
        value = variables.get(token)
        if not (isinstance(value, str) and value.startswith("#")):
            continue
        for surface in surfaces:
            assert _ratio(value, surface) >= AA, (
                f"{theme.name}: {token} {value} is {_ratio(value, surface):.2f}:1 "
                f"against surface {surface} (needs {AA}:1)"
            )


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


@pytest.mark.parametrize(
    "theme", [t for t in ALL_THEMES if t.name in ORB_THEMES], ids=lambda t: t.name
)
def test_generated_readable_tokens_clear_aa_on_orb_themes(theme):
    """The values `$text-error` / `$text-muted` actually resolve to (Textual's
    generated variables) must clear AA on the theme's surfaces — this is what
    the ds readable tokens paint after task-31264's fix."""
    generated = theme.to_color_system().generate()
    surfaces = [Color.parse(generated[k]) for k in ("surface", "panel")]
    for token in ("text-error", "text-muted"):
        color = Color.parse(generated[token])
        for surface in surfaces:
            blended = surface.blend(Color(color.r, color.g, color.b), color.a)
            ratio = _ratio(blended.hex, surface.hex)
            assert ratio >= AA, (
                f"{theme.name}: generated {token} {color.hex} is {ratio:.2f}:1 "
                f"against {surface.hex} (needs {AA}:1)"
            )
