# Pins PR #2374's review fixes: muted/placeholder/error-readable text in any
# theme that defines them must clear WCAG AA (4.5:1) on the surfaces it sits on.
import pytest

from tldw_chatbook.css.Themes.themes import ALL_THEMES

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
