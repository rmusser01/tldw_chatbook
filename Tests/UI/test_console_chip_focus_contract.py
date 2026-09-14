"""Contract test: a focused Console status chip shows its label, not an empty box.

TASK-383: the global ``*:focus { outline: solid }`` fallback OVERLAYS a box onto
the height-1 status chips, replacing the label with border glyphs (the reported
"empty outlined box"). The chip's own focus rule provides a non-obscuring cue
(high-contrast bg/fg + bold underline), so it must also suppress the outline.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CSS_ROOT = ROOT / "tldw_chatbook" / "css"
# ADR-161 task 10: the console-owned chip rules live in the console
# source sheets (union: each rule exists in exactly one of the pair) and
# ride the boot bundle. The source+generated integrity pair is now
# (console sources, bundle).
CONSOLE_SOURCES = (
    _CSS_ROOT / "features" / "_console.tcss",
    _CSS_ROOT / "features" / "_console_panels.tcss",
)
BUNDLE = _CSS_ROOT / "tldw_cli_modular.tcss"


def _detok(value: str) -> str:
    """Resolve the ADR-161 sizing-scale tokens to their literal values."""
    resolved = value
    for pattern, repl in (
        (r"\$ds-space-(\d+)", r"\1"),
        (r"\$ds-size-(\d+)", r"\1"),
        (r"\$ds-percent-(\d+)", r"\1%"),
        (r"\$ds-fr-(\d+)", r"\1fr"),
        (r"\$ds-width-full", "100%"),
        (r"\$ds-height-full", "100%"),
        (r"\$ds-width-fill", "1fr"),
        (r"\$ds-height-fill", "1fr"),
    ):
        resolved = re.sub(pattern, repl, resolved)
    return resolved


def _contract_arms() -> list[tuple[str, str]]:
    """(label, detokenized text) per integrity arm."""
    return [
        (
            " + ".join(p.name for p in CONSOLE_SOURCES),
            _detok(
                "\n".join(p.read_text(encoding="utf-8") for p in CONSOLE_SOURCES)
            ),
        ),
        (BUNDLE.name, _detok(BUNDLE.read_text(encoding="utf-8"))),
    ]


def _chip_focus_body(css_text: str) -> str:
    uncommented = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    match = re.search(
        r"\.console-control-chip:focus\s*\{([^}]*)\}", uncommented
    )
    return match.group(1) if match else ""


def _chip_base_body(css_text: str) -> str:
    """Return the body of the base ``.console-control-chip { ... }`` rule.

    Anchored on ``.console-control-chip`` immediately followed by ``{``
    (only whitespace between) so it does not also match the
    ``.console-control-chip:focus`` variant, which has a ``:focus`` pseudo
    -class between the selector and the brace.
    """
    uncommented = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    match = re.search(r"\.console-control-chip\s*\{([^}]*)\}", uncommented)
    return match.group(1) if match else ""


def test_chip_hit_target_min_width_and_padding_are_widened():
    """RAG-46: a ~10x1-cell chip hit target is easy to miss with mouse/touch.
    ``min-width`` raised 7 -> 12 and horizontal ``padding`` raised ``0 1`` ->
    ``0 2`` (source + regenerated bundle) to widen both the box and its
    clickable interior."""
    for label, css_text in _contract_arms():
        body = _chip_base_body(css_text)
        assert body, f"{label}: no base .console-control-chip rule"

        min_width = re.search(r"\bmin-width\s*:\s*([^;]+);", body)
        assert min_width, f"{label}: chip must set min-width"
        assert min_width.group(1).strip() == "12", (
            f"{label}: chip min-width must be raised to 12"
        )

        padding = re.search(r"\bpadding\s*:\s*([^;]+);", body)
        assert padding, f"{label}: chip must set padding"
        assert padding.group(1).strip() == "0 2", (
            f"{label}: chip padding must be raised to 0 2"
        )


def test_focused_chip_suppresses_the_obscuring_outline_but_keeps_the_cue():
    """The chip focus rule drops the outline while keeping the readable cue."""
    for label, css_text in _contract_arms():
        body = _chip_focus_body(css_text)
        assert body, f"{label}: no .console-control-chip:focus rule"

        outline = re.search(r"\boutline\s*:\s*([^;]+);", body)
        assert outline, f"{label}: chip focus must set outline"
        assert outline.group(1).strip() == "none", (
            f"{label}: chip focus must suppress the box-drawing outline"
        )

        # The non-obscuring focus cue that replaces it stays intact.
        assert "$ds-focus-bg" in body
        assert "$ds-focus-fg" in body
        assert "bold underline" in body
