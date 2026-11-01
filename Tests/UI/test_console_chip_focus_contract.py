"""Contract test: a focused Console status strip shows its text, not an empty box.

TASK-383 (chip era): the global ``*:focus { outline: solid }`` fallback OVERLAID
a box onto the height-1 status row, replacing the label with border glyphs (the
reported "empty outlined box"). TASK-35160 replaced the chips with one text
line (``.console-status-text``); the contract carries over to that rule: the
strip's focus styling must suppress the outline while keeping a non-obscuring
cue (high-contrast bg/fg), so a focused strip still reads as text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CSS_ROOT = ROOT / "tldw_chatbook" / "css"
# ADR-161 task 10: the console-owned rules live in the console
# source sheets (union: each rule exists in exactly one of the pair) and
# ride the boot bundle. The source+generated integrity pair is now
# (console sources, bundle).
CONSOLE_SOURCES = (
    _CSS_ROOT / "features" / "_console.tcss",
    _CSS_ROOT / "features" / "_console_panels.tcss",
    _CSS_ROOT / "features" / "_console_inspector.tcss",
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
    ):
        resolved = re.sub(pattern, repl, resolved)
    return resolved


def _contract_arms() -> list[tuple[str, str]]:
    """(label, detokenized text) per integrity arm."""
    return [
        (
            " + ".join(p.name for p in CONSOLE_SOURCES),
            _detok("\n".join(p.read_text(encoding="utf-8") for p in CONSOLE_SOURCES)),
        ),
        (BUNDLE.name, _detok(BUNDLE.read_text(encoding="utf-8"))),
    ]


def _strip_focus_body(css_text: str) -> str:
    uncommented = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    match = re.search(r"\.console-status-text:focus\s*\{([^}]*)\}", uncommented)
    return match.group(1) if match else ""


def test_focused_strip_suppresses_the_obscuring_outline_but_keeps_the_cue():
    """The strip focus rule drops the outline while keeping the readable cue."""
    for label, css_text in _contract_arms():
        body = _strip_focus_body(css_text)
        assert body, f"{label}: no .console-status-text:focus rule"

        outline = re.search(r"\boutline\s*:\s*([^;]+);", body)
        assert outline, f"{label}: strip focus must set outline"
        assert outline.group(1).strip() == "none", (
            f"{label}: strip focus must suppress the box-drawing outline"
        )

        # The non-obscuring focus cue that replaces it stays intact.
        assert "$ds-focus-bg" in body
        assert "$ds-focus-fg" in body


def test_strip_line_is_a_single_row_of_text():
    """The strip's base rule keeps it a height-1 nowrap text line -- the
    point of TASK-35160 (no per-chip boxes, no ellipsis cap)."""
    for label, css_text in _contract_arms():
        uncommented = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
        match = re.search(r"\.console-status-text\s*\{([^}]*)\}", uncommented)
        assert match, f"{label}: no base .console-status-text rule"
        body = match.group(1)
        assert "height: 1" in body, f"{label}: strip must stay one row"
        assert "nowrap" in body, f"{label}: strip text must not wrap"
