"""Contract test: a focused Console status chip shows its label, not an empty box.

TASK-383: the global ``*:focus { outline: solid }`` fallback OVERLAYS a box onto
the height-1 status chips, replacing the label with border glyphs (the reported
"empty outlined box"). The chip's own focus rule provides a non-obscuring cue
(high-contrast bg/fg + bold underline), so it must also suppress the outline.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTIC = ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
BUNDLE = ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


def _chip_focus_body(css_text: str) -> str:
    uncommented = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    match = re.search(
        r"\.console-control-chip:focus\s*\{([^}]*)\}", uncommented
    )
    return match.group(1) if match else ""


def test_focused_chip_suppresses_the_obscuring_outline_but_keeps_the_cue():
    """The chip focus rule drops the outline while keeping the readable cue."""
    for css_path in (AGENTIC, BUNDLE):
        body = _chip_focus_body(css_path.read_text(encoding="utf-8"))
        assert body, f"{css_path.name}: no .console-control-chip:focus rule"

        outline = re.search(r"\boutline\s*:\s*([^;]+);", body)
        assert outline, f"{css_path.name}: chip focus must set outline"
        assert outline.group(1).strip() == "none", (
            f"{css_path.name}: chip focus must suppress the box-drawing outline"
        )

        # The non-obscuring focus cue that replaces it stays intact.
        assert "$ds-focus-bg" in body
        assert "$ds-focus-fg" in body
        assert "bold underline" in body
