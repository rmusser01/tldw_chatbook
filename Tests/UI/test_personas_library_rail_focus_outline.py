"""Contract test: a focused Personas library rail shows the selected row,
not an empty bordered box.

task-445 (RP UX review P3): the global ``*:focus { outline: solid }``
fallback (core/_reset.tcss) draws its outline around the ``ListView``'s own
box once it takes keyboard focus after a row is selected. That box is
taller than the 1-2 rows it holds (it fills the rail's remaining height),
so the outline rendered as a tall empty bordered rectangle hanging below
the selected row -- the "transient rendering artifact" the review reported.
Live-reproduced via tmux capture before this fix (a `#` outlined box
spanning ~24 rows below "Default Assistant"), and confirmed gone after.

Mirrors the task-383 Console chip focus-outline contract test.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTIC = ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
BUNDLE = ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


def _rule_body(css_text: str, selector_pattern: str) -> str:
    uncommented = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    match = re.search(selector_pattern + r"\s*\{([^}]*)\}", uncommented)
    return match.group(1) if match else ""


def test_focused_library_rail_suppresses_the_obscuring_outline():
    """The library rail's ListView focus rule drops the global outline."""
    for css_path in (AGENTIC, BUNDLE):
        body = _rule_body(
            css_path.read_text(encoding="utf-8"),
            r"#personas-library-rows:focus",
        )
        assert body, f"{css_path.name}: no #personas-library-rows:focus rule"

        outline = re.search(r"\boutline\s*:\s*([^;]+);", body)
        assert outline, f"{css_path.name}: rail focus must set outline"
        assert outline.group(1).strip() == "none", (
            f"{css_path.name}: rail focus must suppress the box-drawing outline"
        )
