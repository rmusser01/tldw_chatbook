"""Contract test for uniform Console transcript message selection styling.

TASK-385: selecting a user/assistant message shows the focus treatment (focus
colours + bold underline), but a selected Tool (or System) message kept its muted
``dim italic`` styling -- the single-class ``.console-transcript-message-tool`` /
``-system`` rules follow ``.console-transcript-message-selected`` in source order
with equal specificity, so they win the cascade for a row carrying both classes
and strip the selection treatment. A selected message of any kind must read the
same.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTIC = ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
BUNDLE = ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"

_SELECTED = "console-transcript-message-selected"


def _rules(css_text: str) -> list[tuple[str, str]]:
    """Return (selector, body) pairs for every rule in the stylesheet.

    Comments are stripped first (mirroring ``test_non_obscuring_focus_contract``)
    so braces inside ``/* ... */`` -- common in the generated bundle -- are not
    mistaken for rule delimiters.
    """
    uncommented = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    return re.findall(r"([^{}]+)\{([^}]*)\}", uncommented)


def _selected_treatment_for(css_text: str, kind: str) -> str:
    """Return the merged body of every rule that targets a selected <kind> row.

    A rule counts if its selector combines the kind class and the selected class
    on the same compound token (higher specificity than the bare kind rule).
    """
    bodies: list[str] = []
    kind_class = f"console-transcript-message-{kind}"
    for selector, body in _rules(css_text):
        for token in selector.split(","):
            token = token.strip()
            if kind_class in token and _SELECTED in token and "." + kind_class + "." in token + ".":
                # both classes on one element (no descendant space between them)
                compound = token.replace(" ", "")
                if f".{kind_class}" in compound and f".{_SELECTED}" in compound:
                    bodies.append(body)
    return "\n".join(bodies)


def test_selected_tool_and_system_messages_share_the_selected_treatment():
    """A selected transcript row of any kind reads the same in source and bundle.

    Guards TASK-385: the muted tool/system role rules must not out-cascade the
    selection treatment, so selected tool/system rows re-assert the focus colour
    and bold underline.
    """
    for css_path in (AGENTIC, BUNDLE):
        css = css_path.read_text(encoding="utf-8")

        # Baseline: the canonical selected treatment the other kinds must match.
        # Match the standalone `.console-transcript-message-selected` rule exactly
        # (a selector-list token), never a compound `-tool…-selected` selector.
        selected = next(
            (
                b
                for s, b in _rules(css)
                if any(tok.strip() == f".{_SELECTED}" for tok in s.split(","))
            ),
            "",
        )
        assert "bold underline" in selected, f"{css_path.name}: baseline selected rule missing"

        for kind in ("tool", "system"):
            treatment = _selected_treatment_for(css, kind)
            assert treatment, (
                f"{css_path.name}: a selected {kind} message must re-assert the "
                f"selection treatment with higher specificity than the muted "
                f".console-transcript-message-{kind} rule"
            )
            assert "$ds-focus-fg" in treatment, (
                f"{css_path.name}: selected {kind} message must use the focus colour"
            )
            assert "bold" in treatment and "underline" in treatment, (
                f"{css_path.name}: selected {kind} message must be bold underline"
            )
