"""Agentic terminal CSS token regressions."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSS = REPO_ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
GENERATED_CSS = REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


PANE_BORDER_RULES = (
    (
        "#schedules-list-pane",
        "#workflows-list-pane",
        "#acp-list-pane",
    ),
    (
        "#schedules-detail-pane",
        "#workflows-detail-pane",
        "#acp-detail-pane",
    ),
    (
        "#schedules-inspector-pane",
        "#workflows-inspector-pane",
        "#acp-inspector-pane",
    ),
)

GRID_BORDER_PATTERN = re.compile(r"border\s*:\s*solid\s+\$ds-grid-line\s*;", re.IGNORECASE)


def _combined_rule_block(css: str, selectors: tuple[str, ...]) -> str:
    selector_pattern = r"\s*,\s*".join(re.escape(selector) for selector in selectors)
    match = re.search(rf"{selector_pattern}\s*\{{(?P<body>.*?)\}}", css, flags=re.DOTALL)
    assert match is not None, f"Missing combined pane rule for {selectors!r}"
    return match.group("body")


def test_agentic_pane_borders_use_design_grid_token() -> None:
    for path in (SOURCE_CSS, GENERATED_CSS):
        css = path.read_text(encoding="utf-8")

        assert "#6f6f6f" not in css.lower()
        for selectors in PANE_BORDER_RULES:
            pane_rule = _combined_rule_block(css, selectors)
            assert GRID_BORDER_PATTERN.search(pane_rule) is not None
