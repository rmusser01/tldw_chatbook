"""Scheduling CSS token regressions."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSS = REPO_ROOT / "tldw_chatbook/css/features/_scheduling.tcss"
GENERATED_CSS = REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"
AGENTIC_TERMINAL_CSS = REPO_ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"

MODULE_BANNER = "/* ===== MODULE: features/_scheduling.tcss ===== */"
BORDER_PATTERN = re.compile(
    r"border\s*:\s*solid\s+\$ds-grid-line\s*;", re.IGNORECASE
)

SELECTORS = (
    "#scheduling-workbench",
    "#scheduling-list-pane",
    "#scheduling-detail-pane",
    "#scheduling-inspector-pane",
)

OLD_SELECTORS = (
    "#schedules-workbench",
    "#schedules-list-pane",
    "#schedules-detail-pane",
    "#schedules-inspector-pane",
)


def _rule_block(css: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}", css, re.DOTALL)
    assert match is not None, f"Missing rule block for {selector!r}"
    return match.group("body")


def test_generated_bundle_contains_scheduling_module_banner() -> None:
    assert MODULE_BANNER in GENERATED_CSS.read_text(encoding="utf-8")


def test_scheduling_panes_declare_design_grid_border() -> None:
    for path in (SOURCE_CSS, GENERATED_CSS):
        css = path.read_text(encoding="utf-8")
        for selector in SELECTORS:
            rule = _rule_block(css, selector)
            assert BORDER_PATTERN.search(rule) is not None, (
                f"{selector} in {path} missing border: solid $ds-grid-line"
            )


def test_scheduling_ids_not_in_agentic_terminal_shared_css() -> None:
    agentic_css = AGENTIC_TERMINAL_CSS.read_text(encoding="utf-8")
    for selector in SELECTORS:
        assert selector not in agentic_css, (
            f"{selector} must live in _scheduling.tcss, not _agentic_terminal.tcss"
        )


def test_legacy_scheduling_ids_removed_from_source_css() -> None:
    """The legacy SchedulesScreen IDs must not appear in the new workbench CSS."""
    source_css = SOURCE_CSS.read_text(encoding="utf-8")
    for selector in OLD_SELECTORS:
        assert selector not in source_css, (
            f"{selector} is a legacy SchedulesScreen ID and must not appear in _scheduling.tcss"
        )
