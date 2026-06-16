"""Agentic terminal CSS token regressions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSS = REPO_ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
GENERATED_CSS = REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"

def test_agentic_pane_borders_use_design_grid_token() -> None:
    for path in (SOURCE_CSS, GENERATED_CSS):
        css = path.read_text(encoding="utf-8")

        assert "#6f6f6f" not in css
        assert css.count("border: solid $ds-grid-line;") >= 3
