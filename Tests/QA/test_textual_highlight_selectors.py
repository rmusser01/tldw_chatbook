"""Textual highlighted-list selector regressions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CSS_ROOT = REPO_ROOT / "tldw_chatbook/css"
GENERATED_CSS = CSS_ROOT / "tldw_cli_modular.tcss"

EXPECTED_HIGHLIGHT_SELECTORS = (
    "ListView ListItem.-highlight",
    "#embeddings-model-list ModelListItem.-highlight",
    "#embeddings-collection-list CollectionListItem.-highlight",
    "#chatbooks-list ListItem.-highlight",
    "ConfigSearchResult.-highlight",
)


def _non_personas_tcss_files() -> list[Path]:
    return sorted(
        path
        for path in CSS_ROOT.rglob("*.tcss")
        if "personas" not in path.as_posix() and path.name != GENERATED_CSS.name
    )


def test_non_personas_source_uses_textual_highlight_selector() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _non_personas_tcss_files()
        if ".--highlight" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_generated_bundle_uses_textual_highlight_selector() -> None:
    css = GENERATED_CSS.read_text(encoding="utf-8")

    assert ".--highlight" not in css
    for selector in EXPECTED_HIGHLIGHT_SELECTORS:
        assert selector in css
