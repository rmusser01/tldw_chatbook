"""Design-token governance (ADR-150, TASK-32475).

Mechanical enforcement of the design-token system:

1. Every ``$ds-*`` token referenced in any TCSS source module must be defined
   in ``core/_variables.tcss`` — agents may not invent tokens ad hoc in
   feature sheets, and renamed/removed tokens fail loudly instead of silently
   falling back.
2. Hex-color ratchet: raw hex literals are legal only in token definitions
   (``core/_variables.tcss``) and the theme catalog (``Themes/``). The two
   legacy source sheets that still carry hex literals are grandfathered at
   their current counts; the count may only go down.
3. New sheets start clean: any ``*.tcss`` module not in the grandfathered
   manifest must use the spacing scale and may not introduce raw numeric
   ``padding``/``margin`` declarations or hex literals at all.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.css.build_css import CSS_MODULES

REPO_ROOT = Path(__file__).resolve().parents[2]
CSS_ROOT = REPO_ROOT / "tldw_chatbook/css"
TOKENS_FILE = CSS_ROOT / "core/_variables.tcss"

# Matches a full token name used anywhere (definition or reference).
_TOKEN_RE = re.compile(r"\$ds-[a-z0-9-]+")
# A token definition line, e.g. `$ds-space-1: 1;`
_DEFINITION_RE = re.compile(r"^\s*(\$ds-[a-z0-9-]+)\s*:", re.MULTILINE)
_HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b")
_RAW_SPACING_RE = re.compile(
    r"\b(?:padding|margin)(?:-(?:top|right|bottom|left))?\s*:\s*[0-9]"
)

# Legacy sheets with pre-existing hex literals in ACTIVE declarations,
# pinned at current counts (ratchet: may only decrease). Comments are
# stripped before counting, so documented measurements never consume
# allowance and removing one never creates allowance for a real literal.
# Active counts verified 2026-09-11 (_lists.tcss has comment-only hexes).
# ADR-161 task 10 (2026-09-14): the agentic monolith's single grandfathered
# hex (#6f7782, the Console frame border) was tokenized to $ds-console-frame
# when the console vocabulary moved to features/_console{,_panels}.tcss, so
# the allowance is gone -- no sheet carries an active hex today.
_HEX_GRANDFATHERED: dict[str, int] = {}

# Every sheet that existed when ADR-150 landed (2026-09-11), including the
# legacy files not in the build manifest (_unified_sidebar, _new_ingest,
# _chatbooks_improved). Anything NOT in this set is a post-ADR sheet and must
# be token-clean.
_LEGACY_SHEETS: frozenset[str] = frozenset(
    """
    core/_variables.tcss core/_reset.tcss core/_base.tcss core/_typography.tcss
    layout/_windows.tcss layout/_tabs.tcss layout/_sidebars.tcss
    layout/_panes.tcss layout/_containers.tcss
    components/_buttons.tcss components/_forms.tcss components/_lists.tcss
    components/_navigation.tcss components/_change_review.tcss
    components/_messages.tcss components/_dialogs.tcss components/_status.tcss
    components/_agentic_terminal.tcss components/_workbench.tcss
    components/_widgets.tcss components/_settings_splash_theme.tcss
    components/_shared_components.tcss components/_unified_sidebar.tcss
    components/_profile_interview.tcss components/_settings_personal_context.tcss
    features/_chat.tcss features/_conversations.tcss features/_notes.tcss
    features/_media.tcss features/_llm-management.tcss
    features/_tools-settings.tcss features/_ingest.tcss
    features/_evaluation_unified.tcss features/_evals.tcss
    features/_metrics.tcss features/_embeddings.tcss features/_splash.tcss
    features/_wizards.tcss features/_chatbooks.tcss features/_scheduling.tcss
    features/_code_repo.tcss features/_coding.tcss features/_tab_dropdown.tcss
    features/_watchlists.tcss features/_lab.tcss features/_logs.tcss
    features/_writing.tcss features/config_search.tcss
    features/feature_alerts.tcss features/_new_ingest.tcss
    features/_chatbooks_improved.tcss features/_research_workspace.tcss
    utilities/_helpers.tcss utilities/_states.tcss utilities/_overrides.tcss
    """.split()
)

_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(css: str) -> str:
    """Comments may mention tokens prose-style ('the $ds-focus contract');
    only real declarations are governed."""
    return _COMMENT_RE.sub("", css)


def _source_modules() -> list[Path]:
    """Active manifest sources and owned TCSS sheets, excluding generated output."""
    modules = {CSS_ROOT / relative for relative in CSS_MODULES}
    for subdir in ("core", "layout", "components", "features", "utilities"):
        modules.update((CSS_ROOT / subdir).glob("*.tcss"))
    return sorted(modules)


def _defined_tokens() -> set[str]:
    return set(_DEFINITION_RE.findall(TOKENS_FILE.read_text(encoding="utf-8")))


@private_profile_test
def test_all_referenced_ds_tokens_are_defined(request) -> None:
    """No sheet may reference a $ds-* token that _variables.tcss lacks."""
    defined = _defined_tokens()
    assert "$ds-space-1" in defined, "token catalog sanity check failed"

    offenders: dict[str, sorted] = {}
    for module in _source_modules():
        text = _strip_comments(module.read_text(encoding="utf-8"))
        referenced = set(_TOKEN_RE.findall(text)) - defined
        if referenced:
            offenders[str(module.relative_to(CSS_ROOT))] = sorted(referenced)

    assert not offenders, (
        "Undefined $ds-* tokens referenced (define them in "
        "core/_variables.tcss, per ADR-150):\n"
        + "\n".join(f"  {mod}: {', '.join(toks)}" for mod, toks in offenders.items())
    )


@private_profile_test
def test_defined_tokens_have_no_orphans_in_catalog(request) -> None:
    """Token definitions must be unique — a redefinition is a silent override."""
    definitions = _DEFINITION_RE.findall(TOKENS_FILE.read_text(encoding="utf-8"))
    duplicates = sorted({tok for tok in definitions if definitions.count(tok) > 1})
    assert not duplicates, f"Duplicate token definitions: {duplicates}"


@private_profile_test
def test_documented_design_vocabulary_is_available(request) -> None:
    """ADR-150's catalog must exist even before a feature consumes a token."""
    required = {
        "$ds-space-0",
        "$ds-space-1",
        "$ds-space-2",
        "$ds-space-3",
        "$ds-space-inline",
        "$ds-space-stack",
        "$ds-space-section",
        "$ds-space-inset",
        "$ds-control-height",
        "$ds-control-height-compact",
        "$ds-textarea-min-height",
        "$ds-duration-fast",
        "$ds-duration-medium",
        "$ds-duration-slow",
        "$ds-opacity-dim",
        "$ds-text-strong",
        "$ds-text-emphasis",
        "$ds-hover-bg",
        "$ds-hover-fg",
        "$ds-disabled-bg",
        "$ds-surface-sunken",
        "$ds-text-disabled-readable",
        "$ds-sidebar-width",
        "$ds-sidebar-min-width",
        "$ds-sidebar-max-width",
    }
    missing = sorted(required - _defined_tokens())
    assert not missing, (
        f"Documented design-language tokens missing from catalog: {missing}"
    )


@private_profile_test
def test_hex_literals_are_ratcheted_outside_token_definitions(request) -> None:
    """Raw hex colors may only shrink, never grow, outside token definitions.

    Comments are stripped before counting: only active declarations are
    governed, so documenting a measured color in a comment can neither fail
    CI nor free up allowance for a real hardcoded color.
    """
    for module in _source_modules():
        relative = str(module.relative_to(CSS_ROOT))
        if module == TOKENS_FILE:
            continue
        text = _strip_comments(module.read_text(encoding="utf-8"))
        count = len(_HEX_RE.findall(text))
        allowance = _HEX_GRANDFATHERED.get(relative, 0)
        assert count <= allowance, (
            f"{relative} has {count} hex literals (allowance {allowance}). "
            "Colors belong in core/_variables.tcss tokens (ADR-150). "
            "The grandfathered allowance may only decrease."
        )


@private_profile_test
def test_hex_ratchet_ignores_comments(request) -> None:
    """Regression: hex strings inside /* */ comments must not be counted."""
    css = "/* measured #51677e at 3:1 */\n.card { color: #ff8fa3; }\n"
    assert _HEX_RE.findall(_strip_comments(css)) == ["#ff8fa3"]


@private_profile_test
def test_new_sheets_must_use_spacing_tokens(request) -> None:
    """Modules added after ADR-150 may not hardcode numeric padding/margin."""
    for module in _source_modules():
        relative = str(module.relative_to(CSS_ROOT))
        if relative in _LEGACY_SHEETS:
            continue
        matches = _RAW_SPACING_RE.findall(
            _strip_comments(module.read_text(encoding="utf-8"))
        )
        assert not matches, (
            f"{relative} is a post-ADR-150 sheet and must use the "
            "$ds-space-* scale instead of raw numeric padding/margin."
        )


@private_profile_test
def test_active_css_extension_is_in_hex_floor(request, tmp_path, monkeypatch):
    sheet = tmp_path / "components" / "active.css"
    sheet.parent.mkdir()
    sheet.write_text("Widget { color: #abcdef; }")
    unused = sheet.parent / "unused.css"
    unused.write_text("Widget { color: #123456; }")
    monkeypatch.setattr(sys.modules[__name__], "CSS_ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "CSS_MODULES", ["components/active.css"])
    assert sheet in _source_modules()
    assert unused not in _source_modules()
    with pytest.raises(AssertionError, match="components/active.css"):
        test_hex_literals_are_ratcheted_outside_token_definitions.__wrapped__(request)


@pytest.mark.asyncio
@private_profile_test
async def test_section_header_preserves_global_and_stats_computed_geometry(request):
    """The former Stats duplicate contributes the same global header geometry."""
    from textual.app import App, ComposeResult
    from textual.containers import Vertical
    from textual.widgets import Label

    source = "\n".join((CSS_ROOT / name).read_text() for name in CSS_MODULES)

    class HeaderProbe(App):
        CSS = source

        def compose(self) -> ComposeResult:
            yield Label("Global", id="global-header", classes="section-header")
            with Vertical(id="stats-container"):
                yield Label("Stats", id="stats-header", classes="section-header")

    async with HeaderProbe().run_test() as pilot:
        await pilot.pause()
        for widget_id in ("global-header", "stats-header"):
            header = pilot.app.query_one(f"#{widget_id}")
            assert tuple(header.styles.padding) == (0, 0, 0, 1)
            # Screen .section-header in _chat.tcss overrides the generic margin.
            assert tuple(header.styles.margin) == (1, 0, 1, 0)
            assert header.styles.border_left[0] == "thick"
            assert header.styles.border_left[1] == header.styles.color
