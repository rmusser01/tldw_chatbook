---
id: TASK-15996
title: 'Update stale DEFAULT_CSS references for consolidated classes'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 01:10'
labels:
  - docs
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four in-tree references still say `DEFAULT_CSS` for classes that now declare `BUNDLED_CSS` after TASK-15450: `Constants.py:1390` (documents the KEEP-IN-SYNC contract from task-264 — a reader will now look in the wrong attribute), `UI/Workbench/help.py:94`, `UI/MCP_Modules/mcp_tools_mode.py:140`, `UI/MCP_Modules/mcp_inspector.py:848`. Cosmetic except the Constants.py contract note. Sweep for others while there (`grep -rn DEFAULT_CSS` filtered to comments/docstrings naming consolidated classes). Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All four cited references updated to name BUNDLED_CSS (and the contract note reads correctly)
- [x] #2 A grep sweep confirms no other stale comment/docstring references remain for the 57 consolidated classes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Enumerate the currently-consolidated classes at HEAD: `grep -rn "BUNDLED_CSS\s*=\|BUNDLED_SCREEN_CSS\s*="` under `tldw_chatbook/`, mapped to their enclosing class name.
2. Re-locate the four cited references at HEAD (lines have moved) and read enough surrounding context to fix the whole sentence, not just the token.
3. Repo-wide sweep: `grep -rn DEFAULT_CSS` over `tldw_chatbook/` and `Tests/`, cross-reference every hit against the enumerated class list (same-line name match), and separately re-grep each of the 53 consolidated-declaring files for any OTHER `DEFAULT_CSS` mention (self-referential comments that don't repeat the class name on the same line, e.g. "this widget's own DEFAULT_CSS").
4. Classify every hit: (a) self-referential to a consolidated class's own former attribute -> fix to BUNDLED_CSS; (b) a generic Textual CSS-tier statement ("app-tier CSS beats widget DEFAULT_CSS on ties") -> leave alone, still accurate Textual terminology; (c) about Textual's own builtin widgets (Button/Footer) -> leave alone; (d) about a different, non-consolidated class in the same file -> leave alone.
5. Apply fixes; for chat_screen.py's harness-behavior claim, rewrite to describe the real ConsolidatedCSSApp mechanism rather than a blind rename, since the old claim ("any bare App loads widget DEFAULT_CSS") no longer holds post-15450.
6. Verify: py_compile every touched file, run `Tests/UI/test_widget_css_consolidation.py` plus the touched tests, ruff check/format --diff on touched files only.
7. Update this task file and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the 4 cited references plus 25 more found by the sweep (29 hit-sites, 20 files: 17 source + 3 tests). Enumerating `BUNDLED_CSS`/`BUNDLED_SCREEN_CSS`-declaring classes at HEAD found 53 (46 `BUNDLED_CSS` + 7 `BUNDLED_SCREEN_CSS`), not 57 — some classes from TASK-15450's original count have since been refactored/retired; not a concern, current state is what matters for accuracy.

Every remaining `DEFAULT_CSS` hit repo-wide was read in context and classified, not blindly renamed:
- **Fixed (self-referential to a consolidated class's own former attribute):** Constants.py:1390, help.py:97, mcp_tools_mode.py:140, mcp_inspector.py:848 (the 4 cited) + AppFooterStatus.py:114, main_navigation.py:824, mcp_workbench.py:314+982, mcp_audit_mode.py:279+696, mcp_permissions_mode.py:89, mcp_inspector.py:329+1124+1130+1133+1137, personas_screen.py:672+9791, persona_profile_editor_widget.py:60, personas_conversation_transcript_widget.py:21, personas_character_card_widget.py:38, personas_character_editor_widget.py:80+329, persona_profile_card_widget.py:24, console_session_surface.py:307, plus 3 test files (test_mcp_tools_mode.py:851, test_screen_footer_hints.py:509/516/518/520/521/569, test_console_tab_strip_budget.py:12).
- **chat_screen.py:1755-1756 rewritten, not just renamed:** the old comment claimed "test harnesses ... load widget DEFAULT_CSS" — true pre-15450 (Textual auto-loads any class's literal `DEFAULT_CSS`), false now (`BUNDLED_CSS` is inert until a harness uses `ConsolidatedCSSApp`, per `Tests/UI/consolidated_css.py`'s own docstring). Rewrote to name that mechanism.
- **Left alone, generic Textual CSS-tier statements** ("app-tier CSS beats widget DEFAULT_CSS on ties/regardless of specificity") — still accurate Textual terminology, not about a specific class's renamed attribute: mcp_workbench.py:673, mcp_servers_mode.py:302, mcp_inspector.py:732+847(Textual's own `Button.DEFAULT_CSS`), mcp_audit_mode.py:280, personas_screen.py:531, lab_mode_strip.py:49, personas_inspector_pane.py:129, personas_library_pane.py:435, main_navigation.py:224-225+233.
- **Left alone, unrelated:** AppFooterStatus.py:106 (Textual's own `Footer.DEFAULT_CSS`), mcp_rail.py:197 (Textual's own `Button.DEFAULT_CSS`), ccp_loading_indicators.py:369 (a different, non-consolidated class — `InlineLoadingIndicator` — in the same file). `Docs/Development/css-consolidation-strategy.md` describes the pre-migration state as design rationale, not a live pointer; left untouched as historical.

Verification: every touched file `py_compile`s; `Tests/UI/test_widget_css_consolidation.py` 17/17 pass (unchanged); `Tests/UI/test_screen_footer_hints.py` and `test_console_tab_strip_budget.py` pass except one pre-existing, unrelated failure (`test_production_routes_own_and_preserve_contextual_footer_hints`, a full-app integration test at a line range untouched by this diff — confirmed via `git show HEAD:...` byte-diff against the edited file, the only differences are the 6 lines this task changed, all in unrelated helper functions). `ruff check`/`format --diff` on the 20 touched files surfaced only pre-existing, unrelated debt (2 unused imports, formatting drift elsewhere in 11 files) on lines this diff never touches.

Modified files: `tldw_chatbook/Constants.py`, `tldw_chatbook/UI/Workbench/help.py`, `tldw_chatbook/UI/MCP_Modules/{mcp_tools_mode,mcp_inspector,mcp_workbench,mcp_audit_mode,mcp_permissions_mode}.py`, `tldw_chatbook/UI/Navigation/main_navigation.py`, `tldw_chatbook/UI/Screens/{chat_screen,personas_screen}.py`, `tldw_chatbook/Widgets/AppFooterStatus.py`, `tldw_chatbook/Widgets/Console/console_session_surface.py`, `tldw_chatbook/Widgets/Persona_Widgets/{persona_profile_editor_widget,persona_profile_card_widget,personas_character_card_widget,personas_character_editor_widget,personas_conversation_transcript_widget}.py`, `Tests/UI/{test_mcp_tools_mode,test_screen_footer_hints,test_console_tab_strip_budget}.py`.
<!-- SECTION:NOTES:END -->
