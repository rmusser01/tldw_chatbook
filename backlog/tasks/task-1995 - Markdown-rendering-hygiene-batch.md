---
id: TASK-1995
title: 'Markdown rendering hygiene: About-pane markup, reading-mode mangling, dead CSS'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-02 22:30'
labels:
  - bug
  - markdown
  - hygiene
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three defects found by the 2026-08-02 markdown-rendering audit (all verified on origin/dev `cf1b345f2`):

1. **About pane feeds Rich markup into a Markdown widget.** `UI/Tools_Settings_Window.py:3356-3381` builds `about_text` out of Rich console markup (`[bold]…[/bold]`, `[italic]`, `[link=…]`) and yields `Markdown(about_text)`. Textual's Markdown does not interpret Rich markup, so the tags render literally. Convert the text to actual markdown (the widget's link handler at `on_markdown_link_clicked` already works) or render via a markup-enabled Static.

2. **Media "reading mode" and search highlighting corrupt markdown/code content.** `Widgets/Media/media_viewer_panel.py:1045-1085`: `_format_content_for_reading` regex-injects paragraph breaks after sentence punctuation and rewrites `^\d+.` lines into `##` headings — mangling code blocks and pre-existing markdown; `_highlight_matches` wraps search hits in backticks/bold, injecting markup into user content (breaks inside fenced blocks). Make both transforms markdown-aware (skip fenced regions at minimum) or restrict them to plain-text content kinds.

3. **Dead CSS.** `Constants.py:1550` styles `Markdown#web-search-results` — no such widget exists anywhere in `UI/` or `Widgets/`. Remove the rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The About markdown source contains no Rich markup tags (verified by test; AMENDED 2026-08-06: the About pane is UNROUTED dead UI since TASK-1346 — `tools_settings` routes to MCPScreen and the F9 Settings screen has no About — so a live capture cannot honestly be produced; reachability gap filed as its own task)
- [x] #2 Reading mode and search highlighting no longer alter the inside of fenced code blocks (test with a code-block-bearing media document)
- [x] #3 The dead Markdown#web-search-results CSS rule is gone (already removed by intervening dev churn between filing and implementation; verified absent repo-wide)
- [x] #4 Existing tests for the touched surfaces stay green
<!-- AC:END -->

## Implementation Plan (the how)

1. Convert the About pane's Rich-markup text to a real-markdown `ABOUT_MARKDOWN` module constant (bullets to `-` list items, `[link=…]` to autolinks) so the existing `Markdown` widget and LinkClicked handler work unchanged.
2. Extract the media reading-mode/search-highlight transforms into fence-aware module functions (`format_reading_text`, `highlight_match_spans`, `_fenced_ranges`); an unclosed fence runs to end-of-content (the safe direction).
3. Unit tests over a fenced document; run the consuming suites.

## Implementation Notes

- `UI/Tools_Settings_Window.py`: `ABOUT_MARKDOWN` constant replaces the inline Rich-markup string. **Reachability finding:** the whole ToolsSettingsWindow (and its About view) is unrouted dead UI since TASK-1346 — `tools_settings` resolves to MCPScreen; the canonical F9 Settings screen carries no About at all. The markup fix stands (module is imported and tested), AC#1 was amended to source-level verification, and the missing-About gap is filed separately.
- `Widgets/Media/media_viewer_panel.py`: transforms are now module-level and fence-aware — sentence-splitting/heading-promotion skip fenced segments byte-identically; search hits inside fences are left unwrapped (backticks/bold inside a fence broke the block rendering).
- Item 3 (dead `Markdown#web-search-results` CSS) had already been removed on dev by intervening churn; verified absent repo-wide.
- Tests: new `Tests/UI/test_markdown_hygiene_1995.py` (5 tests); `test_tools_settings_window.py` (63) and media suites (38) green.
