---
id: TASK-1995
title: 'Markdown rendering hygiene: About-pane markup, reading-mode mangling, dead CSS'
status: To Do
assignee: []
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
- [ ] #1 The About section renders styled text with no literal bracket tags, verified in a live capture
- [ ] #2 Reading mode and search highlighting no longer alter the inside of fenced code blocks (test with a code-block-bearing media document)
- [ ] #3 The dead Markdown#web-search-results CSS rule is gone
- [ ] #4 Existing tests for the touched surfaces stay green
<!-- AC:END -->
