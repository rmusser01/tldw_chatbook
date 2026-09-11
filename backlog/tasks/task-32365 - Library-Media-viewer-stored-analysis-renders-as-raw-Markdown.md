---
id: TASK-32365
title: 'Library Media viewer: stored analysis renders as raw Markdown'
status: Done
assignee: []
created_date: '2026-09-11 06:20'
updated_date: '2026-09-11 07:34'
labels:
  - library
  - media
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A stored analysis shows '## Key contributions' as source text inside its box on the Analysis tab while the reader has a Rendered view elsewhere (A cap 43, B design note). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Analysis tab renders Markdown like the Read tab does, with a Raw toggle
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test: a markdown analysis paints rendered with a Raw toggle
2. _compose_analysis sniffs the analysis with looks_like_markdown_content and passes the viewer's analysis_content_mode
3. Reuse _compose_content_mode_toggle with an id prefix; handle the presses in the viewer widget
4. Green + reader suites, docs
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Analysis body pinned is_markdown=False, mode='raw' with the comment 'Analysis is plain text -> raw mode, not Markdown'. That assumption is false in practice -- a generated analysis is Markdown -- so '## Key contributions' painted as source. _compose_analysis now runs looks_like_markdown_content (the same content sniff task-32234 made the Read tab's sole authority) and passes it, plus the tab's own view mode, to the same LibraryMediaContentBody. _compose_content_mode_toggle grew is_markdown / mode / prefix arguments defaulting to the Read tab's values, so there is still ONE toggle renderer; the Analysis instance yields #library-media-analysis-content-mode-rendered / -raw. A plain-prose analysis gets no toggle, exactly as a plain item gets none on Read. The Find bar's placeholder also now reflects the analysis's markdown-ness ('Search content (raw text)…').

Deviation from the plan's step, deliberate: the plan said to thread analysis_content_mode from the screen the way content_mode is threaded (a new _media_state field, three edits in library_media_controller.py, two controller handlers and two forwarders in library_screen.py -- the last outside this task's ownership ranges). The mode is the widget's own view state: the screen has nothing to keep in step with it, and the Read tab's in-place sync_mode patch exists only to avoid re-parsing a full document on every traversal keystroke, which a deliberate toggle press is not. So the press is handled in the widget and answered with refresh(recompose=True). ~15 lines in one file instead of ~40 across four, and library_screen.py stayed inside its assigned ranges. Consequence: a Raw choice on Analysis persists for the session instead of being reseeded per item.

The strip and separator ids stay UNPREFIXED: only one Reader body composes at a time so they are still unique, and they carry the width rules (#library-media-content-mode-strip in _agentic_terminal.tcss, plus this class's DEFAULT_CSS for the separator) that keep the row out of Textual's bare-1fr non-rendering trap. No CSS change, no bundle regeneration.

Live (tmux, seeded profile, 235x52): 'Andrej Karpathy — Intro to Large Language Models' opens its Analysis tab with 'Rendered (selected) | Raw' and paints 'Summary' as a heading with bulleted list items; pressing Raw paints '## Summary' and '- Pretraining …' as source. Items with a plain analysis show no toggle.

Files: tldw_chatbook/Widgets/Library/library_media_viewer.py, Tests/UI/test_library_crit10_viewer.py (new), Docs/User_Guide/library/media-and-conversations.md (Analysis section, stamped).
<!-- SECTION:NOTES:END -->
