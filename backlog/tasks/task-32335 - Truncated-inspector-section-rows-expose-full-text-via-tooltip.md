---
id: TASK-32335
title: >-
  Truncated inspector section rows expose full text via tooltip
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review C6. Inspector section rows render a truncated secondary line with no tooltip, while changed-files rows carry full un-elided paths in theirs -- inconsistent truncation recovery (console_inspector_section.py rows ~610-617). Add tooltips carrying the untruncated primary+secondary text.

Filed from the 2026-09-10 Console rail UX review (review item C6).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED tests: row tooltip carries full text; patch path refreshes it; markup escaped. 2. Add _refresh_tooltip to the row widget; call from __init__ and _apply_row_update (which now syncs stored texts). 3. Run section + environment + fleet suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every rendered inspector section row has a tooltip containing its full primary and secondary text
- [x] #2 Tooltips are markup-escaped (user content has raised MarkupError before -- PR-T1 I1)
- [x] #3 No per-row layout change
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** `ConsoleInspectorSectionRow` now carries the row's full
untruncated text in a tooltip (`_refresh_tooltip()`: primary + secondary,
joined; `rich.markup.escape`d because tooltips are Rich-parsed content and
row text is user-adjacent). Refreshed at construction and at the end of
`_apply_row_update` — the in-place patch path now also keeps the row
widget's stored `_primary_text`/`_secondary_text` in sync (previously only
the Statics were patched, so a tooltip built from stored text would have
gone stale). No layout change; tooltips only appear on hover.

**ADR check.** Not required — additive display affordance in one widget.

**Modified.**
`tldw_chatbook/Widgets/Console/console_inspector_section.py`,
`Tests/UI/test_console_inspector_section.py` (+2 tests: full-text tooltip
and in-place-patch refresh + markup escape). Verified: inspector-section,
environment-section, fleet-panel suites — 56 passed.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (code): ConsoleInspectorSectionRow sets no tooltip (console_inspector_section.py:820-1003); only chevron and view-all tails have them.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
