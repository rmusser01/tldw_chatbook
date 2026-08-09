---
id: TASK-3312
title: >-
  Ingest copy polish from live verify: F1 esc dupe, egress failure receipt, warning noun echo, panel-header focus
status: To Do
assignee: []
created_date: '2026-08-08 00:30'
labels:
  - library
  - ingest
  - ux
  - copy
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cosmetic findings from the 3300-3305 arc live verification (2026-08-08):

1. F1 help in Ingest lists escape twice ("esc: back to hub" from the shared shortcut set + "escape: Back to Library hub" from BINDINGS).
2. The queue failure receipt for an egress-blocked URL leaks a markup escape (`\[web_security]`), ends mid-sentence with a trailing comma, and is far more technical than the plain-language inline preflight line one row above (task-3305's mapping) — route it through the same plain-language treatment with the remedy intact.
3. Guardrail modal warning line can repeat its noun ("- Audio processing (1 file): Audio processing") when the feature label equals the capability hint — suppress the echo.
4. The collapsible options-panel header is a Tab stop whose focus is color-only (glyph-less) — the one focusable the task-3302 treatment missed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 F1 in Ingest lists one escape row
- [ ] #2 Egress-blocked URL receipts render plain language, no markup-escape leak, no dangling comma, remedy preserved
- [ ] #3 Guardrail warning lines never repeat the feature name as their own hint
- [ ] #4 Focused collapsible panel headers show a glyph-level focus indicator
<!-- AC:END -->
