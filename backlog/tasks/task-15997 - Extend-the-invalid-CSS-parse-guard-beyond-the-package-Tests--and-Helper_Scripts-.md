---
id: TASK-15997
title: 'Extend the invalid-CSS parse guard beyond the package: Tests/ and Helper_Scripts/'
status: To Do
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - tests
  - hardening
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15450's `test_every_class_level_css_block_parses_as_a_stylesheet` runs every class-level CSS block in the tldw_chatbook package through `Stylesheet.parse()` (the API that actually raises — `textual.css.parse.parse` only collects errors), and on its FIRST run found two live crashers nobody knew about (`audio_troubleshooting_dialog`, `dictation_performance_widget` — plus the two selection dialogs that motivated it). Nothing sweeps `Tests/` or `Helper_Scripts/` (including the custom splash-card examples) for the same defect class: an invalid property in any sheet poisons the whole stylesheet at parse time. Extend the guard's walk to those trees (reusing its block-extraction helper), fix what it finds, and record what it found. Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The parse guard covers class-level CSS blocks under Tests/ and Helper_Scripts/
- [ ] #2 Any newly-found invalid blocks are fixed (removal or translation, with the never-parsed rationale applied as in TASK-15450)
- [ ] #3 Notes record the found-count so the sweep's value is measurable
<!-- AC:END -->
