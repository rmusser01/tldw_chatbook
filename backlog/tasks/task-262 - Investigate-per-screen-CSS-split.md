---
id: TASK-262
title: Investigate splitting the monolithic stylesheet per screen (~90-130ms first paint)
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, startup]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The bundled tldw_cli_modular.tcss (16,064 lines / 2,278 rules) parses in 88-130ms before first paint. Textual supports per-Screen CSS_PATH/DEFAULT_CSS. Splitting requires a cross-screen selector-dependency audit and per-screen visual QA (regressions are visual, not crashes) — investigation first, then staged extraction starting with the largest per-screen rule blocks. Related future consideration (no task): chat_screen.py's 11k-line module costs ~161ms of pure import as the default tab. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P2 C4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selector-dependency audit documents which rules are shared vs per-screen
- [ ] #2 A staged split plan (or a reasoned decision not to split) recorded
- [ ] #3 If split: measured first-paint delta + per-screen visual QA
<!-- AC:END -->
