---
id: TASK-1551
title: 'Audit Tests/UI geometry assertions made under CSS-less harnesses'
status: To Do
assignee: []
created_date: '2026-07-30 17:00'
labels: [tests, ui, harness]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleHarness (and its siblings in Tests/UI) is a bare App[None] that pushes ChatScreen
directly and never loads the app CSS bundle, so no rule in tldw_cli_modular.tcss applies
under it. A live-gate failure (composer draft cropped to one row by a height: 1 rule)
shipped through two review-approved fix rounds because every geometry assertion ran under
such a harness — see the 2026-07-30 entry in backlog/docs/lessons-testing-evidence.md and
_CssTrueConsoleHarness in Tests/UI/test_console_composer_overflow.py. Other height /
clipping / visibility assertions across Tests/UI carry the same blind spot and may be
green against defects users can see.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Every Tests/UI assertion about on-screen geometry (heights, regions, clipping, row visibility) is inventoried with whether its harness loads the real CSS bundle.
- [ ] #2 Assertions that are load-bearing for user-visible layout run under a bundle-loading harness (shared helper, not per-file copies).
- [ ] #3 The shared helper lives in one importable place and the lessons entry is updated to point at it.
<!-- AC:END -->
