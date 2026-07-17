---
id: TASK-291
title: >-
  Skills-import test: positive completion signal so repeat imports cannot
  false-pass
status: To Do
assignee: []
created_date: '2026-07-17 17:25'
labels:
  - testing
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Import row's outcome Static is never cleared between imports and all five superpowers fixture skills emit byte-identical success copy, so _run_skills_import_via_ui's change-detection can false-pass on imports 2-5: if a slow-but-successful import finishes after the deadline, the helper returns the unchanged 'previous' text that already equals the expected success string. Deferred from the 2026-07-17 de-flake pass (Docs/superpowers/reviews/2026-07-17-deflake-diagnosis.md §2 + review recommendation). Give the test a positive completion signal — e.g. a per-import sequence/generation counter on the outcome line, or asserting the imported-name set grows — so each import's success is individually attributable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A slow-but-successful repeat import cannot be reported as complete by stale outcome text
- [ ] #2 The skills-import suite still passes deterministically
<!-- AC:END -->
