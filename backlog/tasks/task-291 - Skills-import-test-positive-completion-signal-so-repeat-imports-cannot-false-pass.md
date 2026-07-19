---
id: TASK-291
title: >-
  Skills-import test: positive completion signal so repeat imports cannot
  false-pass
status: Done
assignee:
  - '@claude'
created_date: '2026-07-17 17:25'
updated_date: '2026-07-17 22:45'
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
- [x] #1 A slow-but-successful repeat import cannot be reported as complete by stale outcome text
- [x] #2 The skills-import suite still passes deterministically
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Product-side fix (stronger than a test-only signal): the import success line now names the imported skill ('Imported "<name>" · re-review it in the trust panel') — stale text from the previous import can never equal the next expected line, and users can attribute the outcome. All 7 test assertions are name-specific; duplicate-skip copy already carried the name. Suites: Tests/Skills 9+ passed, Tests/UI/test_library_skills_canvas.py green (168 total across both). Commit a816c65e.
<!-- SECTION:NOTES:END -->
