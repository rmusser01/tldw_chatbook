---
id: TASK-31732
title: Resolve descriptor growth before Canvas V1 integration
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 19:46'
updated_date: '2026-09-05 19:47'
labels:
  - canvas
  - testing
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate the descriptor-growth signal reported by the Canvas acceptance runs so integration is based on understood resource ownership rather than an unexplained warning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The original growth signal is reproduced or bounded with isolated, source-free evidence identifying resource categories and responsible test or runtime lifetimes.
- [ ] #2 Any confirmed in-scope resource leak is corrected with a failing regression and targeted passing controls without hiding the sentinel or weakening cleanup guarantees.
- [ ] #3 The final affected run and independent review document the outcome and retained limitations before the Canvas pull request proceeds.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR for diagnosis or direct lifetime repair; existing ADR-115 applies if Canvas ownership is involved. ADR path: backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md. Reason: investigate and restore existing cleanup guarantees without new resource, storage or security policy; revisit before any architectural change.
1. Reproduce the recorded 207-descriptor growth in the exact ten-module selection under existing pre-import test isolation, collecting bounded per-test/module counts and source-free resource categories.
2. Compare normal collection and TLDW_TEST_GC_EVERY=1; distinguish live retained owners, cycles, SQLite descriptor reuse and sentinel timing. Narrow to the responsible real lifetime before choosing a fix.
3. Add a discriminating regression for any confirmed leak; preserve native and served shutdown, temporary history, threaded DB ownership and default sentinel thresholds. Update this plan with the exact correction before product edits.
4. Root runs targeted RED/GREEN and the affected selection; an independent static reviewer checks the scoped correction. Record commands, warnings, limitations and any generalizable lesson; close only after all ACs are met.
5. Integration authorization is tracked in the Canvas implementation plan: fetch/rebase latest dev, preflight and affected verification, create or update the PR against dev, address Qodo feedback, wait for checks, merge without bypassing protections, then begin V2 design.
<!-- SECTION:PLAN:END -->
