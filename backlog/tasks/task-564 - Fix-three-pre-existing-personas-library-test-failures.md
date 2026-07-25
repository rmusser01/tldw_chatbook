---
id: TASK-564
title: >-
  Fix three pre-existing personas/library test failures
status: To Do
assignee: []
created_date: '2026-07-24 23:34'
updated_date: '2026-07-24 23:34'
labels:
  - tests
  - library
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three test failures surfaced by the image-gen P3 verification sweeps (2026-07-24) and attributed as PRE-EXISTING via a throwaway worktree at the P3 plan base (`133330366` — they fail identically without any P3 changes): a library-scale notify-signature mismatch, a workbench hidden-directory export assertion, and a library footer-hint text drift. They are unrelated to image generation; filing so the baseline stops accreting known-red tests that every branch must re-attribute.

First step per the P3 final review: re-confirm each still reproduces on current `origin/dev` before fixing, so the attribution survives review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Each failure is re-confirmed on current `origin/dev` (or closed as already-fixed) with the failing test name + one-line root cause recorded in the Implementation Notes.
- [ ] The genuine defects (product code or stale test expectation — determine which per case) are fixed; all three tests pass.
- [ ] The relevant suites run green with no new failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
