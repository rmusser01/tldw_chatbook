---
id: TASK-15390
title: Search/RAG gate16 evidence-heading test fails on clean dev
status: To Do
assignee: []
created_date: '2026-08-11 17:25'
labels:
  - library
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the library-queue batch (task-14902's implementation run, 2026-08-11) and
A/B-verified twice — most recently by the batch's whole-branch review against a CLEAN
`origin/dev` checkout (`484d25b5e`) in a temporary worktree, where it fails identically:

`Tests/UI/test_library_shell.py::test_evidence_heading_and_coverage_note_are_mode_aware_and_conditional`
(the "gate16" family) fails on dev with no library-queue changes present. Not caused by, and
not maskable by, the 14902 choice-strip work — the batch's own targeted suites are green
around it.

Nobody in this arc root-caused it (out of scope both times it surfaced); it is NOT in the
long-standing known-ambient list (the old ~45 shell-geometry failures were fixed on dev
separately), so it is presumably a recent regression or a test drifted from a deliberate
Search/RAG copy change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified: production regression vs. stale test assumption, with the introducing commit named
- [ ] #2 The test passes on dev (production fixed, or the pin rewritten to the intended contract with the change documented)
<!-- AC:END -->
