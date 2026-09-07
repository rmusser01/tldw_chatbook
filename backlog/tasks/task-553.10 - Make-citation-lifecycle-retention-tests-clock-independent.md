---
id: TASK-553.10
title: Make citation lifecycle retention tests clock-independent
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 16:26'
updated_date: '2026-07-24 16:31'
labels:
  - rag
  - citations
  - tests
dependencies: []
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fixed-clock GC tests import _persist, which stamps live message and conversation timestamps. After wall clock passes fixed NOW, max-owner retention blocks collection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Lifecycle fixture timestamps are deterministic relative to NOW.
- [x] #2 The six failing collection tests pass regardless of current date.
- [x] #3 The lifecycle suite and foundation gate have no new regression.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. ADR required: no; ADR path: N/A; Reason: test-only fixture correction that preserves production lifecycle, storage, and policy boundaries.
2. Reproduce the six current fixed-clock lifecycle collection failures and trace timestamp ownership.
3. Alias the repository _persist fixture helper and wrap it locally to normalize created message, conversation, and owner timestamps to a deterministic baseline before NOW while preserving per-test overrides.
4. Run the six regressions, the complete lifecycle suite, and the exact foundation gate where feasible.
5. Run Ruff, formatting, and diff checks; self-review, complete task notes and acceptance criteria, set Done, and commit the task plus test-only correction.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a lifecycle-test-local wrapper around the shared citation repository persistence helper. The wrapper preserves repository behavior, then anchors the created conversation, message, and trace-owner timestamps at NOW minus one day so retention assertions are independent of wall-clock date while later per-test timestamp overrides remain authoritative. No production code or lifecycle policy changed. ADR required: no; ADR path: N/A; Reason: test-only fixture correction preserving existing architecture and policy boundaries. Verification: six isolated regressions passed; lifecycle suite 50 passed; exact foundation gate 762 passed with one pre-existing dependency warning; Ruff check and format check passed; git diff --check passed.
<!-- SECTION:NOTES:END -->
