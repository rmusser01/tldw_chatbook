---
id: TASK-15390
title: Search/RAG gate16 evidence-heading test fails on clean dev
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 17:25'
updated_date: '2026-09-17 04:34'
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

`Tests/UI/test_product_maturity_gate16_library_search_rag.py::test_evidence_heading_and_coverage_note_are_mode_aware_and_conditional`
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
- [x] #1 Root cause identified: production regression vs. stale test assumption, with the introducing commit named
- [x] #2 The test passes on dev (production fixed, or the pin rewritten to the intended contract with the change documented)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A; existing backlog/decisions/003-settings-library-rag-defaults.md defines unchanged Library result-display ownership.
Reason: test isolation and task/audit correction only; no production or architectural change.
1. Trace the original result-count ordering drift and its existing correction. Reproduce the current test on the working branch and a clean exported saved origin/dev snapshot.
2. Remove the heading unit test dependency on real profile storage: explicitly provide multiple valid depths, preserve mode suffix, coverage content and child ordering assertions, and verify the same test change against saved dev.
3. Run the focused Search/RAG gate without exclusions plus existing depth-resolution tests. Record history, current failure and green evidence; independently review, update task/audit records and commit locally. No full repository suite or integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The evidence-heading rendering test now controls profile depth at the existing UI helper and checks 5, 15 and 23. Real panel construction, mode suffixes, coverage contents/absence and count-to-coverage ordering remain covered; production behavior is unchanged.

Root cause: 13916d8abe inserted the count line before coverage; 67c667d063 had already corrected that historical ordering assertion. The unchanged branch gate16 suite passes 69 tests. The later f875d8225f profile-depth pin assumed 15; saved dev 24094f23d59c7a9d3cfac964c19fd263bc0393b2 adds storage admission (b5251e9a6eb), exposing a separate fixture dependency where unavailable profile/config storage yields the intended fallback 5. The repaired test supplies its rendering input above the guarded import; separate state/config tests retain actual resolver coverage.

Verification: 277 focused tests pass with no exclusions; the same three repaired heading cases pass against a clean export of saved dev using the existing dependency environment. Import-root, AST and relevant production-byte comparisons verify isolation. No new Ruff diagnostics (five existing remain), changed-range formatting and diff whitespace checks pass. Independent final review reports no findings. No native rerun was needed for this test-only change; no full suite, push or merge.

Modified the gate16 test, audit ledger, prior QA follow-up notes, and testing lesson; evidence and attempt history: Docs/superpowers/qa/2026-09-17-rag-evidence-heading/README.md. Fixture attempts using an unsupported constructor keyword and a lower resolver import were corrected before final verification. No new ADR required; existing backlog/decisions/003-settings-library-rag-defaults.md governs the unchanged profile/display boundary.
<!-- SECTION:NOTES:END -->
