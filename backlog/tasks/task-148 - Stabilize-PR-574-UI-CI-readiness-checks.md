---
id: TASK-148
title: Stabilize PR 574 UI CI readiness checks
status: Done
assignee: []
created_date: '2026-06-30 19:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Harden the PR 574 UI check failures that pass in isolation but fail under CI suite load because Textual screen/category updates are asserted before the target state is mounted or rendered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Settings category tests wait for selected category content before querying or saving
- [x] Library navigation context waits for the requested conversation selection before asserting
- [x] Library mode-switch checks wait for active mode-specific content
- [x] Console stale-search test waits for fresh rendered browser results before asserting stale rows are gone
- [x] Targeted failing CI slice passes locally
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test readiness hardening only; no architectural, storage, provider, or long-lived UX contract change.

1. Add or reuse focused wait helpers for Textual category/mode readiness.
2. Patch only the tests implicated by the latest PR UI failure summary.
3. Run the exact failing CI slice.
4. Run affected UI test files and diff hygiene checks.
5. Record verification in implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added focused Textual readiness waits for the PR 574 UI CI failures instead of adding fixed sleeps. Settings tests now wait for the target category button, active category, and expected mounted selector/text before querying controls or saving. Library navigation waits for the requested conversation ID and visible title before asserting. The Library mode helper now allows full CI render latency and performs a final text sample before failing. The Console stale-search regression waits for the fresh conversation row ID before asserting stale rows are absent. The latest-dev smoke test waits for the Settings provider SelectOverlay child before exporting the screenshot. Follow-up CI passes exposed additional full-suite races, so the Console stale-search test now starts the fresh beta refresh with the current token after the stale task returns, Settings destination tests wait for mounted copy, Library replay waits for mode controls, the cached navigation smoke test presses nav buttons directly instead of relying on bottom-row coordinates, and late Settings provider/revert tests wait for mounted controls and restored input values. Verification run locally: latest 9-test CI failure slice passed; older 9-test PR failure slice passed; remaining 5-test CI failure slice passed; final 3-test Settings CI slice passed; affected UI files passed in 379-test and 445-test runs; full Settings file passed; git diff --check passed.
<!-- SECTION:NOTES:END -->
