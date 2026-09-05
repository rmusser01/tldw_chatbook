---
id: TASK-31746
title: Remove inert legacy Notes auto-sync timer residue
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:56'
updated_date: '2026-09-05 20:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the retired single-root sync cutover by removing the unused Library timer slot, while preserving current autosave and lasting-sync lifecycle ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing cutover guard finds no legacy Library sync timer, worker or mutating handler
- [x] #2 Real fifty-route and unmount tests assert legacy timer absence while retaining their current autosave, worker and controller checks
- [x] #3 No lasting-sync behavior, timer scheduling or size ceiling is changed beyond removing the unused legacy slot
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the sole product reference is an inert None assignment and capture the existing cutover RED plus both lifecycle baselines.
2. Remove only that assignment; migrate the two obsolete None assertions to genuine attribute absence without weakening the AST cutover guard.
3. Run cutover and real lifecycle tests, lint/diffcheck, record exact screen size and parent review, then commit separately.
ADR required: no new ADR
ADR path: backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md
Reason: Existing ADR retires the single-root legacy engine in favor of device-local lasting-sync ownership; this removes dead residue without changing ownership or scheduling.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the sole inert legacy auto-sync timer assignment. Repository-wide product census found no reader, scheduler, worker, or teardown use; current lasting-sync invalidation and the separate note autosave timer remain unchanged. Two real lifecycle tests now require actual absence instead of preserving an unused None slot. The unchanged AST cutover test failed before this removal and passes afterward.
Baseline real lifecycle tests: 2 passed (57.70s). Full cutover and both lifecycle tests after removal: 32 passed (72.63s), including fifty route cycles and unmount cleanup. Shell test Ruff and diffcheck pass; screen retains the same 40 preexisting Ruff findings, no new ones. Screen is now 41302 lines /1301 methods.
ADR: existing ADR-059; no new ownership or scheduling decision. No behavioral guard, timeout, or ceiling was relaxed. Parent review pending before scoped commit.

Parent reviewed the exact diff with no blocking findings. Final Library ratchet tightened to 41302 lines / 1301 methods and its two checks pass; scoped Ruff and git diff --check pass. Existing screen Ruff debt remains unchanged.
<!-- SECTION:NOTES:END -->
