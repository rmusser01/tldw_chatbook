---
id: TASK-28029
title: Fix console_view_hooks crash on pre-wiring screens (13 red tests)
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-02 14:54'
updated_date: '2026-09-02 15:08'
labels:
  - console
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The wave-6 controller-extraction refactors (cd5209b37 retrieval, a315c9220 skill) rewired three console_view_hooks entries to eager instance-attribute derefs (self._retrieval / self._skill) that only exist after build_console_controllers. Any ChatScreen that reaches runtime attach before wiring (bare __new__ screens in tests; the documented early-attach path) crashes with AttributeError, leaving 12 state-restore tests plus the attach/detach slot-set contract test red since Aug 20-21.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 console_view_hooks returns the exact CONSOLE_VIEW_HOOK_SLOTS key set on a bare __new__ ChatScreen
- [x] #2 The 12 test_console_native_chat_flow state-restore tests pass
- [x] #3 test_attach_and_detach_cover_exactly_the_same_slot_set passes
- [x] #4 Fully-wired screens bind identical values as before (no behavior change)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (bug fix restoring an existing documented contract). 1. Defensive getattr for _retrieval/_skill in console_view_hooks with None fallback matching each slot's viewless default. 2. RED->GREEN across the 13 tests. 3. Targeted suites around wiring/hooks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Against `dev` (PR base) the disease has one remaining exposed nerve: dev's wave-6 refactor already made `console_view_hooks` defensive (`getattr(retrieval, "_capture_console_staged_rag", None)`), but `_ensure_console_chat_store` eagerly dereferences the wiring-set `self._workspace` (chat_screen.py:7096/7102) and every bare-screen state-restore test drives it through `_restore_console_snapshot_with_sessions`. Fix: `getattr(self, "_workspace", None)` guard; no workspace controller -> `workspace_context=None` (the same safe runtime default the `resume_pending` branch already uses) and skip the empty-store realignment. One fix heals the whole family: 13/13 state-restore tests green, full chat-flow suite 24 -> 13 failures (the remaining 13 are the pre-existing prompt-improvement/send-path/toast rot on dev, untouched by this change -- see PR body). Original branch-side diagnosis (eager `_retrieval`/`_skill` in the hook dict) is recorded above for the history; on dev that half is already landed.

## Renumbering provenance

Originally created as TASK-28027 on 2026-09-02; renumbered to TASK-28029 before merge
because parallel Library-media PRs landed on `dev` first carrying
task-28027/task-28025 for different
work. Per the TASK-19601 owner rule (older arrival keeps the id), the
landed tasks keep those numbers.
