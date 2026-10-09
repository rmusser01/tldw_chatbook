---
id: TASK-34435
title: Eliminate per-turn double collection of dictionaries and world books on the
  console seam
status: Done
created_date: 2026-10-08 00:35
updated_date: 2026-10-09 06:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up (deferred T3e/T4d during console maintenance): capture_prompt_transform_inputs collects the same books/entries the resolver collects, so every console send pays the collection twice. The interface seams (books=/entries=) landed in TASK-34666/34416 - wire the pre-collected bundles through the console controller appliers. Coordinate with console pipeline maintenance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One collection per turn verified by spies,Console appliers use the pre-collected bundles,Console tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 3 subtask 3e + Task 4 subtask 4d
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
VERIFIED AT TIP (2026-10-08, worktree perf/nonconsole-followups @ b3ffce0c7a): the per-turn double collection does NOT exist anymore; already eliminated by the console-pipeline maintenance absorbed into dev tip. Evidence: (1) code trace -- Chat/console_runtime.py _apply_chat_dictionaries_for_app/_apply_world_info_for_app (frozen_inputs path, landed with b7dd8e53f2 'PR 2504 rebase' integration) consume turn_context.prompt_transform_inputs directly; the controller (Chat/console_chat_controller.py _apply_chat_dictionaries ~22528 / _apply_world_info ~22400) always passes frozen_inputs on provider send paths, so the appliers' self-fetch fallback never runs when a turn snapshot exists. (2) Spy test Tests/Chat/test_console_prompt_transform_single_collection.py (new): one production-wired console send -> collect_active_chatdict_entries runs exactly 1x, _collect_active_world_books exactly 1x, WorldBookManager.get_world_books_for_conversation exactly 1x. (3) RED control: rebinding pre-fix appliers (ignore frozen bundle, always self-fetch via apply_active_chatdicts_to_text/apply_world_info_to_message) yields 2x per side -- proving the committed spy test discriminates a regression. Mid-turn edit semantics (per-turn snapshot consistency) pinned hermetically: a book edited inside the applier call is NOT picked up. No production change needed; seams books=/entries= remain available for other callers. Console appliers effectively use the pre-collected bundles (via the runtime frozen-input path rather than the resolver-level params).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Already fixed at tip by console-pipeline maintenance (b7dd8e53f2 frozen-input applier path): one collection per turn per side verified by committed spy test + RED control (2x under pre-fix shape); mid-turn snapshot consistency pinned. No production change required.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
