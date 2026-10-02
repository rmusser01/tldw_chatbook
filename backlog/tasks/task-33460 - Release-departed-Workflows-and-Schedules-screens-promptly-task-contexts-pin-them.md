---
id: TASK-33460
title: Release departed Workflows and Schedules screens promptly (task contexts pin
  them)
status: To Do
created_date: 2026-09-29 19:04
labels:
- performance
- memory
- perf-audit-2026-09
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while measuring PERF-11 (TASK-33270) on a scratch profile with PERF-05 applied, after a 24-visit destination tour and a full collection. A departed WorkflowsScreen stays alive for the whole session: WorkflowAuthoring keeps its finished _opening task (Workflows/authoring.py, open()), and that task's contextvars context was copied at create_task while the screen was Textual's active_message_pump. Departed SchedulesWorkbench instances (3 after the tour) are held by the contexts of cancelled asyncio TimerHandles, which the loop keeps until their original due time, so each lingers for up to one timer period. Retained screens keep their widgets and render-line caches (Textual Strips) in the heap that every full GC pass walks, so they lengthen gen-2 pauses. Holder chains were captured with a gc referrer walk: Context -> hamt -> WorkflowsScreen (holder: finished Task WorkflowAuthoring._open) and Context -> hamt -> SchedulesWorkbench (holder: TimerHandle with a None callback, due in 27 s).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A departed WorkflowsScreen is collectable after navigating away (weakref test, like Tests/Performance/test_screen_leaks.py)
- [ ] #2 A departed SchedulesWorkbench is collectable within one short settle after navigating away, not only after its timers' original due times
- [ ] #3 The fix does not change what the Workflows authoring store or the Schedules timers do while the screen is mounted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
