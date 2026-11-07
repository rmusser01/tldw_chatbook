---
id: TASK-22506
title: >-
  Retire or mount the dead loading_states module
status: Done
assignee: []
created_date: '2026-08-26'
labels:
  - cleanup
  - owner-decision
priority: low
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Found by TASK-22220 while fixing its forever-timer: `Widgets/loading_states.py` has ZERO
imports anywhere in `tldw_chatbook/` or `Tests/` — only CSS rules reference `InlineLoader`.
The timer leak 22220 fixed therefore had no production mount path. Same family as the perf
review's own dead-timer inventory (Tamagotchi, DetailedProgress — both also unmounted).

Retiring vs mounting is an owner call, which is why this is filed rather than actioned.

## Acceptance Criteria

- [x] A decision is recorded: retire the module (and its CSS) or mount it where the loading affordance is wanted
- [x] If retired, the CSS rules and any orphaned tests go with it and the diagnostic/CSS artifacts are regenerated
- [ ] The sibling dead widgets (Tamagotchi, DetailedProgress) get the same decision or an explicit deferral

## Closeout audit 2026-10-11

Board-prune audit (branch chore/board-prune-c, base a3c47c6c72): completed-but-unmarked. Evidence gathered 2026-10-11 against origin/dev:

AC dispositions:
- AC#1: met - Decision: RETIRE - executed. Widgets/loading_states.py deleted with its dedicated tests (test_inline_loader_timer.py, test_detailed_progress.py, test_toast_notifications.py) by commit 5f3adeca33 (ADR-161 task 2 fix).
- AC#2: met - CSS/diagnostic artifacts regenerated in the same commit (test_widget_css_consolidation.py adjusted with it).
- Sibling DetailedProgress retired in the same commit; Tamagotchi (Widgets/Tamagotchi) still ships and is referenced by Backup_Recovery/owner_registry.py - AC#3's Tamagotchi half has no explicit deferral note found, left unticked.
