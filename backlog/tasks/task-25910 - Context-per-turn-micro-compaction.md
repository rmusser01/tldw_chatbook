---
id: TASK-25910
title: 'Context: per-turn micro-compaction'
status: To Do
assignee: []
created_date: '2026-08-31 15:10'
updated_date: '2026-08-31 15:11'
labels:
  - console
  - context
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Compaction currently happens in one batch at the trigger ratio, producing a visible stall in long sessions. Verified on origin/dev: Chat/console_chat_controller.py:18054 runs the decision at send preflight and Chat/console_context_compaction.py:770 decides tri-state on the trigger ratio; a named grep for micro_compact across tldw_chatbook returns zero. Hermes amortizes instead, folding the oldest un-absorbed exchange into a rolling summary each turn. Chatbook's compaction machinery is otherwise ahead of hermes (ask/auto/off modes, provenance, branch-aware memory, honest model-window reporting) - this closes the one place it is behind, and it sits on plan_manual_range (console_context_compaction.py:1047), which already supports arbitrary-range compaction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a turn completes, the oldest not-yet-absorbed exchange can be folded into the existing memory record without waiting for the trigger ratio
- [ ] #2 Cadence is configurable, including fully off, and off reproduces today's behavior exactly
- [ ] #3 Micro-compaction reuses the existing compaction planner, memory record, and provenance rather than introducing a parallel path
- [ ] #4 It never runs during an active send and never blocks the composer
- [ ] #5 The existing ASK mode is honored: if compaction requires consent, micro-compaction does not silently bypass it
- [ ] #6 Prompt-cache impact is bounded: the change does not break the stable prefix on every turn - measured and recorded in the notes
<!-- AC:END -->
