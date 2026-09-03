---
id: TASK-25910
title: 'Context: per-turn micro-compaction'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:10'
updated_date: '2026-09-01 17:23'
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
- [x] #1 After a turn completes, the oldest not-yet-absorbed exchange can be folded into the existing memory record without waiting for the trigger ratio
- [x] #2 Cadence is configurable, including fully off, and off reproduces today's behavior exactly
- [x] #3 Micro-compaction reuses the existing compaction planner, memory record, and provenance rather than introducing a parallel path
- [x] #4 It never runs during an active send and never blocks the composer
- [x] #5 The existing ASK mode is honored: if compaction requires consent, micro-compaction does not silently bypass it
- [x] #6 Prompt-cache impact is bounded: the change does not break the stable prefix on every turn - measured and recorded in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: plan max_units cap + default byte-identity + cadence helper\n2. plan_compaction(max_units=) caps the existing span loop (iterative prior-memory fold already built in)\n3. Preflight micro_compaction flag: escalate ONLY below-trigger AUTOMATIC (never ASK), cap plan at 1 unit, all refusals silent\n4. compact_context_now(micro=True) reuses the whole off-send assembly; _set_run_state COMPLETED transition drives a per-session cadence counter + fire-and-forget task with an in-flight guard\n5. Config [console] micro_compaction_every_turns (0=off default), read at ctor
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Micro-compaction reuses the ENTIRE automatic pipeline (AC#3): plan_compaction gains max_units (None = byte-identical today, pinned) capping the existing largest-first span loop at the oldest exchange — the iterative prior-memory fold ('replaces prior memory and only post-boundary units') was already the planner's behavior, so folding IS the existing path. The preflight gains micro_compaction: it escalates ONLY a below-trigger AUTOMATIC decision (ASK is structurally never bypassed — AC#5; any other decision is a silent no-op for a background pass), caps the plan at 1 unit, and an unprofitable fold (summary cap + wrapper >= replaced tokens, the planner's own check) silently waits — the min-reclaim analog. compact_context_now(micro=True) reuses the off-send assembly incl. its active-run refusal (AC#4); the trigger is _set_run_state's COMPLETED transition -> cadence counter (micro_compaction_due, cadence<=0/junk = off) -> fire-and-forget loop task with a per-session in-flight guard. Config [console] micro_compaction_every_turns, default 0 = today exactly (AC#2); read once at controller ctor (per-completion get_cli_setting would exhaust finite-side_effect test doubles — found live as 98 StopIterations before switching, which also turned out to be a pre-existing failing suite at the pre-lane-2 base, verified by worktree bisect). AC#6 (cache impact, recorded): a fold rewrites ONLY the memory row — system+tools prefix bytes are untouched but the provider cache breaks from the memory row onward on fold turns; cadence N bounds breaks to 1/N of turns vs one monolithic break + stall at the trigger ratio; default off = zero new breaks. 3 new tests; compaction suite 138 passed; range-to-prefix (GENERATED_RANGE) memories no-op under micro by design (documented in plan_compaction docstring).
<!-- SECTION:NOTES:END -->
