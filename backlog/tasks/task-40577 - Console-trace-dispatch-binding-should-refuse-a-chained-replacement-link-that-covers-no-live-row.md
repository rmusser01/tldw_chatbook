---
id: TASK-40577
title: >-
  Console: trace dispatch binding should refuse a chained-replacement link that
  covers no live row
status: In Progress
assignee:
  - '@claude'
created_date: '2026-11-12 23:16'
updated_date: '2026-11-20 16:56'
labels:
  - console
  - trace
dependencies:
  - TASK-38298
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The chained trace replacement (TASK-38298, ADR-337 amendment 2026-11-06) is revalidated at dispatch binding: each link's start and end nodes, that its range is contiguous, and that no live row lies between links. Binding does not check that each link covers at least one live row of the surface it replaces. The planner never builds such a link, but if a malformed plan reached binding it would commit, and every later read of that trace segment would fail (surface_replacement_target_inactive) instead of one send being refused. A per-link live-row check would keep the failure closed and local to the send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dispatch binding refuses, before writing anything, a chain in which any link's range holds no live row of the predecessor surface
- [x] #2 A test pins the refusal and is red without the check
- [x] #3 Chains the planner builds and single replacements still bind and read back unchanged
<!-- AC:END -->

## Identity

Filed 2026-11-12 from the PR #3515 review (finding F3). The backlog CLI offered TASK-40334, which another branch already holds; a sweep of every ref's task paths and every worktree found a maximum of 40546, so this task takes 40577.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the premise on dev: a link over dead sequences only, reaching binding, commits and then every read of the new head fails (surface_replacement_target_inactive).
2. Write a test that drives the real planner (with a defective link grouping), preparation (its shared check skipped) and dispatch binding; assert the bind is refused as rolled_back, the refusal is replacement_chain_link_inactive raised with zero rows changed in the binding transaction, counts unchanged, the call still reserved, the old head still reads, and the next real send records. Show it red on dev.
3. Add the per-link live-row check to _validate_replacement_range, in the same pass over the live entries as the no-live-row-between check, so it runs before the first write of the binding transaction (and at preparation).
4. Mutation-check the new check; run every trace dispatch/binding/chain/read-back test file and compare reds with origin/dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Dispatch binding now refuses a chain link that holds no live row, before it writes anything.**

**The premise, confirmed on dev d85e90bc9e** with a scratch probe, since deleted. An earlier edit left sequences 1..130 dead. A defective grouping, (0,0) (1,130) (131,255) (256,260), passed preparation and binding, and the call committed as `dispatch_started`. A cold read of the new head then failed with `surface_replacement_target_inactive`, and so did the planning of the chat's next send.

**The change.** `_validate_replacement_range` already makes one pass over the predecessor surface's live entries. It now maps each linked sequence to its link and records the link each live row falls in:
- A live row inside the range but outside every link is still `replacement_chain_gap`.
- A link that no live row falls in is now `replacement_chain_link_inactive`.

The request writer runs this check before the binding transaction's first write, so the send is refused with nothing written, and preparation refuses the same plan. No schema change. The ADR-337 amendment's implementation paragraph now names the check. The size ratchet row for `console_trace_service.py` moves 6739 -> 6746 (+7, exact).

**The test.** `test_dispatch_binding_refuses_a_link_that_covers_no_live_row` in `Tests/Chat/test_console_trace_chained_replacement.py` drives:
- the real planner, with the defective grouping standing in for a planner defect;
- the real preparation, with its copy of the check skipped so that binding is tested alone;
- the real `bind_and_mark_dispatch`.

It asserts:
- `TraceCallPersistenceError` with outcome `rolled_back`;
- the refusal is `replacement_chain_link_inactive`, raised with 0 rows changed in the binding transaction (`connection.total_changes` at `persist_request` entry against its value at the refusal);
- trace row counts unchanged;
- the call still `reserved` with no surface node;
- the previous head still reads back;
- preparation refuses the same plan.

The real send then records the planner's own chain, (0,255) (256,260), and reads back warm, cold and through the native reader.

**Evidence.**
- Red on dev: `Failed: DID NOT RAISE TraceCallPersistenceError`.
- Green with the fix.
- Mutation: with the new `raise` disabled, red again with the same DID NOT RAISE.
- Preparation's copy of the check is skipped in the test, so the binding layer is tested alone, not shadowed by the earlier guard.
**AC#3: chains the planner builds, and single replacements, still bind and read back.** Every trace dispatch, binding, chain and read-back file was run, one file per command, on this branch and on a detached dev d85e90bc9e worktree:
- Chained replacement 12 passed (dev 11; +1 is the new test); rewritten history 14; rewritten-tail planner 4; rewritten tail 26.
- Trace service 77, whose single-replacement cases are among them; trace runtime 87; call lifecycle 15.
- Dispatch continuation handoff 73; discarded tool run 31; failed tool run 51; refusal diagnostics 3; spawn sub-agent capture 20.
- Continuation 29; fork lineage 7; native reader 9; repository 27; projection 72.
- Surface identity 9; surface recovery 8; first-send atomicity 10; compaction 10; compaction admission 10.
- Final values 27; models 250; chat controller 368.

Five reds are dev's own, identical by test id and failure on dev d85e90bc9e:
- `test_console_trace_graph_gc.py::test_reopen_has_no_trace_payload_in_any_durable_owner`;
- `Tests/UI/test_console_surface_refusal_recovery.py::test_surface_refusal_recovers_the_existing_accepted_user[retry|send-without|cancel]` (IndexError);
- `Tests/Benchmarks/test_console_trace_growth.py::test_console_trace_growth_release_gate[append]` (append dispatch refused at turn 104).

None of them reaches a chained replacement.

Not verified in the running app. The planner never builds such a link, so no UI path reaches the check, and launching the TUI would use the owner's real profile.
<!-- SECTION:NOTES:END -->
