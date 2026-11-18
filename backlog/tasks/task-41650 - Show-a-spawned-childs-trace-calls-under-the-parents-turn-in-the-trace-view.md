---
id: TASK-41650
title: "Show a spawned child's trace calls under the parent's turn in the trace view"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 19:05
labels:
- follow-up
- trace
- ui
priority: medium
references:
- tldw_chatbook/UI/Screens/trajectory_screen.py
- tldw_chatbook/Chat/console_trace_repository.py
dependencies:
- TASK-41310
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41310 records each spawned child's requests in its own trace segment under the parent's turn (schema v87, `console_trace_child_segments`). The trace view does not show them yet; on 2026-11-17 the owner chose "later, as a follow-up".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The trace view lists a child run's calls under the parent turn that spawned them
- [x] #2 A test spawns a child with capture on and asserts that the parent's turn shows the child's calls
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (display only; ADR-337's 2026-11-17 amendment already decides ownership and names this reader as the follow-up; a dated note records that the Trace view now reads child calls)

1. Repository: one read, `ConsoleTraceRepository.read_child_calls(cursor, conversation_id)`, returning the calls of every child segment linked to the conversation's attached owner, in creation (spawn) order. Uses the existing segment-order index; no new index.
2. Projection (`Chat/trajectory.py`): `derive_trajectory(child_calls=...)` adapts each call into a `child_call` record on the spawning turn (`turn_id` from the link), depth 1 (indented like other nested rows), label "Child call", preview `sub-agent N · call M · <route>` (N numbered per turn in spawn order), state from the call, model/provider, dispatch and settle times, `actor_kind="subagent"` and `run_id` = the child's chain (so `a`/`s`, the Agent filter and the timeline's agent lane pick them up with no screen change).
3. Snapshot builder (`UI/Console_Modules/review_selection.py`): read child calls through the repository in one transaction; a failure becomes the usual `capture_failed` diagnostic row.
4. Test: extend the TASK-41310 spawn harness to build the trace view's snapshot; mount `TrajectoryScreen` and assert the parent's turn lists each child call (two siblings with a tool round) labelled as a child, and no parent call is labelled as one. Show it red without the builder read.
5. Docs: `Docs/User_Guide/console/semantic-trace-capture.md` (child calls in the Trace); ADR-337 dated note; size ratchet row for the repository if it grows.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Trace view (`y`) lists each spawned child's provider calls as nested **Child call** rows of the turn that spawned it. No screen code changed: the rows reuse the ledger's existing record conventions.

- **Read.** `ConsoleTraceRepository.read_child_calls(cursor, conversation_id)`: the calls of every `console_trace_child_segments` link under the conversation's attached owner, ordered by creation. It is served by `idx_console_trace_calls_segment_order`; no new index. The snapshot builder (`UI/Console_Modules/review_selection.py::_build_trajectory_snapshot`) calls it in one `db.transaction()` and turns a failure into the usual `capture_failed` diagnostic row. No raw SQL in the UI.
- **Projection.** `derive_trajectory(child_calls=...)` -> `_records_from_child_calls`: kind `child_call`, label `Child call`, depth 1 (the `↳` indent), `turn_id` from the link, summary `sub-agent N · call M · <route>`. N numbers children per turn by dispatch time (the order the ledger shows them), so siblings never collide and nothing reads as the parent's. State comes from the call; model, provider, dispatch and settle times are shown, with no first-token time (none is recorded) and no token counts. `actor_kind="subagent"` and `run_id` (the child's chain) make `a`/`s`, the Agent filter and the timeline's agent lane include the rows. Event ids are `trace-call:<call_id>`.
- **Design choices.** Calls interleave with the parent's steps by time, as every other ledger row does: a child's first call follows the spawn, and its later calls may land after the parent's reply, because children outlive the parent's turn. No CSS was needed (DataTable rows only).
- **Tests.** `Tests/Chat/test_console_spawn_subagent_trace_capture.py::test_trace_view_lists_child_calls_under_the_spawning_turn`: real controller, store, bridge, gateway and trace runtime under Capture On. Two sibling children each make a tool round, the view is mounted at 200x60 (full-screen tier), and the test asserts that the spawning turn's block holds exactly the four child calls (none elsewhere, so the parent's two calls are not labelled), that each state is Complete, and that `a` lands on a child call. `Tests/Chat/test_trajectory_projection.py::test_child_calls_nest_under_their_turn_numbered_in_dispatch_order` covers numbering by dispatch time against the input order and skips a call with no turn.
- **Evidence (2026-11-18, branch `fu2/traceui` on dev d0486978dd, machine load 60-95).** The new mounted-view test was red before the change (`assert [] == [...4 child rows]`) and green after. Mutations, each of which fails its test: no dispatch-order sort (projection test); reading every owner call instead of child segments (the view shows `sub-agent 3`); the builder not passing `child_calls` (no rows).
  - Covering files at `-n 4` (spawn capture, projection, trajectory screen/live/timeline/filters/export/import/responsive/capture, v87 child segments, staged-evidence export, semantic mutation inventory, resume active path): 477 passed, 30 failed. 26 failures are `test_console_resume_active_path.py`, and that file has the same 23 failures by id on head and on a detached dev d0486978dd (22 `RecoveryRequired: raw_source_selection_changed`, 1 assertion at :2500). The 4 `test_console_semantic_mutation_inventory.py` failures were load flakes: run alone, 136 passed.
  - `Tests/Architecture`, serial: 1475 passed, 4 failed. Three are red on base too (`test_reviewed_diagnostic_changes_are_metadata_only` and both timer-path inventory tests; TASK-41310 saw the same). The fourth, `test_task_15103_review_ledger_canonical_schema_and_arithmetic`, passed when run alone on head and in the base run, so it is order- or load-dependent. `./scripts/preflight.sh` passed.
- **Size.** `console_trace_repository.py` 4021 -> 4040 (+19), row raised with a dated note. `trajectory.py` (1924) and `review_selection.py` (1186) have no row.
- **Files.** `tldw_chatbook/Chat/console_trace_repository.py`, `tldw_chatbook/Chat/trajectory.py`, `tldw_chatbook/UI/Console_Modules/review_selection.py`, `Tests/Chat/test_console_spawn_subagent_trace_capture.py`, `Tests/Chat/test_trajectory_projection.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/console/semantic-trace-capture.md`, `backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md` (dated note).
- **Left out (follow-ups).** (1) The child's AgentRunsDB run and its steps still render as a separate "Turn 2", keyed by the conversation id, because a sub-agent run has no `assistant_message_id`. (2) Child calls carry no token counts: the stored usage needs the native reader's private normalizer. (3) The Conversation Inspector's per-message reader still omits child calls.
<!-- SECTION:NOTES:END -->
