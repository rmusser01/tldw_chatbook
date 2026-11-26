---
id: TASK-42564
title: "A spawned child agent's preset fallback under Capture On is sent and traced as a new chain"
status: Done
assignee:
- '@claude'
created_date: 2026-11-26 07:39
updated_date: 2026-11-26 08:39
labels:
- follow-up
- console
- agents
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_streaming_request.py
- tldw_chatbook/Chat/console_trace_child_fallback.py
- backlog/decisions/200-preset-pre-tool-fallback-targets.md
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies:
- TASK-42556
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-42556 (round 27, 2026-11-26). Under Capture On, a spawned child whose preset target fails never runs its preset's `fallback_models` (ADR-200). TASK-42556 made that refusal a visible skip before any call, because ADR-337 recorded no fallback call for a child. A primary agent's fallback has been traced as a new chain since TASK-42525.

**Owner decision (2026-11-26), "Yes, new chain (Recommended)".** The question was: "Should a sub-agent's preset fallback instead start a new trace chain in its own segment, as a primary agent's fallback has since TASK-42525?" The chosen option, verbatim: "New ADR-337 amendment. The sub-agent's fallback is sent and traced as a new chain. It only happens before any tool call, so no history rewrite is needed. Built this round on top of the visible-skip fix."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With Capture On, a spawned child whose preset target fails falls back, is traced as a new chain of the child (same actor, in a child segment linked to the same owner and turn), and its result reaches the parent
- [x] #2 The parent's next message is sent
- [x] #3 Every case the new rule does not admit is refused before any reservation: an alternate that keeps the target is a visible skip with the TASK-42556 copy; a target the child never durably selected is refused before preparation; a broken link is refused in the reservation transaction (rolled back), and again at dispatch binding
- [x] #4 Provider-only fallback (ADR-110) and Capture Off are unchanged
- [x] #5 The new tests are red before the change and green after, and run on a PR lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the owner decision here and write the ADR-337 amendment (2026-11-26) before any code, next to TASK-42525's fallback-chain amendment.
2. Adapter (`Chat/console_streaming_request.py`): a child call that names another target mints a new chain only when the child run's durable preset selection (`AgentRunsDB.get_run_fallback_state`) names that target at a later index; otherwise refuse before preparation.
3. Ledger: admit the new chain's first call only as the child's fallback (same owner and turn, no earlier response, the latest earlier call a settled ERROR with the same policy to another target), in the reservation transaction, and recheck it at dispatch binding. Schema v87 ties a child segment to one run identity, so the new chain opens a child segment of its own under the same actor (no schema change).
4. Keep TASK-42556's visible skip only for a ready alternate the rule cannot record (it keeps the target), with the review's minors: an unready alternate keeps its reason, and the copy fits the 200-character step summary.
5. Trace view: number a child's calls by actor, so its fallback chain reads as the same sub-agent.
6. Tests through the real controller (`_spawn_turn`): fallback sent and traced, its answer reaching the parent, the parent's next message; hard negatives (same target, never selected, refused at reservation, refused at binding), each with a mutation; Capture Off unchanged; a runtime-level test for the unready reason. Red on the visible-skip source, green after. Same lane as TASK-42556 (recovered-suites).
7. User Guide, size ratchets, the brief's checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Status:** AC#1-#5 met. One deviation from the coordinator's wording: the fallback chain does not share the child's first segment, because a v87 trigger forbids it (below).

**ADR.** ADR-337 now carries the owner-approved amendment "a spawned child's preset fallback starts a new chain (TASK-42564)", written before the code. It sits after TASK-42525's fallback-chain amendment, whose "still refused" bullet now points to it. The 2026-11-17 child amendment notes the second segment.

**Why a segment per chain.** The first build kept the child's segment and minted a new chain inside it. The reservation then failed with `IntegrityError: provider call must use the active effective owner`. The v87 trigger `console_trace_calls_owner_lineage` refuses a call in a child segment whose `run_id` differs from the segment link's. Sharing one segment would need a schema migration, and the trigger text is also pinned in the recovery schema. Instead the fallback chain opens a child segment of its own, linked to the same owner and turn, and its first call is that segment's AGENT_FIRST. The chains share the child's actor, which is the link. No schema change.

**Changes.**
- *Adapter* (`Chat/console_streaming_request.py`). A captured child's call that names another target mints a new chain only when the child run durably selected that target at a later index. The selection is read with `AgentRunsDB.get_run_fallback_state`, which the bridge now passes to the adapter as `preset_fallback_state`. ADR-200 persists the selection before dispatch, and the index only rises. The call is routed as AGENT_FIRST. Any other target change raises `TraceProvenanceAlignmentError` before preparation, so nothing is reserved. The primary's rule (TASK-42525) is unchanged, only restructured to share the branch.
- *Ledger* (new `Chat/console_trace_child_fallback.py`). `child_fallback_admits` is called by the trace factory before it creates a child segment for a later chain of an existing actor, in the reservation transaction. `recheck_child_fallback` repeats it at dispatch binding, in `ConsoleTraceService.persist_request`, for an AGENT_FIRST call. Every other chain of the actor must:
  - belong to the same owner and turn;
  - have no call that started a response, so no tool call came first;
  - end on a settled ERROR with the same frozen policy, to another provider or model (`falls_back_from`).
- *Fail-closed skip.* `FallbackRuntime` gains a `refuse` hook. The runtime consults it only for a ready alternate; an unready one keeps its own reason (review minor 1). Under Capture On, `AgentService` refuses an alternate whose execution provider and model equal the child's current target, since the trace cannot record a fallback that keeps the target. The skip copy is now `PRESET_FALLBACK_NOT_CAPTURED` = "same target, not traceable while Capture is on; turn Capture off to allow it" (76 characters, review minor 2).
- *Trace view* (`Chat/trajectory.py`). Child calls are numbered by actor (the part of `run_id` before `:`), and their calls are counted across chains. The fallback therefore reads `sub-agent 1 · call 2 · agent first`, not as a second sub-agent.
- *User Guide*: `semantic-trace-capture.md` (Sub-agent calls) and `agent-runs-and-tools.md` (Preset fallback).

**Tests** (`Tests/Chat/test_console_child_preset_fallback_trace.py`, through `_spawn_turn`: the real controller, bridge, gateway and trace factory, only the HTTP socket faked).
- `test_a_spawned_childs_preset_fallback_is_sent_and_traced[capture_on|capture_off]`. The child asks `synthetic-qwen` (404 `model_not_found`), then the fallback model, and completes. Its answer ("child done") is in the parent's next request, and that next message completes (`follow_up` is a new harness option). Under Capture On:
  - the child's calls are `agent_first error` then `agent_first complete`, on two chains of one actor, with no `reserved` call;
  - the parent's calls are all complete;
  - the Trace view rows read `sub-agent 1 · call 1 · agent first` and `sub-agent 1 · call 2 · agent first`.
- `test_an_alternate_that_keeps_the_target_is_a_visible_skip[capture_on|capture_off]`. The alternate is the child's own model. Under Capture On it is skipped with the copy, with no second call; under Capture Off it is asked again.
- `test_a_child_fallback_the_rule_does_not_admit_is_never_sent[adapter|reservation|binding]`:
  - *adapter:* the selection write is skipped; the child's error names the refusal, and only the failed call exists;
  - *reservation:* the ledger's link check reads the fallback as not admitted; the reservation rolls back, and only the failed call exists;
  - *binding:* the same, with the factory's check disabled; the call is refused at binding and is left `reserved` (TASK-42565).

  In every case the fallback model is never asked. A settled call cannot be edited (trigger "terminal provider call settlement is immutable"), so the ledger's read is forged by patching `falls_back_from` in the helper module.
- `test_a_refusal_never_replaces_an_unready_alternates_own_reason` runs at runtime level, for review minor 1.

**A flake in the new tests, found by the soak and fixed.** In soak rounds 2 and 3, `[capture_on]` of the sent-and-traced test and `[binding]` failed with `At index 0 diff: ('agent_first', 'complete') != ('agent_first', 'error')`. `console_trace_calls.created_at` is `CURRENT_TIMESTAMP`, so it only has one-second resolution, and both chains' first calls have `call_sequence` 0. The harness's `ORDER BY created_at, call_sequence` therefore orders them arbitrarily within a second. The tests now find each chain by its state, and their assertions are unchanged. The ledger's own "latest earlier call" in `child_fallback_admits` orders by `created_at DESC, rowid DESC`, so the insertion order breaks a tie.

**Red/green.** On the visible-skip source (the branch before this change), 6 tests fail and the 2 Capture Off cases pass:
- the sent-and-traced test, on the child's status;
- the same-target test, on the copy's length (118 > 80);
- the unit test, with a `TypeError`: no `refuse` hook yet;
- the three negatives, with an `ImportError`: no helper module yet.

All 8 pass with the change.

**Mutations** (each guard disabled alone; its test turned red; the restore was checked):
- the adapter's durable-selection check: `[adapter]`;
- the same-target refuse hook: `[capture_on]` of the same-target test;
- the factory's admission: `[reservation]`;
- the binding recheck: `[binding]`;
- the AGENT_FIRST routing of the new chain: the sent-and-traced test (the v87 trigger refuses);
- actor numbering in the Trace view: the sent-and-traced test;
- the runtime's ready-only refuse: the unit test.

The "no response in the earlier chains" clause has no test of its own. A child call that starts a response either proposes a tool batch, after which ADR-200 never falls back, or ends the run. The same state on the failed call is also refused by `falls_back_from`.

**Runs** (load 10-35):
- Soak: the new file, `test_console_spawn_subagent_trace_capture.py` and `test_trajectory_projection.py`, 20 rounds at `-n 4` after the flake fix: 92 passed in every round, at load 10.6-13.7.
- Mutations, rerun after the flake fix: all 7 red, and the restored tree was checked against the saved diff.
- Covering files (116: every `Tests/Chat/test_console_trace_*.py`, every test file naming the fallback runtime, preset fallback or child calls, and the adapter's tests, including `test_console_failed_call_retry_ownership.py` for the primary rule) at `-n 4`: 4746 passed; 178 nodes failed or errored. All 178 fail the same way on dev 6e41919880, rerun node by node in a scratch worktree at that commit.
- `Tests/CI` at `-n 4`: 2009 passed, 2 skipped.
- `Tests/Architecture` serially: 1507 passed, 5 skipped.
- `./scripts/preflight.sh` passed. `ruff check tldw_chatbook --select F` is clean, and `Tests/lint/ruff_gate.py` exits 0.
- The recovered-suites step via `r19-step.sh` (`-p no:xdist --storage-lease-guard --timeout=180`): 2030 passed, 1 skipped, 1 deselected, in 818.8 s at load 6-12.

**Lane.** The file was already the last target of recovered-suites (TASK-42556). With the step's flags (`-p no:xdist --storage-lease-guard --timeout=180`) at load 12.7-16.2 it takes 20.7 s for 8 tests (was 14.3 s for 2).

**Files.**
- `Agents/agent_runtime.py` (size row 4069 → 4072), `Agents/fallback_chain.py`, `Agents/agent_service.py` (7705 → 7714).
- `Chat/console_agent_bridge.py` (10120 → 10123), `Chat/console_agent_preparation.py`, `Chat/console_streaming_request.py`.
- `Chat/console_trace_child_fallback.py` (new), `Chat/console_trace_runtime.py`, `Chat/console_trace_service.py` (7037 → 7041), `Chat/trajectory.py`.
- The test file, plus the harness options in `Tests/Chat/test_console_spawn_subagent_trace_capture.py`.
- ADR-337, the two User Guide pages, and `Tests/Architecture/test_module_size_ratchet.py`. Every size row is raised by exact growth, dated.
<!-- SECTION:NOTES:END -->
