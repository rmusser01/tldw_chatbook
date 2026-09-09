---
id: TASK-32075
title: Recover captured sends after discarding an interrupted tool run
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 19:00'
updated_date: '2026-09-08 19:29'
labels:
  - console
  - trace
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A third-request trace construction failure can leave a captured tool suffix behind. After reopening and explicitly discarding the pending response, the next captured send fails with surface_replacement_checkpoint_unavailable or unsupported_surface_change. Restore usable subsequent sends while preserving the historical requests and rejecting unproven recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a third-request trace failure, reopening and discarding the pending response allows the next two captured sends with project instructions enabled or disabled.
- [x] #2 Recovery preserves earlier captured calls exactly and requires the exact durable discarded response owner, exact prior call, unchanged saved history and matching capture policy.
- [x] #3 Targeted real SQLite/controller/agent/gateway regressions cover cold recovery and reject missing, mismatched or active recovery evidence.
- [x] #4 Recovery also handles already-failed follow-up turns after each pending response is explicitly discarded; every intervening saved user and discarded owner must be verified.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, amendment to existing ADR097.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: compose durable discard ownership with the existing bounded completed-turn replacement proof; no new storage or schema.
1. Turn the real cold-reload/discard reproduction into regression tests with project instructions on/off, successive sends and immutable capture assertions.
2. Bind an optional discarded assistant owner to the next user and prior trace turn. Revalidate durable terminal state, parent/source identity, absence of active recovery, latest response-bearing trace call, unchanged history and matching policy at preparation and final persistence.
3. Reuse the bounded tool/context replacement and transformed-source restoration contracts without admitting arbitrary history changes or replaying tools.
4. Admit a bounded sequence of intervening failed follow-up users only when every exact saved revision and discarded response owner links the latest traced turn to the current user; recheck the entire chain at final binding.
5. Add negative ownership/state/policy tests, run targeted trace and logging suites, format/lint changed ranges and document verified limits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recover captured sends after an explicitly discarded tool run, including one or more already-failed follow-ups that committed user messages without provider calls. A bounded witness carries the exact discarded assistant owner and any intervening saved-user revision/assistant pairs. Preparation, final persistence and owned reservation recovery recheck every direct-parent link, active-checkpoint absence, source revision, unique response owner (including deleted siblings), latest trace call, policy, surface tail and tool/context lineage. Historical request snapshots and call records stay unchanged; response_started is not relabeled as a successful outcome. No schema or dependency changes.

Real SQLite/controller/agent/gateway regressions mock only provider HTTP and inject the previously shipped third-call construction failure. Coverage includes project instructions on/off, cold reopen/discard/reopen, actual dictionary transforms with verified source pins, one/two previously failed follow-ups, two successful successor sends, and final-boundary rejection of missing/changed/active/ambiguous owners, bad policy, unknown outcomes, substituted saved values, duplicate or over-limit chains. Independent review found and resolved a false-positive transform fixture and ambiguous sibling ownership; review found no remaining actionable issues. The reporter's precise recovery sequence remains unconfirmed, so this establishes reproducible causes of the logged categories rather than proving their local run is fixed.

ADR required: yes; amended backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Updated Docs/User_Guide/console/semantic-trace-capture.md and backlog/docs/lessons-testing-evidence.md. Main changes are console_trace_final_values.py, console_trace_runtime.py and console_trace_service.py; regressions are in test_console_trace_discarded_tool_run.py. The plan expanded after a separate RED reproduction showed already-failed follow-up users also needed exact discard-chain proof.

Validation on dev base 7a7493e529533b31d139f35d13fb4eb543edcf19: 407 targeted trace tests passed, including 31 new discard/repeated-follow-up cases; 177 logging/UI tests passed. All six scripts/preflight.sh checks passed. Changed ranges formatted; git diff --check clean. Ruff comparison against HEAD found no introduced findings (599 baseline, 597 current across modified Python files). No full suite run.
PR #2522 review clarified the public witness and discard-lookup contracts: all parameters, optional evidence meanings, oldest-to-newest follow-up ordering, lookup-window bounds, absence results and actual escaping exceptions. Executable ASTs remain identical; compilation and 27 targeted final-value tests pass. These documentation changes describe existing ADR-097 semantics.
<!-- SECTION:NOTES:END -->
