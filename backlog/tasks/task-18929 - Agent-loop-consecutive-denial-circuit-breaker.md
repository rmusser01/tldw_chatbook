---
id: TASK-18929
title: 'Agent loop: consecutive-denial circuit breaker'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-19 09:55'
updated_date: '2026-09-12 19:49'
labels:
  - agents
  - tools
  - approvals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's consecutive-denial circuit breaker (2026-08-19 hermes-release review). When an agent's tool calls are denied several times in a row, it often re-asks forever — burning turns and tokens. Add a configurable circuit breaker: after N consecutive denials within a run (default small, e.g. 3), the loop stops with an honest terminal message ("stopped after N consecutive denied tool calls — review the denial reasons or rephrase") instead of continuing to re-ask. The counter resets on any successful or approved call; only unbroken denial streaks trip it. Applies per run, including per-child in the fleet (a child's denials trip that child's breaker, not a sibling's).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The streak counter resets on any successful/approved tool call; only consecutive denials trip the breaker — pinned by tests
- [x] #2 Tripping produces a clear terminal state: transcript System row + run-log record naming the count; the user can immediately retry or continue (no silent hang, no lost partial reply)
- [x] #3 Fleet isolation: the breaker is per run/child; one child tripping does not stop a sibling or the supervisor
- [x] #4 Tests cover trip, reset, disabled mode, per-child isolation, and the terminal messaging
- [x] #5 The agents denial_circuit_breaker_limit setting defaults to 3; explicit 0 disables it and invalid values use the documented conservative default.
- [x] #6 Only authoritative explicit user denial or configured permission Off increments the streak; unanswered, timeout, cancellation, stale authority, legacy refusal text and synthetic restored-pending results do not.
- [x] #7 Evaluate the trailing streak after a fully settled tool batch, preserve every tool reply, and stop before another model call; an approved tail resets the streak and the terminal message reports the actual observed count.
- [x] #8 Builtin, local, MCP, virtual CLI and raw-shell authoritative invocation decisions reach the run-local counter even when pending review is bypassed; defaulted unanswered stamps and opaque runtime refusals never fabricate denial authority.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/154-agent-denial-streak-boundary.md. Reason: compatible structured provider/review/runtime provenance and coherent stop boundary. Execute Docs/superpowers/plans/2026-09-12-agent-denial-breaker.md: compatible typed review facts; authoritative producers; run-local budget/counter; Console and real persistence/fleet evidence; docs and independent review. Accepted ADR-078 remains unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented compatible approval facts at the existing review/provider boundaries and a fresh counter for each runtime invocation. Explicit user Deny and configured Off count; unresolved, unavailable, stale-authority, restored-pending and legacy-text refusals do not. Actual invocation results take precedence over review metadata, and success or any non-denial result resets the streak. The default is 3; explicit 0 disables it and invalid values fall back to 3. Child copies and admitted/live intersections preserve the configured limit.

A complete batch settles before enforcement, so approved tails reset the streak and the terminal message reports the observed count. The run stops before another model request with complete native history, STEP_ERROR and a run-local error record. Cancellation and continuation persistence failures retain precedence. Console preserves partial text, adds one System explanation even without the original placeholder, and accepts the next agent submit. Real fleet tests prove sibling/supervisor isolation and fresh resumed-child counters with retained tool replies.

ADR required: yes. Implemented backlog/decisions/154-agent-denial-streak-boundary.md and the linked specification/plan; accepted ADR-078 is unchanged. Source slices 49b5569676, 762941cec0/e85c7d16c3, a56f534fd4 and 86f2a79e76 were independently reviewed. Work touched shared models/runtime, existing producer/stamp owners, Console budget/finalization seams, focused tests and the user guide; no schema migration or new dependency.

Targeted evidence (overlapping runs are not summed): 30 shared-type/trace cases; 686 distinct producer cases across qualified runs, then 81 MCP cases and 6 final cases for the absent-stamp fix; initial 130 runtime/budget/continuation cases and expanded final66; final6 Console/store, real SQLite/segmented log, noncancelled restored-pending, gated sibling and actual resumed-history cases. Nearby15 and root33 terminal/log plus2 fleet baseline cases also passed in separate selections. Final Task4 output was captured directly: 6 passed, exit0. Every edited formatting range and new test file passes; per-file lint counts are unchanged (two existing F811 diagnostic messages only shift their referenced import line).

Evidence qualifications: core runtime preimplementation RED was missed; a later exact-BASE 13-failure control proves sensitivity only. Task4 missing-placeholder RED was observed before editing (1failed/1passed), but only an explicitly labelled output summary was retained. Shared RequestsDependencyWarning and foreign pytest cleanup warnings remain unchanged. No full suite, live provider test, shared dependency repair or foreign cleanup was performed. Full reports/reviews and exact commands remain in .superpowers/sdd/2026-09-12-agent-denial-breaker/ in the preserved orchestration worktree; current completion is reflected in backlog/docs/agent-orchestration-followups-2026-09-12.md.
<!-- SECTION:NOTES:END -->
