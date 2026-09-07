---
id: TASK-16074
title: Make Moonshot live native-tool continuation pass
status: Done
assignee: []
created_date: '2026-08-14 02:20'
updated_date: '2026-08-14 03:43'
labels: []
dependencies:
  - TASK-15676
references:
  - backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md
  - >-
    Docs/superpowers/specs/2026-08-13-task-16074-moonshot-live-tool-uat-fix-design.md
  - Docs/superpowers/plans/2026-08-13-task-16074-moonshot-live-tool-uat-fix.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the post-merge Moonshot Kimi K3 integration defect found by paid UAT so the real Console tool-call and continuation path completes successfully without weakening the provider contract or exposing credentials.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A doubly gated paid Moonshot Kimi K3 probe completes exactly one calculator call, continues with the tool result, and returns the required final marker.
- [x] #2 The exact Moonshot SSE fingerprint, terminal choice usage, and identical trailing usage duplicate that triggered Chatbook's synthetic HTTP 502 errors are accepted under bounded, fail-closed validation and pinned by automated regressions at the real provider boundary.
- [x] #3 Moonshot credentials and captured live/raw provider payloads remain absent from logs, tracebacks, fixtures, and committed files; regressions use only minimal synthetic SSE data.
- [x] #4 Focused Moonshot, hosted Chat, AgentService, and Console continuation regressions remain green without changing unrelated provider behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin Moonshot's bounded `system_fingerprint` and terminal
   `choices[0].usage` plus identical trailing usage streaming shapes in the
   neutral hosted parser and joined Console native-tool fixtures with strict
   RED tests.
2. Apply the minimal provider-neutral streaming validation corrections and
   prove malformed, conflicting, misplaced, unknown, or oversized data still
   fails closed.
3. Run only focused hosted/Moonshot/AgentService/Console/privacy regressions,
   then the doubly gated paid Moonshot UAT.
4. Close task evidence, rebase on `dev`, open the follow-up PR, address its
   checks/review comments, merge it, and clean up the branch/worktree.

ADR required: no

ADR path: backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md

Reason: this is a compatibility correction within ADR-063's existing hosted
provider wire and durable continuation boundaries; it does not introduce a new
storage, sync, security, dependency, or cross-module ownership decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Corrected the neutral hosted stream parser for Moonshot's live Kimi K3 event
  sequence: bounded `system_fingerprint`, terminal choice-level usage, and one
  type-strict identical trailing usage duplicate. Unknown, malformed,
  misplaced, conflicting, repeated, and JSON-type-distinct data remain rejected.
- Added pure parser RED/GREEN coverage and Moonshot-only joined
  Console→AgentService→HTTP fixtures; Z.ai retains its prior wire shape.
- Paid UAT passed on reviewed, rebased code SHA `da2816853`: the doubly gated
  live node completed one calculator call, continued with its result, and
  returned the required final marker (`1 passed` in 16.82s).
- Focused verification: 109 touched-file tests passed; 77 provider/Console
  related tests passed with two opt-in live skips; 60 AgentService/privacy tests
  passed. Ruff, formatter, focused mypy, compileall, and diff checks passed.
- Two independent final reviews approved the code/spec. A filenames-only
  tracked-tree search found no credential match, and no raw live payload or
  temporary diagnostic harness is committed.
- Reused ADR-063; no new ADR was required. The plan expanded only after paid UAT
  revealed later stream events that the first keys-only trace could not see.
<!-- SECTION:NOTES:END -->
