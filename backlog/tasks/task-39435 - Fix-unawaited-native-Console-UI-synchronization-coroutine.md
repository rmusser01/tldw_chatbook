---
id: TASK-39435
title: Fix unawaited native Console UI synchronization coroutine
status: Done
created_date: 2026-11-05 15:12
assignee:
- '@codex'
labels:
- console
- bug
- async
references:
- backlog/tasks/task-34852 - Avoid-leaking-Console-acceptance-refresh-coroutines-before-worker-admission.md
documentation:
- Docs/Architecture/console.md
- backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md
updated_date: 2026-11-05 16:05
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Eliminate the native Console UI synchronization coroutine warning while preserving actual refresh, cancellation and teardown behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A targeted reproduction identifies the actual coroutine allocation/scheduling origin and fails on the warning before correction.
- [x] #2 Affected Console scheduling paths create no abandoned synchronization coroutine on refusal, replacement-before-start or teardown, while admitted refresh still executes.
- [x] #3 Targeted native/mounted regressions and scoped static checks pass, independent review approves the fix, and unrelated worktree/index changes remain intact.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Routine correction of coroutine ownership at the existing Textual scheduling boundary; preserve UI/runtime interfaces and scheduling policy.
1. Trace all production callers of ChatScreen._sync_native_console_chat_ui and read the earlier TASK-34852 fix, current Textual worker implementation and relevant lifecycle lessons.
2. Reproduce the actual mounted warning with allocation/origin tracing and strict warning capture. Submit root-cause evidence to the controller before behavioral edits.
3. Add a deterministic failing regression for the proven origin and relevant refusal/replacement/teardown boundaries, then make the smallest correction using existing scheduling APIs.
4. Run targeted refresh/lifecycle/native mounted checks and scoped Ruff/format comparisons; retain exact commands, RED/GREEN output and source identities without replaying a provider.
5. Self-review and commit exact owned paths, obtain independent spec/quality review of the complete fix and connected scheduling callers, preserve unrelated index/worktree bytes, update task/evidence, and mark Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

Fixed eager native Console UI refresh scheduling through the existing Textual async-callable API, including the original finally replay and stored context-publication callback alias. One local async wiring forwarder preserves late lookup and original return values for the three proxy seams. No new scheduler, dependency, schema, permission owner, visual tokens or turn lifetime changed. ADR required: no; routine existing-boundary correction under ADR-094.

Corroborated native allocation with coroutine-origin tracking and tracemalloc, then reproduced refusal/cancel-before-entry/replacement and stored-alias leaks through actual Textual workers. Positive controls observe original native code entry, real session-tab publication and live-error propagation. Initial code5f03c716fe and scoped correction98e8035315 passed independent spec/quality review with no remaining product findings. Final source-pinned4+1 focused checks pass; before/after13-file maps match committed bytes. Scoped static checks show no introduced diagnostics or edited-range format debt. Original279 unrelated identities and61 staged files remain intact.

Old14 launch identity and interrupted/adverse/profile cohorts retain CV1/CV2 qualifications; no broad-green, historical-baseline, full-suite or provider-replay claim. Exact reports, raw diagnostics, command/source receipts and decisions are retained in Docs/superpowers/qa/2026-11-05-console-sync-coroutine-warning. Updated living Console architecture, Textual incident lessons and prior workflow QA followup. Root byte verification precedes final metadata commit and cleanup of only this task's scratch.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

The native Console synchronization warning is corrected and independently reviewed. Refresh workers defer coroutine creation until execution; refused, replaced and cancelled-before-entry work no longer abandons native sync coroutines. Final source-qualified worker/alias/mounted/public checks pass; earlier broader setup/fixture failures and source-epoch limits remain documented. Unrelated worktree and staged content are preserved.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Renumbering provenance

Originally TASK-38614, introduced by `e08627ac38392fb890984169376b1c242c3a2f3e`. Renumbered while rebasing PR #3480 because current dev `36a21dc4d270940e3576b49487a71b37c78896f8` already carries the distinct task “One time guarded FTS rebuild for media databases with historical overwrite drift” at that ID. The landed task keeps its ID. This move preserves the incoming task's status, criteria and history and updates its inbound references.
