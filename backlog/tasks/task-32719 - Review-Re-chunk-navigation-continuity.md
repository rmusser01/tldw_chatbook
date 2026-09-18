---
id: TASK-32719
title: Review Re-chunk navigation continuity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 15:14'
updated_date: '2026-09-17 15:32'
labels:
  - library
  - search-rag
  - verification
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Library component review so navigating away during Re-chunk and returning gives truthful progress, completion and retry behavior without overlapping work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Navigation during Re-chunk preserves the existing operation and prevents overlapping Re-chunk or backfill work.
- [x] #2 Returning to Search/RAG provides truthful running or completed feedback and current legacy counts, including completion while away and failure recovery.
- [x] #3 Targeted tests, private native checks at both themes and sizes, static checks, independent review and guide/audit evidence qualify the behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/164-rechunk-run-lifetime.md (existing ADR-078/031/150 also apply). Reason: move the transient UI run lifetime from replaceable panel to app session so navigation cannot discard ownership or feedback.
1. Four navigation probes reproduce missing busy state on return and lost completion while away for both canvas and screen replacement. Preserve this red evidence.
2. Introduce one lazy app-owned Re-chunk run/worker with the existing Textual Signal; panels subscribe on mount and unsubscribe on unmount. Keep service/policy/group/slot semantics, atomic completion publication and failure/retry behavior.
3. Verify navigation interleavings, both theme/size classes, duplicate/backfill refusal, failures and scheduling failure with targeted tests. Run real private SQLite native journeys, static checks and independent review. Update guide/audit/task and commit locally; no full suite, push or merge.
Allocation: all reachable paths across 305 refs and 27 worktrees gave task maximum 32718 and ADR maximum 163.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-chunk now has an ephemeral app-session run owner and app-owned thread worker. Search/RAG panels subscribe while mounted and hydrate running/receipt state on entry. This preserves feedback across Library canvas and whole-screen replacement without retaining removed widgets. Completion state and shared-slot release happen in one UI-thread callback; the existing local scope, runtime-policy, worker group and backfill refusal contracts are preserved. Failure, including scheduling failure, clears progress and permits retry.

Four initial navigation probes reproduced lost disabled state and completion notices. All 96 targeted tests pass, including 21 new theme/size/route/return-time, failure-away/retry and scheduler-failure cases. Four private native journeys each return from Notes while active, finish in Console, and return through Ctrl+3 to the receipt and updated census. Eight final captures were inspected. Four real items are re-chunked, one empty item remains skipped, all five source records are unchanged, ten private databases are healthy, defaults are unchanged, and normal exit/PID absence are verified. No conversation messages were added. The initial compact native harness attempted a hidden navigation row; opening the existing Nav grip corrected the harness without a production change.

New owner/test/runner Ruff lint and formatting pass; changed panel block formatting passes, and the panel drops from five existing lint diagnostics to four. Independent review requested the scheduler-failure test and confirmed it covered; final code/runner/documentation review has no unresolved findings. Guide, audit, testing lesson and QA evidence are updated at Docs/superpowers/qa/2026-09-17-rag-rechunk-navigation/README.md.

ADR required: yes. ADR path: backlog/decisions/164-rechunk-run-lifetime.md, created before implementation and linked in the task plan and ADR index; existing ADR-078, ADR-031 and ADR-150 still apply. Receipt retention is app-session-only; no durable job history or restart resumption is introduced. Native semantic indexing was disabled and the receipt disclosed that skip. No stylesheet, token or dependency changes; no full suite, push or merge.
<!-- SECTION:NOTES:END -->
