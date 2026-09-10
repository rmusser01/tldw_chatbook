---
id: TASK-32276
title: Connect Console archive search and exact conversation resumption
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 15:35'
updated_date: '2026-09-10 22:47'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console and Library complete the same original-conversation recovery journey.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library Resume reopens the original conversation and active branch, preserving unrelated drafts and reusing open sessions.
- [x] #2 Console exposes archive and full archived conversation search and filters archived chats from ordinary history.
- [x] #3 Active or queued work is guarded and archive never silently cancels it.
- [x] #4 Documentation and targeted integrated lifecycle tests cover the complete workflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md
Reason: implement shared lifecycle and recovery design.

Inspect pending handoff and Console activation; pin exact resume/draft/busy guard tests; add typed resume and archive/search entry points; run integrated verification and update guide.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
Spec: Docs/superpowers/specs/2026-09-10-console-archive-recovery-design.md

PR #2576 review: guard the initial workspace read during Resume; verify a storage failure preserves recovery state and a retry opens the original conversation. Existing ADR147 applies.
Wave 2 review: preserve honest partial recovery across the separate workspace/conversation stores using fresh confirmation preflight and explicit retry feedback; fence existing-session activation and handoff settlement against navigation/supersession; attach safe identity context to failures; parameterize recovery record types. Add narrow failing regressions before fixes and document the partial-completion policy in ADR-147.
PR #2576 third review: preserve deleted conversation discovery independently of archive scope; prevent failed new mutations from exposing prior Undo; recheck durable state before existing-session Resume; serialize Unicode name checks with restore writes; align workspace-archive action copy and navigation contracts. Add focused regressions and verify affected integrations. ADR required: no new ADR; implements existing ADR147 lifecycle/recovery boundaries.
PR #2576 fourth review: centralize resume-ID validation; keep memory-backed registry enrichment on its owning thread; verify confirmed close retains real saved history; fence recovery publication by request/revision ownership; check durable state for both warm and cold resume paths; reuse async workspace restore for receipt Undo. Add targeted regressions. ADR required: no new ADR; implement ADR147 ownership and recovery rules.
PR #2576 fifth review: verify all 17 findings; retain send drafts and completed archive receipts through cancellation; make remaining workspace lifecycle storage asynchronous; fence the late existing-session hydration branch; synchronize retained reader lifecycle metadata and honest Find navigation; repair behavioral test gaps/flaky waits, batching names, and docs. Use bounded independent domain agents plus local integration. ADR required: no new ADR; correct ADR147 close consent wording and implement existing lifecycle/ownership decisions. Targeted checks only.
PR #2576 sixth review: replace remaining fixed Settings recovery waits with observable predicates, assert actual archived setup before Trash recovery, and exercise Console full-search through the mounted Library destination with matching saved text. ADR required: no; test evidence only under ADR147. Run targeted affected cases.
The real-router regression also exposed broad Library source failure replacing a successfully fetched conversation canvas. Preserve Conversations independent request/error ownership during snapshot reconciliation, matching its existing compose policy, and verify with both the real Console route and a deterministic unrelated-source failure. ADR required: no new ADR; restores the existing Library/ADR147 conversation recovery boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Connected Console archive/search controls to Library and added a typed original-ID resume handoff. Recovery reuses open sessions or hydrates the persisted conversation and branch, retains workspace/global ownership and unrelated drafts, and guards archive/send admission. Current dev asynchronous hydration and switcher modes remain intact.

ADR: backlog/decisions/147-conversation-archive-and-exact-resume.md. User guide: Docs/User_Guide/console/sessions-tabs-workspaces.md. Targeted verification and limitations: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md.

Integrated onto an isolated branch from current dev; original checkout changes are excluded. Task and ADR IDs were reassigned to avoid published collisions.
PR #2576 review: initial workspace-read failures now show recovery guidance without navigation; a successful retry stages the original ID. Added a failing-before/passing-after regression. The mounted rename-collision workflow waits for its actual action control before interacting.
Wave 2 review #3983108362/#3983108372/#3983108334: confirmed recovery rereads the conversation version and workspace before writes. If workspace restoration completes but conversation recovery fails, feedback explicitly reports the restored workspace and directs Resume retry; ADR-147 records why cross-store rollback could overwrite another writer and is avoided. Existing-session resume now passes a current-screen/current-claim guard through activation and retrieval refresh, checks it after awaited boundaries, suppresses stale final focus/paint, and releases interrupted claims while preserving newer requests. Error logs retain tracebacks with conversation/workspace IDs only as structured context. New tests cover stale/deleted/rehomed confirmation state, optimistic/storage partial failure and retry, interrupted activation, latest-request draining, retrieval paint, and all six error-log paths. Evidence: first nine regressions failed before fixes; 34 boundary/Library tests then passed, plus two existing default-activation and two mounted restore/resume/send variants. Parent owns the remaining mounted archive send-gate assertions and final combined run. New/changed focused files are Ruff-clean; baseline comparison for session/retrieval/wiring adds zero diagnostics; compilation and diff checks pass. No full sweep, commits, or review replies performed by this worker.
Final integrated review verification and baseline limits are recorded in Docs/superpowers/qa/console/2026-09-10-archive-recovery.md. All modified archive flows pass their targeted tests; the unrelated compact Overview assertion reproduces with the prior Settings implementation. Started-write cancellation preserves storage completion publication. Existing ADR147 applies.
PR #2576 third review: fixed Trash archive independence, failed-mutation Undo ownership, durable existing-tab Resume checks, serialized Unicode restore names, workspace recovery labels, and archive navigation contracts. Targeted real SQLite, recovery and mounted checks pass; third-review evidence and temporary host-disk interruption are recorded in the QA report. ADR147 applies and documents deletion-oriented scope. Self-review and scoped static checks complete.
PR #2576 fourth review: centralized resume validation, preserved in-memory SQLite ownership, verified confirmed close retains stored history, fenced recovery against newer request revisions, checked both warm/cold lifecycle state, and routed Console receipt Undo through async restore with expected-record checks. Evidence: 99 focused tests, 2 supersession-boundary cases, confirmed-close SQLite test, and 14 mounted lifecycle cases pass; static and diagnostic checks pass. See fourth-review QA section; ADR147 applies.

PR #2576 fifth review: retained committed archive/Undo completion through cancellation, preserved live owning keyboard drafts, made remaining workspace lifecycle storage asynchronous, fenced late Resume and retained reader generations, corrected Find feedback, and strengthened mounted source/import/wait assertions. ADR147 close-consent wording clarified; existing lifecycle design applies. Targeted results and baseline fixture limitations are recorded in Docs/superpowers/qa/console/2026-09-10-archive-recovery.md. New diagnostics were reviewed; scoped Ruff and artifact checks pass. Self-review caught and covered closed-owner cancellation and same-ID reader-generation races.

PR #2576 sixth review: Settings and Trash assertions now prove mounted/durable completion and archived setup. Console Full search boots the real application and reaches matching Library results/original reader in both sizes, excluding an unrelated chat. This exposed and fixed broad-source reconciliation hiding a successful conversation page; a deterministic cold-entry regression pins independent request ownership. Final targeted run: 26 passed; separate source-failure/reader/reconciliation checks and baseline focus limitation are documented in the QA report. Existing ADR147 applies; no new ADR. Scoped lint/artifact checks and self-review complete.
<!-- SECTION:NOTES:END -->
