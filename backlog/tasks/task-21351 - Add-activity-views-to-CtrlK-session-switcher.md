---
id: TASK-21351
title: Add activity views to Ctrl+K session switcher
status: In Progress
assignee: []
created_date: '2026-08-23 22:37'
updated_date: '2026-09-02 21:57'
labels: []
dependencies:
  - TASK-20937
references:
  - >-
    Docs/superpowers/specs/2026-08-22-console-edge-rails-workspace-tree-design.md#follow-up-ctrlk-active-conversation-view
  - >-
    Docs/superpowers/specs/2026-08-23-console-session-switcher-activity-views-design.md
  - >-
    Docs/superpowers/plans/2026-08-23-task-21351-console-session-switcher-activity-views.md
  - backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md
  - backlog/tasks/task-28125 - Harden-CtrlK-switcher-trust-and-terminal-fit.md
  - >-
    backlog/tasks/task-21351.1 -
    Implement-local-Active-History-CtrlK-activity-switcher.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users quickly switch to local Console conversations that are active now,
need attention, or have unseen outcomes without losing complete historical
conversation browsing. Deliver a local-first activity model whose durable
acknowledgement state survives restart; correlated server workflows remain a
separately releasable later phase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ctrl+K opens immediately in an activity-focused view whose ordered groups are Waiting for you, Working, New results, Current, and Other open; conversation lifecycle never hides otherwise active execution or attention.
- [x] #2 Every selectable conversation result is one canonical local Console session or persisted conversation; duplicate open sessions merge deterministically, unbound drafts remain distinct, and activation uses an explicit immutable destination rather than nullable-field or positional inference. A vanished session-only outcome is instead a separately labeled, receipt-keyed Mark seen action with no inferred destination.
- [x] #3 History can browse and search every persisted local conversation, groups results by local calendar recency, pages bounded results, and remains available when local activity storage is degraded.
- [x] #4 A nonblank Active search widens to History only after zero Active matches; the documented Active/History toggle retains the modal-lifetime query, while closing resets mode and query.
- [x] #5 Ordinary inactive-session and post-turn FLEET outcomes persist as idempotent, revisioned receipts across restart; effective corrections supersede obsolete revisions, in-turn FLEET children do not duplicate ordinary outcomes, and pre-migration terminal history does not create an upgrade flood.
- [x] #6 Successful outcomes are acknowledged only after the exact destination and receipt-keyed notice visibly load, except that a vanished session-only destination can clear only through its receipt-keyed Session unavailable / Mark seen action; failed, stuck, stopped, and cancelled outcomes require an explicit Mark seen action, and newer outcomes arriving during navigation remain unseen.
- [x] #7 Activity and compatibility-mark publication is race-safe and never interrupts direct-run, queue-chain, or FLEET execution settlement; migration/read failure fails closed without deleting or rebuilding AgentRunsDB, while orphan reconciliation rolls status and receipt forward atomically or not at all.
- [x] #8 Ordering, focus retention, paging, async-result rejection, F2 rename scope, keyboard/pointer activation, literal-safe two-row labels, narrow-width omission, and the unconditional Cancel action are deterministic and accessible; scope/current/destination are unambiguous, rows are left-aligned and theme-semantic, and the content-sized modal never exceeds 35 terminal rows.
- [x] #9 Opening Ctrl+K or changing local modes performs no network request, mounts at most 50 conversation/unavailable result rows plus bounded modal controls, does not reconstruct transcripts, and leaves Console Context/Inspector rail projections and ownership unchanged.
- [ ] #10 Automated model, migration, producer, race, production-path activation, and production-stylesheet compositor tests cover the approved behavior; final evidence includes equal-cell iTerm2 and Windows Terminal parity after TASK-20937.6 is complete.
- [x] #11 Blank-query Enter targets the MRU other open tab, explicit navigation and nonblank search target the highlighted committed result, and contextual onboarding plus plain-language deterministic domain-semantic state/workspace search expose the exact next consequence and distilled metadata without transcript inspection, embeddings, or network calls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the Phase 1 receipt/projection ownership and modal-scoped F3
   exception in ADR-085.
2. Keep TASK-21351 In Progress as the approved design/umbrella owner. The user
   explicitly waived TASK-20937 as an implementation-start gate on 2026-09-02;
   it remains the final parity-evidence gate. Execute Phase 1 under the linked
   In Progress child TASK-21351.1, which carries the implementation criteria and
   links the approved spec, plan, ADR, and completed TASK-28125 trust baseline.
3. Add AgentRunsDB v15 receipt persistence, revision/supersession operations,
   and atomic orphan reconciliation with genuine v14-to-v15 coverage.
4. Route ordinary direct, queue-chain, and post-turn FLEET outcomes through one
   non-throwing activity-receipt service while retaining the FLEET mark only as
   a compatibility badge.
5. Build the canonical local Active projection and bounded persisted History
   adapter without changing the Console rail projections.
6. Replace the eager switcher with the 35-row Active/History modal, immutable
   activation payloads, generation-gated async search/paging, stable focus, and
   truthful keyboard/pointer controls.
7. Add the destination outcome notice and receipt-specific acknowledgement
   coordinator, then verify the production activation path and visible-paint
   boundary.
8. Run focused and reachable suites, production-stylesheet compositor checks,
   performance measurements, and equal-cell terminal parity before closeout.
9. Complete the child acceptance criteria, notes, and Done transition first;
   then complete the parent acceptance criteria, notes, and Done transition
   after the child and all parent-level evidence are complete.
10. Apply the approved Impeccable critique to the existing modal without changing ADR-085 ownership: clarify visible scope and Enter consequence, use adaptive left-aligned semantic-token presentation, simplify search teaching and row metadata, and verify keyboard/pointer Mark seen parity plus page navigation.

Detailed plan:
`Docs/superpowers/plans/2026-08-23-task-21351-console-session-switcher-activity-views.md`.

ADR required: yes
ADR path: `backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md`
Reason: ADR-085 records durable acknowledgement ownership, the derived FLEET
mark relationship, canonical switcher subjects/targets, and the deliberate
modal-scoped F3 exception to ADR-031. Phase 2 server correlation requires its
own later task and ADR.

## Renumbering provenance

Two delayed Backlog CLI attempts assigned TASK-21201 and TASK-21202, both
already claimed on remote branches at filing time. The duplicate TASK-21201
file was removed, and the surviving TASK-21202 follow-up moved to TASK-21351
before implementation or review references were created.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the local Phase 1 activity-focused Ctrl+K switchboard through child TASK-21351.1: canonical Active groups and exact destinations, bounded all-local History, durable revisioned receipts, consequence-aware acknowledgement, race-safe FLEET compatibility coordination, 35-row bounded UI, MRU-other blank Enter, inline onboarding, and deterministic metadata-only semantic search. Updated the user guide, ADR-085, and reproducible QA artifacts.

Focused automated and production-shaped verification passes, independent review reports no remaining actionable findings, and static checks pass. The reachable feature rail result and its non-switcher failures are documented without claiming baseline equivalence.

ADR required: yes. ADR path: backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md.

Remaining closeout gate: parent AC #10 equal-cell native terminal parity. iTerm2 evidence is complete; Windows Terminal remains blocked by TASK-20937.6. Parent intentionally remains In Progress.

2026-09-02 Impeccable refinement: clarified visible scope and exact Enter consequence; adopted content-sized, left-aligned, theme-semantic rows; distilled update metadata; simplified search guidance; added contextual hints, Home/End/Page navigation, and keyboard/pointer Mark seen confirmation parity. Fresh evidence: 90 targeted switcher/trust UI tests passed, 2 hermetic production/compositor capture tests passed, the Impeccable detector returned no findings, and scoped Ruff plus git diff --check passed. No new ADR was required because these presentation refinements stay within ADR-085. AC #10 remains open solely for equal-cell native terminal parity.

2026-09-02 PR review follow-up: rebased onto dev 25478303e and addressed Qodo findings with bounded keyset receipt pagination, a stats-free plan-pinned unseen index, strict Pydantic receipt/query boundaries, explicit Mark seen handling for missing persisted destinations, authority-token fencing for unavailable notices, and complete public API documentation. Kept the read on AgentRunsDB.connection() because transaction() issues BEGIN IMMEDIATE and is intentionally reserved for writes. Rebuilt generated CSS and the reviewed persistent-diagnostic inventory. Verification: 12 receipt DB tests, 177 Chat projection/receipt tests, 50 activity switcher/outcome UI tests, 58 switcher trust/keyboard/rail tests, and all 23 PR performance guards pass; CSS, index-plan, diagnostic, profile-path, backlog-ID, schema-allowlist, Ruff, Impeccable, and diff checks pass. No new ADR is required; these review corrections remain within ADR-085. Native-terminal parity remains the sole closeout gate.

2026-09-02 CI follow-up: the Linux UI-ready census measured 973/972 while repeated macOS runs measured 970-971/972. Root cause was the new receipt coordinator implementation module importing during Console bridge mount. Added a thread-safe lazy proxy in the already-resident ConsoleRuntime and pinned console_activity_receipts as absent at UI readiness, so first switcher or settlement use constructs the same service without spending first-paint module budget. Verification: the born-red census now passes at 970/972 locally, all 23 PR performance guards pass, and 135 receipt/switcher/runtime ownership tests pass. No new ADR is required; the fix follows the existing ADR-097 ratchet policy and preserves ADR-085 ownership.

2026-09-02 native iTerm2 UAT: completed the macOS equal-cell walkthrough at 160x45. Reciprocal MRU blank Enter, History fit/grouping, and exact saved-destination activation passed. The walkthrough found and drove the repair of History workspace semantics: `workspace:roleplay` had been forwarded to conversation FTS even though workspace labels are not indexed. Rebased commit `5206daca1` separates the domain query from FTS, resolves labels to bounded workspace filters, handles the visible `Chats` global/default union, and consumes saved-state aliases. The post-fix focused rail passes 193 tests; scoped Ruff and `git diff --check` pass. Checksummed evidence is in `Docs/superpowers/qa/task-21351-console-switcher-activity/task-21351-iterm2-uat-2026-09-02.md`. Parent AC #10 remains open only for Windows Terminal evidence through TASK-20937.6. No new ADR is required; this enforces ADR-085's existing semantic-search boundary.

2026-09-02 pre-PR review follow-up: reproduced and fixed all three Important findings. History now preserves Unicode literal storage queries, resolves ordinary and multi-word workspace labels while retaining neighboring title terms, and encodes arbitrarily many deduplicated workspace IDs through one parameterized `json_each(?)` value instead of an expanding placeholder list. Added controller-to-real-SQLite coverage for workspace-only, Unicode title, and Unicode message matches plus large-ID and service-boundary tests. Verification now passes 199 focused tests, scoped Ruff, and `git diff --check`. Fix: `ca8a57990`. Windows Terminal parity remains the only open parent criterion; no new ADR is required because the changes complete ADR-085's approved deterministic local search boundary.
<!-- SECTION:NOTES:END -->
