---
id: TASK-21351
title: Add activity views to Ctrl+K session switcher
status: In Progress
assignee: []
created_date: '2026-08-23 22:37'
updated_date: '2026-08-24 05:04'
labels: []
dependencies:
  - TASK-20937
references:
  - >-
    Docs/superpowers/specs/2026-08-22-console-edge-rails-workspace-tree-design.md#follow-up-ctrlk-active-conversation-view
  - Docs/superpowers/specs/2026-08-23-console-session-switcher-activity-views-design.md
  - Docs/superpowers/plans/2026-08-23-task-21351-console-session-switcher-activity-views.md
  - backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md
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
- [ ] #1 Ctrl+K opens immediately in an activity-focused view whose ordered groups are Waiting for you, Working, New results, Current, and Other open; conversation lifecycle never hides otherwise active execution or attention.
- [ ] #2 Every selectable conversation result is one canonical local Console session or persisted conversation; duplicate open sessions merge deterministically, unbound drafts remain distinct, and activation uses an explicit immutable destination rather than nullable-field or positional inference. A vanished session-only outcome is instead a separately labeled, receipt-keyed Mark seen action with no inferred destination.
- [ ] #3 History can browse and search every persisted local conversation, groups results by local calendar recency, pages bounded results, and remains available when local activity storage is degraded.
- [ ] #4 A nonblank Active search widens to History only after zero Active matches; the documented Active/History toggle retains the modal-lifetime query, while closing resets mode and query.
- [ ] #5 Ordinary inactive-session and post-turn FLEET outcomes persist as idempotent, revisioned receipts across restart; effective corrections supersede obsolete revisions, in-turn FLEET children do not duplicate ordinary outcomes, and pre-migration terminal history does not create an upgrade flood.
- [ ] #6 Successful outcomes are acknowledged only after the exact destination and receipt-keyed notice visibly load, except that a vanished session-only destination can clear only through its receipt-keyed Session unavailable / Mark seen action; failed, stuck, stopped, and cancelled outcomes require an explicit Mark seen action, and newer outcomes arriving during navigation remain unseen.
- [ ] #7 Activity and compatibility-mark publication is race-safe and never interrupts direct-run, queue-chain, or FLEET execution settlement; migration/read failure fails closed without deleting or rebuilding AgentRunsDB, while orphan reconciliation rolls status and receipt forward atomically or not at all.
- [ ] #8 Ordering, focus retention, paging, async-result rejection, F2 rename scope, keyboard/pointer activation, literal-safe two-row labels, narrow-width omission, and the unconditional Cancel action are deterministic and accessible; the complete modal never exceeds 35 terminal rows.
- [ ] #9 Opening Ctrl+K or changing local modes performs no network request, mounts at most 50 conversation/unavailable result rows plus bounded modal controls, does not reconstruct transcripts, and leaves Console Context/Inspector rail projections and ownership unchanged.
- [ ] #10 Automated model, migration, producer, race, production-path activation, and production-stylesheet compositor tests cover the approved behavior; final evidence includes equal-cell iTerm2 and Windows Terminal parity after TASK-20937.6 is complete.
<!-- AC:END -->

## Implementation Plan

1. Record the Phase 1 receipt/projection ownership and modal-scoped F3
   exception in ADR-085.
2. Keep TASK-21351 In Progress as the approved design/umbrella owner. After
   TASK-20937 is Done, create one atomic Phase 1 implementation child with
   TASK-20937 as its dependency, copy all implementation acceptance criteria
   into that child, and link the approved spec, plan, and ADR before changing
   code. Put the child In Progress while the parent remains In Progress.
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
