---
id: TASK-31741
title: Complete Canvas integration artifact guards
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 20:02'
updated_date: '2026-09-05 20:41'
labels:
  - canvas
  - testing
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Canvas merge candidate satisfy repository artifact guards with reviewed diagnostic metadata and measured SQLite index coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every changed diagnostic inventory owner is reviewed for content or credential disclosure and the generated inventory reproduces.
- [ ] #2 All six Canvas indexes are classified in the census with real no-statistics query-plan evidence for read indexes and explicit constraint rationale for uniqueness enforcers.
- [ ] #3 Targeted checks and repository preflight pass on the latest-dev merge candidate; independent review and implementation notes document evidence.
- [ ] #4 Canvas startup integration stays within existing app-import, first-paint module, CSS, and pre-import payload ratchets without raising thresholds or hiding new residents.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md. Reason: verify existing schema and privacy contracts, no new indexes or runtime privileges.
1. Review each diagnostic statement drift against the pinned inventory revision, including exception traceback and user-controlled argument paths; add failing privacy regression and bounded correction for any newly confirmed disclosure before regenerating metadata.
2. Add isolated real-SQLite tests asserting sqlite_stat1 absent and representative Canvas queries use the four query indexes; document the two composite unique foreign-key target constraints as correctness enforcers in the census. Reuse existing repository fixtures; do not add or remove indexes to appease the guard.
3. Run targeted tests and artifact generators, inspect inventory/census diff, and repeat preflight. Root owns executable checks; agents stay static-only.
4. Rebase latest dev with recoverable branch state, resolve only scoped conflicts, rerun affected checks and preflight, obtain independent static review and record commands/limitations. Complete before opening the authorized PR.
5. Post-rebase fixture reconciliation: preserve the required selection-generation contract by completing the three served compiler snapshot doubles, and remove duplicate prefill/generation-settings imports retained during conflict resolution while keeping dev's endpoint-policy imports. Loopback route failures under the restricted executor are environmental: the unchanged gateway module passes all72 tests with local-listener permission; do not alter network markers or guards to hide them. Run the corrected affected selection with that permission and retain the initial results honestly.
6. Complete the existing bare TldwCli shutdown fake with the current meeting-session ownership slot exposed by the rebased lifecycle test; preserve the shutdown ordering assertion and production lifecycle. This is fixture compatibility, not a new fallback or a weakened ownership invariant.
7. Startup ratchet repair (existing ADR-097 plus ADR-121, no new architecture or threshold increase): compare isolated untouched-dev and feature first-paint module censuses; retain the real app/store Canvas lifecycle owner, defer unused browser/control implementation imports to served admission or the first actual Canvas publication/open, and preserve view rebinding, settlement-driven auto-open, kill-switch revocation, and shutdown. Add first-use/no-use regression controls before implementation, then measure rather than assume the reduction. If necessary, defer existing Chatbook archive import/export implementation edges to their existing first-use entry points while preserving public exports and service ownership; do not invent a replacement Canvas owner or relax validation. Root alone runs isolated pytest/browser checks. Independently review the bounded changes and rerun all four startup guards plus affected lifecycle, tool, archive, and browser tests before PR creation.
<!-- SECTION:PLAN:END -->
