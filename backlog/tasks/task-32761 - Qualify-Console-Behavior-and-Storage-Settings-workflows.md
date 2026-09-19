---
id: TASK-32761
title: Qualify Console Behavior and Storage Settings workflows
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 22:43'
updated_date: '2026-09-17 23:26'
labels:
  - ui
  - settings
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the next Settings review with visible keyboard controls, truthful staged versus immediate save behavior, validation and recovery, and restart-only Storage defaults. Current older tests fail before the UI because their selected configuration changes after import.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console Behavior and Storage controls remain reachable and visibly readable under production styles at compact and wide sizes in dark and light themes.
- [x] #2 Representative Console staged and immediate settings preserve drafts, runtime ownership and saved values through validation, save failure, retry, revert and category navigation.
- [x] #3 Storage validation and Check Storage do not create or move files; valid saves update private configuration defaults only and preserve active database handles until restart.
- [x] #4 Original affected tests retain their assertions under correct private-profile isolation; focused regressions, token governance and static checks pass.
- [x] #5 Private native dark/light and compact/wide journeys verify the reviewed flows, clean lifecycle and unchanged default profile; remaining limitations are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/004-settings-storage-defaults-restart-boundary.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: Review and repair the established settings and token contracts without changing ownership, storage migration, or runtime boundaries. Read additional existing feature ADRs before touching their instant-apply controls.
1. Preserve original Console/Storage test assertions while selecting a private profile before imports; measure actual production-CSS control paint across both themes and terminal sizes.
2. Add representative keyboard regressions for Console staged/instant save ownership, failure/retry and navigation; Storage validation, non-mutating checks and persisted next-launch defaults.
3. Fix only confirmed gaps using existing token-backed controls and existing settings mutation boundaries, rebuild source-derived styles, and run affected tests plus static/governance checks.
4. Verify the same representative flows in a real terminal with fresh private roots, inspect captures, verify process shutdown, database integrity and unchanged default fingerprints.
5. Obtain independent review, update the completion ledger and exact QA evidence, close the task only when all criteria pass, and save to draft PR #2704 against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Console and Storage compact fields now use token-backed stacked rows and readable checkbox labels. Remote-image and status-row changes use an app-owned ordered queue: failures restore saved runtime state, screen recreation cannot abandon the latest choice, configuration reload refreshes the baseline, and current save/cache-warning receipts survive coalesced toggles. Original assertions retain private-profile isolation; Storage ownership remains restart-only. Source and generated styles, diagnostic inventory, lesson and completion ledger updated.

92 distinct targeted cases pass (18 immediate, 8 geometry, 4 staged journeys, 30 originals, 32 governance/boot); all seven derived checks pass across restricted preflight plus verified-input Mermaid retry. Scoped Ruff passes; large legacy files introduce no lint findings. Four final native dark/light x compact/wide journeys passed with eight inspected captures, real private saves, unchanged active handles, normal exit/released lock, 11 healthy private databases and unchanged default-profile fingerprints. Independent review findings were fixed and re-reviewed clear. Existing ADR-004, ADR-020, ADR-150 and ADR-161 apply; no new ADR.

QA receipt: Docs/superpowers/qa/2026-09-17-settings-console-storage/README.md. The wider review remains open with explicit Console subflow limits. User-requested historical conflict comparison: Docs/superpowers/reports/2026-09-17-pr-2704-conflict-review.md (four merges; 20 file entries / 30 blocks). PR #2704 stays draft; merging into dev requires the user's visual review and explicit green light.
<!-- SECTION:NOTES:END -->
