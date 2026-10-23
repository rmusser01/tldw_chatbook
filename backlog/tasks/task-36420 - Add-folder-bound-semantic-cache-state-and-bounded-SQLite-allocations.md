---
id: TASK-36420
title: Add folder-bound semantic cache state and bounded SQLite allocations
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-23 03:26'
updated_date: '2026-10-23 03:52'
labels: []
dependencies:
  - TASK-36006
references:
  - backlog/decisions/241-workspace-project-context-and-folder-bound-index.md
  - Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare optional semantic storage without changing existing project access, keyword behavior or saved Library choices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fresh v4 and supported upgrade chains preserve keyword rows, policies, frontier, claims and authority; injected DDL failures leave the prior valid catalog intact.
- [ ] #2 Semantic settings and consent have an independent revision; their changes preserve keyword epoch, readiness and Library policies/scopes, and status opens no embedding/model objects.
- [ ] #3 Semantic-capable allocations and all competing new-version mutations use serialized binding-wide admission with a source-qualified pre-write main/journal/temp/control regular-file-length bound within1GiB; unsupported geometry, unretired ownership or observed mixed writers refuse before semantic mutation.
- [ ] #4 Body, commit, rollback or native-close failure retains uncertain ownership and blocks reuse; read/status/search paths perform no hidden unreserved writes.
- [ ] #5 Legacy cleanup v2/v3 and newv4 remain available without migration/allocation; existing v1 quarantine is preserved, and old compatible Ready storage is never discarded to fund staging.
- [ ] #6 Pure keyword-only operation retains established behavior without embedding imports or semantic-specific runtime qualification; keyword search remains usable during scanner/provider waits outside finite SQL units, and missing/degraded semantics preserves saved Library choices.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Task36420, amended semantic plan/ADR241, current project-index architecture, quota-foundation-proposal source-pinned refinement and actual baseline/store/lifetime tests. Existing ADR applies; no duplicate decision.
2. Establish meaningful RED tests using real SQLite/native binding ownership for additive v4/fresh and old migration preservation, independent settings/revision, model-free status and unchanged keyword-only availability; then bounded allocation, all-control-write, pre-open journal, two-namespace contention and commit/rollback/close uncertainty cases.
3. Implement narrow project allocation.py using stable allocation-v1.lock through BindingLifetime for finite try-only SQL units outside registry fences (retained job lifetime may precede it), source-qualified main/journal/temp/control allowance on actual held connections and positive native settlement. Admit every semantic-capable competing allocation; preserve legacy keyword-only mode and record unsupported/mixed-version boundary honestly. No generic kernel change or embedding/provider/model construction.
4. Add004/fresh-v4 exact catalog and semantic immutable settings/generation/member records under same owner, read-only status, independent semantic revision CAS and preserved prior keyword state. Extend exact v2/v3/v4 cleanup without migration/allocation, retain v1quarantine and oldReady headroom.
5. Run targeted new/affected keyword DB/service/recovery/cleanup/native nodes only, scoped lint/format/source guards and diff. Retain raw/child receipts as each process retires outside rotating temp; no dependency/model install, benchmark corpus rebuild, fullsuite or external operations. Include analytical+actual cap evidence and unresolved limits; commit exact task paths and report for independent spec/quality review.
ADR required: yes, existing amended decision.
ADR path: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md.
Reason: direct implementation of accepted folder-bound semantic storage/native allocation contract; amendment written before code. No new Library owner, dependency or generic runtime boundary.
6. Final pre-code consistency correction: actual project SQLite timeout/busy_timeout0; no allocation gate/store/cursor across scanner IPC, provider wait, Capture, joins or retry sleep. Positively close each native store and inventory before releasing its gate; every gate-held lifetime/receipt acquisition is try-only and unwinds on busy. Verify keyword lookup while the real scanner is waiting. This supersedes the draft allocation-first whole-job order; source-backed proposal found finite units feasible.
<!-- SECTION:PLAN:END -->
