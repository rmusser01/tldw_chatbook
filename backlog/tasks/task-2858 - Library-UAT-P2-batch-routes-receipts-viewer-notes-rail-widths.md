---
id: TASK-2858
title: 'Library UAT P2 batch: routes, receipts, viewer, notes, rail, widths'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 12:45'
labels:
  - library
  - ux
  - uat-2026-08-06
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 P2 findings (LIB-03/09/11/12/13/14/15/17/18/19), critique snapshot
`.impeccable/critique/2026-08-07T01-01-42Z__tldw-chatbook-ui-screens-library-screen-py.md`,
observed at dev `6ffa56516`. Grouped for one pass; split if any item grows.

1. LIB-03 — Entry-route-dependent landing: palette "Library" lands the Import canvas while
   "Switch to Library" lands the hub; re-entering Library resets the previously visited canvas.
2. LIB-09 — Help/advertisement contract: F1 on the Media canvas lists skills/evidence keys that
   do nothing there; media-viewer footer advertises `u` which is inert in the viewer.
3. LIB-11 — Empty "Export chatbook" click is a silent no-op (no toast/disabled styling/reason).
4. LIB-12 — Successful export leaves no durable receipt (zip written; canvas pixel-identical).
5. LIB-13 — Media viewer renders raw markdown while Notes Preview renders it properly.
6. LIB-14 — Note lifecycle: "Blank note" commits a DB row before typing; literal "Untitled" text
   must be hand-deleted; version bumps from clicking Preview.
7. LIB-15 — Rail gloss/count lifecycle is nondeterministic ("Collections — item sets" → "(0)" →
   bare → "(1)"; some rows keep glosses, some lose them).
8. LIB-17 — Click into a prefilled query lands the cursor at position 0 (typed text prepends);
   the rail search box retains stale queries across screen switches.
9. LIB-18 — Width degradation: row labels truncate mid-word at ≤120 ("Conversa... (0)",
   "Flash... due: 0"); at ≤100 the footer's screen-specific keys hide behind a leading "…"; at 80
   the nav hard-cuts a tab label mid-word.
10. LIB-19 — Three folder-notes concepts (Database mode, Files mode, Sync) are never related to
    each other anywhere; at minimum one sentence on each should place it relative to the others.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library landing is route-independent (one canonical landing) or intentionally routed with the difference stated on-screen; revisits restore the last canvas or intentionally reset (decision recorded)
- [ ] #2 F1 and footers advertise only keys that work on the current surface
- [ ] #3 The Export button is never a silent no-op, and a successful export leaves a durable on-canvas receipt with the output path
- [ ] #4 The media viewer renders markdown (with a raw toggle) for markdown media
- [ ] #5 Blank notes no longer commit literal "Untitled" rows that require hand-deletion; version stamps change only on content saves
- [ ] #6 Rail glosses/counts follow one deterministic rule across all rows
- [ ] #7 Prefilled search inputs are editable without cursor traps, and stale rail queries do not survive screen switches
- [ ] #8 At 120/100/80 columns no rail row label truncates mid-word, and each finding's surface is re-verified live
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Four-group SDD arc on fix/library-uat-p2-batch (plan: docs/superpowers/plans/library-uat-p2-batch.md):\n1. Entry routing + canvas restoration (LIB-03, AC1) — generic entries land one canonical surface, deep links keep their labeled destinations, revisits restore.\n2. Honest advertisement + export feedback (LIB-09/11/12, AC2-3) — full BINDINGS gate audit, disabled-with-reason export button, durable last-export receipt.\n3. Content surfaces (LIB-13/14, AC4-5) — viewer markdown via the existing Notes renderer + Raw toggle; note lifecycle (placeholder title, no premature commit, version on save only).\n4. Rail determinism + input traps + widths + folder-notes copy (LIB-15/17/18/19, AC6-8) + close-out.\nEvery item re-verified at HEAD first (P1 arc + task-1993 may have moved several). Same process as the P1 arc: TDD, task review + scoped re-reviews per fix round, live tmux verification per task, final whole-branch review.
<!-- SECTION:PLAN:END -->
