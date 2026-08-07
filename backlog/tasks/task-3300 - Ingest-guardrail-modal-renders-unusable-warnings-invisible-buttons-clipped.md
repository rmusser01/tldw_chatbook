---
id: TASK-3300
title: >-
  Ingest guardrail modal renders unusable — warnings invisible, buttons clipped
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - ux
  - uat
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finding MI-01 (P0) of the 2026-08-07 Media Ingestion UX review (dual-agent live critique; tracking file `.impeccable/critique/2026-08-07-media-ingest-ux-options-review.md` in the main checkout). The Library ingest surface's only blocking consent dialog — `IngestGuardrailModal` (`library_screen.py:980`) — opened live as an empty black full-height column: "Some files may fail to import:" and the per-warning lines never rendered, because each warning sits in a bare `with Vertical():` (default `height: 1fr`) inside the `height: auto` modal, starving every Static (the exact "bare Container starving its sibling" defect DESIGN.md §7 names). The confirm label "Start import anyway" wraps to two lines inside `width: 14` while Cancel doesn't (misaligned baselines). The modal uses off-token `background: black; border: tall gray`. The warning line renders `({count} files)` — "(1 files)". Cancel carries `variant="error"` (a red safe action).

The user is asked to consent to a partially-doomed ingest they cannot read about — a "no hidden recovery states" violation at the surface's highest-stakes moment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 With one and with three tooling warnings, every warning line and its Copy-install-command button are visible inside a compact modal (no full-height empty band), verified by a rendered-geometry test that fails on the pre-fix CSS
- [ ] #2 Both action buttons render their full labels on one line with aligned baselines
- [ ] #3 Modal styling uses theme tokens (no `background: black` / `border: tall gray` literals)
- [ ] #4 Warning count is grammatically correct for 1 vs many ("1 file", "2 files")
- [ ] #5 Cancel is not styled as the destructive action; the confirm carries the action emphasis
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the defect on the worktree (dev tip 023a04a48) with a mounted-modal test asserting warning Statics have nonzero rendered height — expect RED.
2. `height: auto` on the per-warning Verticals (or drop the wrapper); widen/auto-size the confirm button; replace color literals with `$panel`/`$surface` + border tokens; fix plural; swap variant emphasis.
3. Rendered-geometry assertions (region heights, label single-line) under a CSS-true harness; mutation-check by reverting the height rule and confirming RED.
<!-- SECTION:PLAN:END -->
