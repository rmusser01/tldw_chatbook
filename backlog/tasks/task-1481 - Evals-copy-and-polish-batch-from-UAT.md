---
id: TASK-1481
title: >-
  Evals copy and polish batch from UAT
status: Done
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small fixes found by live UAT (2026-07-30), batched:

1. Detail/inspector copy says "library rail" but the rail is labeled "Catalog" (`evals_screen.py` empty-state and blocked-reason strings).
2. The same strings use ASCII `--` where the rail copy uses real em-dashes.
3. The Δ baseline lens on a single-target run renders every cell as the word "baseline" with an empty Spread column; it should state that the lens needs a second target to compare against.
4. The snippet table header advertises columns (`#  Snippet  Group  Chars  Flags`) that do not align with the actual row layout (meta renders as a right-aligned blob).
5. Stale comment `lab_frame.py:91` claims EvalsScreen binds Escape to a back action; it does not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] No user-facing Evals string refers to the "library rail"; copy names the rail as it is labeled
- [x] User-facing Evals copy uses em-dashes consistently (no ASCII `--` in rendered strings)
- [x] A single-target run's Δ baseline lens states the lens needs at least two targets
- [x] The snippet table's header matches what its rows actually render
- [x] The stale Escape comment in `lab_frame.py` is corrected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rename "library rail" to the painted "Catalog" name in rendered strings; em-dash sweep
2. Explain the Δ lens instead of rendering a column of the word "baseline"
3. Align the snippet table header with its rows; correct the stale Escape comment
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commits b439fdde7, 0b76e4b19. Rendered copy now names the rail "Catalog rail" (three sites) and uses em-dashes throughout (five sites — the fix round completed the sweep into degenerate_canary_text, cost_text, and a RuntimeError that reaches a toast). The Δ-lens guard was narrowed after review: it fires ONLY for column-baseline mode with <2 targets ("needs at least two targets to compare"), because row-baseline single-target Δ computes genuine per-snippet divergence and the blanket gate had silently removed it (and blanked failed cells' em-dash mark); a scope-inversion mutation proves the gate discriminates the modes. Snippet header now mirrors the actual row shape ("#   Snippet   Group · Chars · Flags"). The lab_frame comment states the verified truth: Escape is unbound throughout the destination — nothing in the chain handles it.
<!-- SECTION:NOTES:END -->
