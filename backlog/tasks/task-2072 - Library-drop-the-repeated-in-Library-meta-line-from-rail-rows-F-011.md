---
id: TASK-2072
title: 'Library: drop the repeated ''in Library'' meta line from rail rows (F-011)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 02:15'
labels:
  - ux-review
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every rail row repeats 'in Library' on a second line, making rows 3 lines tall; Create is unreachable at 100x30 and Details is clipped even at 170x50. Evidence: library_rail.py:221-226. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rail rows are one line by default,Meta line remains only where it discriminates handoff rows,Create section and Details are reachable at 100x30,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (rail row presentation + bundle CSS). Steps: 1. RED tests: rail row labels have no second line by default ('in Library' gone); handoff rows keep a discriminating meta line ('opens Study'); rendered 100x30 test asserting Create section header and Details status group are inside the viewport. 2. library_rail.py compose: one-line label unless target_kind == 'handoff'; inline height 1 (2 for handoffs). 3. Bundle .library-rail-row height/min-height 2 -> 1 and drop the per-row bottom margin (the 3-line-tall math: 2 height + 1 margin); regenerate tldw_cli_modular.tcss; keep lockstep comment. 4. Run rail/shell/parity/destination tests + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One-line rail rows by default: the blanket second line ('in Library' on all ~11 rows) was pure stutter and the reason Create was unreachable at 100x30 and Details clipped at 170x50. A meta line survives only where it discriminates -- the three Study handoff rows keep 'opens Study'. library_rail.py keys the meta line on target_kind == 'handoff' (no 'screen' rows exist); inline height 1 (2 for handoffs). Bundle CSS .library-rail-row height/min-height 2 -> 1, per-row bottom margin dropped; tldw_cli_modular.tcss regenerated via build_css.py. Tests: new F-011 pins in Tests/UI/test_library_shell.py (one-line default + handoff discriminator; Create header and Details status reachable at 100x30 rendered test). Also hardened _open_library_ingest_canvas to wait for the rail row before pressing (snapshot recompose race surfaced by F-010) -- cures test_library_ingest_clear_finished_requires_second_press and test_preflight_forecasts_already_ingested_text_files. Verified: rail/shell-state/shell files green (3 timing-flaky notes tests pass in isolation; 3 test_library_screen.py failures pre-exist on dev, confirmed at F-010 parent). Ruff clean on changed files (1 pre-existing F401 in test_library_shell.py untouched). ADR: not required (presentation + copy only, no schema/boundary changes). Commit 05cc32a0e.
<!-- SECTION:NOTES:END -->
