---
id: TASK-2960
title: MCP forced-loading parity regression from the PR-1385 tall-section change
status: Done
assignee: []
created_date: '2026-08-07 21:00'
updated_date: '2026-11-12 19:38'
labels:
  - mcp
  - tests
  - dev-baseline
  - regression
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`test_destination_visual_parity_correction.py::test_mcp_forced_loading_state_stays_inside_workbench` fails on current dev: the MCP inspector renders 45 rows tall at y=6 in a 42-row viewport (`assert (6 + 45) <= 42`), so the triad's geometry contract breaks in the forced-loading state.

Bisected across worktrees running the single test: **passes at `acaae68e9` (#1389), fails at `39232202b` (#1385, feat/rag-truth-mcp-honesty)** — introduced inside that ~70-commit merge, most plausibly `e9dbaf6e0` ("fix(mcp): the Advanced runner is reachable again under a tall section payload"), which deliberately changed inspector/section height behavior. Explicitly NOT from the defer-past-first-paint series: #1392/#1395/#1396 were probed individually and are clean for this test.

This needs the PR-1385 authors' design call, not a blind fix: either the tall-section reachability change must respect the viewport contract (scroll inside the inspector rather than growing it), or the parity contract must be updated to the new intended geometry — the commit history suggests the reachability behavior was deliberate.

For context, the other five failures in this parity file (Schedules geometry ×4 + nav overflow hint) predate `b0185749c` — older baseline debt, separate from this regression.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] The forced-loading MCP triad fits the viewport again (or the parity contract is deliberately re-pinned to the new geometry, with the rationale recorded).
- [x] The tall-section reachability behavior from `e9dbaf6e0` is preserved.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

**Closed 2026-11-12 (round-13 closeout).** At the round-13 stack top 8b633d0101 both pins pass (run.sh, `-n 4`, load ~9-11): `Tests/UI/test_destination_visual_parity_correction.py::test_mcp_forced_loading_state_stays_inside_workbench` and `Tests/UI/test_mcp_inspector.py::test_run_action_is_reachable_under_a_tall_section_payload`.
- AC#1: MCP now lives in Settings > Tools & MCP (ADR-240), and the forced-loading test pins that stacked layout (`test_destination_visual_parity_correction.py:2333-2392`); it passed in every TASK-39170 run and is on the UI census (`scripts/ui_pr_gate_census.txt:2845`).
- AC#2: the tall-section reachability pin passes. Its file runs on no PR lane; that is not this task's criterion.
