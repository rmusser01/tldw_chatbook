---
id: TASK-2960
title: MCP forced-loading parity regression from the PR-1385 tall-section change
status: To Do
assignee: []
created_date: '2026-08-07 21:00'
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
- [ ] The forced-loading MCP triad fits the viewport again (or the parity contract is deliberately re-pinned to the new geometry, with the rationale recorded).
- [ ] The tall-section reachability behavior from `e9dbaf6e0` is preserved.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
