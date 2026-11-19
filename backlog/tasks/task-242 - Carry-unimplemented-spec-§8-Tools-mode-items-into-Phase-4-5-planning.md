---
id: TASK-242
title: Carry unimplemented spec-§8 Tools-mode items into Phase 4/5 planning
status: Done
assignee: []
created_date: '2026-07-16 15:20'
labels:
  - mcp
  - phase4
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec §8 items neither implemented in Phase 3 nor recorded in its out-of-scope list: inspector detail 'server + transport, last used / last error' lines, a risk filter in the filter bar, and a 'needs auth → Open credentials' empty-state bucket. Recorded here so Phase 4/5 planning picks them up deliberately (PR #639 final review N3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each item lands in a Phase 4 or Phase 5 plan or is explicitly descoped in the spec
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Two of the three spec §8 items landed in the shipped Phase 5 surface: the risk/severity filter (mcp_audit_mode.py Severity column + _severity_kind, T8) and the needs-auth affordance (HubAction.OPEN_CREDENTIALS, mcp_inspector.py:270); the 'last used / last error' inspector lines were overtaken by the redesign's audit canvas.
