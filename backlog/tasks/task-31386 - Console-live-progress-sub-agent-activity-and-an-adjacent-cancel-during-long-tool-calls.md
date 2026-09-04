---
id: TASK-31386
title: >-
  Console live progress: sub-agent activity and an adjacent cancel during long
  tool calls
status: To Do
assignee: []
created_date: '2026-09-04 19:29'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sub-project E of the design spec (2026-08-19-console-user-interaction-design.md section 4) asked for 'what the agent is doing now, elapsed time, visible cancel during long tool calls'. Part of it has since shipped: the unfinished Assistant row shows a live activity line during a turn (tool name and elapsed seconds, then Thinking and Generating states; see Docs/User_Guide/console/agent-runs-and-tools.md 'In the reply row itself'), and the composer's Stop button cancels the run. What remains from E: a sub-agent's work never appears in that line, so a fleet turn reads as idle while children run; and the Stop button is the run-wide stop, with no per-tool-call cancel next to the activity line for a single long tool call the user wants abandoned without ending the turn. This task is the residual, re-scoped against what exists rather than the spec's original ground truth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 During a fleet turn the activity line reflects the running child agents (count and the longest-running tool) rather than reading idle
- [ ] #2 A long-running tool call exposes a cancel adjacent to the activity line that abandons that call and lets the turn continue, using the existing per-call abandon path
- [ ] #3 Single-agent turns are unchanged
<!-- AC:END -->
