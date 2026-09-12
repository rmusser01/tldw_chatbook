---
id: TASK-32497
title: 'Agent rail: show each child''s resolved target (ADR-147 follow-up)'
status: To Do
assignee: []
created_date: '2026-09-12 07:55'
labels:
  - agents
  - console
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-32477 (agent provider routing, ADR-147, final-review finding I-1). Spec §Error handling promises "The rail summary shows each child's resolved target (e.g. 'qwen-local · qwen3.8-27b')" and the user guide repeats the claim — but nothing implements it: SubAgentSummary (console_agent_bridge.py:1079-1107) carries only text/status/run_id/handle_id, and _fleet_row_from_summary/_fleet_row_from_record (UI/Console_Modules/agent.py:397-443) render no target (secondary_text=""). Plumb the resolved target (from the run row's v16 resolved_* snapshot, falling back to the live definition for legacy rows) through FleetHandle/SubAgentSummary into the rail row's secondary_text, or correct the guide sentence if the rail format is deliberately deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each live/finished child's rail line shows its resolved target (provider · model) or the guide claim is corrected
- [ ] #2 Legacy (NULL-snapshot) rows render something honest (no fabricated target)
- [ ] #3 Widget/pilot test pins the rendered target for a routed child
<!-- AC:END -->
