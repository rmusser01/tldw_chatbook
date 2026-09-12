---
id: TASK-32499
title: 'Agent-routing polish bundle: unnamed-refusal loop counter, error level surfacing, helper placement (ADR-147 follow-up)'
status: To Do
assignee: []
created_date: '2026-09-12 07:55'
labels:
  - agents
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-32477 (agent provider routing, final-review minors M-1/M-2/M-3). Three small items, one PR: (1) agent_runtime.py:1473-1474 increments the loop's secondary spawn counter on `result.ok or not agent_name`, so refused UNNAMED spawns burn the loop bound and after max_subagents refused generic spawns, legal spawns see "budget exhausted" with zero children admitted (fail-closed, turn-bounded, pre-existing shape — but the routing feature made refusals common; the authoritative service counter is untouched). (2) RoutingError.level (override/preset/default) exists but is not surfaced in refusal strings (agent_service.py SpawnAdmissionRefusal f"[{code}] {err}"; settings panel f"{label} -> [{code}] {exc}") — spec says each error names the failing level; include it, which also softens the inherit-readiness UX (an inherit refusal then reads as "the parent's provider just became unready"). (3) parse_params_text lives in Widgets/settings_agents_panel.py and is imported by Widgets/Console/console_endpoint_template_modal.py — move it to Chat/sampling_params.py or a forms helper and update both import sites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Refused unnamed spawns do not consume the loop's secondary spawn bound (test pinned)
- [ ] #2 Refusal messages surface the failing level (override/preset/default/inherit) in both spawn and settings-panel paths
- [ ] #3 parse_params_text moved to a shared module; both widgets import it; suites green
<!-- AC:END -->
